from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import utm

cam_topic = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image"
gps_topic = "/world/aruco/model/x500_mono_cam_down_0/link/base_link/sensor/navsat_sensor/navsat"

FLANN_INDEX_LSH = 6
index_param = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_param = dict(checks=50)

class VisualOdometry(Node):
    def __init__(self, ):
        super().__init__('vo')

        self.bridge = CvBridge()
        self.vo_traj = Path()
        self.gps_traj = Path()

        self.camera_height = 45.72

        q_world_body = [0.00099683850809800432,
                        0.017431699200411887,
                        0.318973505651108,
                        0.9476027885504652]

        # Drone body -> Gazebo ENU world
        R_world_body = R.from_quat(q_world_body).as_matrix()

        # Camera link -> drone body
        R_body_camera = R.from_euler("xyz", [0.0, 1.5707, 0.0]).as_matrix()

        # OpenCV optical -> Gazebo camera link
        R_camera_optical = np.array([[0,  0,  1],
                                     [-1, 0,  0],
                                     [0, -1,  0]], dtype=np.float64)

        # Initial camera optical frame -> ENU
        self.R_camera_to_enu = (R_world_body @ R_body_camera @ R_camera_optical)

        #subscriber
        self.image_sub = self.create_subscription(Image, cam_topic, self.image_callback, qos_profile_sensor_data)
        # self.gps_sub = self.create_subscription(NavSatFix, gps_topic, self.navsat_callback, 10)

        #publisher
        self.vo_traj_pub = self.create_publisher(Path, "/vo/path", 10)
        # self.gps_traj_pub = self.create_publisher(Path, "/gps/path", 10) 

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_bgr: Optional[np.ndarray] = None
        self.prev_kp: Optional[List[cv2.KeyPoint]] = None
        self.prev_des: Optional[np.ndarray] = None

        #orb feature detector
        self.orb = cv2.ORB_create(nfeatures=3000)

        #kd tree matcher
        self.flann = cv2.FlannBasedMatcher(index_param, search_param)

        self.pose = np.eye(4, dtype=np.float64)
        self.vo_trajectory = []
        # self.gt_trajectory = []

        self.K = np.array([[539.9363327026367, 0.0, 640.0], [0.0, 539.9363708496094, 480.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        self.origin_x = None
        self.origin_y = None
        self.origin_z = None

        # 2D trajectory visualization
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(9, 7))

        self.ax.set_title("Visual Odom - ORB")
        self.ax.set_xlabel("X Displacement (m)")
        self.ax.set_ylabel("Y Displacement (m)")

        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_aspect("equal", adjustable="box")

        self.vo_line, = self.ax.plot([], [], color="orange", linewidth=1.5, label="VO Trajectory")
 #       self.gt_line, = self.ax.plot([], [], color="green", linewidth=1.5, label="GT Trajectory")

        self.vo_start, = self.ax.plot([], [], "bo", markersize=6)
#        self.gt_start, = self.ax.plot([], [], "go", markersize=6)

        self.vo_current, = self.ax.plot([], [], "ro", markersize=5)
        # self.gt_current, = self.ax.plot([], [], "mo", markersize=5)

        self.ax.legend(loc="upper right")
        plt.show(block=False)

    def extract_features(self, gray_frame):
        #clahe
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_frame = clahe.apply(gray_frame)

        #orb
        kp_frame, desc_frame = self.orb.detectAndCompute(enhanced_frame, None)

        return kp_frame, desc_frame

    def image_callback(self, msg:Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"cv_bridge conversion failed: {exc}")
            return
        
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypts, desc = self.extract_features(gray_frame)

        if self.prev_des is None:
            self.prev_gray = gray_frame
            self.prev_bgr = image.copy()
            self.prev_kp = keypts
            self.prev_des = desc
            return

        if desc is None or self.prev_des is None:
            self.get_logger().debug("Skipping frame with missing ORB descriptors")
            return

        matches = self.flann.knnMatch(self.prev_des, desc, k=2)
        good_match = []

        for pair in matches:
            if len(pair) < 2:
                continue
            m,n = pair
            if m.distance < float(0.6) * n.distance:
                good_match.append(m)

        if len(good_match) > 12:
            src_pts = np.array([self.prev_kp[m.queryIdx].pt for m in good_match])
            dst_pts = np.array([keypts[m.trainIdx].pt for m in good_match])

            # E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, confidence=0.995, maxIters=5000,)

            if H is not None and mask is not None:
                inlier_mask = mask.ravel().astype(bool)

                src_inliers = src_pts[inlier_mask]

                num_inliers = len(src_inliers)

                if num_inliers >= 12:
                    num_sol, rot, trans, normals = cv2.decomposeHomographyMat(H, self.K)

                    best_idx = None
                    best_score = float("inf")

                    expected_norm = np.array([0.0, 0.0, 1.0])

                    for i in range(num_sol):
                        normal = normals[i].reshape(3)
                        normal /= (np.linalg.norm(normal) + 1e-12)

                        score = min(np.linalg.norm(normal - expected_norm),
                            np.linalg.norm(normal + expected_norm))

                        if score < best_score:
                            best_score = score
                            best_idx = i

                    if best_idx is not None:
                        R_rel = rot[best_idx]
                        t_rel = (trans[best_idx].reshape(3) * self.camera_height)

                    self.rel_pose = np.eye(4)

                    self.rel_pose[:3, :3] = R_rel
                    self.rel_pose[:3, 3] = t_rel.reshape(3)
                    self.pose = self.pose @ np.linalg.inv(self.rel_pose)
                    position = self.pose[:3, 3].copy()

                    position_enu = (self.R_camera_to_enu @ position)

                    self.vo_trajectory.append(position_enu.copy())

                    self.display()
                    self.publish_traj(msg.header.stamp, self.pose)

        self.prev_gray = gray_frame
        self.prev_bgr = image.copy()
        self.prev_kp = keypts
        self.prev_des = desc

    def display(self):
        if len(self.vo_trajectory) >= 2:
            vo_path = np.array(self.vo_trajectory)

            x_vo = vo_path[:, 0]
            y_vo = vo_path[:, 1]

            self.vo_line.set_data(x_vo, y_vo)

            self.vo_start.set_data([x_vo[0]], [y_vo[0]])
            self.vo_current.set_data([x_vo[-1]], [y_vo[-1]])

        # if len(self.gt_trajectory) >= 2:
        #     gt_path = np.array(self.gt_trajectory)

        #     x_gt = gt_path[:, 0]
        #     y_gt = gt_path[:, 1]

        #     self.gt_line.set_data(x_gt, y_gt)

        #     self.gt_start.set_data([x_gt[0]], [y_gt[0]])
        #     self.gt_current.set_data([x_gt[-1]], [y_gt[-1]])

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def publish_traj(self, stamp, curr_pose):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"

        position_vo = curr_pose[:3, 3]
        position_enu = (self.R_camera_to_enu @ position_vo)

        pose.pose.position.x = float(position_enu[0])
        pose.pose.position.y = float(position_enu[1])
        pose.pose.position.z = float(position_enu[2])

        R_vo = curr_pose[:3, :3]
        R_enu = (self.R_camera_to_enu @ R_vo)

        qw, qx, qy, qz = R.from_matrix(R_enu).as_quat(scalar_first=True)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        self.vo_traj.header.stamp = stamp
        self.vo_traj.header.frame_id = "map"
        self.vo_traj.poses.append(pose)
        self.vo_traj_pub.publish(self.vo_traj)

    # def navsat_callback(self, msg: NavSatFix):
    #     utm_x, utm_y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
    #     utm_z = msg.altitude

    #     if self.origin_x is None:
    #         self.origin_x = utm_x
    #         self.origin_y = utm_y
    #         self.origin_z = utm_z
    #         self.get_logger().info(f'Origin set to UTM X: {utm_x}, Y: {utm_y}')

    #     local_x = utm_x - self.origin_x
    #     local_y = utm_y - self.origin_y
    #     local_z = utm_z - self.origin_z

    #     # Create a new PoseStamped point for the trajectory
    #     pose = PoseStamped()
    #     pose.header = msg.header
    #     pose.header.frame_id = 'map'
        
    #     pose.pose.position.x = local_x
    #     pose.pose.position.y = local_y
    #     pose.pose.position.z = local_z
        
    #     # GPS does not give orientation, default to identity quaternion
    #     pose.pose.orientation.x = 0.0
    #     pose.pose.orientation.y = 0.0
    #     pose.pose.orientation.z = 0.0
    #     pose.pose.orientation.w = 1.0

    #     # Append the new pose to our ongoing path array
    #     self.gps_traj.header.stamp = msg.header.stamp
    #     self.gps_traj.header.frame_id = "map"
    #     self.gps_traj.poses.append(pose)

    #     # Publish the updated path
    #     self.gps_traj_pub.publish(self.gps_traj)

    #     self.gt_trajectory.append(np.array([local_x, local_y, local_z], dtype=np.float64))

        self.display()

    def metrics(self,):
        pass

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()