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

        self.camera_height = 60.96

        #subscriber
        self.image_sub = self.create_subscription(Image, cam_topic, self.image_callback, qos_profile_sensor_data)

        #publisher
        self.vo_traj_pub = self.create_publisher(Path, "/vo/path", 10)

        self.prev_gray = None
        self.prev_pts = None

        #feature params
        self.feature_params = dict(maxCorners=300,
                             qualityLevel=0.01,
                             minDistance=7,
                             blockSize=7)

        #opticalflow params
        self.lk_params = dict(winSize = (21,21),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.pose = np.eye(4, dtype=np.float64)
        self.vo_trajectory = []

        self.K = np.array([[539.9363327026367, 0.0, 640.0], [0.0, 539.9363708496094, 480.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        self.origin_x = None
        self.origin_y = None
        self.origin_z = None

        # 2D trajectory visualization
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(9, 7))

        self.ax.set_title("Visual Odom")
        self.ax.set_xlabel("X Displacement (m)")
        self.ax.set_ylabel("Y Displacement (m)")

        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_aspect("equal", adjustable="box")

        self.vo_line, = self.ax.plot([], [], color="orange", linewidth=1.5, label="VO Trajectory")
        self.vo_start, = self.ax.plot([], [], "bo", markersize=6)
        self.vo_current, = self.ax.plot([], [], "ro", markersize=5)

        self.ax.legend(loc="upper right")
        plt.show(block=False)

    def extract_features(self, gray_frame):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_frame = clahe.apply(gray_frame)

        mask = np.full_like(gray_frame, 255)

        #shi-tomasi
        points = cv2.goodFeaturesToTrack(enhanced_frame, mask=mask, **self.feature_params)

        return points

    def image_callback(self, msg:Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"cv_bridge conversion failed: {exc}")
            return
        
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_pts = self.extract_features(gray_frame)
            self.prev_gray = gray_frame
            return

        if self.prev_pts is None or len(self.prev_pts) < 4:
            self.prev_pts = self.extract_features(gray_frame)
            self.prev_gray = gray_frame
            return

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_frame, self.prev_pts, None, **self.lk_params)

        if p1 is None or st is None:
            self.prev_pts = self.extract_features(gray_frame)
            self.prev_gray = gray_frame
            return
        
        valid = st.reshape(-1) == 1

        src_pts = self.prev_pts.reshape(-1, 2)[valid]
        dst_pts = p1.reshape(-1, 2)[valid]

        finite = (np.isfinite(src_pts).all(axis=1) & np.isfinite(dst_pts).all(axis=1))

        src_pts = src_pts[finite]
        dst_pts = dst_pts[finite]

        if len(src_pts) >= 12:
            # E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None and mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                num_inliers = np.count_nonzero(inlier_mask)
                inlier_ratio = num_inliers/len(src_pts)

                if num_inliers >= 12 and inlier_ratio >= 0.5:
                    num_sol, rot, trans, normals = cv2.decomposeHomographyMat(H, self.K)

                    best_idx = None
                    best_score = float("inf")

                    expected_norm = np.array([0.0, 0.0, 1.0])

                    for i in range(num_sol):
                        normal = normals[i].reshape(3)
                        normal /= (np.linalg.norm(normal) + 1e-12)

                        score = min(np.linalg.norm(normal - expected_norm), np.linalg.norm(normal + expected_norm))

                        if score < best_score:
                            best_score = score
                            best_idx = i

                    if best_idx is not None:
                        R_rel = rot[best_idx]
                        t_rel = (trans[best_idx].reshape(3) * self.camera_height)

                    self.rel_pose = np.eye(4, dtype=np.float64)

                    self.rel_pose[:3, :3] = R_rel
                    self.rel_pose[:3, 3] = t_rel

                    self.pose = self.pose @ np.linalg.inv(self.rel_pose)
                    position = self.pose[:3, 3].copy()

                    self.vo_trajectory.append(position.copy())

                    self.display()
                    self.publish_traj(msg.header.stamp)

        curr_pts = dst_pts.astype(np.float32).reshape(-1, 1, 2)

        if len(curr_pts) < 100:
            new_pts = self.extract_features(gray_frame)

            if new_pts is not None:
                curr_pts = np.concatenate((curr_pts, new_pts), axis=0)

        self.prev_pts = curr_pts
        self.prev_gray = gray_frame

    def display(self):
        if len(self.vo_trajectory) >= 2:
            vo_path = np.array(self.vo_trajectory)

            x_vo = vo_path[:, 0]
            y_vo = vo_path[:, 1]

            self.vo_line.set_data(x_vo, y_vo)

            self.vo_start.set_data([x_vo[0]], [y_vo[0]])
            self.vo_current.set_data([x_vo[-1]], [y_vo[-1]])

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def publish_traj(self, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"

        position_vo = self.pose[:3, 3]

        pose.pose.position.x = float(position_vo[0])
        pose.pose.position.y = float(position_vo[1])
        pose.pose.position.z = float(-position_vo[2])
        R_vo = self.pose[:3, :3]

        qw, qx, qy, qz = R.from_matrix(R_vo).as_quat(scalar_first=True)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        self.vo_traj.header.stamp = stamp
        self.vo_traj.header.frame_id = "map"
        self.vo_traj.poses.append(pose)
        self.vo_traj_pub.publish(self.vo_traj)

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