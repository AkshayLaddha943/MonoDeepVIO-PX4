import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import utm
import message_filters

imu_topic = "/world/aruco/model/x500_mono_cam_down_0/link/base_link/sensor/imu_sensor/imu"
gps_topic = "/world/aruco/model/x500_mono_cam_down_0/link/base_link/sensor/navsat_sensor/navsat"

class IMUEstimator(Node):
    def __init__(self,):
        super().__init__('imu')

        self.imu_path = Path()
        self.gps_path = Path()
        self.init_msg = Imu()

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        self.orientation = None

        self.imu_filter = message_filters.SimpleFilter()
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, qos_profile_sensor_data)
        self.gps_sub = self.create_subscription(NavSatFix, gps_topic, self.navsat_callback, 10)

        self.gps_filter = message_filters.Subscriber(self, NavSatFix, gps_topic, qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.imu_filter, self.gps_filter], queue_size=100, slop=0.02)
        self.sync.registerCallback(self.sync_callback)

        self.pose_pub = self.create_publisher(Path, "/ins/path", 10)
        self.gt_pub = self.create_publisher(Path, "/gps/path", 10)

        self.imu_poses = []
        self.gt_poses = []

        self.synced_imu = []
        self.synced_gps = []
        self.synced_timestamp = []

        self.gravity = np.array([0.0, 0.0, -9.81])
        self.last_time = None

        #to add bias to gyro and accel model (optional)
        self.range = np.random.default_rng(42)
        self.gyro_bias = self.range.normal(0.0, np.deg2rad(0.1), size=3)
        self.accel_bias = self.range.normal(0.0, 0.005 * 9.80665, size=3)

        self.origin_x = None
        self.origin_y = None
        self.origin_z = None

        plt.ion()

        self.fig, self.ax = plt.subplots()

        self.imu_line, = self.ax.plot([], [], 'b-', label='IMU')
        self.gt_line, = self.ax.plot([], [], 'r-', label='GPS')

        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.set_title("IMU vs GPS Trajectory")
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.grid(True)
        self.ax.legend()

        self.create_timer(0.2, self.plot)

    def imu_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.orientation is None:
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            if np.linalg.norm(q) < 1e-8:
                return

            self.orientation = R.from_quat(q)
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t
        if dt <= 0:
            return

        #orientation from gyroscope-integration
        # omega = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        
        # delta_theta = omega * dt
        # delta_rot = R.from_rotvec(delta_theta)
        # self.orientation *= delta_rot

        #orientation in quaternion from Gazebo directly
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        orientation_ref = R.from_quat(q)

        #accel (body frame -> navigation frame)
        accel_body = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]) + self.accel_bias
        accel_enu = orientation_ref.apply(accel_body)

        accel_net = accel_enu + self.gravity

        #position
        self.position = self.position + self.velocity * dt + 0.5 * accel_net * dt**2
        self.velocity += accel_net * dt

        self.publish_imu_path(msg.header.stamp)

    def navsat_callback(self, msg: NavSatFix):
        utm_x, utm_y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        utm_z = msg.altitude

        if self.origin_x is None:
            self.origin_x = utm_x
            self.origin_y = utm_y
            self.origin_z = utm_z
            self.get_logger().info(f'Origin set to UTM X: {utm_x}, Y: {utm_y}')

        self.local_x = utm_x - self.origin_x
        self.local_y = utm_y - self.origin_y
        self.local_z = utm_z - self.origin_z

        self.gt_poses.append([self.local_x, self.local_y, self.local_z])
        self.publish_gps_path(msg.header.stamp)
    
    def plot(self):
        if len(self.imu_poses) >= 2:
            imu_traj = np.array(self.imu_poses)

            self.imu_line.set_data(imu_traj[:, 0], imu_traj[:, 1])

        if len(self.gt_poses) >= 2:
            gt_traj = np.array(self.gt_poses)

            self.gt_line.set_data(gt_traj[:, 0], gt_traj[:, 1])

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def publish_imu_path(self, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        
        pose.pose.position.x = float(self.position[0])
        pose.pose.position.y = float(self.position[1])
        pose.pose.position.z = float(self.position[2])

        qx, qy, qz, qw = self.orientation.as_quat()
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        self.imu_path.header.stamp = stamp
        self.imu_path.header.frame_id = "map"

        self.imu_poses.append(self.position.copy())
        self.imu_filter.signalMessage(pose)

        if len(self.imu_poses) % 25 == 0:
            self.imu_path.header.stamp = stamp
            self.imu_path.header.frame_id = "map"

            self.imu_path.poses.append(pose)

            self.pose_pub.publish(self.imu_path)

    def publish_gps_path(self, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        
        pose.pose.position.x = self.local_x
        pose.pose.position.y = self.local_y
        pose.pose.position.z = self.local_z

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.gps_path.header.stamp = stamp
        self.gps_path.header.frame_id = "map"
        self.gps_path.poses.append(pose)

        self.gt_pub.publish(self.gps_path)

    def sync_callback(self,imu_msg, gps_msg):
        imu_pos = np.array([imu_msg.pose.position.x, imu_msg.pose.position.y, imu_msg.pose.position.z])

        utm_x, utm_y, _, _ = utm.from_latlon(gps_msg.latitude, gps_msg.longitude)
        gps_pos = np.array([utm_x, utm_y, gps_msg.altitude])

        t = (gps_msg.header.stamp.sec + gps_msg.header.stamp.nanosec * 1e-9)

        # Store synchronized data
        self.synced_imu.append(imu_pos)
        self.synced_gps.append(gps_pos)
        self.synced_timestamp.append(t)

    def metrics(self,):
        if len(self.synced_timestamp) < 2:
            self.get_logger().warn("Insufficient synchronized IMU-GPS measurements")
            return
    
        gps = np.array(self.synced_gps)
        pred_ins = np.array(self.synced_imu)
        time = np.array(self.synced_timestamp)

        order = np.argsort(time)

        pred_ins = pred_ins[order]
        gps = gps[order]
        time = time[order]

        pred_ins = pred_ins - pred_ins[0]
        gps = gps - gps[0]
        time = time - time[0]

        #ate
        ate = np.linalg.norm(gps - pred_ins, axis=1)
        
        #rmse
        rmse = np.sqrt(np.mean((ate) ** 2))
        final_error = ate[-1]

        self.get_logger().info(f"ATE RMSE: {rmse:.3f} m")
        self.get_logger().info(f"Final Position Error: {final_error:.3f} m")

        #position error wrt time
        plt.figure()
        plt.plot(time, ate)

        plt.xlabel("Time (s)")
        plt.ylabel("Position Error (m)")
        plt.title("IMU Position Error vs Time")
        plt.grid(True)

def main(args=None):
    rclpy.init(args=args)
    node = IMUEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.metrics()
        plt.ioff()
        plt.show(block=True)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()