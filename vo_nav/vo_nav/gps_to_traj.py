import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import utm

gps_topic = "/world/baylands/model/x500_mono_cam_down_0/link/base_link/sensor/navsat_sensor/navsat"

class GPSTrajPublisher(Node):
    def __init__(self,):
        super().__init__('gps_traj')

        self.gps_sub = self.create_subscription(NavSatFix, gps_topic, self.navsat_callback, 10)
        self.traj_pub = self.create_publisher(Path, "/gps/path", 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        self.origin_x = None
        self.origin_y = None
        self.origin_z = None

    def navsat_callback(self, msg: NavSatFix):
        utm_x, utm_y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        utm_z = msg.altitude

        if self.origin_x is None:
            self.origin_x = utm_x
            self.origin_y = utm_y
            self.origin_z = utm_z
            self.get_logger().info(f'Origin set to UTM X: {utm_x}, Y: {utm_y}')

        local_x = utm_x - self.origin_x
        local_y = utm_y - self.origin_y
        local_z = utm_z - self.origin_z

        # Create a new PoseStamped point for the trajectory
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = 'map'
        
        pose.pose.position.x = local_x
        pose.pose.position.y = local_y
        pose.pose.position.z = local_z
        
        # GPS does not give orientation, default to identity quaternion
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        # Append the new pose to our ongoing path array
        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.poses.append(pose)

        # Publish the updated path
        self.traj_pub.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GPSTrajPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()