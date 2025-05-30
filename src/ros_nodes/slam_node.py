#!/usr/bin/env python3
"""
SLAM Node for ORB-SLAM2 Integration
Subscribes to camera images and publishes SLAM pose, trajectory, and map points.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import json
import time
import threading
import struct

class SLAMNode:
    def __init__(self):
        rospy.init_node('slam_node', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.vocab_path = rospy.get_param('~vocab_path', 'models/orbslam2/ORBvoc.txt')
        self.settings_path = rospy.get_param('~settings_path', 'config/orbslam2_config.yaml')
        self.use_viewer = rospy.get_param('~use_viewer', False)
        self.scale_factor = rospy.get_param('~scale_factor', 1.2)
        self.num_features = rospy.get_param('~num_features', 1000)
        
        # SLAM system initialization
        self.initialize_slam()
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Subscribers
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, 
                                         self.color_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, 
                                         self.depth_callback, queue_size=1)
        
        # Publishers
        self.pose_pub = rospy.Publisher('/slam/pose', PoseStamped, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/slam/trajectory', Path, queue_size=10)
        self.map_points_pub = rospy.Publisher('/slam/map_points', PointCloud2, queue_size=10)
        self.keyframes_pub = rospy.Publisher('/slam/keyframes', MarkerArray, queue_size=10)
        self.status_pub = rospy.Publisher('/slam/status', String, queue_size=10)
        self.map_pub = rospy.Publisher('/slam/occupancy_map', OccupancyGrid, queue_size=1)
        
        # State management
        self.current_pose = None
        self.trajectory = Path()
        self.trajectory.header.frame_id = "map"
        self.map_points = []
        self.keyframes = []
        
        # Synchronization
        self.color_image = None
        self.depth_image = None
        self.image_lock = threading.Lock()
        self.slam_lock = threading.Lock()
        
        # Performance monitoring
        self.frame_count = 0
        self.tracking_state = "NOT_INITIALIZED"
        self.start_time = time.time()
        self.processing_times = []
        
        rospy.loginfo("SLAM Node Initialized")
    
    def initialize_slam(self):
        """Initialize ORB-SLAM2 system"""
        try:
            # Try to import and initialize ORB-SLAM2
            # This would require proper ORB-SLAM2 Python bindings
            # For now, we'll use a dummy implementation
            rospy.logwarn("Using dummy SLAM implementation - install ORB-SLAM2 Python bindings for full functionality")
            self.slam_system = None
            self.use_dummy_slam = True
            
            # Initialize dummy SLAM state
            self.dummy_pose = np.eye(4)
            self.dummy_trajectory = []
            self.dummy_map_points = []
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize SLAM system: {e}")
            self.use_dummy_slam = True
    
    def color_callback(self, msg):
        """Process color image for SLAM"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.image_lock:
                self.color_image = cv_image
                self.color_timestamp = msg.header.stamp
                
            # Process SLAM if we have both color and depth
            if self.depth_image is not None:
                self.process_slam_frame(msg.header.stamp)
                
        except Exception as e:
            rospy.logerr(f"Color callback error: {e}")
    
    def depth_callback(self, msg):
        """Process depth image for SLAM"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            
            with self.image_lock:
                self.depth_image = cv_image
                self.depth_timestamp = msg.header.stamp
                
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")
    
    def process_slam_frame(self, timestamp):
        """Process SLAM with current color and depth images"""
        start_time = time.time()
        
        try:
            with self.image_lock:
                if self.color_image is None or self.depth_image is None:
                    return
                
                color_img = self.color_image.copy()
                depth_img = self.depth_image.copy()
            
            with self.slam_lock:
                if self.use_dummy_slam:
                    pose_matrix = self.process_dummy_slam(color_img, depth_img)
                else:
                    pose_matrix = self.process_orbslam(color_img, depth_img, timestamp)
                
                if pose_matrix is not None:
                    self.update_pose(pose_matrix, timestamp)
                    self.update_trajectory(timestamp)
                    self.publish_slam_data(timestamp)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            # Publish status periodically
            if self.frame_count % 30 == 0:
                self.publish_status()
                
        except Exception as e:
            rospy.logerr(f"SLAM processing error: {e}")
    
    def process_dummy_slam(self, color_img, depth_img):
        """Dummy SLAM implementation for testing"""
        # Simulate camera movement
        movement = 0.01 * np.sin(self.frame_count * 0.1)
        
        # Update dummy pose (move forward with slight oscillation)
        self.dummy_pose[0, 3] += 0.01  # Move forward
        self.dummy_pose[1, 3] += movement  # Oscillate sideways
        
        # Add some rotation
        angle = 0.001 * self.frame_count
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.dummy_pose = rotation @ self.dummy_pose
        
        # Simulate map points
        if len(self.dummy_map_points) < 100:
            # Add random map points
            for _ in range(5):
                point = np.random.rand(3) * 10 - 5  # Random points in [-5, 5] range
                self.dummy_map_points.append(point)
        
        self.tracking_state = "OK"
        return self.dummy_pose.copy()
    
    def process_orbslam(self, color_img, depth_img, timestamp):
        """Process with actual ORB-SLAM2 (placeholder)"""
        # This would be the actual ORB-SLAM2 processing
        # For implementation, you would need ORB-SLAM2 Python bindings
        
        # Convert timestamp to seconds
        timestamp_sec = timestamp.to_sec()
        
        # Process frame with ORB-SLAM2
        # pose_matrix = self.slam_system.track_rgbd(color_img, depth_img, timestamp_sec)
        
        # For now, return None to indicate no tracking
        return None
    
    def update_pose(self, pose_matrix, timestamp):
        """Update current pose from 4x4 transformation matrix"""
        # Extract position and orientation from transformation matrix
        position = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]
        
        # Convert rotation matrix to quaternion
        quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        # Create pose message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        
        self.current_pose = pose_msg
        
        # Publish transform
        self.publish_transform(pose_msg)
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
    
    def publish_transform(self, pose_msg):
        """Publish TF transform"""
        transform = TransformStamped()
        transform.header = pose_msg.header
        transform.child_frame_id = "camera_link"
        
        transform.transform.translation.x = pose_msg.pose.position.x
        transform.transform.translation.y = pose_msg.pose.position.y
        transform.transform.translation.z = pose_msg.pose.position.z
        
        transform.transform.rotation = pose_msg.pose.orientation
        
        self.tf_broadcaster.sendTransform(transform)
    
    def update_trajectory(self, timestamp):
        """Update trajectory path"""
        if self.current_pose is not None:
            self.trajectory.poses.append(self.current_pose)
            
            # Limit trajectory length to avoid memory issues
            if len(self.trajectory.poses) > 1000:
                self.trajectory.poses = self.trajectory.poses[-500:]
            
            self.trajectory.header.stamp = timestamp
    
    def publish_slam_data(self, timestamp):
        """Publish all SLAM-related data"""
        if self.current_pose is not None:
            # Publish current pose
            self.pose_pub.publish(self.current_pose)
            
            # Publish trajectory
            self.trajectory_pub.publish(self.trajectory)
            
            # Publish map points
            self.publish_map_points(timestamp)
            
            # Publish keyframes
            self.publish_keyframes(timestamp)
    
    def publish_map_points(self, timestamp):
        """Publish map points as PointCloud2"""
        if not self.dummy_map_points:
            return
            
        # Create PointCloud2 message
        header = Header()
        header.stamp = timestamp
        header.frame_id = "map"
        
        # Define point cloud fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        
        # Pack point data
        cloud_data = []
        for point in self.dummy_map_points:
            # Pack x, y, z coordinates
            cloud_data.append(struct.pack('fff', point[0], point[1], point[2]))
            # Pack RGB (white color)
            rgb = struct.pack('I', 0xFFFFFF)
            cloud_data.append(rgb)
        
        # Create PointCloud2 message
        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(self.dummy_map_points)
        pc2_msg.fields = fields
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.data = b''.join(cloud_data)
        pc2_msg.is_dense = True
        
        self.map_points_pub.publish(pc2_msg)
    
    def publish_keyframes(self, timestamp):
        """Publish keyframes as markers"""
        marker_array = MarkerArray()
        
        # Create marker for current pose as keyframe
        if self.current_pose is not None and self.frame_count % 30 == 0:  # Every 30 frames
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = timestamp
            marker.ns = "keyframes"
            marker.id = len(self.keyframes)
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            marker.pose = self.current_pose.pose
            
            marker.scale.x = 0.3
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime = rospy.Duration(0)  # Persistent
            
            marker_array.markers.append(marker)
            self.keyframes.append(marker)
        
        # Add existing keyframes
        for keyframe in self.keyframes[-50:]:  # Show last 50 keyframes
            marker_array.markers.append(keyframe)
        
        self.keyframes_pub.publish(marker_array)
    
    def publish_status(self):
        """Publish SLAM status information"""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_processing_time = np.mean(self.processing_times[-30:]) if self.processing_times else 0
        
        status = {
            'tracking_state': self.tracking_state,
            'frames_processed': self.frame_count,
            'map_points': len(self.dummy_map_points),
            'keyframes': len(self.keyframes),
            'average_fps': avg_fps,
            'average_processing_time': avg_processing_time,
            'elapsed_time': elapsed_time
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)
        
        rospy.loginfo(f"SLAM Status - State: {self.tracking_state}, FPS: {avg_fps:.2f}, "
                     f"Map Points: {len(self.dummy_map_points)}, Keyframes: {len(self.keyframes)}")

if __name__ == '__main__':
    try:
        node = SLAMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass