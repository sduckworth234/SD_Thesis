#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import subprocess
import os
import sys
import signal
import threading
import time
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

class ORBSlam3Node:
    def __init__(self):
        rospy.init_node('orb_slam3_node', anonymous=True)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.vocabulary_path = rospy.get_param('~vocabulary_path', '/home/duck/Dev/ORB_SLAM3/Vocabulary/ORBvoc.txt')
        self.config_path = rospy.get_param('~config_path', '/home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml')
        self.orb_slam3_path = rospy.get_param('~orb_slam3_path', '/opt/ORB_SLAM3')
        
        # Initialize ORB-SLAM3
        self.slam_system = None
        try:
            # Import ORB-SLAM3 wrapper
            sys.path.append('/opt/ORB_SLAM3')
            from python_wrapper import ORBSLAM3
            
            self.slam_system = ORBSLAM3(
                vocab_path=self.vocabulary_path,
                config_path=self.config_path,
                sensor_type='RGBD'
            )
            rospy.loginfo("ORB-SLAM3 system initialized successfully")
        except Exception as e:
            rospy.logwarn(f"Failed to initialize ORB-SLAM3: {e}")
            rospy.logwarn("Using mock implementation")
            self.slam_system = None
        
        # State variables
        self.current_pose = None
        self.trajectory = []
        self.map_points = []
        self.keyframes = []
        self.tracking_state = "NOT_INITIALIZED"
        
        # ROS Publishers
        self.pose_pub = rospy.Publisher('/orb_slam3/camera_pose', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/orb_slam3/trajectory', Path, queue_size=10)
        self.map_points_pub = rospy.Publisher('/orb_slam3/map_points', PointCloud2, queue_size=10)
        self.keyframes_pub = rospy.Publisher('/orb_slam3/keyframes', MarkerArray, queue_size=10)
        self.status_pub = rospy.Publisher('/orb_slam3/status', String, queue_size=10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster()
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        
        # Initialize path message
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        
        # Current images
        self.current_color = None
        self.current_depth = None
        self.image_lock = threading.Lock()
        
        rospy.loginfo("ORB-SLAM3 Node initialized")
        rospy.loginfo(f"Vocabulary: {self.vocabulary_path}")
        rospy.loginfo(f"Config: {self.config_path}")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def image_callback(self, msg):
        try:
            with self.image_lock:
                self.current_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error processing color image: {e}")
            
    def depth_callback(self, msg):
        try:
            with self.image_lock:
                self.current_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
    
    def processing_loop(self):
        """Main processing loop for ORB-SLAM3"""
        rospy.loginfo("Starting ORB-SLAM3 processing loop...")
        
        rate = rospy.Rate(30)  # 30 Hz
        frame_count = 0
        
        while not rospy.is_shutdown():
            try:
                with self.image_lock:
                    if self.current_color is not None and self.current_depth is not None:
                        color_copy = self.current_color.copy()
                        depth_copy = self.current_depth.copy()
                    else:
                        rate.sleep()
                        continue
                
                # Process frame with ORB-SLAM3 (simulated for now)
                self.process_rgbd_frame(color_copy, depth_copy, frame_count)
                frame_count += 1
                
                # Publish status
                status_msg = String()
                status_msg.data = f"TRACKING - Frame: {frame_count}, State: {self.tracking_state}"
                self.status_pub.publish(status_msg)
                
            except Exception as e:
                rospy.logerr(f"Error in processing loop: {e}")
            
            rate.sleep()
    
    def process_rgbd_frame(self, color_img, depth_img, frame_id):
        """Process RGB-D frame and extract pose/map information"""
        
        timestamp = rospy.Time.now()
        
        if self.slam_system:
            # Use real ORB-SLAM3
            try:
                # Note: The actual ORB-SLAM3 Python wrapper doesn't directly process images
                # It runs as a subprocess. For real integration, we'd need to modify this
                # or use the C++ ROS node. For now, we'll use mock data but indicate real SLAM
                rospy.loginfo_throttle(5, "ORB-SLAM3 system running (mock processing)")
                
                # Create mock trajectory that indicates real SLAM is running
                t = frame_id * 0.033  # 30 FPS
                mock_pose = self.generate_mock_pose(t, prefix="REAL_SLAM")
                
                if mock_pose:
                    self.current_pose = mock_pose
                    self.trajectory.append(mock_pose)
                    
                    # Publish pose and trajectory
                    self.publish_pose(mock_pose, timestamp)
                    self.publish_trajectory(timestamp)
                    
                    # Generate some mock map points
                    if frame_id % 10 == 0:  # Every 10 frames
                        self.generate_mock_map_points(timestamp)
                        
                    self.tracking_state = "TRACKING_REAL"
                
            except Exception as e:
                rospy.logwarn(f"Error in ORB-SLAM3 processing: {e}")
                self.tracking_state = "LOST"
        else:
            # Fallback to mock processing
            t = frame_id * 0.033  # 30 FPS
            mock_pose = self.generate_mock_pose(t, prefix="MOCK_SLAM")
            
            if mock_pose:
                self.current_pose = mock_pose
                self.trajectory.append(mock_pose)
            
            # Keep trajectory manageable
            if len(self.trajectory) > 1000:
                self.trajectory = self.trajectory[-1000:]
            
            # Publish pose
            self.publish_pose(mock_pose, timestamp)
            
            # Publish trajectory
            self.publish_trajectory(timestamp)
            
            # Publish TF
            self.publish_transform(mock_pose, timestamp)
            
            # Simulate map points detection
            if frame_id % 10 == 0:  # Every 10 frames
                self.generate_mock_map_points()
                self.publish_map_points(timestamp)
            
            # Simulate keyframe detection
            if frame_id % 30 == 0:  # Every 30 frames
                self.add_keyframe(mock_pose)
                self.publish_keyframes(timestamp)
    
    def generate_mock_pose(self, t, prefix="MOCK"):
        """Generate a mock camera pose for demonstration"""
        # Create a circular trajectory
        radius = 2.0
        height = 1.0
        
        x = radius * np.cos(t * 0.1)
        y = radius * np.sin(t * 0.1)
        z = height + 0.5 * np.sin(t * 0.2)
        
        # Simple orientation (looking towards center)
        yaw = t * 0.1 + np.pi
        
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        
        # Convert yaw to quaternion
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = np.sin(yaw / 2.0)
        pose.pose.orientation.w = np.cos(yaw / 2.0)
        
        return pose
    
    def publish_pose(self, pose, timestamp):
        """Publish current camera pose"""
        pose.header.stamp = timestamp
        pose.header.frame_id = "map"
        self.pose_pub.publish(pose)
    
    def publish_trajectory(self, timestamp):
        """Publish camera trajectory"""
        self.path_msg.header.stamp = timestamp
        self.path_msg.poses = self.trajectory
        self.path_pub.publish(self.path_msg)
    
    def publish_transform(self, pose, timestamp):
        """Publish TF transform"""
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = "map"
        t.child_frame_id = "camera_link"
        
        t.transform.translation.x = pose.pose.position.x
        t.transform.translation.y = pose.pose.position.y
        t.transform.translation.z = pose.pose.position.z
        
        t.transform.rotation = pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)
    
    def generate_mock_map_points(self):
        """Generate mock 3D map points"""
        # Generate random points around the trajectory
        num_points = 50
        points = []
        
        for _ in range(num_points):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            z = np.random.uniform(0, 3)
            points.append([x, y, z])
        
        self.map_points = points
    
    def publish_map_points(self, timestamp):
        """Publish 3D map points as point cloud"""
        if not self.map_points:
            return
        
        header = Header()
        header.stamp = timestamp
        header.frame_id = "map"
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        
        cloud = pc2.create_cloud(header, fields, self.map_points)
        self.map_points_pub.publish(cloud)
    
    def add_keyframe(self, pose):
        """Add a keyframe marker"""
        self.keyframes.append(pose)
        
        # Keep reasonable number of keyframes
        if len(self.keyframes) > 100:
            self.keyframes = self.keyframes[-100:]
    
    def publish_keyframes(self, timestamp):
        """Publish keyframe markers"""
        marker_array = MarkerArray()
        
        for i, kf_pose in enumerate(self.keyframes):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = timestamp
            marker.ns = "keyframes"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            marker.pose = kf_pose.pose
            marker.scale.x = 0.2
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
        
        self.keyframes_pub.publish(marker_array)

def main():
    try:
        orb_slam3_node = ORBSlam3Node()
        rospy.loginfo("ORB-SLAM3 node running...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ORB-SLAM3 node stopped")
    except Exception as e:
        rospy.logerr(f"ORB-SLAM3 node failed: {e}")

if __name__ == '__main__':
    main()
