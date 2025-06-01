#!/usr/bin/env python3
"""
Camera Node for Intel RealSense D435i
Publishes color and depth images for the SD_Thesis project.
"""

import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import time

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.color_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=1)
        self.color_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=1)
        self.depth_info_pub = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/camera/depth/color/points', PointCloud2, queue_size=1)
        
        # Parameters
        self.width = rospy.get_param('~width', 640)
        self.height = rospy.get_param('~height', 480)
        self.fps = rospy.get_param('~fps', 30)
        self.enable_depth = rospy.get_param('~enable_depth', True)
        self.enable_pointcloud = rospy.get_param('~enable_pointcloud', False)
        self.mock_mode = rospy.get_param('~mock_mode', False)
        
        if self.mock_mode:
            rospy.loginfo("Camera node running in mock mode")
            self.pipeline = None
        if self.mock_mode:
            rospy.loginfo("Camera node running in mock mode")
            self.pipeline = None
            self.profile = None
        else:
            # Initialize RealSense pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure color stream
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            if self.enable_depth:
                # Configure depth stream
                self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # Start pipeline
            try:
                self.profile = self.pipeline.start(self.config)
                rospy.loginfo("RealSense camera initialized successfully")
            except Exception as e:
                rospy.logerr(f"Failed to initialize RealSense camera: {e}")
                self.use_webcam_fallback()
                
        # Get camera intrinsics
        if hasattr(self, 'profile') and self.profile:
            self.setup_camera_info()
        
        # Point cloud processing (if enabled)
        if self.enable_pointcloud and self.enable_depth:
            self.pc = rs.pointcloud()
            self.colorizer = rs.colorizer()
            
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        
    def use_webcam_fallback(self):
        """Fallback to webcam if RealSense is not available"""
        rospy.logwarn("Falling back to webcam")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.profile = None
        
    def setup_camera_info(self):
        """Setup camera info messages from RealSense intrinsics"""
        # Get color stream profile
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        
        # Create color camera info
        self.color_info = CameraInfo()
        self.color_info.width = color_intrinsics.width
        self.color_info.height = color_intrinsics.height
        self.color_info.distortion_model = "plumb_bob"
        self.color_info.K = [color_intrinsics.fx, 0, color_intrinsics.ppx,
                           0, color_intrinsics.fy, color_intrinsics.ppy,
                           0, 0, 1]
        self.color_info.P = [color_intrinsics.fx, 0, color_intrinsics.ppx, 0,
                           0, color_intrinsics.fy, color_intrinsics.ppy, 0,
                           0, 0, 1, 0]
        self.color_info.D = [0, 0, 0, 0, 0]  # Assuming no distortion for simplicity
        
        if self.enable_depth:
            # Get depth stream profile
            depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
            depth_intrinsics = depth_profile.get_intrinsics()
            
            # Create depth camera info
            self.depth_info = CameraInfo()
            self.depth_info.width = depth_intrinsics.width
            self.depth_info.height = depth_intrinsics.height
            self.depth_info.distortion_model = "plumb_bob"
            self.depth_info.K = [depth_intrinsics.fx, 0, depth_intrinsics.ppx,
                               0, depth_intrinsics.fy, depth_intrinsics.ppy,
                               0, 0, 1]
            self.depth_info.P = [depth_intrinsics.fx, 0, depth_intrinsics.ppx, 0,
                               0, depth_intrinsics.fy, depth_intrinsics.ppy, 0,
                               0, 0, 1, 0]
            self.depth_info.D = [0, 0, 0, 0, 0]
    
    def create_default_camera_info(self):
        """Create default camera info for webcam fallback"""
        info = CameraInfo()
        info.width = self.width
        info.height = self.height
        info.distortion_model = "plumb_bob"
        # Default camera matrix (rough approximation)
        fx = fy = self.width
        cx = self.width / 2
        cy = self.height / 2
        info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        info.D = [0, 0, 0, 0, 0]
        return info
    
    def run(self):
        """Main camera loop"""
        rate = rospy.Rate(self.fps)
        
        while not rospy.is_shutdown():
            try:
                if self.mock_mode:
                    # Generate mock images
                    self.process_mock_frame()
                elif hasattr(self, 'profile') and self.profile:
                    # RealSense camera
                    self.process_realsense_frame()
                else:
                    # Webcam fallback
                    self.process_webcam_frame()
                    
                # Performance monitoring
                self.frame_count += 1
                if self.frame_count % (self.fps * 5) == 0:  # Every 5 seconds
                    elapsed = time.time() - self.start_time
                    fps_actual = self.frame_count / elapsed
                    rospy.loginfo(f"Camera FPS: {fps_actual:.2f}")
                    
            except Exception as e:
                rospy.logerr(f"Camera error: {e}")
                
            rate.sleep()
    
    def process_mock_frame(self):
        """Generate mock camera frames for testing"""
        # Create mock color image with moving pattern
        mock_color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add time-varying pattern
        t = time.time()
        center_x = int(self.width/2 + 100 * np.sin(t))
        center_y = int(self.height/2 + 50 * np.cos(t))
        
        cv2.circle(mock_color, (center_x, center_y), 50, (0, 255, 0), -1)
        cv2.putText(mock_color, f"Mock Camera - Frame {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create mock depth image
        mock_depth = np.full((self.height, self.width), 1000, dtype=np.uint16)
        
        # Add some depth variation
        for i in range(self.height):
            for j in range(self.width):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                mock_depth[i, j] = int(800 + 200 * np.sin(dist / 20))
        
        # Convert to ROS messages
        now = rospy.Time.now()
        
        color_msg = self.bridge.cv2_to_imgmsg(mock_color, "bgr8")
        color_msg.header.stamp = now
        color_msg.header.frame_id = "camera_color_optical_frame"
        
        depth_msg = self.bridge.cv2_to_imgmsg(mock_depth, "16UC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_depth_optical_frame"
        
        # Publish images
        self.color_pub.publish(color_msg)
        if self.enable_depth:
            self.depth_pub.publish(depth_msg)
        
        # Publish camera info
        color_info = self.create_mock_camera_info()
        color_info.header.stamp = now
        color_info.header.frame_id = "camera_color_optical_frame"
        self.color_info_pub.publish(color_info)
        
        if self.enable_depth:
            depth_info = self.create_mock_camera_info()
            depth_info.header.stamp = now
            depth_info.header.frame_id = "camera_depth_optical_frame"
            self.depth_info_pub.publish(depth_info)
    
    def create_mock_camera_info(self):
        """Create mock camera info"""
        info = CameraInfo()
        info.width = self.width
        info.height = self.height
        info.distortion_model = "plumb_bob"
        # Mock camera intrinsics
        fx = fy = 600.0
        cx = self.width / 2.0
        cy = self.height / 2.0
        info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        info.D = [0, 0, 0, 0, 0]
        return info
    
    def process_realsense_frame(self):
        """Process frame from RealSense camera"""
        frames = self.pipeline.wait_for_frames()
        
        # Get color frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
            
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create and publish color image message
        color_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        color_msg.header.stamp = rospy.Time.now()
        color_msg.header.frame_id = "camera_color_optical_frame"
        self.color_pub.publish(color_msg)
        
        # Publish color camera info
        if hasattr(self, 'color_info'):
            self.color_info.header.stamp = color_msg.header.stamp
            self.color_info.header.frame_id = color_msg.header.frame_id
            self.color_info_pub.publish(self.color_info)
        
        if self.enable_depth:
            # Get depth frame
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create and publish depth image message
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, "16UC1")
                depth_msg.header.stamp = color_msg.header.stamp
                depth_msg.header.frame_id = "camera_depth_optical_frame"
                self.depth_pub.publish(depth_msg)
                
                # Publish depth camera info
                if hasattr(self, 'depth_info'):
                    self.depth_info.header.stamp = depth_msg.header.stamp
                    self.depth_info.header.frame_id = depth_msg.header.frame_id
                    self.depth_info_pub.publish(self.depth_info)
    
    def process_webcam_frame(self):
        """Process frame from webcam fallback"""
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to capture webcam frame")
            return
            
        # Create and publish image message
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "camera_color_optical_frame"
        self.color_pub.publish(img_msg)
        
        # Publish camera info
        info = self.create_default_camera_info()
        info.header.stamp = img_msg.header.stamp
        info.header.frame_id = img_msg.header.frame_id
        self.color_info_pub.publish(info)
    
    def shutdown(self):
        """Clean shutdown"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        if hasattr(self, 'cap'):
            self.cap.release()
        rospy.loginfo("Camera node shutdown complete")

if __name__ == '__main__':
    try:
        camera_node = CameraNode()
        rospy.on_shutdown(camera_node.shutdown)
        camera_node.run()
    except rospy.ROSInterruptException:
        pass