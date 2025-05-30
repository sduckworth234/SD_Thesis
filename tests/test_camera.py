#!/usr/bin/env python3
"""
Test script for Intel RealSense D435i camera functionality
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs

class RealSenseTest:
    def __init__(self):
        rospy.init_node('realsense_test', anonymous=True)
        self.bridge = CvBridge()
        
        # ROS subscribers
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_callback)
        
        # Test variables
        self.color_received = False
        self.depth_received = False
        self.info_received = False
        
        rospy.loginfo("RealSense test node initialized")
    
    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.color_received = True
            
            # Display basic image info
            height, width, channels = cv_image.shape
            rospy.loginfo_once(f"Color image: {width}x{height}, {channels} channels")
            
            # Show image (for visualization testing)
            cv2.imshow("Color Image", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing color image: {e}")
    
    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_received = True
            
            # Display basic depth info
            height, width = cv_image.shape
            rospy.loginfo_once(f"Depth image: {width}x{height}")
            
            # Convert to displayable format
            depth_display = cv2.convertScaleAbs(cv_image, alpha=0.03)
            cv2.imshow("Depth Image", depth_display)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
    
    def info_callback(self, msg):
        self.info_received = True
        rospy.loginfo_once(f"Camera info received: {msg.width}x{msg.height}")
    
    def run_test(self):
        rospy.loginfo("Starting RealSense camera test...")
        rospy.loginfo("Make sure the camera is connected and ROS driver is running:")
        rospy.loginfo("roslaunch realsense2_camera rs_camera.launch")
        
        rate = rospy.Rate(1)  # 1 Hz
        test_duration = 10  # seconds
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < test_duration:
            # Check test status
            if self.color_received and self.depth_received and self.info_received:
                rospy.loginfo("✓ All camera streams working correctly!")
                break
            else:
                missing = []
                if not self.color_received:
                    missing.append("color")
                if not self.depth_received:
                    missing.append("depth")
                if not self.info_received:
                    missing.append("camera_info")
                
                rospy.logwarn(f"Waiting for: {', '.join(missing)}")
            
            rate.sleep()
        
        # Final test report
        rospy.loginfo("=== RealSense Test Results ===")
        rospy.loginfo(f"Color stream: {'✓' if self.color_received else '✗'}")
        rospy.loginfo(f"Depth stream: {'✓' if self.depth_received else '✗'}")
        rospy.loginfo(f"Camera info: {'✓' if self.info_received else '✗'}")
        
        if all([self.color_received, self.depth_received, self.info_received]):
            rospy.loginfo("🎉 RealSense camera test PASSED!")
            return True
        else:
            rospy.logerr("❌ RealSense camera test FAILED!")
            return False

def test_direct_connection():
    """Test direct connection to RealSense camera without ROS"""
    rospy.loginfo("Testing direct camera connection...")
    
    try:
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline.start(config)
        
        for i in range(10):  # Capture 10 frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            rospy.loginfo(f"Frame {i+1}: Color and depth received")
        
        pipeline.stop()
        rospy.loginfo("✓ Direct camera connection test PASSED!")
        return True
        
    except Exception as e:
        rospy.logerr(f"Direct camera connection test FAILED: {e}")
        return False

if __name__ == '__main__':
    try:
        # First test direct connection
        direct_test = test_direct_connection()
        
        if direct_test:
            # Test ROS integration
            tester = RealSenseTest()
            ros_test = tester.run_test()
            
            if ros_test:
                rospy.loginfo("All RealSense tests completed successfully!")
            else:
                rospy.logwarn("ROS integration test failed. Check if ROS driver is running.")
        else:
            rospy.logerr("Direct camera test failed. Check camera connection.")
            
        cv2.destroyAllWindows()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by user")
    except Exception as e:
        rospy.logerr(f"Test failed with error: {e}")
