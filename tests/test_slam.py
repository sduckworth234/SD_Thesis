#!/usr/bin/env python3
"""
SLAM Test Node
Tests ORB-SLAM3 functionality and provides visualization feedback
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray

class SLAMTest:
    def __init__(self):
        rospy.init_node('slam_test', anonymous=True)
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Test parameters
        self.camera_type = rospy.get_param('~camera_type', 'rgbd')
        self.test_duration = rospy.get_param('~test_duration', 60.0)  # seconds
        
        # Statistics
        self.pose_count = 0
        self.image_count = 0
        self.start_time = rospy.Time.now()
        self.last_pose = None
        self.trajectory = []
        
        # Publishers for visualization
        self.trajectory_pub = rospy.Publisher('/slam_test/trajectory', Path, queue_size=1)
        self.marker_pub = rospy.Publisher('/slam_test/markers', MarkerArray, queue_size=1)
        self.status_pub = rospy.Publisher('/slam_test/status', Marker, queue_size=1)
        
        # Subscribers
        self.pose_sub = rospy.Subscriber('/slam/pose', PoseStamped, self.pose_callback)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        rospy.loginfo(f"SLAM Test initialized for {self.camera_type} camera")
        rospy.loginfo(f"Test duration: {self.test_duration} seconds")
        
        # Start test
        self.run_test()
    
    def pose_callback(self, msg):
        """Handle pose updates from SLAM"""
        self.pose_count += 1
        self.last_pose = msg
        
        # Add to trajectory
        self.trajectory.append(msg)
        
        # Publish trajectory
        self.publish_trajectory()
        
        # Log pose information
        if self.pose_count % 10 == 0:  # Every 10th pose
            pos = msg.pose.position
            rospy.loginfo(f"Pose {self.pose_count}: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")
    
    def image_callback(self, msg):
        """Handle image updates"""
        self.image_count += 1
        
        if self.image_count % 30 == 0:  # Every 30th image (roughly 1 second at 30 FPS)
            rospy.loginfo(f"Processed {self.image_count} images")
    
    def publish_trajectory(self):
        """Publish trajectory for visualization"""
        if len(self.trajectory) < 2:
            return
            
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        
        # Add recent poses to path (limit to last 100 for performance)
        recent_poses = self.trajectory[-100:]
        for pose_stamped in recent_poses:
            path.poses.append(pose_stamped)
        
        self.trajectory_pub.publish(path)
    
    def publish_status(self):
        """Publish test status as text marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "slam_test"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0
        
        # Scale
        marker.scale.z = 0.5
        
        # Color (green if good, yellow if issues, red if bad)
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        pose_rate = self.pose_count / elapsed if elapsed > 0 else 0
        image_rate = self.image_count / elapsed if elapsed > 0 else 0
        
        if pose_rate > 10 and image_rate > 20:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            status = "GOOD"
        elif pose_rate > 5 and image_rate > 10:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            status = "WARNING"
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            status = "ERROR"
        
        marker.color.a = 1.0
        
        # Text
        marker.text = f"SLAM Test Status: {status}\n"
        marker.text += f"Poses: {self.pose_count} ({pose_rate:.1f} Hz)\n"
        marker.text += f"Images: {self.image_count} ({image_rate:.1f} Hz)\n"
        marker.text += f"Elapsed: {elapsed:.1f}s"
        
        self.status_pub.publish(marker)
    
    def run_test(self):
        """Main test loop"""
        rate = rospy.Rate(1.0)  # 1 Hz for status updates
        
        rospy.loginfo("Starting SLAM test...")
        rospy.loginfo("Waiting for SLAM pose data...")
        
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - self.start_time).to_sec()
            
            # Publish status
            self.publish_status()
            
            # Check test completion
            if elapsed >= self.test_duration:
                self.print_final_results()
                break
            
            # Check for issues
            if elapsed > 10 and self.pose_count == 0:
                rospy.logwarn("No poses received after 10 seconds - check SLAM node")
            
            if elapsed > 5 and self.image_count == 0:
                rospy.logwarn("No images received after 5 seconds - check camera")
            
            rate.sleep()
    
    def print_final_results(self):
        """Print final test results"""
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("SLAM TEST RESULTS")
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"Test Duration: {elapsed:.1f} seconds")
        rospy.loginfo(f"Total Poses: {self.pose_count}")
        rospy.loginfo(f"Total Images: {self.image_count}")
        rospy.loginfo(f"Pose Rate: {self.pose_count/elapsed:.1f} Hz")
        rospy.loginfo(f"Image Rate: {self.image_count/elapsed:.1f} Hz")
        
        if self.trajectory:
            # Calculate trajectory length
            total_distance = 0.0
            for i in range(1, len(self.trajectory)):
                p1 = self.trajectory[i-1].pose.position
                p2 = self.trajectory[i].pose.position
                dist = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)
                total_distance += dist
            
            rospy.loginfo(f"Trajectory Length: {total_distance:.3f} meters")
            rospy.loginfo(f"Trajectory Points: {len(self.trajectory)}")
        
        # Test evaluation
        pose_rate = self.pose_count / elapsed
        image_rate = self.image_count / elapsed
        
        rospy.loginfo("=" * 50)
        if pose_rate > 10 and image_rate > 20:
            rospy.loginfo("✅ SLAM TEST PASSED")
        elif pose_rate > 5 and image_rate > 10:
            rospy.logwarn("⚠️  SLAM TEST WARNING - Performance issues")
        else:
            rospy.logerr("❌ SLAM TEST FAILED")
        rospy.loginfo("=" * 50)

if __name__ == '__main__':
    try:
        slam_test = SLAMTest()
    except rospy.ROSInterruptException:
        pass
