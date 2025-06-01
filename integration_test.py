#!/usr/bin/env python3

import rospy
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image
import time

class SystemIntegrationMonitor:
    def __init__(self):
        rospy.init_node('system_integration_monitor', anonymous=True)
        
        # System status tracking
        self.camera_active = False
        self.slam_active = False
        self.detection_active = False
        
        # Message counters
        self.camera_count = 0
        self.slam_count = 0
        self.detection_count = 0
        
        # Subscribers
        self.camera_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.camera_callback)
        self.slam_sub = rospy.Subscriber('/slam/status', String, self.slam_callback)
        self.detection_sub = rospy.Subscriber('/detection/status', String, self.detection_callback)
        
        rospy.loginfo("System Integration Monitor initialized")
        
    def camera_callback(self, msg):
        self.camera_active = True
        self.camera_count += 1
        
    def slam_callback(self, msg):
        self.slam_active = True
        self.slam_count += 1
        
    def detection_callback(self, msg):
        self.detection_active = True
        self.detection_count += 1
        
    def run_integration_test(self, duration=30):
        """Run a comprehensive integration test"""
        rospy.loginfo(f"Starting {duration}s integration test...")
        
        start_time = time.time()
        test_passed = True
        
        # Wait for all systems to come online
        rospy.loginfo("Waiting for all systems to come online...")
        timeout = 10
        start_wait = time.time()
        
        while not (self.camera_active and self.slam_active and self.detection_active):
            if time.time() - start_wait > timeout:
                rospy.logerr("TIMEOUT: Not all systems came online within 10 seconds")
                rospy.logerr(f"Camera: {self.camera_active}, SLAM: {self.slam_active}, Detection: {self.detection_active}")
                return False
            time.sleep(0.1)
        
        rospy.loginfo("✅ All systems online!")
        
        # Reset counters
        self.camera_count = 0
        self.slam_count = 0
        self.detection_count = 0
        
        # Monitor for the test duration
        test_start = time.time()
        while time.time() - test_start < duration:
            elapsed = time.time() - test_start
            if elapsed > 0:
                camera_rate = self.camera_count / elapsed
                slam_rate = self.slam_count / elapsed
                detection_rate = self.detection_count / elapsed
                
                rospy.loginfo(f"[{elapsed:.1f}s] Rates - Camera: {camera_rate:.1f}Hz, SLAM: {slam_rate:.2f}Hz, Detection: {detection_rate:.1f}Hz")
            
            time.sleep(5)  # Log every 5 seconds
        
        # Final performance analysis
        total_time = time.time() - test_start
        final_camera_rate = self.camera_count / total_time
        final_slam_rate = self.slam_count / total_time
        final_detection_rate = self.detection_count / total_time
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("FINAL INTEGRATION TEST RESULTS")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Test Duration: {total_time:.1f} seconds")
        rospy.loginfo(f"Camera Rate: {final_camera_rate:.2f} Hz (Target: ~30 Hz)")
        rospy.loginfo(f"SLAM Rate: {final_slam_rate:.2f} Hz (Target: ~1 Hz)")
        rospy.loginfo(f"Detection Rate: {final_detection_rate:.2f} Hz (Target: ~25 Hz)")
        
        # Performance validation
        camera_ok = 25 <= final_camera_rate <= 35  # Allow some tolerance
        slam_ok = 0.5 <= final_slam_rate <= 2.0
        detection_ok = 20 <= final_detection_rate <= 35
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("PERFORMANCE VALIDATION")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Camera Performance: {'✅ PASS' if camera_ok else '❌ FAIL'}")
        rospy.loginfo(f"SLAM Performance: {'✅ PASS' if slam_ok else '❌ FAIL'}")
        rospy.loginfo(f"Detection Performance: {'✅ PASS' if detection_ok else '❌ FAIL'}")
        
        overall_pass = camera_ok and slam_ok and detection_ok
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"OVERALL TEST RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        rospy.loginfo("=" * 60)
        
        return overall_pass

if __name__ == '__main__':
    try:
        monitor = SystemIntegrationMonitor()
        
        # Run 30-second integration test
        success = monitor.run_integration_test(30)
        
        if success:
            rospy.loginfo("🎉 SYSTEM INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            rospy.logerr("❌ SYSTEM INTEGRATION TEST FAILED!")
            sys.exit(1)
            
    except rospy.ROSInterruptException:
        rospy.loginfo("Integration test interrupted")
    except Exception as e:
        rospy.logerr(f"Integration test failed: {e}")
        sys.exit(1)
