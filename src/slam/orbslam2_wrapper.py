# ORB-SLAM3 Wrapper for Python Integration
import cv2
import numpy as np
import subprocess
import os
import sys

# Add ORB-SLAM3 Python wrapper to path
sys.path.append('/opt/ORB_SLAM3')

try:
    from python_wrapper import ORBSLAM3
    ORBSLAM3_AVAILABLE = True
except ImportError:
    ORBSLAM3_AVAILABLE = False

class ORB_SLAM3_Wrapper:
    """
    Wrapper for ORB-SLAM3 system
    Provides simplified interface for ROS integration
    """
    
    def __init__(self, vocab_path, config_path, sensor_type="RGBD"):
        """
        Initialize ORB-SLAM3 wrapper
        
        Args:
            vocab_path: Path to ORB vocabulary file
            config_path: Path to configuration YAML file
            sensor_type: "MONOCULAR", "STEREO", "RGBD", "MONOCULAR_INERTIAL", "STEREO_INERTIAL"
        """
        self.vocab_path = vocab_path
        self.config_path = config_path
        self.sensor_type = sensor_type
        self.slam_system = None
        self.initialized = False
        
        if ORBSLAM3_AVAILABLE:
            try:
                self.slam_system = ORBSLAM3(vocab_path, config_path, sensor_type)
                self.slam_system.start()
                self.initialized = True
            except Exception as e:
                print(f"Failed to initialize ORB-SLAM3: {e}")
                self.initialized = False
        else:
            print("ORB-SLAM3 not available - install using scripts/install_orbslam3.sh")
            self.initialized = False

    def is_initialized(self):
        """Check if SLAM system is properly initialized"""
        return self.initialized and self.slam_system and self.slam_system.is_alive()

    def process_frame(self, image, timestamp):
        """
        Process frame with ORB-SLAM3
        Note: ORB-SLAM3 ROS nodes handle frame processing internally
        This is a placeholder for direct API integration
        """
        if not self.is_initialized():
            return None
        
        # In practice, ORB-SLAM3 processes frames through ROS topics
        # Direct API integration would require C++ bindings
        return None

    def shutdown(self):
        """Shutdown ORB-SLAM3 system"""
        if self.slam_system:
            self.slam_system.stop()
            self.initialized = False

    def get_map(self):
        """Get current map points"""
        # Placeholder - would need direct API access
        return []

    def reset(self):
        """Reset SLAM system"""
        if self.slam_system:
            self.slam_system.stop()
            self.slam_system.start()

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.shutdown()

# Legacy compatibility
def initialize_slam(config_file):
    """Legacy function for compatibility"""
    vocab_path = "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    return ORB_SLAM3_Wrapper(vocab_path, config_file)