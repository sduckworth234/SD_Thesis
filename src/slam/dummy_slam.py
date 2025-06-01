#!/usr/bin/env python3
"""
Dummy SLAM implementation for testing purposes
Provides basic pose estimation without actual SLAM
"""

import numpy as np
import time

class DummySLAM:
    """Simple dummy SLAM for testing pipeline integration"""
    
    def __init__(self, vocab_path=None, config_path=None, sensor_type="RGBD"):
        self.is_initialized = False
        self.pose = np.eye(4)  # Identity matrix
        self.trajectory = []
        self.start_time = time.time()
        
    def start(self):
        """Initialize dummy SLAM"""
        self.is_initialized = True
        print("Dummy SLAM started - providing mock pose data")
        
    def stop(self):
        """Stop dummy SLAM"""
        self.is_initialized = False
        print("Dummy SLAM stopped")
        
    def is_alive(self):
        """Check if SLAM is running"""
        return self.is_initialized
        
    def get_pose(self):
        """Get current pose (mock data - simple forward motion)"""
        if not self.is_initialized:
            return None
            
        # Simulate simple forward motion
        elapsed = time.time() - self.start_time
        x = elapsed * 0.1  # Move forward at 0.1 m/s
        
        pose = np.eye(4)
        pose[0, 3] = x  # X translation
        
        self.pose = pose
        self.trajectory.append(pose.copy())
        
        return pose
        
    def get_trajectory(self):
        """Get full trajectory"""
        return self.trajectory
        
    def get_map_points(self):
        """Get map points (mock data)"""
        # Return some random 3D points
        num_points = 100
        points = np.random.randn(num_points, 3) * 2.0  # Random points in 3D
        return points

# For backward compatibility
ORBSLAM3 = DummySLAM
