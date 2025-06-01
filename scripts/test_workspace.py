#!/usr/bin/env python3

import sys
import os

print("=== Workspace Verification ===")

# Check Python version
print(f"Python version: {sys.version}")

# Check current working directory
print(f"Current directory: {os.getcwd()}")

# Check if we're in the right workspace
expected_files = ["CMakeLists.txt", "package.xml", "src", "launch"]
for file in expected_files:
    if os.path.exists(file):
        print(f"✓ {file} found")
    else:
        print(f"✗ {file} missing")

# Test basic imports
print("\n=== Import Tests ===")

# Test numpy and opencv
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

# Test dummy SLAM
try:
    sys.path.append(os.path.join("src", "slam"))
    from dummy_slam import DummySLAM
    print("✓ Dummy SLAM imported")
    
    # Test basic functionality
    slam = DummySLAM()
    slam.start()
    pose = slam.get_pose()
    print(f"✓ Dummy SLAM pose generated: {pose is not None}")
    slam.stop()
    
except Exception as e:
    print(f"✗ Dummy SLAM: {e}")

print("\n=== ROS Environment ===")
ros_vars = ["ROS_DISTRO", "ROS_MASTER_URI", "ROS_PACKAGE_PATH"]
for var in ros_vars:
    value = os.environ.get(var, "Not set")
    print(f"{var}: {value}")

print("\nVerification complete!")
