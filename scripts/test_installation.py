#!/usr/bin/env python3
"""
Test script to verify the complete installation and setup
"""

import sys
import os
import subprocess
import importlib.util

def test_python_dependencies():
    """Test Python package dependencies"""
    print("Testing Python dependencies...")
    
    # Essential packages that should be available
    required_packages = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'rospy': 'rospy',
        'pyrealsense2': 'pyrealsense2',
        'torch': 'torch',
        'torchvision': 'torchvision'
    }
    
    failed_imports = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} imported successfully")
        except ImportError as e:
            print(f"✗ {package_name} import failed: {e}")
            failed_imports.append(package_name)
    
    return len(failed_imports) == 0

def test_orbslam3_installation():
    """Test ORB-SLAM3 installation"""
    print("\nTesting ORB-SLAM3 installation...")
    
    # Check if ORB-SLAM3 directory exists
    orbslam3_path = "/home/duck/Desktop/SD_Thesis/third_party/ORB_SLAM3"
    if not os.path.exists(orbslam3_path):
        print(f"✗ ORB-SLAM3 directory not found at {orbslam3_path}")
        return False
    
    print(f"✓ ORB-SLAM3 directory found at {orbslam3_path}")
    
    # Check for essential files
    essential_files = [
        "CMakeLists.txt",
        "Examples/Monocular/mono_realsense_D435i",
        "Examples/RGB-D/rgbd_realsense_D435i",
        "lib/libORB_SLAM3.so"
    ]
    
    missing_files = []
    for file_path in essential_files:
        full_path = os.path.join(orbslam3_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ Found {file_path}")
        else:
            print(f"✗ Missing {file_path}")
            missing_files.append(file_path)
    
    # Check Python wrapper
    python_wrapper_path = os.path.join(orbslam3_path, "python_wrapper.so")
    if os.path.exists(python_wrapper_path):
        print("✓ Python wrapper found")
        
        # Try importing the wrapper
        try:
            sys.path.append(orbslam3_path)
            import python_wrapper
            print("✓ Python wrapper imported successfully")
            return len(missing_files) == 0
        except ImportError as e:
            print(f"✗ Python wrapper import failed: {e}")
            return False
    else:
        print("✗ Python wrapper not found")
        return False

def test_ros_setup():
    """Test ROS setup"""
    print("\nTesting ROS setup...")
    
    # Check if ROS is sourced
    ros_package_path = os.environ.get('ROS_PACKAGE_PATH')
    if ros_package_path:
        print(f"✓ ROS_PACKAGE_PATH set: {ros_package_path[:100]}...")
    else:
        print("✗ ROS_PACKAGE_PATH not set")
        return False
    
    # Check if roscore is available
    try:
        result = subprocess.run(['which', 'roscore'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ roscore command found")
        else:
            print("✗ roscore command not found")
            return False
    except Exception as e:
        print(f"✗ Error checking roscore: {e}")
        return False
    
    return True

def test_camera_setup():
    """Test camera setup"""
    print("\nTesting camera setup...")
    
    try:
        import pyrealsense2 as rs
        
        # Try to get RealSense context
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) > 0:
            for device in devices:
                print(f"✓ Found RealSense device: {device.get_info(rs.camera_info.name)}")
            return True
        else:
            print("⚠ No RealSense devices found (this is OK if testing without hardware)")
            return True  # Return True as this might be expected
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== Installation Test Suite ===\n")
    
    test_results = {
        "Python Dependencies": test_python_dependencies(),
        "ORB-SLAM3 Installation": test_orbslam3_installation(),
        "ROS Setup": test_ros_setup(),
        "Camera Setup": test_camera_setup()
    }
    
    print("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The installation appears to be working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
