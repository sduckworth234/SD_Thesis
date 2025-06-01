#!/usr/bin/env python3
"""
Quick verification script for ORB-SLAM3 installation
Tests if ORB-SLAM3 components are properly installed
"""

import os
import sys
import subprocess

def check_file_exists(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (NOT FOUND)")
        return False

def check_orbslam3_installation():
    """Check ORB-SLAM3 installation"""
    print("🔍 Checking ORB-SLAM3 Installation...")
    print("=" * 50)
    
    success = True
    
    # Check main installation directory
    success &= check_file_exists("/opt/ORB_SLAM3", "ORB-SLAM3 Installation Directory")
    
    # Check library
    success &= check_file_exists("/opt/ORB_SLAM3/lib/libORB_SLAM3.so", "ORB-SLAM3 Library")
    
    # Check vocabulary
    success &= check_file_exists("/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt", "ORB Vocabulary")
    
    # Check standalone executables (working)
    standalone_executables = [
        "/opt/ORB_SLAM3/Examples/Monocular/mono_euroc",
        "/opt/ORB_SLAM3/Examples/Stereo/stereo_euroc", 
        "/opt/ORB_SLAM3/Examples/RGB-D/rgbd_realsense_D435i"
    ]
    
    standalone_ok = True
    for exe in standalone_executables:
        if check_file_exists(exe, f"Standalone Executable {os.path.basename(exe)}"):
            standalone_ok = True
        else:
            standalone_ok = False
    
    # Check ROS executables (optional - may not be built)
    ros_executables = [
        "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Mono",
        "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Stereo", 
        "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/RGBD"
    ]
    
    ros_ok = True
    for exe in ros_executables:
        if not check_file_exists(exe, f"ROS Executable {os.path.basename(exe)}"):
            ros_ok = False
    
    if not ros_ok:
        print("ℹ️  ROS executables not available - using standalone executables")
        
    # Update success based on standalone availability
    success = success and standalone_ok
    
    # Check Python wrapper
    success &= check_file_exists("/opt/ORB_SLAM3/python_wrapper.py", "Python Wrapper")
    
    # Check project configuration files
    project_files = [
        "/home/duck/Desktop/SD_Thesis/models/orbslam3/ORBvoc.txt",
        "/home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml",
        "/home/duck/Desktop/SD_Thesis/config/orbslam3_jetson.yaml"
    ]
    
    for file_path in project_files:
        success &= check_file_exists(file_path, f"Project Config {os.path.basename(file_path)}")
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 ORB-SLAM3 installation appears to be complete!")
        print("\n📝 Next steps:")
        print("1. Run 'source ~/.bashrc' to load environment variables")
        print("2. Test ROS integration: roslaunch sd_thesis test_camera.launch")
        print("3. Test complete pipeline: roslaunch sd_thesis complete_pipeline.launch")
    else:
        print("⚠️  ORB-SLAM3 installation has missing components")
        print("\n🔧 To fix:")
        print("1. Re-run: ./scripts/install_orbslam3.sh")
        print("2. Check installation log: /tmp/orbslam3_install.log")
        print("3. Install missing dependencies manually")
    
    return success

def check_ros_environment():
    """Check ROS environment setup"""
    print("\n🤖 Checking ROS Environment...")
    print("=" * 50)
    
    # Check ROS installation
    try:
        result = subprocess.run(['rosversion', '-d'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ ROS Distribution: {result.stdout.strip()}")
        else:
            print("❌ ROS not properly installed")
            return False
    except FileNotFoundError:
        print("❌ ROS not found in PATH")
        return False
    
    # Check ROS package path
    ros_package_path = os.environ.get('ROS_PACKAGE_PATH', '')
    if '/opt/ORB_SLAM3/Examples/ROS' in ros_package_path:
        print("✅ ORB-SLAM3 ROS package path configured")
    else:
        print("⚠️  ORB-SLAM3 not in ROS_PACKAGE_PATH")
        print("   Add this line to ~/.bashrc:")
        print("   export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/opt/ORB_SLAM3/Examples/ROS")
    
    return True

def check_dependencies():
    """Check key dependencies"""
    print("\n📦 Checking Dependencies...")
    print("=" * 50)
    
    dependencies = [
        ('opencv-python', 'import cv2'),
        ('numpy', 'import numpy'),
        ('yaml', 'import yaml'),
        ('rospy', 'import rospy')
    ]
    
    success = True
    for name, import_cmd in dependencies:
        try:
            exec(import_cmd)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (missing)")
            success = False
    
    return success

def main():
    """Main verification function"""
    print("🚀 SD_Thesis ORB-SLAM3 Verification")
    print("=" * 50)
    
    orbslam3_ok = check_orbslam3_installation()
    ros_ok = check_ros_environment() 
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if orbslam3_ok and ros_ok and deps_ok:
        print("🎉 ALL CHECKS PASSED!")
        print("Your system is ready to run the SD_Thesis pipeline with ORB-SLAM3")
        sys.exit(0)
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("Please fix the issues above before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()
