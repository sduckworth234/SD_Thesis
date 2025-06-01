#!/usr/bin/env python3
"""
Comprehensive ORB-SLAM3 Integration Test
Tests the complete ORB-SLAM3 installation and ROS integration
"""

import sys
import os
import rospy

def test_orbslam3_installation():
    """Test ORB-SLAM3 basic installation"""
    print("🔍 Testing ORB-SLAM3 Installation...")
    
    # Test Python wrapper import
    sys.path.append('/opt/ORB_SLAM3')
    try:
        from python_wrapper import ORBSLAM3
        print("✅ ORB-SLAM3 Python wrapper imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import ORB-SLAM3: {e}")
        return False
    
    # Test wrapper instantiation
    try:
        slam = ORBSLAM3(
            vocab_path='/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt',
            config_path='/home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml',
            sensor_type='RGBD'
        )
        print("✅ ORB-SLAM3 wrapper instantiates successfully")
        
        # Check executables
        if os.path.exists(slam.executables['RGBD']):
            print(f"✅ RGBD executable found: {slam.executables['RGBD']}")
        else:
            print(f"❌ RGBD executable not found: {slam.executables['RGBD']}")
            return False
            
        slam.stop()  # Clean shutdown
        return True
        
    except Exception as e:
        print(f"❌ Failed to create ORB-SLAM3 wrapper: {e}")
        return False

def test_project_integration():
    """Test project-specific integration"""
    print("\n🔧 Testing Project Integration...")
    
    # Test project wrapper
    sys.path.append('/home/duck/Desktop/SD_Thesis/src/slam')
    try:
        from orbslam2_wrapper import ORB_SLAM3_Wrapper
        print("✅ Project ORB-SLAM3 wrapper imports successfully")
        
        wrapper = ORB_SLAM3_Wrapper(
            '/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt',
            '/home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml'
        )
        print("✅ Project wrapper instantiates successfully")
        wrapper.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Failed to test project integration: {e}")
        return False

def test_config_files():
    """Test configuration files"""
    print("\n📋 Testing Configuration Files...")
    
    configs = [
        '/home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml',
        '/home/duck/Desktop/SD_Thesis/config/orbslam3_jetson.yaml',
        '/home/duck/Desktop/SD_Thesis/models/orbslam3/ORBvoc.txt'
    ]
    
    all_good = True
    for config in configs:
        if os.path.exists(config):
            print(f"✅ {os.path.basename(config)}")
        else:
            print(f"❌ {os.path.basename(config)} not found")
            all_good = False
    
    return all_good

def main():
    """Main test function"""
    print("🚀 ORB-SLAM3 Integration Test")
    print("=" * 50)
    
    test1 = test_orbslam3_installation()
    test2 = test_project_integration() 
    test3 = test_config_files()
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS")
    print("=" * 50)
    
    if test1 and test2 and test3:
        print("🎉 ALL TESTS PASSED!")
        print("\n✨ Your ORB-SLAM3 installation is ready!")
        print("\n📝 Next Steps:")
        print("1. Test camera: roslaunch sd_thesis test_camera.launch")
        print("2. Test SLAM: roslaunch sd_thesis test_slam.launch")
        print("3. Run full pipeline: roslaunch sd_thesis complete_pipeline.launch")
        print("\n💡 For Jetson deployment, use config/orbslam3_jetson.yaml")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        
    return test1 and test2 and test3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
