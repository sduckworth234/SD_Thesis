#!/usr/bin/env python3
"""Quick ORB-SLAM3 status check"""

import os
import sys

def main():
    print("🔍 Quick ORB-SLAM3 Status Check")
    print("=" * 40)
    
    # Check installation
    if os.path.exists("/opt/ORB_SLAM3"):
        print("✅ ORB-SLAM3 installation directory found")
    else:
        print("❌ ORB-SLAM3 installation directory missing")
        return False
    
    # Check library
    if os.path.exists("/opt/ORB_SLAM3/lib/libORB_SLAM3.so"):
        print("✅ ORB-SLAM3 library found")
    else:
        print("❌ ORB-SLAM3 library missing")
        return False
    
    # Check executables
    rgbd_exe = "/opt/ORB_SLAM3/Examples/RGB-D/rgbd_realsense_D435i"
    if os.path.exists(rgbd_exe):
        print("✅ RGB-D executable found")
    else:
        print("❌ RGB-D executable missing")
        return False
    
    # Check vocabulary
    vocab_path = "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    if os.path.exists(vocab_path):
        print("✅ ORB vocabulary found")
    else:
        print("❌ ORB vocabulary missing")
        return False
    
    # Check Python wrapper
    wrapper_path = "/opt/ORB_SLAM3/python_wrapper.py"
    if os.path.exists(wrapper_path):
        print("✅ Python wrapper found")
    else:
        print("❌ Python wrapper missing")
        return False
    
    print("\n🎉 All components found! ORB-SLAM3 is ready!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
