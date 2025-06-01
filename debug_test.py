#!/usr/bin/env python3

import sys
import os

print("Starting debug test...")

# Test basic functionality
try:
    import numpy as np
    print("✓ NumPy imported")
    
    # Test dummy SLAM import
    sys.path.insert(0, os.path.join(os.getcwd(), "src", "slam"))
    print("Added path:", os.path.join(os.getcwd(), "src", "slam"))
    
    from dummy_slam import DummySLAM
    print("✓ DummySLAM imported")
    
    # Test instantiation
    slam = DummySLAM()
    print("✓ DummySLAM instantiated")
    
    # Test methods
    slam.start()
    print("✓ SLAM started")
    
    pose = slam.get_pose()
    print("✓ Pose obtained:", type(pose))
    
    slam.stop()
    print("✓ SLAM stopped")
    
    print("All tests passed!")
    
except Exception as e:
    print("Error:", str(e))
    import traceback
    traceback.print_exc()
