#!/bin/bash
# Add ORB-SLAM3 library paths to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib:/opt/ORB_SLAM3/Thirdparty/Sophus/lib

# Source ROS setup
source /home/duck/Desktop/orb_slam3_ws/devel/setup.bash

# Run the ORB-SLAM3 node
rosrun sd_thesis orb_slam3_node "$@"
