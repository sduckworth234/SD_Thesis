#!/bin/bash
# Comprehensive script to launch ORB-SLAM3 with RealSense D435i and proper library paths

# Kill any existing ROS processes
echo "Stopping any existing ROS processes..."
killall -9 rosmaster roscore || true
sleep 1

# Set up environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib:/opt/ORB_SLAM3/Thirdparty/Sophus/lib
source /opt/ros/noetic/setup.bash
source /home/duck/Desktop/orb_slam3_ws/devel/setup.bash

# Start roscore
echo "Starting roscore..."
roscore &
ROSCORE_PID=$!
sleep 2

# Start RealSense camera with optimized parameters
echo "Starting RealSense camera..."
roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  enable_pointcloud:=false \
  color_width:=640 color_height:=480 \
  depth_width:=640 depth_height:=480 \
  color_fps:=30 depth_fps:=30 \
  initial_reset:=false \
  reconnect_timeout:=6.0 \
  wait_for_device_timeout:=10.0 &
CAMERA_PID=$!

# Wait for camera to initialize
echo "Waiting for camera to initialize..."
sleep 5

# Publish static transforms
echo "Publishing static transforms..."
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
MAPTOODOM_PID=$!
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link &
ODOMTOBASE_PID=$!
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link camera_link &
BASETOCOLOR_PID=$!

# Check if camera topics are available
echo "Checking camera topics..."
for i in {1..10}; do
  if rostopic list | grep -q "/camera/color/image_raw"; then
    echo "Camera topics found!"
    break
  fi
  echo "Waiting for camera topics... ($i/10)"
  sleep 1
  if [ $i -eq 10 ]; then
    echo "ERROR: Camera topics not found after waiting. Check camera connection."
    exit 1
  fi
done

# Start ORB-SLAM3 node
echo "Starting ORB-SLAM3 C++ node..."
rosrun sd_thesis orb_slam3_node \
  _vocabulary_file:=/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  _settings_file:=/home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml \
  _orb_slam3_path:=/opt/ORB_SLAM3 \
  /camera/color/image_raw:=/camera/color/image_raw \
  /camera/depth/image_rect_raw:=/camera/aligned_depth_to_color/image_raw &
SLAM_PID=$!

# Start RViz with visualization configuration
echo "Starting RViz..."
rosrun rviz rviz -d /home/duck/Desktop/SD_Thesis/config/orb_slam3_visualization.rviz &
RVIZ_PID=$!

# Register cleanup function
cleanup() {
  echo "Shutting down..."
  kill $RVIZ_PID 2>/dev/null
  kill $SLAM_PID 2>/dev/null
  kill $BASETOCOLOR_PID 2>/dev/null
  kill $ODOMTOBASE_PID 2>/dev/null
  kill $MAPTOODOM_PID 2>/dev/null
  kill $CAMERA_PID 2>/dev/null
  kill $ROSCORE_PID 2>/dev/null
  killall -9 rosmaster roscore 2>/dev/null
  echo "Cleanup complete."
  exit 0
}

# Register trap
trap cleanup INT TERM

# Keep script running
echo "Complete pipeline is running! Press Ctrl+C to exit."
while true; do
  sleep 1
done
