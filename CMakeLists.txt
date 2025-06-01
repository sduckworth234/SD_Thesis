cmake_minimum_required(VERSION 3.10)
project(sd_thesis)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DCOMPILEDWITHC11")

# Find catkin and required components
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  message_filters
  std_msgs
)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find Pangolin (optional)
find_package(Pangolin QUIET)
if(NOT Pangolin_FOUND)
    message(WARNING "Pangolin not found, some features may be disabled")
endif()

# ORB-SLAM3 paths
set(ORB_SLAM3_ROOT_PATH /opt/ORB_SLAM3)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${ORB_SLAM3_ROOT_PATH}/include
  ${ORB_SLAM3_ROOT_PATH}/include/CameraModels
  ${ORB_SLAM3_ROOT_PATH}
  ${ORB_SLAM3_ROOT_PATH}/Thirdparty/DBoW2
  ${ORB_SLAM3_ROOT_PATH}/Thirdparty/g2o
  ${ORB_SLAM3_ROOT_PATH}/Thirdparty/Sophus
)

# Catkin package setup
catkin_package(
  CATKIN_DEPENDS 
    roscpp 
    rospy 
    sensor_msgs 
    geometry_msgs 
    nav_msgs 
    visualization_msgs 
    cv_bridge 
    tf2 
    tf2_ros 
    tf2_geometry_msgs
    message_filters
    std_msgs
)

# ORB-SLAM3 ROS Node
add_executable(orb_slam3_node src/ros_nodes/orb_slam3_node.cpp)

# Set RPATH for runtime library loading
set_target_properties(orb_slam3_node PROPERTIES
  INSTALL_RPATH "${ORB_SLAM3_ROOT_PATH}/lib:${ORB_SLAM3_ROOT_PATH}/Thirdparty/DBoW2/lib:${ORB_SLAM3_ROOT_PATH}/Thirdparty/g2o/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

target_link_libraries(orb_slam3_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${ORB_SLAM3_ROOT_PATH}/lib/libORB_SLAM3.so
  ${ORB_SLAM3_ROOT_PATH}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_ROOT_PATH}/Thirdparty/g2o/lib/libg2o.so
)

if(Pangolin_FOUND)
  target_link_libraries(orb_slam3_node ${Pangolin_LIBRARIES})
endif()

# Install Python scripts
catkin_install_python(PROGRAMS
  src/ros_nodes/orb_slam3_node.py
  src/ros_nodes/camera_node.py
  src/ros_nodes/detection_node.py
  src/ros_nodes/enhancement_node.py
  src/ros_nodes/slam_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)