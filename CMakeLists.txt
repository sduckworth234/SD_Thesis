cmake_minimum_required(VERSION 3.10)
project(SD_Thesis)

set(CMAKE_CXX_STANDARD 14)

# Include directories
include_directories(include)

# Add subdirectories for each module
add_subdirectory(src/slam)
add_subdirectory(src/enhancement)
add_subdirectory(src/detection)
add_subdirectory(src/camera)
add_subdirectory(src/ros_nodes)

# Specify the executable for the main application
add_executable(thesis_app src/main.cpp)

# Link libraries
target_link_libraries(thesis_app
    slam_lib
    enhancement_lib
    detection_lib
    camera_lib
    ros_nodes_lib
)

# Install rules
install(TARGETS thesis_app DESTINATION bin)