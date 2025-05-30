# Low-Light Vision for Search and Rescue UAV Systems

**SD Thesis Project - 2025**

This repository contains the complete codebase, documentation, and experimental data for a thesis project focused on developing an autonomous UAV system for search and rescue operations in low-light environments. The project integrates multiple computer vision technologies including ORB-SLAM2, YOLO v4, ZERO-DCE++, SCI, and other low-light image enhancement models.

## 🎯 Project Objective

Develop and benchmark a modular UAV system equipped with an Intel RealSense D435i camera capable of:
- **Autonomous navigation** in GPS-denied environments using visual SLAM
- **Person detection** in low-light conditions using enhanced YOLO v4
- **Real-time image enhancement** for improved visibility in dark disaster scenarios
- **Edge deployment** on Jetson Xavier NX for real-world applications

## 🚁 Target Application

Search and rescue operations in dark, disaster-stricken environments where traditional lighting is unavailable and GPS signals are compromised.

## 🏗️ System Architecture

### Hardware Platforms
- **Development Environment**: Ubuntu 20.04, AMD CPU, NVIDIA RTX 4070
- **Target Deployment**: Jetson Xavier NX (Ubuntu 18.04, aarch64)
- **Camera**: Intel RealSense D435i depth camera

### Software Stack
- **ROS Noetic** for system integration and communication
- **ORB-SLAM2** for visual simultaneous localization and mapping
- **YOLO v4** for real-time person detection
- **ZERO-DCE++** and **SCI** for low-light image enhancement
- **Gazebo** and **RViz** for simulation and visualization

## 📁 Project Structure

- **src/**: Contains the source code for various functionalities.
  - **camera/**: Code for capturing images and calibrating the camera.
  - **detection/**: Implementation of person detection using YOLO v4.
  - **enhancement/**: Low-light image enhancement models including ZERO-DCE++ and SCI.
  - **slam/**: SLAM functionalities for mapping and localization.
  - **ros_nodes/**: ROS nodes for managing camera, detection, and SLAM operations.

- **config/**: Configuration files for camera parameters, YOLO model, and ORB-SLAM2.

- **scripts/**: Shell scripts for setting up the environment, installing dependencies, and deploying to Jetson Xavier NX.

- **tests/**: Unit and integration tests for ensuring code quality and functionality.

- **data/**: Directories for datasets, benchmarks, and results.

- **docs/**: Documentation including methodology, setup instructions, deployment guides, and benchmarking plans.

- **logs/**: Directory for storing experiment logs.

- **launch/**: Launch files for running simulations and real hardware setups.

- **requirements.txt**: List of Python dependencies required for the project.

- **CMakeLists.txt**: Build configuration for C++ components.

- **package.xml**: Metadata for the ROS package.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd SD_Thesis
   ```

2. Set up the environment:
   ```
   ./scripts/setup_environment.sh
   ```

3. Install dependencies:
   ```
   ./scripts/install_dependencies.sh
   ```

## Usage

- To run the camera node:
  ```
  rosrun ros_nodes camera_node.py
  ```

- To perform person detection:
  ```
  rosrun ros_nodes detection_node.py
  ```

- To run SLAM:
  ```
  rosrun ros_nodes slam_node.py
  ```

## Future Work

- Implement additional low-light enhancement models.
- Optimize the system for deployment on Jetson Xavier NX.
- Develop a comprehensive testing and benchmarking methodology.

## Acknowledgments

Special thanks to all contributors and resources that made this project possible.