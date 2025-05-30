# Setup Instructions for SD_Thesis Project

This document provides step-by-step instructions for setting up the SD_Thesis project environment.

## Prerequisites

Before you begin, ensure you have the following installed:

- Ubuntu 20.04 (for development)
- Python 3.x
- ROS Noetic
- CMake
- Git

## Step 1: Clone the Repository

Clone the GitHub repository to your local machine:

```bash
git clone <repository_url>
cd SD_Thesis
```

## Step 2: Set Up the Python Environment

Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Step 3: Install ROS Dependencies

Make sure you have the necessary ROS packages installed. You can install them using:

```bash
sudo apt-get update
sudo apt-get install ros-noetic-<package_name>
```

Replace `<package_name>` with the required packages for your project.

## Step 4: Build the Project

If you have C++ components, navigate to the root of the project and build it using CMake:

```bash
mkdir build
cd build
cmake ..
make
```

## Step 5: Run Setup Scripts

Run the setup scripts to configure the environment:

```bash
bash scripts/setup_environment.sh
bash scripts/install_dependencies.sh
```

## Step 6: Camera Calibration

Before using the camera, ensure it is calibrated. Run the calibration script:

```bash
python src/camera/camera_calibration.py
```

## Step 7: Launch the ROS Nodes

You can launch the ROS nodes using the provided launch files:

For simulation:

```bash
roslaunch launch/simulation.launch
```

For real hardware:

```bash
roslaunch launch/real_hardware.launch
```

## Step 8: Testing

Run the unit tests to ensure everything is working correctly:

```bash
pytest tests/unit_tests/test_detection.py
```

## Step 9: Deployment on Jetson Xavier NX

Follow the instructions in `docs/jetson_deployment.md` for deploying the project on the Jetson Xavier NX.

## Conclusion

You are now set up to work on the SD_Thesis project. Refer to the other documentation files for more detailed instructions on methodology, benchmarking, and deployment.