# Complete Setup Instructions for Ubuntu 20.04

This guide provides step-by-step instructions to set up the complete SD_Thesis environment on Ubuntu 20.04, including all dependencies for ORB-SLAM2, YOLO v4, low-light enhancement models, and Intel RealSense camera integration.

## 🖥️ System Requirements

- **OS**: Ubuntu 20.04 LTS (Desktop or Server)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060+ or better)
- **Storage**: 50GB+ free space
- **Camera**: Intel RealSense D435i
- **Internet**: Required for downloading dependencies

## 📋 Pre-Installation Checklist

Before starting, ensure you have:
- [ ] Fresh Ubuntu 20.04 installation
- [ ] sudo privileges
- [ ] Stable internet connection
- [ ] NVIDIA GPU drivers installed (if using GPU)
- [ ] Intel RealSense D435i camera available

## 🚀 Quick Start (Automated Setup)

For experienced users, run the automated setup script:

```bash
git clone https://github.com/sduckworth234/SD_Thesis.git
cd SD_Thesis
chmod +x scripts/complete_setup.sh
./scripts/complete_setup.sh
```

For detailed step-by-step setup, continue with the manual installation below. for SD_Thesis Project

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

## 🔧 Manual Installation Guide

### Step 1: Update System and Install Basic Dependencies

```bash
# Update package lists and upgrade system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python development tools
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel
```

### Step 2: Install NVIDIA CUDA (GPU Support)

⚠️ **Important**: Only if you have an NVIDIA GPU and want GPU acceleration.

```bash
# Remove any existing CUDA installations
sudo apt remove --purge nvidia-* -y
sudo apt autoremove -y

# Install NVIDIA driver repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA 11.8 (compatible with most deep learning frameworks)
sudo apt install -y cuda-11-8

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Step 3: Install ROS Noetic

```bash
# Setup ROS Noetic repository
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update

# Install ROS Noetic Desktop Full
sudo apt install -y ros-noetic-desktop-full

# Install additional ROS packages
sudo apt install -y \
    ros-noetic-vision-opencv \
    ros-noetic-image-transport \
    ros-noetic-cv-bridge \
    ros-noetic-sensor-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-rosbridge-server \
    ros-noetic-usb-cam \
    ros-noetic-camera-info-manager

# Initialize rosdep
sudo rosdep init
rosdep update

# Setup ROS environment
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify ROS installation
roscore &
sleep 5
pkill -f roscore
echo "ROS Noetic installation complete"
```

### Step 4: Install Intel RealSense SDK

```bash
# Install Intel RealSense repository key
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

# Add Intel RealSense repository
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt update

# Install Intel RealSense SDK
sudo apt install -y \
    librealsense2-dkms \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg

# Install Python bindings
pip3 install pyrealsense2

# Test RealSense installation (connect camera first)
# realsense-viewer
```

### Step 5: Install OpenCV with Additional Modules

```bash
# Remove any existing OpenCV installations
sudo apt remove python3-opencv -y
pip3 uninstall opencv-python opencv-contrib-python -y

# Install OpenCV dependencies
sudo apt install -y \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libtbb-dev \
    libeigen3-dev \
    libglew-dev \
    libpostproc-dev \
    libopenexr-dev \
    libgdal-dev

# Install OpenCV with contrib modules (for additional features)
pip3 install opencv-contrib-python==4.5.5.64
```

### Step 6: Install Deep Learning Frameworks

```bash
# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# For CPU-only PyTorch (if no GPU):
# pip3 install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow with GPU support
pip3 install tensorflow-gpu==2.8.0

# Install additional ML dependencies
pip3 install \
    scikit-learn \
    scikit-image \
    imgaug \
    albumentations \
    timm \
    einops
```

### Step 7: Install ORB-SLAM2

```bash
# Install ORB-SLAM2 dependencies
sudo apt install -y \
    libglew-dev \
    libeigen3-dev \
    libsuitesparse-dev

# Install Pangolin (visualization library)
cd ~/
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Install ORB-SLAM2
cd ~/
git clone https://github.com/raulmur/ORB_SLAM2.git
cd ORB_SLAM2

# Build DBoW2
cd Thirdparty/DBoW2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build g2o
cd ../../g2o
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build ORB-SLAM2
cd ../../..
chmod +x build.sh
./build.sh

# Build ROS package
chmod +x build_ros.sh
./build_ros.sh

# Add ORB-SLAM2 to environment
echo "export ORB_SLAM2_ROOT_PATH=~/ORB_SLAM2" >> ~/.bashrc
source ~/.bashrc
```

### Step 8: Install YOLO v4

```bash
# Install Darknet (YOLO v4 implementation)
cd ~/
git clone https://github.com/AlexeyAB/darknet.git
cd darknet

# Edit Makefile for GPU support (if available)
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile

# Compile Darknet
make -j$(nproc)

# Download YOLO v4 weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

# Test YOLO v4 installation
# ./darknet detect cfg/yolov4.cfg yolov4.weights data/dog.jpg

# Install Python YOLO wrapper
pip3 install ultralytics
```

### Step 9: Install Low-Light Enhancement Models

#### ZERO-DCE++

```bash
# Create models directory
mkdir -p ~/SD_Thesis_Models/zero_dce
cd ~/SD_Thesis_Models/zero_dce

# Clone ZERO-DCE++ repository
git clone https://github.com/Li-Chongyi/Zero-DCE_extension.git
cd Zero-DCE_extension

# Install specific dependencies
pip3 install \
    kornia==0.6.12 \
    lpips==0.1.4 \
    pytorch-lightning==1.8.6

# Download pre-trained weights
mkdir -p checkpoints
# Note: You'll need to download weights from the official repository or train your own
```

#### SCI (Self-Calibrating Illumination)

```bash
# Create SCI directory
mkdir -p ~/SD_Thesis_Models/sci
cd ~/SD_Thesis_Models/sci

# Clone SCI repository
git clone https://github.com/vis-opt-group/SCI.git
cd SCI

# Install dependencies
pip3 install \
    colour-science \
    rawpy \
    imageio

# Download pre-trained weights
# Note: Follow the official repository instructions for downloading weights
```

#### Additional Low-Light Models

```bash
# Install common dependencies for low-light enhancement
pip3 install \
    retinex-py \
    lime-py \
    clahe
```

### Step 10: Setup Project Environment

```bash
# Clone the SD_Thesis repository
cd ~/
git clone https://github.com/sduckworth234/SD_Thesis.git
cd SD_Thesis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Setup catkin workspace
cd ~/
mkdir -p catkin_ws/src
cd catkin_ws/src
ln -s ~/SD_Thesis .
cd ..
catkin_make

# Source the workspace
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Create necessary directories
mkdir -p ~/SD_Thesis/data/{datasets,models,checkpoints,results}
mkdir -p ~/SD_Thesis/logs/experiment_logs
```

## 🔌 Hardware Setup: Intel RealSense D435i

### Step 1: Connect the Camera

1. **Connect the camera** to a USB 3.0 port (blue connector)
2. **Wait for recognition** (should appear in `lsusb`)
3. **Verify connection**:

```bash
# Check if camera is detected
lsusb | grep Intel

# Test camera with RealSense viewer
realsense-viewer

# Test Python interface
python3 -c "import pyrealsense2 as rs; print('RealSense Python OK')"
```

### Step 2: Camera Permissions

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Set udev rules for RealSense
sudo cp ~/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Log out and log back in for group changes to take effect
```

### Step 3: Test Camera Integration

```bash
# Activate environment
cd ~/SD_Thesis
source venv/bin/activate

# Test ROS camera node
roslaunch launch/test_camera.launch

# In another terminal, check camera topics
rostopic list | grep camera
rostopic echo /camera/color/image_raw/header
```

## 🧪 Testing Individual Components

### Test 1: RealSense Camera

```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Run camera node
cd ~/SD_Thesis
source venv/bin/activate
python src/ros_nodes/camera_node.py

# Terminal 3: Verify camera data
rostopic hz /camera/color/image_raw
rostopic hz /camera/depth/image_raw
```

### Test 2: YOLO v4 Detection

```bash
# Test standalone YOLO
cd ~/darknet
./darknet detect cfg/yolov4.cfg yolov4.weights data/dog.jpg

# Test with Python wrapper
cd ~/SD_Thesis
source venv/bin/activate
python scripts/test_yolo.py
```

### Test 3: Low-Light Enhancement

```bash
# Test ZERO-DCE++
cd ~/SD_Thesis
source venv/bin/activate
python scripts/test_enhancement.py --model zero_dce --input data/test_images/dark.jpg

# Test SCI
python scripts/test_enhancement.py --model sci --input data/test_images/dark.jpg
```

### Test 4: ORB-SLAM2

```bash
# Test with sample dataset
cd ~/ORB_SLAM2
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml data/rgbd_dataset_freiburg1_xyz

# Test with RealSense
cd ~/SD_Thesis
roslaunch launch/test_slam.launch
```

## 🔗 Running the Complete Pipeline

### Option 1: Individual Nodes

```bash
# Terminal 1: ROS Core
roscore

# Terminal 2: Camera Node
cd ~/SD_Thesis && source venv/bin/activate
rosrun ros_nodes camera_node.py

# Terminal 3: Enhancement Node
rosrun ros_nodes enhancement_node.py

# Terminal 4: Detection Node
rosrun ros_nodes detection_node.py

# Terminal 5: SLAM Node
rosrun ros_nodes slam_node.py
```

### Option 2: Launch File (Integrated)

```bash
# Run complete pipeline
cd ~/SD_Thesis
roslaunch launch/complete_pipeline.launch
```

## 🐛 Troubleshooting Common Issues

### CUDA Issues

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall CUDA drivers if needed
sudo apt remove --purge nvidia-* -y
sudo ubuntu-drivers autoinstall
sudo reboot
```

### RealSense Issues

```bash
# Check camera connection
lsusb | grep Intel

# Restart udev rules
sudo service udev restart
sudo udevadm control --reload-rules

# Check camera permissions
groups $USER | grep video
```

### ROS Issues

```bash
# Check ROS environment
echo $ROS_PACKAGE_PATH
printenv | grep ROS

# Reinstall ROS if needed
sudo apt remove ros-noetic-* -y
sudo apt autoremove -y
# Repeat ROS installation steps
```

### Python Dependencies

```bash
# Update pip and reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Build Issues

```bash
# Clean and rebuild catkin workspace
cd ~/catkin_ws
catkin_make clean
catkin_make

# Clean ORB-SLAM2 build
cd ~/ORB_SLAM2
rm -rf build
chmod +x build.sh
./build.sh
```

## 📊 Performance Verification

After successful installation, verify system performance:

```bash
# Run benchmark script
cd ~/SD_Thesis
source venv/bin/activate
python scripts/benchmark_system.py

# Expected performance targets:
# - Camera capture: 30 FPS
# - YOLO detection: >10 FPS
# - Enhancement: >5 FPS
# - SLAM tracking: >20 FPS
```

## 🎯 Next Steps

1. **Calibrate your RealSense camera** using the calibration script
2. **Collect test datasets** in various lighting conditions
3. **Run baseline experiments** to establish performance metrics
4. **Begin thesis experiments** following the methodology

## 📞 Getting Help

If you encounter issues:

1. Check the **troubleshooting section** above
2. Review **logs** in `logs/experiment_logs/`
3. Check **GitHub Issues** for known problems
4. Create a **new issue** with detailed error information

Your Ubuntu 20.04 environment is now ready for comprehensive thesis research! 🚀