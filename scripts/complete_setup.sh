#!/bin/bash

# SD_Thesis Complete Setup Script for Ubuntu 20.04
# This script automates the complete installation of all dependencies
# for the low-light vision search and rescue UAV system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running on Ubuntu 20.04
check_ubuntu_version() {
    log "Checking Ubuntu version..."
    
    if [[ ! -f /etc/lsb-release ]]; then
        error "Cannot determine Ubuntu version. This script is designed for Ubuntu 20.04."
    fi
    
    source /etc/lsb-release
    if [[ "$DISTRIB_RELEASE" != "20.04" ]]; then
        warn "This script is optimized for Ubuntu 20.04. You are running $DISTRIB_RELEASE."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log "Ubuntu version check passed"
}

# Update system packages
update_system() {
    log "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    sudo apt install -y curl wget git build-essential cmake pkg-config
    log "System update completed"
}

# Install basic development tools
install_dev_tools() {
    log "Installing development tools..."
    
    sudo apt install -y \
        gcc g++ \
        python3 python3-pip python3-dev python3-venv \
        libeigen3-dev \
        libopencv-dev \
        libopencv-contrib-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
        libblas-dev \
        liblapack-dev \
        libgtest-dev \
        libssl-dev \
        libusb-1.0-0-dev \
        libudev-dev \
        libgtk-3-dev \
        libglfw3-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        freeglut3-dev
        
    log "Development tools installed"
}

# Check for NVIDIA GPU and install CUDA if needed
setup_cuda() {
    log "Checking for NVIDIA GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected. Setting up CUDA..."
        
        # Check if CUDA is already installed
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
            log "CUDA $CUDA_VERSION already installed"
        else
            warn "NVIDIA GPU detected but CUDA not installed."
            info "Please install CUDA manually from: https://developer.nvidia.com/cuda-downloads"
            info "Recommended version: CUDA 11.8 for maximum compatibility"
            read -p "Have you installed CUDA? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                warn "CUDA installation skipped. Some GPU features may not work."
            fi
        fi
    else
        warn "No NVIDIA GPU detected. GPU acceleration will not be available."
    fi
}

# Install ROS Noetic
install_ros_noetic() {
    log "Installing ROS Noetic..."
    
    # Check if ROS is already installed
    if command -v roscore &> /dev/null; then
        log "ROS already installed"
        return
    fi
    
    # Setup sources.list
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    
    # Setup keys
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    
    # Update and install
    sudo apt update
    sudo apt install -y ros-noetic-desktop-full
    
    # Initialize rosdep
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
        sudo rosdep init
    fi
    rosdep update
    
    # Setup environment
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source /opt/ros/noetic/setup.bash
    
    # Install additional ROS packages
    sudo apt install -y \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        build-essential \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-camera-info-manager \
        ros-noetic-realsense2-camera \
        ros-noetic-rgbd-launch \
        ros-noetic-ddynamic-reconfigure
    
    log "ROS Noetic installation completed"
}

# Install Intel RealSense SDK
install_realsense_sdk() {
    log "Installing Intel RealSense SDK..."
    
    # Check if already installed
    if dpkg -l | grep -q librealsense2; then
        log "RealSense SDK already installed"
        return
    fi
    
    # Register the server's public key
    sudo mkdir -p /etc/apt/keyrings
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    
    # Add the server to the list of repositories
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list
    
    # Update and install
    sudo apt update
    sudo apt install -y \
        librealsense2-dkms \
        librealsense2-utils \
        librealsense2-dev \
        librealsense2-dbg
    
    log "RealSense SDK installation completed"
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        error "requirements.txt not found in current directory"
    fi
    
    log "Python dependencies installed"
}

# Install Pangolin (for ORB-SLAM2)
install_pangolin() {
    log "Installing Pangolin..."
    
    cd /tmp
    if [[ ! -d "Pangolin" ]]; then
        git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
    fi
    cd Pangolin
    
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    
    log "Pangolin installation completed"
}

# Install OpenCV with contrib modules
install_opencv() {
    log "Installing OpenCV with contrib modules..."
    
    # Check if OpenCV is already compiled with contrib
    if python3 -c "import cv2; print(cv2.__version__)" &>/dev/null; then
        CV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)")
        log "OpenCV $CV_VERSION already installed"
        
        # Check if contrib modules are available
        if python3 -c "import cv2; cv2.xfeatures2d.SIFT_create()" &>/dev/null; then
            log "OpenCV contrib modules already available"
            return
        fi
    fi
    
    warn "Installing OpenCV from source with contrib modules..."
    
    cd /tmp
    
    # Download OpenCV and contrib
    if [[ ! -d "opencv" ]]; then
        git clone https://github.com/opencv/opencv.git
        git clone https://github.com/opencv/opencv_contrib.git
        
        cd opencv
        git checkout 4.5.5
        cd ../opencv_contrib
        git checkout 4.5.5
        cd ..
    fi
    
    cd opencv
    mkdir -p build
    cd build
    
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D PYTHON_EXECUTABLE=$(which python3) \
          -D BUILD_EXAMPLES=OFF \
          -D WITH_CUDA=ON \
          -D CUDA_ARCH_BIN="6.1,7.5,8.6" \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D ENABLE_FAST_MATH=ON \
          -D CUDA_FAST_MATH=ON \
          -D WITH_CUBLAS=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=ON \
          -D WITH_GSTREAMER=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          ..
    
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    
    log "OpenCV installation completed"
}

# Download and compile ORB-SLAM2
install_orb_slam2() {
    log "Installing ORB-SLAM2..."
    
    cd /tmp
    if [[ ! -d "ORB_SLAM2" ]]; then
        git clone https://github.com/raulmur/ORB_SLAM2.git
    fi
    cd ORB_SLAM2
    
    # Build DBoW2
    cd Thirdparty/DBoW2
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    # Build g2o
    cd ../../g2o
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    # Build ORB-SLAM2
    cd ../../..
    chmod +x build.sh
    ./build.sh
    
    # Build ROS node
    export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:$(pwd)/Examples/ROS
    chmod +x build_ros.sh
    ./build_ros.sh
    
    log "ORB-SLAM2 installation completed"
}

# Download YOLO v4 weights and config
setup_yolo() {
    log "Setting up YOLO v4..."
    
    mkdir -p models/yolo
    cd models/yolo
    
    # Download YOLO v4 config and weights
    if [[ ! -f "yolov4.cfg" ]]; then
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
    fi
    
    if [[ ! -f "yolov4.weights" ]]; then
        wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    fi
    
    if [[ ! -f "coco.names" ]]; then
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
    fi
    
    cd ../..
    log "YOLO v4 setup completed"
}

# Create workspace and build
setup_workspace() {
    log "Setting up workspace..."
    
    # Source ROS environment
    source /opt/ros/noetic/setup.bash
    
    # Create catkin workspace if it doesn't exist
    if [[ ! -d "catkin_ws" ]]; then
        mkdir -p catkin_ws/src
        cd catkin_ws
        catkin_make
        echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
        cd ..
    fi
    
    # Copy package to catkin workspace
    if [[ ! -L "catkin_ws/src/sd_thesis" ]]; then
        ln -s $(pwd) catkin_ws/src/sd_thesis
    fi
    
    # Build the workspace
    cd catkin_ws
    catkin_make
    cd ..
    
    log "Workspace setup completed"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Check ROS
    if command -v roscore &> /dev/null; then
        info "✓ ROS Noetic installed"
    else
        warn "✗ ROS not found"
    fi
    
    # Check RealSense
    if command -v realsense-viewer &> /dev/null; then
        info "✓ RealSense SDK installed"
    else
        warn "✗ RealSense SDK not found"
    fi
    
    # Check Python packages
    source venv/bin/activate
    if python3 -c "import torch; import cv2; import numpy" &> /dev/null; then
        info "✓ Python dependencies installed"
    else
        warn "✗ Some Python dependencies missing"
    fi
    
    # Check CUDA (if GPU available)
    if command -v nvidia-smi &> /dev/null; then
        if command -v nvcc &> /dev/null; then
            info "✓ CUDA available"
        else
            warn "✗ CUDA not installed"
        fi
    fi
    
    log "Installation verification completed"
}

# Print final instructions
print_final_instructions() {
    log "Installation completed successfully!"
    echo
    info "=== Next Steps ==="
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Activate the Python environment: source venv/bin/activate"
    echo "3. Connect your Intel RealSense D435i camera"
    echo "4. Test the camera: realsense-viewer"
    echo "5. Run the test scripts: python tests/test_camera.py"
    echo
    info "=== Useful Commands ==="
    echo "• Start ROS master: roscore"
    echo "• Launch camera: roslaunch realsense2_camera rs_camera.launch"
    echo "• Run complete pipeline: roslaunch sd_thesis complete_pipeline.launch"
    echo
    info "=== Documentation ==="
    echo "• Setup guide: docs/setup_instructions.md"
    echo "• Project overview: docs/project_overview.md"
    echo "• Development workflow: docs/development_workflow.md"
    echo
    warn "If you encounter any issues, check the troubleshooting section in docs/setup_instructions.md"
}

# Main execution
main() {
    log "Starting SD_Thesis complete setup for Ubuntu 20.04..."
    
    # Change to script directory
    cd "$(dirname "$0")/.."
    
    check_ubuntu_version
    update_system
    install_dev_tools
    setup_cuda
    install_ros_noetic
    install_realsense_sdk
    install_python_deps
    install_pangolin
    install_opencv
    install_orb_slam2
    setup_yolo
    setup_workspace
    verify_installation
    print_final_instructions
    
    log "Setup completed! 🎉"
}

# Run main function
main "$@"
