#!/bin/bash

# ORB-SLAM3 Installation Script for Ubuntu 20.04 and Jetson Xavier NX
# This script compiles ORB-SLAM3 from source with optimizations for low power consumption

set -e

LOG_FILE="/tmp/orbslam3_install.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a $LOG_FILE
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a $LOG_FILE
    exit 1
}

# Check if running on Jetson
check_jetson() {
    if grep -q "JETSON" /etc/nv_tegra_release 2>/dev/null; then
        log "Detected Jetson device - applying Jetson-specific optimizations"
        export IS_JETSON=true
        # Set performance mode
        sudo nvpmodel -m 0 2>/dev/null || warn "Could not set nvpmodel"
        sudo jetson_clocks 2>/dev/null || warn "Could not set jetson_clocks"
    else
        log "Running on regular Ubuntu system"
        export IS_JETSON=false
    fi
}

# Install dependencies
install_dependencies() {
    log "Installing ORB-SLAM3 dependencies..."
    
    sudo apt update
    sudo apt install -y \
        cmake \
        git \
        build-essential \
        libeigen3-dev \
        libglew-dev \
        libgl1-mesa-dev \
        libglfw3-dev \
        libopencv-dev \
        libopencv-contrib-dev \
        python3-dev \
        python3-numpy \
        libyaml-cpp-dev \
        libboost-all-dev \
        libssl-dev \
        libusb-1.0-0-dev \
        pkg-config \
        libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        gfortran \
        openexr \
        libatlas-base-dev \
        libtbb2 \
        libtbb-dev \
        libdc1394-22-dev
        
    log "Dependencies installed successfully"
}

# Install Pangolin with optimizations
install_pangolin() {
    log "Installing Pangolin..."
    
    cd /tmp
    if [[ -d "Pangolin" ]]; then
        rm -rf Pangolin
    fi
    
    git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    
    mkdir -p build
    cd build
    
    # Configure with optimizations for low power consumption
    CMAKE_FLAGS=""
    if [[ "$IS_JETSON" == "true" ]]; then
        CMAKE_FLAGS="-DCUDA_ARCH_BIN=7.2 -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF"
    else
        CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF"
    fi
    
    # Fix compiler warnings for GCC 9.4.0 (Ubuntu 20.04)
    CMAKE_CXX_FLAGS="-Wno-error=deprecated-copy -Wno-error=deprecated-declarations -Wno-error=unused-parameter -Wno-error=unused-variable"
    
    cmake $CMAKE_FLAGS -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS" ..
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    
    log "Pangolin installation completed"
}

# Download and compile ORB-SLAM3
install_orbslam3() {
    log "Installing ORB-SLAM3..."
    
    cd /tmp
    if [[ -d "ORB_SLAM3" ]]; then
        rm -rf ORB_SLAM3
    fi
    
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
    cd ORB_SLAM3
    
    # Apply Jetson-specific optimizations if needed
    if [[ "$IS_JETSON" == "true" ]]; then
        log "Applying Jetson Xavier NX optimizations..."
        
        # Modify CMakeLists.txt for power efficiency
        sed -i 's/-O3/-O2/g' CMakeLists.txt
        sed -i '/set(CMAKE_CXX_FLAGS/a set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native -march=native")' CMakeLists.txt
    fi
    
    # Build Thirdparty libraries
    log "Building DBoW2..."
    cd Thirdparty/DBoW2
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error=deprecated-copy -Wno-error=deprecated-declarations"
    make -j$(nproc)
    
    log "Building g2o..."
    cd ../../g2o
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error=deprecated-copy -Wno-error=deprecated-declarations"
    make -j$(nproc)
    
    # Build Sophus
    log "Building Sophus..."
    cd ../../Sophus
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-error=deprecated-copy -Wno-error=deprecated-declarations"
    make -j$(nproc)
    
    # Build ORB-SLAM3
    log "Building ORB-SLAM3..."
    cd ../../..
    chmod +x build.sh
    ./build.sh
    
    # Build ROS node
    log "Building ROS wrapper..."
    export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:$(pwd)/Examples/ROS
    chmod +x build_ros.sh
    ./build_ros.sh
    
    # Move to installation directory
    sudo mkdir -p /opt/ORB_SLAM3
    sudo cp -r * /opt/ORB_SLAM3/
    sudo chown -R $USER:$USER /opt/ORB_SLAM3
    
    # Add to environment
    echo "export ORB_SLAM3_ROOT_PATH=/opt/ORB_SLAM3" >> ~/.bashrc
    echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/opt/ORB_SLAM3/Examples/ROS" >> ~/.bashrc
    
    log "ORB-SLAM3 installation completed"
}

# Create Python wrapper for ORB-SLAM3
create_python_wrapper() {
    log "Creating Python wrapper for ORB-SLAM3..."
    
    cat > /opt/ORB_SLAM3/python_wrapper.py << 'EOF'
#!/usr/bin/env python3
"""
Python wrapper for ORB-SLAM3
Provides a simplified interface for ROS integration
"""

import subprocess
import os
import signal
import time
import threading

class ORBSLAM3:
    def __init__(self, vocab_path, config_path, sensor_type="RGBD"):
        """
        Initialize ORB-SLAM3
        
        Args:
            vocab_path: Path to ORB vocabulary file
            config_path: Path to configuration YAML file
            sensor_type: "MONOCULAR", "STEREO", "RGBD", "MONOCULAR_INERTIAL", "STEREO_INERTIAL"
        """
        self.vocab_path = vocab_path
        self.config_path = config_path
        self.sensor_type = sensor_type
        self.process = None
        self.is_running = False
        
        # Executable paths
        self.executables = {
            "MONOCULAR": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Mono",
            "STEREO": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Stereo", 
            "RGBD": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/RGBD",
            "MONOCULAR_INERTIAL": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Mono_Inertial",
            "STEREO_INERTIAL": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Stereo_Inertial"
        }
        
    def start(self):
        """Start ORB-SLAM3 process"""
        if self.is_running:
            return
            
        executable = self.executables.get(self.sensor_type)
        if not executable or not os.path.exists(executable):
            raise RuntimeError(f"ORB-SLAM3 executable not found for sensor type: {self.sensor_type}")
            
        cmd = [executable, self.vocab_path, self.config_path]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            self.is_running = True
            
            # Monitor process in separate thread
            threading.Thread(target=self._monitor_process, daemon=True).start()
            
        except Exception as e:
            raise RuntimeError(f"Failed to start ORB-SLAM3: {e}")
    
    def stop(self):
        """Stop ORB-SLAM3 process"""
        if not self.is_running or not self.process:
            return
            
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            self.process.wait(timeout=5)
            
        except subprocess.TimeoutExpired:
            # Force kill if needed
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            
        finally:
            self.is_running = False
            self.process = None
    
    def _monitor_process(self):
        """Monitor ORB-SLAM3 process"""
        if self.process:
            self.process.wait()
            self.is_running = False
    
    def is_alive(self):
        """Check if ORB-SLAM3 process is running"""
        return self.is_running and self.process and self.process.poll() is None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop()

EOF

    sudo chmod +x /opt/ORB_SLAM3/python_wrapper.py
    log "Python wrapper created"
}

# Setup configuration files
setup_config() {
    log "Setting up configuration files..."
    
    # Copy vocabulary to project
    mkdir -p /home/duck/Desktop/SD_Thesis/models/orbslam3
    cp /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /home/duck/Desktop/SD_Thesis/models/orbslam3/
    
    # Create RealSense configuration for ORB-SLAM3
    cat > /home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml << 'EOF'
%YAML:1.0

# Camera Parameters. Adjust them!
Camera.type: "PinHole"

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: 615.3900146484375
Camera.fy: 615.3900146484375
Camera.cx: 320.0
Camera.cy: 240.0

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 640
Camera.height: 480

# Camera frames per second 
Camera.fps: 30.0

# IR projector baseline times fx (aprox.)
Camera.bf: 40.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 40.0

# Depthmap values factor 
DepthMapFactor: 1000.0

#--------------------------------------------------------------------------------------------
# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid 	
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid	
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast			
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500

#--------------------------------------------------------------------------------------------
# Map parameters
# Map.mapfile: "map.bin"
# Map.loadMap: false
# Map.saveMap: false

EOF

    log "Configuration files created"
}

# Main installation process
main() {
    log "Starting ORB-SLAM3 installation for Ubuntu 20.04..."
    log "Installation log: $LOG_FILE"
    
    check_jetson
    install_dependencies
    install_pangolin
    install_orbslam3
    create_python_wrapper
    setup_config
    
    log "ORB-SLAM3 installation completed successfully!"
    log "Please run 'source ~/.bashrc' to load environment variables"
    
    echo ""
    echo -e "${GREEN}=== Installation Summary ===${NC}"
    echo -e "${BLUE}ORB-SLAM3 installed to: /opt/ORB_SLAM3${NC}"
    echo -e "${BLUE}Vocabulary file: /home/duck/Desktop/SD_Thesis/models/orbslam3/ORBvoc.txt${NC}"
    echo -e "${BLUE}Configuration: /home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml${NC}"
    echo -e "${BLUE}Python wrapper: /opt/ORB_SLAM3/python_wrapper.py${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Run 'source ~/.bashrc'"
    echo "2. Test with: rosrun ORB_SLAM3 RGBD /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml"
}

# Run main function
main "$@"
