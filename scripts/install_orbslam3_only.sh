#!/bin/bash

# ORB-SLAM3 Installation Script (Pangolin already installed)
# Ubuntu 20.04 with GCC 9.4 compatibility

set -e

LOG_FILE="/tmp/orbslam3_only_install.log"

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

# Install ORB-SLAM3
install_orbslam3() {
    log "Installing ORB-SLAM3 (Pangolin already installed)..."
    
    cd /tmp
    if [[ -d "ORB_SLAM3" ]]; then
        log "Removing previous ORB_SLAM3 build..."
        rm -rf ORB_SLAM3
    fi
    
    log "Cloning ORB-SLAM3..."
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
    cd ORB_SLAM3
    
    # Apply GCC 9.4 compatibility patches
    log "Applying GCC 9.4 compatibility patches..."
    
    # Patch CMakeLists.txt for main project
    sed -i '/cmake_minimum_required/a\\n# GCC 9.4 compatibility\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy -Wno-deprecated-declarations")\nset(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")' CMakeLists.txt
    
    # Common compiler flags for all third-party libraries
    COMMON_CXX_FLAGS="-Wno-deprecated-copy -Wno-deprecated-declarations -Wno-unused-parameter -Wno-unused-variable"
    
    # Build Thirdparty libraries with compatible flags
    log "Building DBoW2..."
    cd Thirdparty/DBoW2
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$COMMON_CXX_FLAGS"
    make -j$(nproc)
    
    log "Building g2o..."
    cd ../../g2o
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$COMMON_CXX_FLAGS"
    make -j$(nproc)
    
    log "Building Sophus..."
    cd ../../Sophus
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="$COMMON_CXX_FLAGS"
    make -j$(nproc)
    
    # Build ORB-SLAM3 main project
    log "Building ORB-SLAM3 main project..."
    cd ../../..
    
    # Make sure build script is executable
    chmod +x build.sh
    
    # Modify build.sh to use our compiler flags
    log "Patching build.sh for GCC 9.4 compatibility..."
    cp build.sh build.sh.backup
    sed -i 's/cmake \.\./cmake .. -DCMAKE_CXX_FLAGS="'"$COMMON_CXX_FLAGS"'"/' build.sh
    
    # Run the build
    ./build.sh
    
    # Build ROS wrapper if ROS is available
    if command -v roscore &> /dev/null; then
        log "Building ROS wrapper..."
        export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:$(pwd)/Examples/ROS
        chmod +x build_ros.sh
        
        # Patch build_ros.sh too
        cp build_ros.sh build_ros.sh.backup
        sed -i 's/cmake \.\./cmake .. -DCMAKE_CXX_FLAGS="'"$COMMON_CXX_FLAGS"'"/' build_ros.sh
        
        ./build_ros.sh
    else
        warn "ROS not detected, skipping ROS wrapper build"
    fi
    
    # Install to system
    log "Installing ORB-SLAM3 to /opt/ORB_SLAM3..."
    sudo mkdir -p /opt/ORB_SLAM3
    sudo cp -r * /opt/ORB_SLAM3/
    sudo chown -R $USER:$USER /opt/ORB_SLAM3
    
    # Add to environment
    if ! grep -q "ORB_SLAM3_ROOT_PATH" ~/.bashrc; then
        echo "export ORB_SLAM3_ROOT_PATH=/opt/ORB_SLAM3" >> ~/.bashrc
        echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/opt/ORB_SLAM3/Examples/ROS" >> ~/.bashrc
        log "Added ORB-SLAM3 environment variables to ~/.bashrc"
    fi
    
    log "ORB-SLAM3 installation completed successfully!"
}

# Create Python wrapper
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
            "MONOCULAR": "/opt/ORB_SLAM3/Examples/Monocular/mono_euroc",
            "STEREO": "/opt/ORB_SLAM3/Examples/Stereo/stereo_euroc", 
            "RGBD": "/opt/ORB_SLAM3/Examples/RGB-D/rgbd_tum",
            "MONOCULAR_INERTIAL": "/opt/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc",
            "STEREO_INERTIAL": "/opt/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc"
        }
        
        # ROS executables (if available)
        self.ros_executables = {
            "MONOCULAR": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Mono",
            "STEREO": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Stereo", 
            "RGBD": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/RGBD",
            "MONOCULAR_INERTIAL": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Mono_Inertial",
            "STEREO_INERTIAL": "/opt/ORB_SLAM3/Examples/ROS/ORB_SLAM3/Stereo_Inertial"
        }
        
    def start(self, use_ros=True):
        """Start ORB-SLAM3 process"""
        if self.is_running:
            return
            
        # Try ROS executables first if requested
        executable = None
        if use_ros:
            executable = self.ros_executables.get(self.sensor_type)
            if executable and not os.path.exists(executable):
                executable = None
        
        # Fallback to standalone executables
        if not executable:
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
    log "Python wrapper created successfully"
}

# Setup configuration files
setup_config() {
    log "Setting up configuration files..."
    
    # Create directories
    mkdir -p /home/duck/Desktop/SD_Thesis/models/orbslam3
    
    # Copy vocabulary file
    if [[ -f "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt" ]]; then
        cp /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /home/duck/Desktop/SD_Thesis/models/orbslam3/
        log "Vocabulary file copied to project"
    else
        warn "Vocabulary file not found at expected location"
    fi
    
    log "Configuration setup completed"
}

# Main function
main() {
    log "Starting ORB-SLAM3 installation (Pangolin already available)..."
    log "Installation log: $LOG_FILE"
    
    install_orbslam3
    create_python_wrapper
    setup_config
    
    log "ORB-SLAM3 installation completed successfully!"
    
    echo ""
    echo -e "${GREEN}=== Installation Summary ===${NC}"
    echo -e "${BLUE}ORB-SLAM3 installed to: /opt/ORB_SLAM3${NC}"
    echo -e "${BLUE}Python wrapper: /opt/ORB_SLAM3/python_wrapper.py${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Run 'source ~/.bashrc'"
    echo "2. Test installation with the verification script"
    echo ""
    echo -e "${GREEN}Installation log saved to: $LOG_FILE${NC}"
}

# Run main function
main "$@"
