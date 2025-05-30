#!/bin/bash

# SD_Thesis System Verification Script
# Comprehensive verification of all system components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="$3"
    
    ((TESTS_TOTAL++))
    echo -n "Testing $test_name... "
    
    if eval "$test_command" &>/dev/null; then
        if [[ "$expected_result" == "pass" ]] || [[ -z "$expected_result" ]]; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${RED}✗ FAIL (unexpected pass)${NC}"
            ((TESTS_FAILED++))
            return 1
        fi
    else
        if [[ "$expected_result" == "fail" ]]; then
            echo -e "${GREEN}✓ PASS (expected fail)${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${RED}✗ FAIL${NC}"
            ((TESTS_FAILED++))
            return 1
        fi
    fi
}

# System Information
print_system_info() {
    log "=== System Information ==="
    echo "OS: $(lsb_release -d | cut -f2)"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "CPU: $(nproc) cores"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    else
        echo "GPU: No NVIDIA GPU detected"
    fi
    echo
}

# Test Ubuntu version
test_ubuntu_version() {
    log "=== Testing Ubuntu Version ==="
    
    if [[ -f /etc/lsb-release ]]; then
        source /etc/lsb-release
        if [[ "$DISTRIB_RELEASE" == "20.04" ]]; then
            run_test "Ubuntu 20.04" "true"
        else
            warn "Running on Ubuntu $DISTRIB_RELEASE instead of 20.04"
            run_test "Ubuntu version" "false"
        fi
    else
        run_test "Ubuntu detection" "false"
    fi
}

# Test basic development tools
test_dev_tools() {
    log "=== Testing Development Tools ==="
    
    run_test "GCC compiler" "command -v gcc"
    run_test "G++ compiler" "command -v g++"
    run_test "CMake" "command -v cmake"
    run_test "Git" "command -v git"
    run_test "Python3" "command -v python3"
    run_test "pip3" "command -v pip3"
    run_test "pkg-config" "command -v pkg-config"
}

# Test CUDA
test_cuda() {
    log "=== Testing CUDA ==="
    
    if command -v nvidia-smi &> /dev/null; then
        run_test "NVIDIA driver" "nvidia-smi"
        run_test "NVCC compiler" "command -v nvcc"
        
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
            info "CUDA version: $CUDA_VERSION"
        fi
    else
        warn "No NVIDIA GPU detected, skipping CUDA tests"
    fi
}

# Test ROS Noetic
test_ros() {
    log "=== Testing ROS Noetic ==="
    
    run_test "ROS core" "command -v roscore"
    run_test "ROS launch" "command -v roslaunch"
    run_test "ROS node" "command -v rosnode"
    run_test "catkin_make" "command -v catkin_make"
    
    # Test ROS environment
    if [[ -f "/opt/ros/noetic/setup.bash" ]]; then
        run_test "ROS Noetic installation" "true"
        source /opt/ros/noetic/setup.bash
        run_test "ROS environment" "[[ -n \$ROS_DISTRO ]]"
    else
        run_test "ROS Noetic installation" "false"
    fi
    
    # Test ROS packages
    run_test "cv_bridge" "dpkg -l | grep -q ros-noetic-cv-bridge"
    run_test "image_transport" "dpkg -l | grep -q ros-noetic-image-transport"
    run_test "realsense2_camera" "dpkg -l | grep -q ros-noetic-realsense2-camera"
}

# Test Intel RealSense
test_realsense() {
    log "=== Testing Intel RealSense ==="
    
    run_test "librealsense2" "dpkg -l | grep -q librealsense2"
    run_test "realsense-viewer" "command -v realsense-viewer"
    
    # Test Python bindings
    if source venv/bin/activate 2>/dev/null; then
        run_test "pyrealsense2" "python3 -c 'import pyrealsense2'"
        deactivate
    else
        warn "Virtual environment not found, skipping Python tests"
    fi
    
    # Test camera connection (if available)
    if command -v rs-enumerate-devices &> /dev/null; then
        if rs-enumerate-devices | grep -q "Intel RealSense"; then
            info "RealSense camera detected"
            run_test "Camera connection" "true"
        else
            warn "No RealSense camera connected"
            run_test "Camera connection" "false"
        fi
    fi
}

# Test OpenCV
test_opencv() {
    log "=== Testing OpenCV ==="
    
    run_test "OpenCV library" "pkg-config --exists opencv4"
    
    if source venv/bin/activate 2>/dev/null; then
        run_test "OpenCV Python" "python3 -c 'import cv2'"
        
        # Test OpenCV version and features
        if python3 -c "import cv2" &>/dev/null; then
            CV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)")
            info "OpenCV version: $CV_VERSION"
            
            # Test contrib modules
            if python3 -c "import cv2; cv2.xfeatures2d.SIFT_create()" &>/dev/null; then
                run_test "OpenCV contrib modules" "true"
            else
                run_test "OpenCV contrib modules" "false"
            fi
            
            # Test DNN module
            run_test "OpenCV DNN module" "python3 -c 'import cv2; cv2.dnn.readNet'"
        fi
        
        deactivate
    fi
}

# Test Python dependencies
test_python_deps() {
    log "=== Testing Python Dependencies ==="
    
    if source venv/bin/activate 2>/dev/null; then
        run_test "NumPy" "python3 -c 'import numpy'"
        run_test "PyTorch" "python3 -c 'import torch'"
        run_test "TensorFlow" "python3 -c 'import tensorflow'"
        run_test "Matplotlib" "python3 -c 'import matplotlib'"
        run_test "scikit-learn" "python3 -c 'import sklearn'"
        run_test "Pillow" "python3 -c 'import PIL'"
        
        # Test PyTorch CUDA
        if python3 -c "import torch" &>/dev/null; then
            if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                run_test "PyTorch CUDA" "true"
                TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
                info "PyTorch CUDA version: $TORCH_CUDA_VERSION"
            else
                run_test "PyTorch CUDA" "false"
            fi
        fi
        
        deactivate
    else
        warn "Virtual environment not found, creating one..."
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        
        if [[ -f "requirements.txt" ]]; then
            pip install -r requirements.txt
            run_test "Requirements installation" "true"
        else
            warn "requirements.txt not found"
            run_test "Requirements installation" "false"
        fi
        
        deactivate
    fi
}

# Test ORB-SLAM2
test_orb_slam2() {
    log "=== Testing ORB-SLAM2 ==="
    
    # Check if ORB-SLAM2 is compiled
    if [[ -f "/tmp/ORB_SLAM2/build/lib/libORB_SLAM2.so" ]]; then
        run_test "ORB-SLAM2 library" "true"
    else
        run_test "ORB-SLAM2 library" "false"
    fi
    
    # Check vocabulary file
    if [[ -f "models/orb_slam2/ORBvoc.txt" ]] || [[ -f "/tmp/ORB_SLAM2/Vocabulary/ORBvoc.txt" ]]; then
        run_test "ORB vocabulary" "true"
    else
        run_test "ORB vocabulary" "false"
    fi
    
    # Check configuration
    run_test "ORB-SLAM2 config" "[[ -f 'config/orb_slam2/realsense_d435i.yaml' ]]"
}

# Test YOLO
test_yolo() {
    log "=== Testing YOLO ==="
    
    run_test "YOLO config" "[[ -f 'models/yolo/yolov4.cfg' ]]"
    run_test "YOLO weights" "[[ -f 'models/yolo/yolov4.weights' ]]"
    run_test "YOLO names" "[[ -f 'models/yolo/coco.names' ]]"
    
    if source venv/bin/activate 2>/dev/null; then
        # Test YOLO with OpenCV DNN
        run_test "YOLO loading" "python3 -c 'import cv2; cv2.dnn.readNet'"
        deactivate
    fi
}

# Test workspace
test_workspace() {
    log "=== Testing Workspace ==="
    
    run_test "Catkin workspace" "[[ -d 'catkin_ws' ]]"
    run_test "Source directory" "[[ -d 'src' ]]"
    run_test "Launch files" "[[ -d 'launch' ]]"
    run_test "Config files" "[[ -d 'config' ]]"
    run_test "Test scripts" "[[ -d 'tests' ]]"
    
    # Test package files
    run_test "package.xml" "[[ -f 'package.xml' ]]"
    run_test "CMakeLists.txt" "[[ -f 'CMakeLists.txt' ]]"
    
    # Test if package is in catkin workspace
    if [[ -d "catkin_ws/src" ]]; then
        if [[ -L "catkin_ws/src/sd_thesis" ]] || [[ -d "catkin_ws/src/sd_thesis" ]]; then
            run_test "Package in workspace" "true"
        else
            run_test "Package in workspace" "false"
        fi
    fi
}

# Test individual components
test_individual_components() {
    log "=== Testing Individual Components ==="
    
    # Test camera
    if [[ -f "tests/test_camera.py" ]]; then
        run_test "Camera test script" "true"
    else
        run_test "Camera test script" "false"
    fi
    
    # Test YOLO
    if [[ -f "tests/test_yolo.py" ]]; then
        run_test "YOLO test script" "true"
    else
        run_test "YOLO test script" "false"
    fi
    
    # Test enhancement
    if [[ -f "tests/test_enhancement.py" ]]; then
        run_test "Enhancement test script" "true"
    else
        run_test "Enhancement test script" "false"
    fi
    
    # Test launch files
    run_test "Camera launch file" "[[ -f 'launch/test_camera.launch' ]]"
    run_test "SLAM launch file" "[[ -f 'launch/test_slam.launch' ]]"
    run_test "Complete pipeline launch" "[[ -f 'launch/complete_pipeline.launch' ]]"
}

# Performance benchmarks
run_benchmarks() {
    log "=== Running Performance Benchmarks ==="
    
    if source venv/bin/activate 2>/dev/null; then
        # CPU benchmark
        echo -n "CPU performance test... "
        start_time=$(date +%s.%N)
        python3 -c "
import numpy as np
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)
c = np.dot(a, b)
"
        end_time=$(date +%s.%N)
        cpu_time=$(echo "$end_time - $start_time" | bc)
        echo "${cpu_time}s"
        
        # GPU benchmark (if available)
        if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            echo -n "GPU performance test... "
            start_time=$(date +%s.%N)
            python3 -c "
import torch
a = torch.rand(1000, 1000).cuda()
b = torch.rand(1000, 1000).cuda()
c = torch.mm(a, b)
torch.cuda.synchronize()
"
            end_time=$(date +%s.%N)
            gpu_time=$(echo "$end_time - $start_time" | bc)
            echo "${gpu_time}s"
        fi
        
        deactivate
    fi
}

# Generate test report
generate_report() {
    log "=== Test Report ==="
    echo "Tests passed: $TESTS_PASSED"
    echo "Tests failed: $TESTS_FAILED"
    echo "Total tests: $TESTS_TOTAL"
    echo "Success rate: $(echo "scale=2; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)%"
    echo
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log "🎉 All tests passed! System is ready for development."
    else
        warn "Some tests failed. Check the output above for details."
        echo
        echo "Common solutions:"
        echo "• Install missing packages: sudo apt install <package-name>"
        echo "• Run setup script: ./scripts/complete_setup.sh"
        echo "• Check documentation: docs/setup_instructions.md"
        echo "• Verify hardware connections"
    fi
}

# Main execution
main() {
    log "SD_Thesis System Verification"
    log "Starting comprehensive system check..."
    echo
    
    # Change to project directory
    cd "$(dirname "$0")/.."
    
    print_system_info
    test_ubuntu_version
    test_dev_tools
    test_cuda
    test_ros
    test_realsense
    test_opencv
    test_python_deps
    test_orb_slam2
    test_yolo
    test_workspace
    test_individual_components
    
    # Optional benchmarks
    read -p "Run performance benchmarks? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_benchmarks
    fi
    
    echo
    generate_report
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
