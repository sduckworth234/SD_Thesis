# Troubleshooting Guide for SD_Thesis

This guide provides solutions to common issues you may encounter when setting up and running the SD_Thesis project on Ubuntu 20.04.

## 🚨 Quick Diagnostics

If you're experiencing issues, run the verification script first:

```bash
cd SD_Thesis
chmod +x scripts/verify_setup.sh
./scripts/verify_setup.sh
```

This will identify which components are working and which need attention.

## 📋 Common Setup Issues

### 1. Ubuntu Version Compatibility

**Issue**: Script designed for Ubuntu 20.04 but running on different version
**Solution**: 
- For Ubuntu 18.04: Use `scripts/jetson_deployment.sh` (adapted for older versions)
- For Ubuntu 22.04: Most packages should work, but may need version adjustments
- Check specific package versions in requirements.txt

### 2. CUDA Installation Problems

**Issue**: NVIDIA GPU detected but CUDA not working
**Symptoms**: 
- `nvcc --version` command not found
- PyTorch not detecting GPU
- OpenCV compiled without CUDA support

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA 11.8 (recommended)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. ROS Noetic Installation Issues

**Issue**: ROS packages not found or installation fails
**Solutions**:
```bash
# Refresh package lists
sudo apt update

# Fix broken packages
sudo apt --fix-broken install

# Reinstall ROS if necessary
sudo apt remove ros-noetic-*
sudo apt autoremove
sudo apt install ros-noetic-desktop-full

# Re-initialize rosdep
sudo rm -rf /etc/ros/rosdep/sources.list.d/20-default.list
sudo rosdep init
rosdep update
```

### 4. Intel RealSense SDK Problems

**Issue**: Camera not detected or driver installation fails
**Symptoms**:
- `realsense-viewer` not working
- "No device connected" error
- USB permission issues

**Solutions**:
```bash
# Check if camera is detected
lsusb | grep Intel

# Fix USB permissions
sudo usermod -a -G plugdev $USER
sudo udevadm control --reload-rules && sudo udevadm trigger

# Reinstall RealSense SDK
sudo apt remove librealsense2-*
sudo apt autoremove

# Re-add repository
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev

# Logout and login again
```

### 5. OpenCV Compilation Issues

**Issue**: OpenCV missing contrib modules or CUDA support
**Solutions**:
```bash
# Remove existing OpenCV
sudo apt remove libopencv-*
sudo apt autoremove

# Compile from source with contrib modules
cd /tmp
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
git checkout 4.5.5
cd ../opencv_contrib
git checkout 4.5.5
cd ../opencv

mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D PYTHON_EXECUTABLE=$(which python3) \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="6.1,7.5,8.6" \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      ..

make -j$(nproc)
sudo make install
sudo ldconfig
```

### 6. Python Environment Issues

**Issue**: Package installation failures or import errors
**Solutions**:
```bash
# Reset virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# If specific packages fail:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow-gpu==2.8.0  # or latest compatible version
```

### 7. ORB-SLAM2 Compilation Errors

**Issue**: ORB-SLAM2 build fails
**Common errors**:
- Missing Eigen3
- Pangolin compilation issues
- OpenCV version conflicts

**Solutions**:
```bash
# Install dependencies
sudo apt install libeigen3-dev libgl1-mesa-dev libglew-dev

# Build Pangolin first
cd /tmp
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Build ORB-SLAM2
cd /tmp
git clone https://github.com/raulmur/ORB_SLAM2.git
cd ORB_SLAM2

# Fix CMakeLists.txt if needed
sed -i 's/++11/++14/g' CMakeLists.txt

# Build
chmod +x build.sh
./build.sh

# Build ROS wrapper
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:$(pwd)/Examples/ROS
chmod +x build_ros.sh
./build_ros.sh
```

### 8. Memory and Performance Issues

**Issue**: System running out of memory during compilation
**Solutions**:
```bash
# Check memory usage
free -h

# Add swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Reduce compilation parallelism
make -j2  # instead of make -j$(nproc)
```

## 🔧 Runtime Issues

### 1. Camera Connection Problems

**Issue**: Camera not streaming or poor quality
**Diagnostics**:
```bash
# Test camera directly
realsense-viewer

# Check ROS topics
rostopic list | grep camera
rostopic hz /camera/color/image_raw
```

**Solutions**:
- Check USB 3.0 connection
- Try different USB ports
- Adjust camera parameters in launch files
- Check power supply if using USB hub

### 2. SLAM Tracking Loss

**Issue**: ORB-SLAM2 loses tracking or fails to initialize
**Solutions**:
- Ensure sufficient lighting
- Move camera slowly for initialization
- Check camera calibration parameters
- Adjust ORB extractor parameters in config

### 3. YOLO Detection Issues

**Issue**: No detections or poor accuracy
**Solutions**:
- Verify model weights are downloaded correctly
- Check confidence thresholds
- Ensure proper lighting conditions
- Test with known images containing people

### 4. Performance Issues

**Issue**: Low FPS or high latency
**Solutions**:
```bash
# Monitor system resources
htop
nvidia-smi

# Reduce image resolution
# Edit launch files to use 320x240 instead of 640x480

# Optimize OpenCV build flags
# Rebuild with optimizations enabled

# Check thermal throttling
sensors
```

## 🐛 ROS-Specific Issues

### 1. Package Not Found Errors

**Solutions**:
```bash
# Source ROS environment
source /opt/ros/noetic/setup.bash

# Build workspace
cd catkin_ws
catkin_make
source devel/setup.bash

# Add to bashrc
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

### 2. Transform Frame Errors

**Solutions**:
```bash
# Check TF tree
rosrun tf2_tools view_frames.py

# Debug transform issues
rosrun tf tf_echo source_frame target_frame

# Ensure static transforms are published
```

### 3. Topic Communication Issues

**Solutions**:
```bash
# Check topic types
rostopic type /camera/color/image_raw

# Monitor message flow
rostopic echo /camera/color/image_raw --noarr

# Check node graph
rosrun rqt_graph rqt_graph
```

## 🏥 Emergency Recovery

If your system becomes unstable:

### 1. Reset to Clean State
```bash
# Stop all ROS processes
killall -9 roscore rosmaster roslaunch

# Clean build artifacts
cd catkin_ws
rm -rf build devel
catkin_make clean

# Reset Python environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. System Recovery
```bash
# Fix broken packages
sudo apt --fix-broken install
sudo apt autoremove
sudo apt autoclean

# Reset network (if ROS networking issues)
sudo systemctl restart NetworkManager
```

### 3. Complete Reinstall
```bash
# Remove project
rm -rf SD_Thesis

# Re-clone and setup
git clone https://github.com/sduckworth234/SD_Thesis.git
cd SD_Thesis
chmod +x scripts/complete_setup.sh
./scripts/complete_setup.sh
```

## 📞 Getting Help

### 1. Check Logs
```bash
# ROS logs
ls ~/.ros/log/latest/

# System logs
journalctl -xe
dmesg | tail
```

### 2. Report Issues
When reporting issues, include:
- Output of `./scripts/verify_setup.sh`
- Relevant error messages
- System specifications
- Steps to reproduce

### 3. Useful Commands for Debugging
```bash
# Environment variables
env | grep ROS
echo $PYTHONPATH
echo $LD_LIBRARY_PATH

# Package versions
dpkg -l | grep ros-noetic
pip list | grep torch

# Hardware info
lscpu
lsusb
nvidia-smi
```

## 🎯 Performance Optimization

### 1. CPU Optimization
```bash
# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Increase process priority
nice -n -10 roslaunch sd_thesis complete_pipeline.launch
```

### 2. GPU Optimization
```bash
# Set GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit
```

### 3. Network Optimization
```bash
# Increase ROS message queue sizes
export ROS_MESSAGE_QUEUE_SIZE=1000

# Use local transport for large messages
export ROS_TRANSPORT_HINT=tcp
```

Remember: Most issues can be resolved by carefully following the setup instructions and ensuring all dependencies are properly installed. When in doubt, run the verification script to identify the root cause.
