#!/bin/bash

# Update package list and install necessary packages
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    ros-noetic-ros-base \
    ros-noetic-image-transport \
    ros-noetic-cv-bridge \
    ros-noetic-realsense2-camera \
    ros-noetic-realsense2-description \
    libopencv-dev \
    libboost-all-dev \
    libeigen3-dev \
    libatlas-base-dev \
    cmake \
    git

# Install Python dependencies
pip3 install -r requirements.txt

# Install YOLOv4 dependencies
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make
cd ..

# Install additional dependencies for image enhancement models
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Clean up
sudo apt autoremove -y

echo "All dependencies have been installed successfully."