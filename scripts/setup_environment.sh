#!/bin/bash

# This script sets up the development environment for the SD_Thesis project.

# Update package lists
sudo apt-get update

# Install necessary system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    ros-noetic-desktop-full \
    ros-noetic-rosbridge-server \
    ros-noetic-vision-opencv \
    ros-noetic-image-transport \
    ros-noetic-cv-bridge

# Create a virtual environment for Python dependencies
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Additional setup for Intel RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F9B3B3D3
sudo add-apt-repository "deb http://librealsense.intel.com/ubuntu $(lsb_release -cs) main"
sudo apt-get update
sudo apt-get install -y librealsense2-dev librealsense2-dkms

# Print completion message
echo "Development environment setup complete."