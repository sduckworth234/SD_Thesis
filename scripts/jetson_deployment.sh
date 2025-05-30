#!/bin/bash

# Script to handle the deployment process to the Jetson Xavier NX

# Update and upgrade the system
sudo apt-get update
sudo apt-get upgrade -y

# Install necessary dependencies
sudo apt-get install -y python3-pip
sudo apt-get install -y ros-noetic-ros-base
sudo apt-get install -y ros-noetic-cv-bridge
sudo apt-get install -y ros-noetic-image-transport
sudo apt-get install -y ros-noetic-vision-opencv

# Install Python dependencies
pip3 install -r ../requirements.txt

# Set up environment variables
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/../src" >> ~/.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Additional setup for Jetson specific configurations can be added here

echo "Deployment to Jetson Xavier NX completed."