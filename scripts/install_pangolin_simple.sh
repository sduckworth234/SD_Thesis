#!/bin/bash

# Simple Pangolin installation with proper error handling

set -e

echo "Installing Pangolin with compiler warning fixes..."

cd /tmp
rm -rf Pangolin
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

mkdir -p build
cd build

# Set environment variables instead of using cmake flags with quotes
export CXXFLAGS="-Wno-error=deprecated-copy -Wno-error=deprecated-register -Wno-error=null-pointer-subtraction -Wno-error=null-pointer-arithmetic -Wno-error"
export CFLAGS="-Wno-error"

echo "Configuring Pangolin..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF

echo "Building Pangolin..."
make -j4

echo "Installing Pangolin..."
sudo make install
sudo ldconfig

echo "Pangolin installation completed!"
