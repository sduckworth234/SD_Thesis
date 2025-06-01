#!/bin/bash

# Quick fix for Pangolin compilation issues with newer GCC

set -e

echo "Fixing Pangolin compilation..."

cd /tmp
rm -rf Pangolin
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

mkdir -p build
cd build

# Configure with compiler warning fixes
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PANGOLIN_PYTHON=OFF \
    -DCMAKE_CXX_FLAGS="-Wno-error=deprecated-copy -Wno-error=deprecated-register -Wno-error=null-pointer-subtraction -Wno-error=null-pointer-arithmetic -Wno-error" \
    -DCMAKE_C_FLAGS="-Wno-error"

echo "Building Pangolin with fixed compiler flags..."
make -j4

echo "Installing Pangolin..."
sudo make install
sudo ldconfig

echo "Pangolin installation completed successfully!"
