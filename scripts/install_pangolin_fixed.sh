#!/bin/bash

# Fixed Pangolin installation for Ubuntu 20.04 with GCC 9.4

set -e

LOG_FILE="/tmp/pangolin_install.log"

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

install_pangolin() {
    log "Installing Pangolin with Ubuntu 20.04 GCC 9.4 compatibility..."
    
    # Clean up any previous attempts
    cd /tmp
    sudo rm -rf Pangolin
    
    # Clone Pangolin
    git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    
    # Create patch for CMakeLists.txt to disable problematic warnings
    log "Applying GCC 9.4 compatibility patches..."
    
    # Create a patch to modify CMakeLists.txt
    cat > pangolin_gcc94_patch.txt << 'EOF'
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -50,0 +50,3 @@
+# Disable problematic warnings for GCC 9.4 (Ubuntu 20.04)
+set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy -Wno-deprecated-declarations")
+set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
EOF

    # Apply the patch manually by modifying CMakeLists.txt
    if grep -q "Disable problematic warnings" CMakeLists.txt; then
        log "Patch already applied"
    else
        log "Applying GCC 9.4 compatibility patch to CMakeLists.txt..."
        
        # Find a good place to insert our flags (after cmake_minimum_required)
        sed -i '/cmake_minimum_required/a\\n# Disable problematic warnings for GCC 9.4 (Ubuntu 20.04)\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy -Wno-deprecated-declarations")\nset(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")' CMakeLists.txt
    fi
    
    # Also patch any component CMakeFiles that might have issues
    log "Patching component CMakeFiles..."
    find . -name "CMakeLists.txt" -exec sed -i '/cmake_minimum_required/a\\n# Disable problematic warnings for GCC 9.4\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy -Wno-deprecated-declarations")\nset(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")' {} \; 2>/dev/null || true
    
    mkdir -p build
    cd build
    
    # Configure with very conservative flags
    log "Configuring Pangolin build..."
    
    CMAKE_FLAGS=(
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_PANGOLIN_PYTHON=OFF"
        "-DBUILD_EXAMPLES=OFF"
        "-DBUILD_TOOLS=OFF"
        "-DCMAKE_CXX_FLAGS=-Wno-deprecated-copy -Wno-deprecated-declarations -Wno-error"
        "-DCMAKE_C_FLAGS=-Wno-deprecated-declarations -Wno-error"
    )
    
    log "Running cmake with flags: ${CMAKE_FLAGS[*]}"
    cmake "${CMAKE_FLAGS[@]}" ..
    
    log "Building Pangolin..."
    make -j$(nproc) 2>&1 | tee -a $LOG_FILE
    
    log "Installing Pangolin..."
    sudo make install
    sudo ldconfig
    
    log "Pangolin installation completed successfully"
}

# Run the installation
install_pangolin
