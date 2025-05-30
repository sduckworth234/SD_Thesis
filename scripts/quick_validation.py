#!/usr/bin/env python3
"""
Quick Setup Validation for SD_Thesis
Run this script after setup to ensure everything is working correctly
"""

import subprocess
import sys
import os
import importlib
import signal
import time
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status):
    """Print status with colors"""
    if status == "PASS":
        print(f"{Colors.GREEN}✓ {message}{Colors.END}")
    elif status == "FAIL":
        print(f"{Colors.RED}✗ {message}{Colors.END}")
    elif status == "WARN":
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
    else:
        print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def run_command(cmd, timeout=10):
    """Run shell command with timeout"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, 
                              text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_python_import(module_name):
    """Check if Python module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    print(f"{Colors.BOLD}SD_Thesis Quick Setup Validation{Colors.END}")
    print("=" * 50)
    
    # Change to project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)
    
    validation_results = []
    
    # 1. Check basic system requirements
    print(f"\n{Colors.BOLD}1. System Requirements{Colors.END}")
    
    # Ubuntu version
    success, stdout, stderr = run_command("lsb_release -r")
    if success and "20.04" in stdout:
        print_status("Ubuntu 20.04", "PASS")
        validation_results.append(True)
    else:
        print_status("Ubuntu 20.04 (current: unknown)", "WARN")
        validation_results.append(False)
    
    # Python 3
    success, stdout, stderr = run_command("python3 --version")
    if success:
        version = stdout.strip().split()[-1]
        print_status(f"Python 3 ({version})", "PASS")
        validation_results.append(True)
    else:
        print_status("Python 3", "FAIL")
        validation_results.append(False)
    
    # 2. Check development tools
    print(f"\n{Colors.BOLD}2. Development Tools{Colors.END}")
    
    tools = ["gcc", "g++", "cmake", "git"]
    for tool in tools:
        success, _, _ = run_command(f"which {tool}")
        print_status(tool, "PASS" if success else "FAIL")
        validation_results.append(success)
    
    # 3. Check ROS installation
    print(f"\n{Colors.BOLD}3. ROS Noetic{Colors.END}")
    
    # ROS core installation
    if os.path.exists("/opt/ros/noetic/setup.bash"):
        print_status("ROS Noetic installation", "PASS")
        validation_results.append(True)
        
        # Source ROS and check environment
        success, stdout, stderr = run_command("source /opt/ros/noetic/setup.bash && echo $ROS_DISTRO")
        if success and "noetic" in stdout:
            print_status("ROS environment", "PASS")
            validation_results.append(True)
        else:
            print_status("ROS environment", "FAIL")
            validation_results.append(False)
    else:
        print_status("ROS Noetic installation", "FAIL")
        validation_results.append(False)
    
    # Check ROS tools
    ros_tools = ["roscore", "roslaunch", "catkin_make"]
    for tool in ros_tools:
        success, _, _ = run_command(f"which {tool}")
        print_status(f"ROS {tool}", "PASS" if success else "FAIL")
        validation_results.append(success)
    
    # 4. Check Intel RealSense
    print(f"\n{Colors.BOLD}4. Intel RealSense{Colors.END}")
    
    # RealSense library
    success, _, _ = run_command("dpkg -l | grep librealsense2")
    print_status("librealsense2", "PASS" if success else "FAIL")
    validation_results.append(success)
    
    # RealSense viewer
    success, _, _ = run_command("which realsense-viewer")
    print_status("realsense-viewer", "PASS" if success else "FAIL")
    validation_results.append(success)
    
    # 5. Check Python environment
    print(f"\n{Colors.BOLD}5. Python Environment{Colors.END}")
    
    # Virtual environment
    venv_path = project_dir / "venv"
    if venv_path.exists():
        print_status("Virtual environment", "PASS")
        validation_results.append(True)
        
        # Activate venv and check packages
        venv_python = venv_path / "bin" / "python3"
        if venv_python.exists():
            # Key packages
            packages = ["numpy", "opencv-python", "torch", "tensorflow"]
            for package in packages:
                success, _, _ = run_command(f"{venv_python} -c 'import {package.replace('-', '_')}'")
                print_status(package, "PASS" if success else "FAIL")
                validation_results.append(success)
        else:
            print_status("Virtual environment Python", "FAIL")
            validation_results.append(False)
    else:
        print_status("Virtual environment", "FAIL")
        validation_results.append(False)
    
    # 6. Check project structure
    print(f"\n{Colors.BOLD}6. Project Structure{Colors.END}")
    
    required_dirs = ["src", "launch", "config", "tests", "scripts", "docs"]
    for dir_name in required_dirs:
        dir_path = project_dir / dir_name
        print_status(f"{dir_name}/ directory", "PASS" if dir_path.exists() else "FAIL")
        validation_results.append(dir_path.exists())
    
    required_files = ["package.xml", "CMakeLists.txt", "requirements.txt"]
    for file_name in required_files:
        file_path = project_dir / file_name
        print_status(file_name, "PASS" if file_path.exists() else "FAIL")
        validation_results.append(file_path.exists())
    
    # 7. Check models and data
    print(f"\n{Colors.BOLD}7. Models and Data{Colors.END}")
    
    # YOLO files
    yolo_files = [
        "models/yolo/yolov4.cfg",
        "models/yolo/coco.names"
    ]
    
    for file_path in yolo_files:
        full_path = project_dir / file_path
        print_status(file_path, "PASS" if full_path.exists() else "FAIL")
        validation_results.append(full_path.exists())
    
    # YOLO weights (large file, may not be present)
    yolo_weights = project_dir / "models/yolo/yolov4.weights"
    if yolo_weights.exists():
        print_status("YOLO weights", "PASS")
        validation_results.append(True)
    else:
        print_status("YOLO weights (will be downloaded)", "WARN")
        validation_results.append(False)
    
    # 8. Test basic functionality
    print(f"\n{Colors.BOLD}8. Basic Functionality Tests{Colors.END}")
    
    # Test Python imports in virtual environment
    if venv_path.exists():
        venv_python = venv_path / "bin" / "python3"
        
        # OpenCV with DNN
        success, _, _ = run_command(f"{venv_python} -c 'import cv2; print(cv2.__version__); cv2.dnn.readNet'")
        print_status("OpenCV DNN module", "PASS" if success else "FAIL")
        validation_results.append(success)
        
        # PyTorch CUDA (if available)
        success, stdout, _ = run_command(f"{venv_python} -c 'import torch; print(torch.cuda.is_available())'")
        if success and "True" in stdout:
            print_status("PyTorch CUDA", "PASS")
            validation_results.append(True)
        elif success:
            print_status("PyTorch (CPU only)", "WARN")
            validation_results.append(True)
        else:
            print_status("PyTorch", "FAIL")
            validation_results.append(False)
    
    # 9. Summary
    print(f"\n{Colors.BOLD}Validation Summary{Colors.END}")
    print("=" * 50)
    
    total_tests = len(validation_results)
    passed_tests = sum(validation_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 System is ready for development!{Colors.END}")
        print("\nNext steps:")
        print("1. Connect your Intel RealSense D435i camera")
        print("2. Test camera: realsense-viewer")
        print("3. Run ROS tests: roslaunch sd_thesis test_camera.launch")
        return 0
    elif success_rate >= 70:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ System mostly ready, minor issues detected{Colors.END}")
        print("\nRecommended actions:")
        print("1. Review failed tests above")
        print("2. Run: ./scripts/complete_setup.sh")
        print("3. Check troubleshooting guide: docs/troubleshooting.md")
        return 1
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ System requires attention{Colors.END}")
        print("\nRequired actions:")
        print("1. Run setup script: ./scripts/complete_setup.sh")
        print("2. Follow setup guide: docs/setup_instructions.md")
        print("3. Check troubleshooting: docs/troubleshooting.md")
        return 2

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {e}{Colors.END}")
        sys.exit(1)
