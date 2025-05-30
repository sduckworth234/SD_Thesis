# Contributing to SD_Thesis

Thank you for your interest in contributing to the SD_Thesis project! This document provides guidelines for contributing to this research project.

## 📋 Project Overview

This is a master's thesis project focused on developing low-light vision systems for search and rescue UAV operations. The project integrates multiple computer vision technologies including:

- Intel RealSense D435i camera integration
- Low-light image enhancement (ZERO-DCE++, SCI)
- Person detection (YOLO v4)
- Visual SLAM (ORB-SLAM2)
- Edge deployment on Jetson Xavier NX

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Include detailed descriptions and reproducible steps
- Specify your hardware/software environment
- Include relevant log files and error messages

### Suggesting Enhancements
- Open an issue with the "enhancement" label
- Clearly describe the proposed feature
- Explain how it aligns with the project objectives
- Provide examples or mockups if applicable

### Code Contributions
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

## 📝 Coding Standards

### Python Code
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Add type hints where appropriate
- Maintain test coverage above 80%

### ROS Nodes
- Follow ROS naming conventions
- Use appropriate message types
- Include proper error handling
- Add comprehensive logging

### Documentation
- Use clear, concise language
- Include code examples
- Update README.md for major changes
- Document experimental procedures

## 🧪 Testing

### Running Tests
```bash
# Unit tests
python -m pytest tests/unit_tests/

# Integration tests
python -m pytest tests/integration_tests/

# ROS tests
catkin_make run_tests
```

### Test Requirements
- All new features must include tests
- Tests should cover edge cases
- Integration tests for ROS nodes
- Performance benchmarks for critical paths

## 📊 Experimental Validation

### Data Collection
- Follow the data collection protocols in `docs/methodology.md`
- Use consistent lighting conditions
- Document all experimental parameters
- Include control experiments

### Benchmarking
- Run standardized benchmarks
- Compare with baseline methods
- Document performance metrics
- Include statistical significance tests

## 🚀 Deployment Testing

### Development Environment
- Ubuntu 20.04 with ROS Noetic
- NVIDIA RTX 4070 (or equivalent)
- Intel RealSense D435i

### Target Environment
- Jetson Xavier NX
- Ubuntu 18.04
- aarch64 architecture

## 📖 Documentation

### Required Documentation
- API documentation for new modules
- Setup instructions for new dependencies
- Experimental procedures and results
- Performance analysis and optimization notes

### Documentation Format
- Use Markdown for all documentation
- Include diagrams and flowcharts where helpful
- Provide code examples
- Link to relevant academic papers

## 🔍 Review Process

### Peer Review
- All contributions require peer review
- Focus on code quality and experimental rigor
- Verify reproducibility of results
- Check alignment with thesis objectives

### Acceptance Criteria
- Code passes all tests
- Documentation is complete
- Experimental validation is provided
- Performance impact is assessed

## 📞 Getting Help

### Communication Channels
- GitHub Issues for bug reports and feature requests
- GitHub Discussions for questions and brainstorming
- Direct contact with project maintainer for urgent issues

### Resources
- Project documentation in `docs/`
- Academic papers in `Literature/`
- Experimental procedures in `docs/methodology.md`
- Setup guides in `docs/setup_instructions.md`

## 🎯 Research Focus Areas

### Priority Areas
1. **Real-time performance optimization**
2. **Jetson Xavier NX compatibility**
3. **Low-light enhancement algorithms**
4. **SLAM robustness in challenging conditions**
5. **Person detection accuracy improvements**

### Out of Scope
- Non-RealSense camera support
- Non-ROS frameworks
- GUI development (focus on command-line tools)
- Platforms other than Ubuntu

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

- Research supervisors and advisors
- Open-source communities (ROS, OpenCV, PyTorch)
- Academic researchers in computer vision and robotics
- Contributors and collaborators

Thank you for helping advance research in autonomous search and rescue systems!
