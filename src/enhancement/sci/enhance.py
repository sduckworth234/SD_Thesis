#!/usr/bin/env python3
"""
SCI Enhancement Interface
High-level interface for Self-Calibrated Illumination low-light image enhancement.
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import Union, Optional, Tuple, List
from .model import SCIInference, SelfCalibratedIllumination


class SCIEnhancer:
    """
    High-level interface for SCI (Self-Calibrated Illumination) low-light image enhancement.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto', stage: int = 3):
        """
        Initialize SCI enhancer.
        
        Args:
            model_path: Path to pre-trained model weights. If None, uses default model.
            device: Computing device ('auto', 'cuda', or 'cpu')
            stage: Number of enhancement stages
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.stage = stage
        
        # Set default model path if not provided
        if model_path is None:
            model_dir = Path(__file__).parent / "models"
            model_path = model_dir / "sci_model.pth"
            
            # Create directory if it doesn't exist
            model_dir.mkdir(exist_ok=True)
            
            # Download model if it doesn't exist
            if not model_path.exists():
                print(f"Model not found at {model_path}")
                print("Using randomly initialized model for demonstration.")
                print("For production use, please provide a trained model.")
                self._create_demo_model(str(model_path))
        
        # Initialize inference engine
        try:
            self.inference_engine = SCIInference(str(model_path), device)
            print(f"SCI model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating demo model for testing...")
            self._create_demo_model(str(model_path))
            self.inference_engine = SCIInference(str(model_path), device)
    
    def _create_demo_model(self, save_path: str):
        """Create a demo model with random weights for testing."""
        model = SelfCalibratedIllumination(stage=self.stage)
        torch.save(model.state_dict(), save_path)

def enhance_image(image):
    """
    Legacy function for backward compatibility.
    Enhance the input image using the SCI model.
    
    Parameters:
    image (numpy.ndarray): Input low-light image to be enhanced.
    
    Returns:
    numpy.ndarray: Enhanced image.
    """
    try:
        enhancer = SCIEnhancer()
        return enhancer.inference_engine.enhance_image(image)
    except Exception as e:
        print(f"Enhancement failed: {e}, using simple contrast enhancement")
        # Fallback to simple enhancement
        enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        return enhanced

def process_video(video_path):
    """
    Process a video file and enhance each frame using the SCI model.
    
    Parameters:
    video_path (str): Path to the input video file.
    """
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        enhanced_frame = enhance_image(frame)
        cv2.imshow('Enhanced Frame', enhanced_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()