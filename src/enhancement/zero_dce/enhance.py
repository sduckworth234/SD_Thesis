#!/usr/bin/env python3
"""
ZERO-DCE++ Enhancement Interface
High-level interface for low-light image enhancement using ZERO-DCE++ model.
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import Union, Optional, Tuple
from .model import ZeroDCEInference, ZeroDCEPlusPlus


class ZeroDCEEnhancer:
    """
    High-level interface for ZERO-DCE++ low-light image enhancement.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize ZERO-DCE++ enhancer.
        
        Args:
            model_path: Path to pre-trained model weights. If None, uses default model.
            device: Computing device ('auto', 'cuda', or 'cpu')
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        # Set default model path if not provided
        if model_path is None:
            model_dir = Path(__file__).parent / "models"
            model_path = model_dir / "zero_dce_plus_plus.pth"
            
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
            self.inference_engine = ZeroDCEInference(str(model_path), device)
            print(f"ZERO-DCE++ model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating demo model for testing...")
            self._create_demo_model(str(model_path))
            self.inference_engine = ZeroDCEInference(str(model_path), device)
    
    def _create_demo_model(self, save_path: str):
        """Create a demo model with random weights for testing."""
        model = ZeroDCEPlusPlus()
        torch.save(model.state_dict(), save_path)
    
    def enhance_image(self, image: Union[str, np.ndarray], 
                     output_path: Optional[str] = None) -> np.ndarray:
        """
        Enhance a single low-light image.
        
        Args:
            image: Input image (file path or numpy array)
            output_path: Optional output file path
            
        Returns:
            Enhanced image as numpy array
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            input_image = cv2.imread(image)
            if input_image is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            input_image = image.copy()
        
        # Enhance image
        enhanced_image = self.inference_engine.enhance_image(input_image)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, enhanced_image)
            print(f"Enhanced image saved to: {output_path}")
        
        return enhanced_image
    
    def enhance_video(self, input_path: str, output_path: str, 
                     fps: Optional[float] = None) -> bool:
        """
        Enhance a video file frame by frame.
        
        Args:
            input_path: Path to input video
            output_path: Path to output enhanced video
            fps: Output video FPS (uses input FPS if None)
            
        Returns:
            Success status
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps is None:
                fps = input_fps
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print(f"Processing {total_frames} frames...")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhance frame
                enhanced_frame = self.inference_engine.enhance_image(frame)
                out.write(enhanced_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Cleanup
            cap.release()
            out.release()
            
            print(f"Enhanced video saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return False
    
    def batch_enhance(self, input_dir: str, output_dir: str, 
                     extensions: list = ['.jpg', '.jpeg', '.png', '.bmp']) -> int:
        """
        Enhance all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for enhanced images
            extensions: List of supported image extensions
            
        Returns:
            Number of images processed
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return 0
        
        print(f"Found {len(image_files)} images to process...")
        
        processed_count = 0
        for i, image_file in enumerate(image_files):
            try:
                # Load and enhance image
                input_image = cv2.imread(str(image_file))
                if input_image is None:
                    print(f"Skipping {image_file.name}: Could not load")
                    continue
                
                enhanced_image = self.inference_engine.enhance_image(input_image)
                
                # Save enhanced image
                output_file = output_path / f"enhanced_{image_file.name}"
                cv2.imwrite(str(output_file), enhanced_image)
                
                processed_count += 1
                progress = ((i + 1) / len(image_files)) * 100
                print(f"Progress: {progress:.1f}% - Processed {image_file.name}")
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
        
        print(f"Batch processing completed. {processed_count}/{len(image_files)} images processed.")
        return processed_count
    
    def evaluate_enhancement(self, original_path: str, enhanced_path: str = None) -> dict:
        """
        Evaluate enhancement quality metrics.
        
        Args:
            original_path: Path to original low-light image
            enhanced_path: Path to enhanced image (if None, enhances original)
            
        Returns:
            Dictionary of quality metrics
        """
        # Load original image
        original_image = cv2.imread(original_path)
        if original_image is None:
            raise ValueError(f"Could not load original image: {original_path}")
        
        # Get enhanced image
        if enhanced_path is None:
            enhanced_image = self.enhance_image(original_image)
        else:
            enhanced_image = cv2.imread(enhanced_path)
            if enhanced_image is None:
                raise ValueError(f"Could not load enhanced image: {enhanced_path}")
        
        # Calculate metrics
        metrics = self.inference_engine.calculate_metrics(original_image, enhanced_image)
        
        # Print results
        print("\n=== Enhancement Quality Metrics ===")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"Brightness Enhancement: {metrics['brightness_ratio']:.2f}x")
        print(f"Contrast Enhancement: {metrics['contrast_ratio']:.2f}x")
        print(f"Original Brightness: {metrics['original_brightness']:.2f}")
        print(f"Enhanced Brightness: {metrics['enhanced_brightness']:.2f}")
        
        return metrics


def main():
    """
    Command-line interface for ZERO-DCE++ enhancement.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="ZERO-DCE++ Low-Light Image Enhancement")
    parser.add_argument("--input", "-i", required=True, help="Input image/video/directory path")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--model", "-m", help="Path to model weights")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"], 
                       help="Computing device")
    parser.add_argument("--mode", default="image", choices=["image", "video", "batch"],
                       help="Processing mode")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate enhancement quality")
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = ZeroDCEEnhancer(model_path=args.model, device=args.device)
    
    # Process based on mode
    if args.mode == "image":
        if args.output is None:
            args.output = "enhanced_" + os.path.basename(args.input)
        
        enhanced = enhancer.enhance_image(args.input, args.output)
        
        if args.evaluate:
            enhancer.evaluate_enhancement(args.input)
            
    elif args.mode == "video":
        if args.output is None:
            args.output = "enhanced_" + os.path.basename(args.input)
        
        success = enhancer.enhance_video(args.input, args.output)
        if not success:
            print("Video enhancement failed!")
            
    elif args.mode == "batch":
        if args.output is None:
            args.output = "enhanced_output"
        
        count = enhancer.batch_enhance(args.input, args.output)
        print(f"Processed {count} images")


if __name__ == "__main__":
    main()