#!/usr/bin/env python3
"""
ZERO-DCE++ Model Implementation
Paper: Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation
Authors: Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, Runmin Cong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class ZeroDCEPlusPlus(nn.Module):
    """
    ZERO-DCE++ (Zero-Reference Deep Curve Estimation) model for low-light image enhancement.
    
    This model learns to enhance low-light images without requiring paired training data.
    It estimates pixel-wise curve parameters that can be applied to enhance the input image.
    """
    
    def __init__(self, input_channels: int = 3, num_iterations: int = 8):
        """
        Initialize ZERO-DCE++ model.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            num_iterations: Number of curve estimation iterations
        """
        super(ZeroDCEPlusPlus, self).__init__()
        
        self.num_iterations = num_iterations
        
        # Enhanced feature extraction with depth-wise separable convolutions
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = self._make_depthwise_conv(32, 32)
        self.conv3 = self._make_depthwise_conv(32, 32)
        self.conv4 = self._make_depthwise_conv(32, 32)
        
        # Curve parameter estimation
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(32, 24, kernel_size=3, stride=1, padding=1, bias=True)  # 8 iterations * 3 channels
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_depthwise_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create depth-wise separable convolution block."""
        return nn.Sequential(
            # Depth-wise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=True),
            nn.ReLU(inplace=True),
            # Point-wise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ZERO-DCE++ model.
        
        Args:
            x: Input low-light image tensor [B, C, H, W]
            
        Returns:
            Tuple of (enhanced_image, curve_parameters)
        """
        # Feature extraction
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        # Curve parameter estimation
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        curves = self.tanh(self.conv7(x6))
        
        # Apply curve enhancement iteratively
        enhanced = x
        for i in range(self.num_iterations):
            # Extract curve parameters for current iteration
            curve = curves[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + enhanced * (1 - enhanced) * curve
        
        return enhanced, curves


class ZeroDCELoss(nn.Module):
    """
    Comprehensive loss function for ZERO-DCE++ training.
    Combines spatial consistency, exposure control, color constancy, and illumination smoothness.
    """
    
    def __init__(self, spa_weight: float = 1.0, exp_weight: float = 10.0, 
                 col_weight: float = 5.0, tvs_weight: float = 1600.0):
        """
        Initialize loss function with weights.
        
        Args:
            spa_weight: Spatial consistency loss weight
            exp_weight: Exposure control loss weight  
            col_weight: Color constancy loss weight
            tvs_weight: Total variation smoothness loss weight
        """
        super(ZeroDCELoss, self).__init__()
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tvs_weight = tvs_weight
    
    def forward(self, enhanced: torch.Tensor, original: torch.Tensor, 
                curves: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Calculate total loss and individual loss components.
        
        Args:
            enhanced: Enhanced image tensor
            original: Original low-light image tensor
            curves: Estimated curve parameters
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Spatial consistency loss
        spa_loss = self._spatial_consistency_loss(enhanced, original)
        
        # Exposure control loss
        exp_loss = self._exposure_control_loss(enhanced)
        
        # Color constancy loss
        col_loss = self._color_constancy_loss(enhanced)
        
        # Total variation smoothness loss
        tvs_loss = self._total_variation_loss(curves)
        
        # Combine losses
        total_loss = (self.spa_weight * spa_loss + 
                     self.exp_weight * exp_loss + 
                     self.col_weight * col_loss + 
                     self.tvs_weight * tvs_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'spatial': spa_loss.item(),
            'exposure': exp_loss.item(), 
            'color': col_loss.item(),
            'smoothness': tvs_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _spatial_consistency_loss(self, enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Calculate spatial consistency loss between enhanced and original images."""
        # Convert to grayscale
        def rgb_to_gray(img):
            return 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        
        enhanced_gray = rgb_to_gray(enhanced)
        original_gray = rgb_to_gray(original)
        
        # Calculate gradients
        def gradient(img):
            grad_x = img[:, :, 1:] - img[:, :, :-1]
            grad_y = img[:, 1:, :] - img[:, :-1, :]
            return grad_x, grad_y
        
        enhanced_grad_x, enhanced_grad_y = gradient(enhanced_gray)
        original_grad_x, original_grad_y = gradient(original_gray)
        
        # Spatial consistency loss
        loss_x = torch.mean(torch.abs(enhanced_grad_x - original_grad_x))
        loss_y = torch.mean(torch.abs(enhanced_grad_y - original_grad_y))
        
        return loss_x + loss_y
    
    def _exposure_control_loss(self, enhanced: torch.Tensor, target_exposure: float = 0.6) -> torch.Tensor:
        """Calculate exposure control loss to achieve proper illumination."""
        # Calculate mean intensity
        mean_intensity = torch.mean(enhanced, dim=(2, 3))
        
        # Exposure loss - encourage proper exposure level
        exp_loss = torch.mean(torch.abs(mean_intensity - target_exposure))
        
        return exp_loss
    
    def _color_constancy_loss(self, enhanced: torch.Tensor) -> torch.Tensor:
        """Calculate color constancy loss to preserve color relationships."""
        # Calculate mean values for each channel
        mean_rgb = torch.mean(enhanced, dim=(2, 3))
        
        # Color constancy loss - minimize difference between color channels
        rg_loss = torch.mean(torch.abs(mean_rgb[:, 0] - mean_rgb[:, 1]))
        rb_loss = torch.mean(torch.abs(mean_rgb[:, 0] - mean_rgb[:, 2]))
        gb_loss = torch.mean(torch.abs(mean_rgb[:, 1] - mean_rgb[:, 2]))
        
        return rg_loss + rb_loss + gb_loss
    
    def _total_variation_loss(self, curves: torch.Tensor) -> torch.Tensor:
        """Calculate total variation loss for curve smoothness."""
        # Calculate TV loss for spatial smoothness
        diff_i = torch.abs(curves[:, :, 1:, :] - curves[:, :, :-1, :])
        diff_j = torch.abs(curves[:, :, :, 1:] - curves[:, :, :, :-1])
        
        tv_loss = torch.mean(diff_i) + torch.mean(diff_j)
        
        return tv_loss


class ZeroDCEInference:
    """
    Inference wrapper for ZERO-DCE++ model with optimizations for real-time processing.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model weights
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ZeroDCEPlusPlus()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Optimization for inference
        if hasattr(torch, 'compile') and device == 'cuda':
            self.model = torch.compile(self.model)
    
    def enhance_image(self, image: np.ndarray, return_curves: bool = False) -> np.ndarray:
        """
        Enhance a single low-light image.
        
        Args:
            image: Input image as numpy array [H, W, C] in BGR format
            return_curves: Whether to return estimated curves
            
        Returns:
            Enhanced image as numpy array [H, W, C] in BGR format
        """
        # Convert BGR to RGB and normalize
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            enhanced_tensor, curves = self.model(input_tensor)
        
        # Convert back to numpy
        enhanced_image = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        
        if return_curves:
            return enhanced_bgr, curves.squeeze(0).cpu().numpy()
        return enhanced_bgr
    
    def enhance_batch(self, images: list) -> list:
        """
        Enhance a batch of images for improved efficiency.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of enhanced images
        """
        enhanced_images = []
        
        for image in images:
            enhanced = self.enhance_image(image)
            enhanced_images.append(enhanced)
        
        return enhanced_images
    
    def calculate_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> dict:
        """
        Calculate enhancement quality metrics.
        
        Args:
            original: Original low-light image
            enhanced: Enhanced image
            
        Returns:
            Dictionary of quality metrics
        """
        # Convert to grayscale for some metrics
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse = np.mean((original_gray - enhanced_gray) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Structural Similarity Index (SSIM)
        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return np.mean(ssim_map)
        
        ssim_score = ssim(original_gray, enhanced_gray)
        
        # Brightness enhancement ratio
        original_brightness = np.mean(original_gray)
        enhanced_brightness = np.mean(enhanced_gray)
        brightness_ratio = enhanced_brightness / original_brightness if original_brightness > 0 else 1.0
        
        # Contrast enhancement
        original_contrast = np.std(original_gray)
        enhanced_contrast = np.std(enhanced_gray)
        contrast_ratio = enhanced_contrast / original_contrast if original_contrast > 0 else 1.0
        
        return {
            'psnr': psnr,
            'ssim': ssim_score,
            'brightness_ratio': brightness_ratio,
            'contrast_ratio': contrast_ratio,
            'original_brightness': original_brightness,
            'enhanced_brightness': enhanced_brightness
        }


def demo_enhancement():
    """
    Demonstration function for ZERO-DCE++ enhancement.
    """
    # Create sample low-light image (dark image)
    sample_image = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    
    # Initialize model (would normally load pre-trained weights)
    print("Creating ZERO-DCE++ model...")
    model = ZeroDCEPlusPlus()
    
    # Convert to tensor for demonstration
    input_tensor = torch.from_numpy(sample_image.transpose(2, 0, 1)).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0)
    
    # Forward pass
    print("Performing enhancement...")
    with torch.no_grad():
        enhanced_tensor, curves = model(input_tensor)
    
    # Convert back to image
    enhanced_image = enhanced_tensor.squeeze(0).numpy().transpose(1, 2, 0)
    enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)
    
    print(f"Input shape: {sample_image.shape}")
    print(f"Enhanced shape: {enhanced_image.shape}")
    print(f"Curve parameters shape: {curves.shape}")
    print("Enhancement completed successfully!")
    
    return enhanced_image


if __name__ == "__main__":
    demo_enhancement()
