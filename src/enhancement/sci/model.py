#!/usr/bin/env python3
"""
SCI (Self-Calibrated Illumination) Model Implementation
Paper: Toward Fast, Flexible, and Robust Low-Light Image Enhancement
Authors: Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, Zhongxuan Luo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List


class SelfCalibratedIllumination(nn.Module):
    """
    SCI (Self-Calibrated Illumination) model for low-light image enhancement.
    
    This model uses a self-calibrated illumination learning framework that can
    adaptively learn illumination-aware transformations for enhancement.
    """
    
    def __init__(self, input_channels: int = 3, stage: int = 3):
        """
        Initialize SCI model.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            stage: Number of enhancement stages
        """
        super(SelfCalibratedIllumination, self).__init__()
        
        self.stage = stage
        
        # Feature extraction backbone
        self.conv_input = nn.Conv2d(input_channels, 32, 3, 1, 1, bias=True)
        
        # Multi-stage enhancement modules
        self.stages = nn.ModuleList()
        for i in range(stage):
            self.stages.append(EnhancementStage(32, 32))
        
        # Output layer
        self.conv_output = nn.Conv2d(32, input_channels, 3, 1, 1, bias=True)
        
        # Illumination-aware attention modules
        self.illumination_attention = nn.ModuleList()
        for i in range(stage):
            self.illumination_attention.append(IlluminationAttention(32))
        
        # Self-calibration modules
        self.self_calibration = SelfCalibrationModule(32, stage)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of SCI model.
        
        Args:
            x: Input low-light image tensor [B, C, H, W]
            
        Returns:
            Tuple of (final_enhanced_image, intermediate_results)
        """
        # Initial feature extraction
        features = F.relu(self.conv_input(x))
        
        # Store intermediate results
        intermediate_results = []
        
        # Multi-stage enhancement
        enhanced = x
        for i in range(self.stage):
            # Apply illumination-aware attention
            attended_features = self.illumination_attention[i](features, enhanced)
            
            # Enhancement stage
            delta = self.stages[i](attended_features)
            enhanced = enhanced + delta
            enhanced = torch.clamp(enhanced, 0, 1)
            
            intermediate_results.append(enhanced)
            
            # Update features for next stage
            if i < self.stage - 1:
                features = features + self.conv_input(enhanced)
        
        # Self-calibration
        calibrated_features = self.self_calibration(features, intermediate_results)
        
        # Final output
        final_delta = self.conv_output(calibrated_features)
        final_enhanced = enhanced + final_delta
        final_enhanced = torch.clamp(final_enhanced, 0, 1)
        
        intermediate_results.append(final_enhanced)
        
        return final_enhanced, intermediate_results


class EnhancementStage(nn.Module):
    """
    Individual enhancement stage with residual learning.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(EnhancementStage, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(out_channels, 3, 3, 1, 1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of enhancement stage."""
        residual = x
        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        
        return out


class IlluminationAttention(nn.Module):
    """
    Illumination-aware attention mechanism.
    """
    
    def __init__(self, channels: int):
        super(IlluminationAttention, self).__init__()
        
        self.channels = channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Illumination-aware weights
        self.illumination_conv = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Apply illumination-aware attention.
        
        Args:
            features: Feature maps
            image: Current enhanced image
            
        Returns:
            Attended features
        """
        # Channel attention
        channel_att = self.channel_attention(features)
        features_ca = features * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(features_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(features_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        features_sa = features_ca * spatial_att
        
        # Illumination-aware weighting
        illumination_weight = self.illumination_conv(image)
        features_illumination = features_sa * illumination_weight
        
        return features_illumination


class SelfCalibrationModule(nn.Module):
    """
    Self-calibration module for adaptive enhancement.
    """
    
    def __init__(self, channels: int, num_stages: int):
        super(SelfCalibrationModule, self).__init__()
        
        self.num_stages = num_stages
        
        # Stage fusion
        self.stage_fusion = nn.Conv2d(channels * num_stages, channels, 1, 1, 0)
        
        # Calibration network
        self.calibration_net = nn.Sequential(
            nn.Conv2d(channels + 3 * num_stages, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    
    def forward(self, features: torch.Tensor, intermediate_results: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply self-calibration based on intermediate results.
        
        Args:
            features: Current feature maps
            intermediate_results: List of intermediate enhanced images
            
        Returns:
            Calibrated features
        """
        # Concatenate intermediate results
        concat_images = torch.cat(intermediate_results, dim=1)
        
        # Combine features and intermediate results
        combined_input = torch.cat([features, concat_images], dim=1)
        
        # Apply calibration
        calibrated_features = self.calibration_net(combined_input)
        
        # Residual connection
        calibrated_features = features + calibrated_features
        
        return calibrated_features


class SCILoss(nn.Module):
    """
    Loss function for SCI model training.
    """
    
    def __init__(self, stage_weights: Optional[List[float]] = None):
        super(SCILoss, self).__init__()
        
        if stage_weights is None:
            self.stage_weights = [1.0, 1.0, 1.0, 2.0]  # Higher weight for final result
        else:
            self.stage_weights = stage_weights
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: List[torch.Tensor], target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Calculate multi-stage loss.
        
        Args:
            predictions: List of predictions from each stage
            target: Ground truth enhanced image
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        total_loss = 0
        loss_dict = {}
        
        for i, pred in enumerate(predictions):
            # L1 loss
            l1 = self.l1_loss(pred, target)
            
            # MSE loss
            mse = self.mse_loss(pred, target)
            
            # Perceptual loss (simplified)
            perceptual = self._perceptual_loss(pred, target)
            
            # Combine losses for this stage
            stage_loss = l1 + 0.1 * mse + 0.01 * perceptual
            weighted_loss = self.stage_weights[i] * stage_loss
            
            total_loss += weighted_loss
            
            loss_dict[f'stage_{i}_l1'] = l1.item()
            loss_dict[f'stage_{i}_mse'] = mse.item()
            loss_dict[f'stage_{i}_perceptual'] = perceptual.item()
            loss_dict[f'stage_{i}_total'] = stage_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Simplified perceptual loss using gradient differences.
        """
        # Convert to grayscale
        def rgb_to_gray(img):
            return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        
        pred_gray = rgb_to_gray(pred)
        target_gray = rgb_to_gray(target)
        
        # Calculate gradients
        pred_grad_x = pred_gray[:, :, :, 1:] - pred_gray[:, :, :, :-1]
        pred_grad_y = pred_gray[:, :, 1:, :] - pred_gray[:, :, :-1, :]
        
        target_grad_x = target_gray[:, :, :, 1:] - target_gray[:, :, :, :-1]
        target_grad_y = target_gray[:, :, 1:, :] - target_gray[:, :, :-1, :]
        
        # Gradient loss
        grad_loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y


class SCIInference:
    """
    Inference wrapper for SCI model with optimizations.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize SCI inference engine.
        
        Args:
            model_path: Path to trained model weights
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SelfCalibratedIllumination()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Optimization for inference
        if hasattr(torch, 'compile') and device == 'cuda':
            self.model = torch.compile(self.model)
    
    def enhance_image(self, image: np.ndarray, return_stages: bool = False) -> np.ndarray:
        """
        Enhance a single low-light image using SCI.
        
        Args:
            image: Input image as numpy array [H, W, C] in BGR format
            return_stages: Whether to return intermediate enhancement stages
            
        Returns:
            Enhanced image as numpy array [H, W, C] in BGR format
        """
        # Convert BGR to RGB and normalize
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(rgb_image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            enhanced_tensor, intermediate_results = self.model(input_tensor)
        
        # Convert back to numpy
        enhanced_image = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        
        if return_stages:
            stages = []
            for stage_result in intermediate_results:
                stage_image = stage_result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                stage_image = np.clip(stage_image * 255.0, 0, 255).astype(np.uint8)
                stage_bgr = cv2.cvtColor(stage_image, cv2.COLOR_RGB2BGR)
                stages.append(stage_bgr)
            return enhanced_bgr, stages
        
        return enhanced_bgr
    
    def calculate_enhancement_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> dict:
        """
        Calculate comprehensive enhancement quality metrics.
        
        Args:
            original: Original low-light image
            enhanced: Enhanced image
            
        Returns:
            Dictionary of quality metrics
        """
        # Convert to float for calculations
        original_float = original.astype(np.float32) / 255.0
        enhanced_float = enhanced.astype(np.float32) / 255.0
        
        # Brightness metrics
        original_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        enhanced_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        brightness_improvement = enhanced_brightness / original_brightness if original_brightness > 0 else 1.0
        
        # Contrast metrics
        original_contrast = np.std(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        enhanced_contrast = np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        contrast_improvement = enhanced_contrast / original_contrast if original_contrast > 0 else 1.0
        
        # Information entropy (measure of detail preservation)
        def calculate_entropy(img):
            hist, _ = np.histogram(img, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))
        
        original_entropy = calculate_entropy(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
        enhanced_entropy = calculate_entropy(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        
        # Color naturalness (simplified)
        def color_naturalness(img):
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # Calculate standard deviation of a and b channels
            a_std = np.std(lab[:, :, 1])
            b_std = np.std(lab[:, :, 2])
            return (a_std + b_std) / 2
        
        original_naturalness = color_naturalness(original)
        enhanced_naturalness = color_naturalness(enhanced)
        
        return {
            'brightness_original': original_brightness,
            'brightness_enhanced': enhanced_brightness,
            'brightness_improvement': brightness_improvement,
            'contrast_original': original_contrast,
            'contrast_enhanced': enhanced_contrast,
            'contrast_improvement': contrast_improvement,
            'entropy_original': original_entropy,
            'entropy_enhanced': enhanced_entropy,
            'entropy_improvement': enhanced_entropy / original_entropy if original_entropy > 0 else 1.0,
            'naturalness_original': original_naturalness,
            'naturalness_enhanced': enhanced_naturalness,
            'naturalness_preservation': min(enhanced_naturalness / original_naturalness, 
                                          original_naturalness / enhanced_naturalness) if original_naturalness > 0 else 1.0
        }


def demo_sci_enhancement():
    """
    Demonstration function for SCI enhancement.
    """
    # Create sample low-light image
    sample_image = np.random.randint(0, 60, (480, 640, 3), dtype=np.uint8)
    
    # Initialize model
    print("Creating SCI model...")
    model = SelfCalibratedIllumination(stage=3)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(sample_image.transpose(2, 0, 1)).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0)
    
    # Forward pass
    print("Performing multi-stage enhancement...")
    with torch.no_grad():
        enhanced_tensor, intermediate_results = model(input_tensor)
    
    # Convert back to image
    enhanced_image = enhanced_tensor.squeeze(0).numpy().transpose(1, 2, 0)
    enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)
    
    print(f"Input shape: {sample_image.shape}")
    print(f"Enhanced shape: {enhanced_image.shape}")
    print(f"Number of enhancement stages: {len(intermediate_results)}")
    print("SCI enhancement completed successfully!")
    
    return enhanced_image, intermediate_results


if __name__ == "__main__":
    demo_sci_enhancement()
