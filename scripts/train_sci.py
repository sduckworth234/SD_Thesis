#!/usr/bin/env python3
"""
Training Script for SCI (Sequential Color Illumination) Model
Low-light image enhancement model training for the SD_Thesis project.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt

class SelfCalibratedIlluminationModule(nn.Module):
    """Self-Calibrated Illumination Module"""
    
    def __init__(self, in_channels, reduction=16):
        super(SelfCalibratedIlluminationModule, self).__init__()
        
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid(avg_out + max_out)
        
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        
        x = x * spatial_att
        
        return x

class ResidualBlock(nn.Module):
    """Residual Block with Self-Calibrated Illumination"""
    
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.sci_module = SelfCalibratedIlluminationModule(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply self-calibrated illumination
        out = self.sci_module(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class SCINet(nn.Module):
    """Self-Calibrated Illumination Network"""
    
    def __init__(self, num_channels=3, num_features=64, num_blocks=8):
        super(SCINet, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(num_channels, num_features, 3, padding=1)
        
        # Residual blocks with SCI modules
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_features, num_features) for _ in range(num_blocks)
        ])
        
        # Feature refinement
        self.conv_refine = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Output layer
        self.conv_last = nn.Conv2d(num_features, num_channels, 3, padding=1)
        
        # Illumination map estimation
        self.illum_conv1 = nn.Conv2d(num_features, num_features // 2, 3, padding=1)
        self.illum_conv2 = nn.Conv2d(num_features // 2, 1, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Feature extraction
        features = self.relu(self.conv_first(x))
        
        # Residual processing
        for res_block in self.res_blocks:
            features = res_block(features)
        
        # Feature refinement
        features = self.relu(self.conv_refine(features))
        
        # Generate enhanced image
        enhanced = self.conv_last(features)
        enhanced = x + enhanced  # Residual learning
        
        # Generate illumination map
        illum_features = self.relu(self.illum_conv1(features))
        illumination_map = self.sigmoid(self.illum_conv2(illum_features))
        
        # Apply illumination enhancement
        enhanced = enhanced * illumination_map + enhanced * (1 - illumination_map)
        
        return enhanced, illumination_map

class SCILoss(nn.Module):
    """SCI Loss Functions"""
    
    def __init__(self):
        super(SCILoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, enhanced, illumination_map, target, original):
        # Reconstruction loss
        reconstruction_loss = self.l1_loss(enhanced, target)
        
        # Perceptual loss (simplified)
        perceptual_loss = self._perceptual_loss(enhanced, target)
        
        # Illumination smoothness loss
        illumination_loss = self._illumination_smoothness_loss(illumination_map)
        
        # Color consistency loss
        color_loss = self._color_consistency_loss(enhanced, target)
        
        # Total variation loss
        tv_loss = self._total_variation_loss(enhanced)
        
        # Combine losses
        total_loss = reconstruction_loss + 0.1 * perceptual_loss + 0.01 * illumination_loss + \
                    0.1 * color_loss + 0.001 * tv_loss
        
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'perceptual': perceptual_loss.item(),
            'illumination': illumination_loss.item(),
            'color': color_loss.item(),
            'tv': tv_loss.item()
        }
    
    def _perceptual_loss(self, enhanced, target):
        """Simplified perceptual loss using gradients"""
        # Convert to grayscale
        gray_enhanced = 0.299 * enhanced[:, 0] + 0.587 * enhanced[:, 1] + 0.114 * enhanced[:, 2]
        gray_target = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # Compute gradients
        grad_enhanced_x = gray_enhanced[:, :, :-1] - gray_enhanced[:, :, 1:]
        grad_enhanced_y = gray_enhanced[:, :-1, :] - gray_enhanced[:, 1:, :]
        
        grad_target_x = gray_target[:, :, :-1] - gray_target[:, :, 1:]
        grad_target_y = gray_target[:, :-1, :] - gray_target[:, 1:, :]
        
        loss_x = self.l1_loss(grad_enhanced_x, grad_target_x)
        loss_y = self.l1_loss(grad_enhanced_y, grad_target_y)
        
        return loss_x + loss_y
    
    def _illumination_smoothness_loss(self, illumination_map):
        """Encourage smooth illumination maps"""
        grad_x = torch.abs(illumination_map[:, :, :, :-1] - illumination_map[:, :, :, 1:])
        grad_y = torch.abs(illumination_map[:, :, :-1, :] - illumination_map[:, :, 1:, :])
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def _color_consistency_loss(self, enhanced, target):
        """Maintain color consistency"""
        mean_enhanced = torch.mean(enhanced, dim=[2, 3])
        mean_target = torch.mean(target, dim=[2, 3])
        return self.l1_loss(mean_enhanced, mean_target)
    
    def _total_variation_loss(self, image):
        """Total variation loss for smoothness"""
        tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        return tv_h + tv_w

class SCIDataset(Dataset):
    """Dataset for SCI model training"""
    
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # Load image pairs
        self.image_pairs = self._load_image_pairs()
        
    def _load_image_pairs(self):
        """Load low-light and normal light image pairs"""
        pairs = []
        
        low_light_dir = os.path.join(self.data_dir, 'low_light_samples')
        normal_dir = os.path.join(self.data_dir, 'normal_light_samples')
        
        if not os.path.exists(low_light_dir) or not os.path.exists(normal_dir):
            print(f"Warning: Sample directories not found. Run generate_sample_data.py first.")
            return pairs
        
        # Match low-light images with normal images
        low_light_files = sorted([f for f in os.listdir(low_light_dir) if f.endswith('.jpg')])
        
        for low_light_file in low_light_files:
            # Extract image ID
            img_id = low_light_file.split('_')[-1].replace('.jpg', '')
            normal_file = f'normal_{img_id}.jpg'
            
            low_light_path = os.path.join(low_light_dir, low_light_file)
            normal_path = os.path.join(normal_dir, normal_file)
            
            if os.path.exists(normal_path):
                pairs.append((low_light_path, normal_path))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        low_light_path, normal_path = self.image_pairs[idx]
        
        # Load images
        low_light = Image.open(low_light_path).convert('RGB')
        normal = Image.open(normal_path).convert('RGB')
        
        if self.transform:
            low_light = self.transform(low_light)
            normal = self.transform(normal)
        
        return low_light, normal

class SCITrainer:
    """Training manager for SCI model"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = SCINet(
            num_channels=3,
            num_features=args.num_features,
            num_blocks=args.num_blocks
        )
        self.model.to(self.device)
        
        # Loss function
        self.criterion = SCILoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=args.epochs
        )
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def load_data(self):
        """Load training and validation datasets"""
        # Training dataset
        train_dataset = SCIDataset(
            data_dir=self.args.data_dir,
            transform=self.transform,
            is_train=True
        )
        
        # Split into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {
            'reconstruction': 0,
            'perceptual': 0,
            'illumination': 0,
            'color': 0,
            'tv': 0
        }
        
        for batch_idx, (low_light, normal) in enumerate(self.train_loader):
            low_light = low_light.to(self.device)
            normal = normal.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            enhanced, illumination_map = self.model(low_light)
            
            # Compute loss
            loss, components = self.criterion(enhanced, illumination_map, normal, low_light)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in components.items():
                loss_components[key] += value
            
            # Print progress
            if batch_idx % self.args.log_interval == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.6f}')
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, loss_components
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for low_light, normal in self.val_loader:
                low_light = low_light.to(self.device)
                normal = normal.to(self.device)
                
                enhanced, illumination_map = self.model(low_light)
                loss, _ = self.criterion(enhanced, illumination_map, normal, low_light)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        print(f'Validation Loss: {avg_loss:.6f}')
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'args': self.args
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.save_dir, f'sci_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'sci_best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Best SCI model saved to {best_path}')
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SCI Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SCI Training Loss')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.args.save_dir, 'sci_training_curves.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f'SCI training curves saved to {plot_path}')
    
    def train(self):
        """Main training loop"""
        print(f"Starting SCI training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.args.epochs + 1):
            print(f'\nEpoch {epoch}/{self.args.epochs}')
            
            # Train
            train_loss, loss_components = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Print epoch summary
            print(f'Epoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(f'  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # Print loss components
            print('  Loss Components:')
            for key, value in loss_components.items():
                print(f'    {key}: {value:.6f}')
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f'\nSCI training completed!')
        print(f'Best validation loss: {best_val_loss:.6f}')
        print(f'Models saved to: {self.args.save_dir}')

def main():
    parser = argparse.ArgumentParser(description='SCI Model Training')
    
    # Data arguments
    parser.add_argument('--data-dir', default='../data/datasets', 
                       help='Path to training data directory')
    parser.add_argument('--save-dir', default='../models/sci', 
                       help='Directory to save trained models')
    
    # Model arguments
    parser.add_argument('--num-features', type=int, default=64,
                       help='Number of feature channels')
    parser.add_argument('--num-blocks', type=int, default=8,
                       help='Number of residual blocks')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='How often to log training progress')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save training configuration
    config_path = os.path.join(args.save_dir, 'sci_training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize trainer
    trainer = SCITrainer(args)
    
    # Load data
    trainer.load_data()
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
