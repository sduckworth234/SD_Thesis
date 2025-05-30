#!/usr/bin/env python3
"""
Training Script for ZERO-DCE++ Model
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

class ZeroDCEPlusNet(nn.Module):
    """Zero-DCE++ Network Implementation"""
    
    def __init__(self, num_iterations=8):
        super(ZeroDCEPlusNet, self).__init__()
        self.num_iterations = num_iterations
        
        # Deep Curve Estimation Network
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        
        # Curve parameter estimation
        self.conv5 = nn.Conv2d(32, 3 * num_iterations, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        # Curve parameters
        curves = self.tanh(self.conv5(x4))
        
        # Apply curve iteratively
        enhanced = x
        for i in range(self.num_iterations):
            curve = curves[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)
        
        return enhanced, curves

class LowLightDataset(Dataset):
    """Dataset for low-light image enhancement training"""
    
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

class ZeroDCELoss(nn.Module):
    """Zero-DCE Loss Functions"""
    
    def __init__(self):
        super(ZeroDCELoss, self).__init__()
        
    def forward(self, enhanced, curves, original):
        # Reconstruction loss
        reconstruction_loss = torch.mean(torch.pow(enhanced - original, 2))
        
        # Illumination smoothness loss
        illumination_loss = self._illumination_smoothness_loss(enhanced)
        
        # Spatial consistency loss
        spatial_loss = self._spatial_consistency_loss(enhanced, original)
        
        # Color constancy loss
        color_loss = self._color_constancy_loss(enhanced)
        
        # Exposure loss
        exposure_loss = self._exposure_loss(enhanced)
        
        # Total loss
        total_loss = reconstruction_loss + 0.5 * illumination_loss + 0.1 * spatial_loss + \
                    0.5 * color_loss + 0.1 * exposure_loss
        
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'illumination': illumination_loss.item(),
            'spatial': spatial_loss.item(),
            'color': color_loss.item(),
            'exposure': exposure_loss.item()
        }
    
    def _illumination_smoothness_loss(self, enhanced):
        """Encourage smooth illumination changes"""
        gray = 0.299 * enhanced[:, 0, :, :] + 0.587 * enhanced[:, 1, :, :] + 0.114 * enhanced[:, 2, :, :]
        grad_x = torch.abs(gray[:, :, :-1] - gray[:, :, 1:])
        grad_y = torch.abs(gray[:, :-1, :] - gray[:, 1:, :])
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def _spatial_consistency_loss(self, enhanced, original):
        """Maintain spatial consistency"""
        # Convert to grayscale
        gray_enhanced = 0.299 * enhanced[:, 0, :, :] + 0.587 * enhanced[:, 1, :, :] + 0.114 * enhanced[:, 2, :, :]
        gray_original = 0.299 * original[:, 0, :, :] + 0.587 * original[:, 1, :, :] + 0.114 * original[:, 2, :, :]
        
        # Compute gradients
        grad_enhanced_x = gray_enhanced[:, :, :-1] - gray_enhanced[:, :, 1:]
        grad_enhanced_y = gray_enhanced[:, :-1, :] - gray_enhanced[:, 1:, :]
        
        grad_original_x = gray_original[:, :, :-1] - gray_original[:, :, 1:]
        grad_original_y = gray_original[:, :-1, :] - gray_original[:, 1:, :]
        
        loss_x = torch.mean(torch.pow(grad_enhanced_x - grad_original_x, 2))
        loss_y = torch.mean(torch.pow(grad_enhanced_y - grad_original_y, 2))
        
        return loss_x + loss_y
    
    def _color_constancy_loss(self, enhanced):
        """Maintain color constancy"""
        mean_rgb = torch.mean(enhanced, dim=[2, 3], keepdim=True)
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        return torch.sqrt(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2))
    
    def _exposure_loss(self, enhanced):
        """Exposure loss to avoid over/under exposure"""
        gray = 0.299 * enhanced[:, 0, :, :] + 0.587 * enhanced[:, 1, :, :] + 0.114 * enhanced[:, 2, :, :]
        mean_val = 0.6  # Target mean exposure
        return torch.mean(torch.pow(gray - mean_val, 2))

class ZeroDCETrainer:
    """Training manager for Zero-DCE++ model"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = ZeroDCEPlusNet(num_iterations=args.num_iterations)
        self.model.to(self.device)
        
        # Loss function
        self.criterion = ZeroDCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        
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
        train_dataset = LowLightDataset(
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
            'illumination': 0,
            'spatial': 0,
            'color': 0,
            'exposure': 0
        }
        
        for batch_idx, (low_light, normal) in enumerate(self.train_loader):
            low_light = low_light.to(self.device)
            normal = normal.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            enhanced, curves = self.model(low_light)
            
            # Compute loss
            loss, components = self.criterion(enhanced, curves, normal)
            
            # Backward pass
            loss.backward()
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
                
                enhanced, curves = self.model(low_light)
                loss, _ = self.criterion(enhanced, curves, normal)
                
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
        checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'Best model saved to {best_path}')
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.args.save_dir, 'training_curves.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f'Training curves saved to {plot_path}')
    
    def train(self):
        """Main training loop"""
        print(f"Starting Zero-DCE++ training on {self.device}")
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
        
        print(f'\nTraining completed!')
        print(f'Best validation loss: {best_val_loss:.6f}')
        print(f'Models saved to: {self.args.save_dir}')

def main():
    parser = argparse.ArgumentParser(description='Zero-DCE++ Training')
    
    # Data arguments
    parser.add_argument('--data-dir', default='../data/datasets', 
                       help='Path to training data directory')
    parser.add_argument('--save-dir', default='../models/zero_dce', 
                       help='Directory to save trained models')
    
    # Model arguments
    parser.add_argument('--num-iterations', type=int, default=8,
                       help='Number of curve estimation iterations')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--step-size', type=int, default=30,
                       help='Learning rate scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Learning rate scheduler gamma')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='How often to log training progress')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save training configuration
    config_path = os.path.join(args.save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize trainer
    trainer = ZeroDCETrainer(args)
    
    # Load data
    trainer.load_data()
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
