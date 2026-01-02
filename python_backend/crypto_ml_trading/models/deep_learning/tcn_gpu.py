"""
GPU-Accelerated Temporal Convolutional Network (TCN) using PyTorch.

This module provides a fully GPU-optimized implementation of TCN
for time series prediction in cryptocurrency trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class CausalConv1d(nn.Module):
    """Causal 1D convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove future values to ensure causality
        return self.conv(x)[:, :, :-self.padding] if self.padding > 0 else self.conv(x)


class ResidualBlock(nn.Module):
    """TCN residual block with dilated causal convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        
        # First convolution
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolution
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual path
        res = x if self.residual is None else self.residual(x)
        
        return self.relu(out + res)


class TCNBlock(nn.Module):
    """Stack of residual blocks with exponentially increasing dilation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 num_layers: int, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels
            layers.append(
                ResidualBlock(in_ch, out_channels, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalConvNetGPU(nn.Module):
    """
    GPU-Accelerated Temporal Convolutional Network for time series prediction.
    
    Features:
    - Dilated causal convolutions
    - Residual connections
    - Batch normalization
    - Dropout regularization
    - Mixed precision training support
    - Multi-GPU support
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_channels: List[int] = [64, 128, 256],
                 output_dim: int = 1,
                 kernel_size: int = 3,
                 num_layers_per_block: int = 2,
                 dropout: float = 0.2,
                 use_attention: bool = True):
        """
        Initialize TCN model.
        
        Args:
            input_dim: Number of input features
            hidden_channels: List of hidden channel sizes for each TCN block
            output_dim: Number of output features
            kernel_size: Kernel size for convolutions
            num_layers_per_block: Number of residual layers per TCN block
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention
        
        # Initialize GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Input projection
        self.input_projection = nn.Conv1d(input_dim, hidden_channels[0], 1)
        
        # TCN blocks
        tcn_blocks = []
        for i in range(len(hidden_channels)):
            in_ch = hidden_channels[i-1] if i > 0 else hidden_channels[0]
            out_ch = hidden_channels[i]
            tcn_blocks.append(
                TCNBlock(in_ch, out_ch, kernel_size, num_layers_per_block, dropout)
            )
        
        self.tcn_blocks = nn.ModuleList(tcn_blocks)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_channels[-1], num_heads=8, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_channels[-1])
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, output_dim)
        )
        
        # Move to GPU if available
        if self.gpu_manager.device.type == 'cuda':
            self.to(self.gpu_manager.device)
            logger.info(f"TCN model moved to {self.gpu_manager.device}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Move to device if needed
        if x.device != self.gpu_manager.device:
            x = x.to(self.gpu_manager.device)
        
        # Transpose for Conv1d (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # Apply attention if enabled
        if self.use_attention:
            # Transpose back for attention (batch, sequence, features)
            x_att = x.transpose(1, 2)
            x_att, _ = self.attention(x_att, x_att, x_att)
            x_att = self.attention_norm(x_att + x.transpose(1, 2))
            x = x_att.transpose(1, 2)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on numpy array.
        
        Args:
            x: Input array of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Predictions as numpy array
        """
        self.eval()
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float()
        
        with torch.no_grad():
            with self.gpu_manager.autocast():
                output = self.forward(x_tensor)
        
        return output.cpu().numpy()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 10,
            **kwargs) -> Dict[str, List[float]]:
        """
        Train the TCN model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        if validation_data:
            X_val_tensor = torch.from_numpy(validation_data[0]).float()
            y_val_tensor = torch.from_numpy(validation_data[1]).float()
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.gpu_manager.device)
                batch_y = batch_y.to(self.gpu_manager.device)
                
                # Mixed precision training
                with self.gpu_manager.autocast():
                    optimizer.zero_grad()
                    output = self.forward(batch_X)
                    
                    if output.shape != batch_y.shape:
                        if batch_y.ndim == 1:
                            batch_y = batch_y.unsqueeze(1)
                    
                    loss = criterion(output, batch_y)
                
                # Backward pass with gradient scaling
                self.gpu_manager.scale_loss(loss).backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.gpu_manager.optimizer_step(optimizer)
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['loss'].append(avg_train_loss)
            
            # Validation phase
            if validation_data:
                self.eval()
                with torch.no_grad():
                    with self.gpu_manager.autocast():
                        X_val_device = X_val_tensor.to(self.gpu_manager.device)
                        y_val_device = y_val_tensor.to(self.gpu_manager.device)
                        
                        val_output = self.forward(X_val_device)
                        
                        if val_output.shape != y_val_device.shape:
                            if y_val_device.ndim == 1:
                                y_val_device = y_val_device.unsqueeze(1)
                        
                        val_loss = criterion(val_output, y_val_device).item()
                
                history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    if best_state:
                        self.load_state_dict(best_state)
                    break
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")
            
            history['lr'].append(optimizer.param_groups[0]['lr'])
        
        return history
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the network."""
        receptive_field = 1
        
        for block_idx, hidden_ch in enumerate(self.hidden_channels):
            num_layers = len(self.tcn_blocks[block_idx].network)
            for layer_idx in range(num_layers):
                dilation = 2 ** layer_idx
                receptive_field += (self.tcn_blocks[block_idx].network[layer_idx].conv1.kernel_size - 1) * dilation
        
        return receptive_field
    
    def save(self, filepath: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_channels': self.hidden_channels,
                'output_dim': self.output_dim,
                'use_attention': self.use_attention
            }
        }, filepath)
        logger.info(f"TCN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.gpu_manager.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"TCN model loaded from {filepath}")
    
    def get_layer_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate layer outputs for analysis."""
        outputs = {}
        
        # Move to device
        if x.device != self.gpu_manager.device:
            x = x.to(self.gpu_manager.device)
        
        # Transpose for Conv1d
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        outputs['input_projection'] = x.clone()
        
        # TCN blocks
        for i, tcn_block in enumerate(self.tcn_blocks):
            x = tcn_block(x)
            outputs[f'tcn_block_{i}'] = x.clone()
        
        # Attention if enabled
        if self.use_attention:
            x_att = x.transpose(1, 2)
            x_att, attention_weights = self.attention(x_att, x_att, x_att)
            outputs['attention_weights'] = attention_weights
            outputs['attention_output'] = x_att.clone()
        
        return outputs