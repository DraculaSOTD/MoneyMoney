"""
GPU-Accelerated Transformer Model using PyTorch.

This module provides a fully GPU-optimized implementation of Transformer
architecture for time series prediction in cryptocurrency trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformerGPU(nn.Module):
    """
    GPU-Accelerated Transformer for time series prediction.
    
    Features:
    - Multi-head self-attention
    - Positional encoding
    - Feed-forward networks
    - Layer normalization
    - Residual connections
    - Mixed precision training support
    - Causal masking for autoregressive prediction
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 output_dim: int = 1,
                 max_seq_length: int = 1024,
                 use_temporal_fusion: bool = True):
        """
        Initialize Transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Dimension of model
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            d_ff: Dimension of feedforward network
            dropout: Dropout rate
            output_dim: Number of output features
            max_seq_length: Maximum sequence length
            use_temporal_fusion: Whether to use temporal fusion mechanism
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.use_temporal_fusion = use_temporal_fusion
        
        # Initialize GPU manager
        self.gpu_manager = get_gpu_manager()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        # Decoder (for sequence-to-sequence tasks)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Temporal fusion (optional)
        if use_temporal_fusion:
            self.temporal_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.temporal_transform = nn.Linear(d_model * 2, d_model)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move to GPU if available
        if self.gpu_manager.device.type == 'cuda':
            self.to(self.gpu_manager.device)
            logger.info(f"Transformer model moved to {self.gpu_manager.device}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create padding mask for variable length sequences."""
        if lengths is None:
            return None
        
        batch_size, max_len = x.size(0), x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        
        return mask
    
    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal mask for autoregressive prediction."""
        mask = torch.triu(torch.ones(size, size, device=self.gpu_manager.device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, src: torch.Tensor, 
                tgt: Optional[torch.Tensor] = None,
                src_lengths: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            src: Source sequence (batch_size, seq_len, input_dim)
            tgt: Target sequence for decoder (optional)
            src_lengths: Actual lengths of sequences for masking
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        # Move to device if needed
        if src.device != self.gpu_manager.device:
            src = src.to(self.gpu_manager.device)
        
        batch_size, seq_len, _ = src.shape
        
        # Input embedding
        src_embedded = self.input_embedding(src)
        
        # Positional encoding (transpose for positional encoding module)
        src_embedded = src_embedded.transpose(0, 1)
        src_embedded = self.positional_encoding(src_embedded)
        src_embedded = src_embedded.transpose(0, 1)
        
        # Create masks
        src_padding_mask = self.create_padding_mask(src, src_lengths)
        
        # Encoder
        memory = self.encoder(
            src_embedded,
            mask=None,  # No causal mask for encoder
            src_key_padding_mask=src_padding_mask
        )
        
        # For sequence prediction, use the encoder output directly
        if tgt is None:
            # Use last valid position for each sequence
            if src_lengths is not None:
                # Gather last valid positions
                indices = (src_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
                last_outputs = memory.gather(1, indices).squeeze(1)
            else:
                # Use mean pooling
                last_outputs = memory.mean(dim=1)
            
            output = self.output_projection(last_outputs)
            
            if return_attention:
                # Return dummy attention for compatibility
                return output, None
            return output
        
        # Decoder (for sequence-to-sequence tasks)
        tgt_embedded = self.input_embedding(tgt)
        tgt_embedded = tgt_embedded.transpose(0, 1)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        tgt_embedded = tgt_embedded.transpose(0, 1)
        
        tgt_mask = self.create_causal_mask(tgt.size(1))
        
        decoder_output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Temporal fusion if enabled
        if self.use_temporal_fusion:
            # Combine encoder and decoder outputs
            combined = torch.cat([memory.mean(dim=1).unsqueeze(1).expand_as(decoder_output), 
                                decoder_output], dim=-1)
            gate = self.temporal_gate(combined)
            transformed = self.temporal_transform(combined)
            decoder_output = gate * transformed + (1 - gate) * decoder_output
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        if return_attention:
            # Return last layer attention weights
            return output, None  # Attention extraction would require model modification
        
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
            learning_rate: float = 0.0001,
            warmup_steps: int = 4000,
            early_stopping_patience: int = 10,
            **kwargs) -> Dict[str, List[float]]:
        """
        Train the transformer model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Peak learning rate
            warmup_steps: Number of warmup steps for learning rate
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
        
        # Optimizer with learning rate scheduling
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.01
        )
        
        # Custom learning rate scheduler (Transformer schedule)
        def lr_lambda(step):
            if step == 0:
                return 0
            return min(step ** (-0.5), step * warmup_steps ** (-1.5))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
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
        
        # Global step for learning rate scheduling
        global_step = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.gpu_manager.device)
                batch_y = batch_y.to(self.gpu_manager.device)
                
                global_step += 1
                
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
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['loss'].append(avg_train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
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
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        return history
    
    def generate_sequence(self, initial_sequence: torch.Tensor, 
                         n_steps: int,
                         temperature: float = 1.0) -> torch.Tensor:
        """
        Generate future predictions autoregressively.
        
        Args:
            initial_sequence: Starting sequence
            n_steps: Number of steps to predict
            temperature: Sampling temperature
            
        Returns:
            Generated sequence
        """
        self.eval()
        
        if initial_sequence.device != self.gpu_manager.device:
            initial_sequence = initial_sequence.to(self.gpu_manager.device)
        
        generated = initial_sequence.clone()
        
        with torch.no_grad():
            for _ in range(n_steps):
                # Get prediction for next step
                output = self.forward(generated)
                
                # Apply temperature
                if temperature != 1.0:
                    output = output / temperature
                
                # Append prediction
                next_pred = output[:, -1:, :]
                generated = torch.cat([generated, next_pred], dim=1)
        
        return generated
    
    def save(self, filepath: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'd_model': self.d_model,
                'output_dim': self.output_dim,
                'use_temporal_fusion': self.use_temporal_fusion
            }
        }, filepath)
        logger.info(f"Transformer model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.gpu_manager.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Transformer model loaded from {filepath}")
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention weights for visualization."""
        self.eval()
        
        if x.device != self.gpu_manager.device:
            x = x.to(self.gpu_manager.device)
        
        attention_weights = {}
        
        # This would require modifying the forward pass to return attention weights
        # For now, return empty dict
        logger.warning("Attention weight extraction not yet implemented")
        
        return attention_weights