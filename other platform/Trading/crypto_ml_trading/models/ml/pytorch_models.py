"""
PyTorch implementations of ML models for unified training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM model for time series classification."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, num_classes: int = 3, 
                 dropout: float = 0.2):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def attention_net(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to LSTM outputs."""
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_output = self.attention_net(lstm_out)
        
        # Final classification
        out = self.dropout(attn_output)
        out = self.fc(out)
        
        return out


class GRUModel(nn.Module):
    """GRU model for time series classification."""
    
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 3,
                 dropout: float = 0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state
        out = hidden[-1]  # Get last layer's hidden state
        
        # Final classification
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for pattern recognition in time series."""
    
    def __init__(self, input_channels: int = 1, sequence_length: int = 100,
                 num_features: int = 50, cnn_filters: list = [64, 128, 256],
                 kernel_sizes: list = [3, 5, 7], lstm_hidden: int = 128,
                 lstm_layers: int = 2, num_classes: int = 3,
                 dropout: float = 0.2):
        super(CNNLSTMModel, self).__init__()
        
        self.input_channels = input_channels
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, kernel_sizes)):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(2))
            self.conv_layers.append(nn.Dropout(dropout))
            in_channels = filters
        
        # Calculate CNN output size
        cnn_output_length = sequence_length
        for _ in range(len(cnn_filters)):
            cnn_output_length = cnn_output_length // 2
        
        # LSTM layer
        lstm_input_size = cnn_filters[-1]
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, sequence, features)
        # For 1D CNN, we need (batch, features, sequence)
        
        # Reshape for CNN if needed
        if x.dim() == 3:  # (batch, sequence, features)
            x = x.transpose(1, 2)  # (batch, features, sequence)
        elif x.dim() == 4:  # (batch, channels, sequence, features)
            # Flatten channels and features
            batch_size = x.size(0)
            x = x.view(batch_size, -1, x.size(2))  # (batch, channels*features, sequence)
        
        # CNN forward pass
        for layer in self.conv_layers:
            x = layer(x)
        
        # Prepare for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        out = hidden[-1]  # Get last layer's hidden state
        
        # Final classification
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class TCNModel(nn.Module):
    """Temporal Convolutional Network for time series classification."""
    
    def __init__(self, input_size: int, num_channels: list = [64, 128, 256],
                 kernel_size: int = 3, num_classes: int = 3,
                 dropout: float = 0.2):
        super(TCNModel, self).__init__()
        
        # Build TCN blocks
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        
        self.tcn = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.fc = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence, features)
        x = x.transpose(1, 2)  # (batch, features, sequence)
        
        # TCN forward pass
        x = self.tcn(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = self.dropout(x)
        out = self.fc(x)
        
        return out


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with dilated causal convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super(TCNBlock, self).__init__()
        
        # Dilated causal convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        
        # Normalization and activation
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution
        out = self.conv1(x)
        out = out[:, :, :-(self.conv1.padding[0])]  # Remove future values
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :-(self.conv2.padding[0])]  # Remove future values
        out = self.bn2(out)
        
        # Residual connection
        if self.residual is not None:
            x = self.residual(x)
        
        # Skip connection
        out = self.relu(out + x[:, :, -out.size(2):])
        
        return out


class TransformerModel(nn.Module):
    """Transformer model for time series classification."""
    
    def __init__(self, input_size: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3,
                 num_classes: int = 3, dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Use mean pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        out = self.fc(x)
        
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)