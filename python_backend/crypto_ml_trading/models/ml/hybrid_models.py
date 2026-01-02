import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, Optional
from .base_model import BaseMLModel


class HybridCNNLSTM(BaseMLModel):
    """
    Hybrid CNN-LSTM model combining pattern recognition with sequence learning.
    Based on the ML project's hybrid architecture.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize Hybrid CNN-LSTM model with configuration."""
        default_config = {
            'cnn_filters': [64, 128],
            'cnn_kernel_size': 3,
            'pool_size': 2,
            'lstm_units': [50, 50],
            'dense_units': [50],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 512,
            'epochs': 100
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        
    def build_model(self, input_shape: Tuple[int, ...], n_classes: int = 3):
        """Build Hybrid CNN-LSTM model architecture."""
        model = Sequential()
        
        # CNN layers for pattern extraction
        for i, filters in enumerate(self.config['cnn_filters']):
            if i == 0:
                # First conv layer with input shape
                model.add(Conv1D(
                    filters=filters,
                    kernel_size=self.config['cnn_kernel_size'],
                    activation=self.config['activation'],
                    padding='same',
                    input_shape=input_shape
                ))
            else:
                model.add(Conv1D(
                    filters=filters,
                    kernel_size=self.config['cnn_kernel_size'],
                    activation=self.config['activation'],
                    padding='same'
                ))
            
            model.add(BatchNormalization())
            
            # Only add pooling if it won't reduce dimension too much
            if input_shape[0] > self.config['pool_size'] * 2:
                model.add(MaxPooling1D(pool_size=self.config['pool_size']))
            
            model.add(Dropout(self.config['dropout_rate']))
        
        # LSTM layers for sequence learning
        for i, units in enumerate(self.config['lstm_units']):
            model.add(LSTM(
                units,
                activation=self.config['activation'],
                return_sequences=i < len(self.config['lstm_units']) - 1
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate']))
        
        # Dense layers
        for units in self.config['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for CNN-LSTM input.
        Expects X shape: (samples, features) or (samples, timesteps, features)
        """
        # If 2D, reshape to 3D with single timestep
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # For very short sequences, adjust kernel size
        if X.shape[1] < self.config['cnn_kernel_size']:
            self.config['cnn_kernel_size'] = max(1, X.shape[1] // 2)
        
        return X, y


class HybridCNNGRU(BaseMLModel):
    """
    Hybrid CNN-GRU model variant.
    Combines CNN feature extraction with GRU sequence modeling.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize Hybrid CNN-GRU model."""
        default_config = {
            'cnn_filters': [64, 128],
            'cnn_kernel_size': 3,
            'pool_size': 2,
            'gru_units': [50, 50],
            'dense_units': [50],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 512,
            'epochs': 100
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        self.model_name = "HybridCNNGRU"
        
    def build_model(self, input_shape: Tuple[int, ...], n_classes: int = 3):
        """Build Hybrid CNN-GRU model architecture."""
        model = Sequential()
        
        # CNN layers
        for i, filters in enumerate(self.config['cnn_filters']):
            if i == 0:
                model.add(Conv1D(
                    filters=filters,
                    kernel_size=self.config['cnn_kernel_size'],
                    activation=self.config['activation'],
                    padding='same',
                    input_shape=input_shape
                ))
            else:
                model.add(Conv1D(
                    filters=filters,
                    kernel_size=self.config['cnn_kernel_size'],
                    activation=self.config['activation'],
                    padding='same'
                ))
            
            model.add(BatchNormalization())
            
            if input_shape[0] > self.config['pool_size'] * 2:
                model.add(MaxPooling1D(pool_size=self.config['pool_size']))
            
            model.add(Dropout(self.config['dropout_rate']))
        
        # GRU layers
        for i, units in enumerate(self.config['gru_units']):
            model.add(GRU(
                units,
                activation=self.config['activation'],
                return_sequences=i < len(self.config['gru_units']) - 1
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate']))
        
        # Dense layers
        for units in self.config['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for CNN-GRU input."""
        # If 2D, reshape to 3D with single timestep
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Adjust kernel size for short sequences
        if X.shape[1] < self.config['cnn_kernel_size']:
            self.config['cnn_kernel_size'] = max(1, X.shape[1] // 2)
        
        return X, y