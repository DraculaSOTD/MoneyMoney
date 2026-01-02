import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Input, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, Optional
from .base_model import BaseMLModel


class LSTMModel(BaseMLModel):
    """
    LSTM model for time series classification.
    Based on the ML project's LSTM architecture.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize LSTM model with configuration."""
        default_config = {
            'lstm_units': [50, 50],
            'dense_units': [50],
            'dropout_rate': 0.2,
            'activation': 'tanh',
            'learning_rate': 0.001,
            'batch_size': 512,
            'epochs': 100
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        
    def build_model(self, input_shape: Tuple[int, ...], n_classes: int = 3):
        """Build LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.config['lstm_units'][0],
            activation=self.config['activation'],
            return_sequences=len(self.config['lstm_units']) > 1,
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout_rate']))
        
        # Additional LSTM layers
        for i, units in enumerate(self.config['lstm_units'][1:]):
            model.add(LSTM(
                units,
                activation=self.config['activation'],
                return_sequences=i < len(self.config['lstm_units']) - 2
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
        Prepare data for LSTM input.
        Expects X shape: (samples, features) or (samples, timesteps, features)
        """
        # If 2D, reshape to 3D with single timestep
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        return X, y
    
    def get_callbacks(self):
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        return callbacks


class BidirectionalLSTMModel(LSTMModel):
    """
    Bidirectional LSTM model for enhanced sequence learning.
    Based on the ML project's bidirectional architecture.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize Bidirectional LSTM model."""
        super().__init__(config)
        self.model_name = "BidirectionalLSTMModel"
        
    def build_model(self, input_shape: Tuple[int, ...], n_classes: int = 3):
        """Build Bidirectional LSTM model architecture."""
        model = Sequential()
        
        # First Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(
                self.config['lstm_units'][0],
                activation=self.config['activation'],
                return_sequences=len(self.config['lstm_units']) > 1
            ),
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout_rate']))
        
        # Additional Bidirectional LSTM layers
        for i, units in enumerate(self.config['lstm_units'][1:]):
            model.add(Bidirectional(
                LSTM(
                    units,
                    activation=self.config['activation'],
                    return_sequences=i < len(self.config['lstm_units']) - 2
                )
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate']))
        
        # Dense layers
        for units in self.config['dense_units']:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
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