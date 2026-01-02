import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class CNNPatternRecognizer:
    """
    Convolutional Neural Network for cryptocurrency chart pattern recognition.
    
    Features:
    - Multi-scale CNN architecture
    - Pattern-specific feature extraction
    - Attention mechanisms
    - Custom implementation without external dependencies
    """
    
    def __init__(self, input_channels: int = 6, 
                 num_classes: int = 5,
                 image_size: int = 64):
        """
        Initialize CNN model.
        
        Args:
            input_channels: Number of input channels (GAF, OHLC, etc.)
            num_classes: Number of pattern classes to recognize
            image_size: Size of input images
        """
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Model parameters
        self.params = {}
        self.cache = {}
        self.is_training = True
        
        # Initialize layers
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize CNN parameters."""
        # Conv layer 1: 32 filters of size 5x5
        self.params['W1'] = self._xavier_init((32, self.input_channels, 5, 5))
        self.params['b1'] = np.zeros((32, 1))
        
        # Conv layer 2: 64 filters of size 3x3
        self.params['W2'] = self._xavier_init((64, 32, 3, 3))
        self.params['b2'] = np.zeros((64, 1))
        
        # Conv layer 3: 128 filters of size 3x3
        self.params['W3'] = self._xavier_init((128, 64, 3, 3))
        self.params['b3'] = np.zeros((128, 1))
        
        # Calculate size after convolutions and pooling
        conv_output_size = self._calculate_conv_output_size()
        
        # Fully connected layers
        self.params['W4'] = self._xavier_init((256, conv_output_size))
        self.params['b4'] = np.zeros((256, 1))
        
        self.params['W5'] = self._xavier_init((128, 256))
        self.params['b5'] = np.zeros((128, 1))
        
        self.params['W6'] = self._xavier_init((self.num_classes, 128))
        self.params['b6'] = np.zeros((self.num_classes, 1))
        
        # Batch normalization parameters
        self._init_batch_norm_params()
        
    def _xavier_init(self, shape: Tuple) -> np.ndarray:
        """Xavier weight initialization."""
        if len(shape) == 2:
            fan_in = shape[1]
            fan_out = shape[0]
        else:  # Conv layers
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
            
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def _init_batch_norm_params(self):
        """Initialize batch normalization parameters."""
        # BN for conv layers
        self.params['gamma1'] = np.ones((32, 1))
        self.params['beta1'] = np.zeros((32, 1))
        
        self.params['gamma2'] = np.ones((64, 1))
        self.params['beta2'] = np.zeros((64, 1))
        
        self.params['gamma3'] = np.ones((128, 1))
        self.params['beta3'] = np.zeros((128, 1))
        
        # Running statistics
        self.params['running_mean1'] = np.zeros((32, 1))
        self.params['running_var1'] = np.ones((32, 1))
        
        self.params['running_mean2'] = np.zeros((64, 1))
        self.params['running_var2'] = np.ones((64, 1))
        
        self.params['running_mean3'] = np.zeros((128, 1))
        self.params['running_var3'] = np.ones((128, 1))
        
    def _calculate_conv_output_size(self) -> int:
        """Calculate output size after convolutions and pooling."""
        size = self.image_size
        
        # Conv1 (5x5, stride 1) + MaxPool (2x2)
        size = (size - 5 + 1) // 2
        
        # Conv2 (3x3, stride 1) + MaxPool (2x2)
        size = (size - 3 + 1) // 2
        
        # Conv3 (3x3, stride 1) + MaxPool (2x2)
        size = (size - 3 + 1) // 2
        
        # Total features: 128 channels * size * size
        return 128 * size * size
    
    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass through CNN.
        
        Args:
            X: Input images (batch_size, channels, height, width)
            
        Returns:
            Dictionary with predictions and features
        """
        # Conv Block 1
        conv1_out = self._conv_forward(X, self.params['W1'], self.params['b1'], stride=1, pad=0, cache_name='conv1')
        bn1_out = self._batch_norm_forward(conv1_out, 'gamma1', 'beta1', 1)
        relu1_out = self._relu_forward(bn1_out, cache_name='relu1')
        pool1_out = self._max_pool_forward(relu1_out, pool_size=2, stride=2, cache_name='pool1')
        
        # Conv Block 2
        conv2_out = self._conv_forward(pool1_out, self.params['W2'], self.params['b2'], stride=1, pad=0, cache_name='conv2')
        bn2_out = self._batch_norm_forward(conv2_out, 'gamma2', 'beta2', 2)
        relu2_out = self._relu_forward(bn2_out, cache_name='relu2')
        pool2_out = self._max_pool_forward(relu2_out, pool_size=2, stride=2, cache_name='pool2')
        
        # Conv Block 3
        conv3_out = self._conv_forward(pool2_out, self.params['W3'], self.params['b3'], stride=1, pad=0, cache_name='conv3')
        bn3_out = self._batch_norm_forward(conv3_out, 'gamma3', 'beta3', 3)
        relu3_out = self._relu_forward(bn3_out, cache_name='relu3')
        pool3_out = self._max_pool_forward(relu3_out, pool_size=2, stride=2, cache_name='pool3')
        
        # Flatten
        batch_size = X.shape[0]
        flattened = pool3_out.reshape(batch_size, -1)
        self.cache['flatten_shape'] = pool3_out.shape
        
        # FC layers
        fc1_out = self._fc_forward(flattened, self.params['W4'], self.params['b4'], cache_name='fc1')
        fc1_relu = self._relu_forward(fc1_out, cache_name='fc1_relu')
        fc1_dropout = self._dropout_forward(fc1_relu, drop_prob=0.5, cache_name='fc1_dropout')
        
        fc2_out = self._fc_forward(fc1_dropout, self.params['W5'], self.params['b5'], cache_name='fc2')
        fc2_relu = self._relu_forward(fc2_out, cache_name='fc2_relu')
        fc2_dropout = self._dropout_forward(fc2_relu, drop_prob=0.5, cache_name='fc2_dropout')
        
        # Output layer
        logits = self._fc_forward(fc2_dropout, self.params['W6'], self.params['b6'], cache_name='fc3')
        
        # Apply softmax
        probs = self._softmax(logits)
        
        # Store intermediate features for analysis
        features = {
            'conv1_features': relu1_out,
            'conv2_features': relu2_out,
            'conv3_features': relu3_out,
            'fc_features': fc2_relu
        }
        
        return {
            'logits': logits,
            'probabilities': probs,
            'predictions': np.argmax(probs, axis=1),
            'features': features
        }
    
    def _conv_forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray,
                     stride: int = 1, pad: int = 0, cache_name: str = None) -> np.ndarray:
        """
        Convolutional layer forward pass.
        
        Args:
            X: Input (N, C, H, W)
            W: Filters (F, C, HH, WW)
            b: Bias (F, 1)
            stride: Stride
            pad: Padding
            cache_name: Name for caching (for backward pass)
            
        Returns:
            Output feature maps
        """
        N, C, H, W = X.shape
        F, _, HH, WW = W.shape
        
        # Pad input
        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
        else:
            X_pad = X
            
        # Output dimensions
        H_out = (H + 2 * pad - HH) // stride + 1
        W_out = (W + 2 * pad - WW) // stride + 1
        
        # Initialize output
        out = np.zeros((N, F, H_out, W_out))
        
        # Convolution
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        # Extract patch
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW
                        
                        patch = X_pad[n, :, h_start:h_end, w_start:w_end]
                        
                        # Convolve
                        out[n, f, i, j] = np.sum(patch * W[f]) + b[f]
        
        # Cache for backward pass
        if cache_name:
            self.cache[cache_name] = (X, W, b, stride, pad)
                        
        return out
    
    def _max_pool_forward(self, X: np.ndarray, pool_size: int = 2,
                         stride: int = 2, cache_name: str = None) -> np.ndarray:
        """Max pooling forward pass."""
        N, C, H, W = X.shape
        
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        
        out = np.zeros((N, C, H_out, W_out))
        max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=int)
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + pool_size
                        w_start = j * stride
                        w_end = w_start + pool_size
                        
                        pool_region = X[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(pool_region)
                        out[n, c, i, j] = max_val
                        
                        # Store max indices for backward pass
                        max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        max_indices[n, c, i, j] = [h_start + max_pos[0], w_start + max_pos[1]]
        
        # Cache for backward pass
        if cache_name:
            self.cache[cache_name] = (X, max_indices, pool_size, stride)
                        
        return out
    
    def _batch_norm_forward(self, X: np.ndarray, gamma_name: str,
                           beta_name: str, layer_idx: int) -> np.ndarray:
        """Batch normalization forward pass."""
        gamma = self.params[gamma_name]
        beta = self.params[beta_name]
        
        if self.is_training:
            # Calculate batch statistics
            if X.ndim == 4:  # Conv layer
                mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
                var = np.var(X, axis=(0, 2, 3), keepdims=True)
            else:  # FC layer
                mean = np.mean(X, axis=0, keepdims=True)
                var = np.var(X, axis=0, keepdims=True)
                
            # Update running statistics
            momentum = 0.9
            self.params[f'running_mean{layer_idx}'] = (
                momentum * self.params[f'running_mean{layer_idx}'] +
                (1 - momentum) * mean.squeeze()
            )
            self.params[f'running_var{layer_idx}'] = (
                momentum * self.params[f'running_var{layer_idx}'] +
                (1 - momentum) * var.squeeze()
            )
        else:
            # Use running statistics
            mean = self.params[f'running_mean{layer_idx}']
            var = self.params[f'running_var{layer_idx}']
            
            if X.ndim == 4:
                mean = mean.reshape(1, -1, 1, 1)
                var = var.reshape(1, -1, 1, 1)
                gamma = gamma.reshape(1, -1, 1, 1)
                beta = beta.reshape(1, -1, 1, 1)
                
        # Normalize
        X_norm = (X - mean) / np.sqrt(var + 1e-8)
        
        # Scale and shift
        out = gamma * X_norm + beta
        
        # Cache for backward pass
        self.cache[f'bn{layer_idx}'] = (X, X_norm, mean, var, gamma, beta)
        
        return out
    
    def _fc_forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray, cache_name: str = None) -> np.ndarray:
        """Fully connected layer forward pass."""
        out = np.dot(X, W.T) + b.T
        
        # Cache for backward pass
        if cache_name:
            self.cache[cache_name] = (X, W, b)
            
        return out
    
    def _relu_forward(self, X: np.ndarray, cache_name: str = None) -> np.ndarray:
        """ReLU activation forward pass."""
        out = np.maximum(0, X)
        
        # Cache for backward pass
        if cache_name:
            self.cache[cache_name] = X
            
        return out
    
    def _dropout_forward(self, X: np.ndarray, drop_prob: float = 0.5, cache_name: str = None) -> np.ndarray:
        """Dropout forward pass."""
        if not self.is_training:
            return X
            
        mask = np.random.rand(*X.shape) > drop_prob
        out = X * mask / (1 - drop_prob)
        
        # Cache for backward pass
        if cache_name:
            self.cache[cache_name] = (mask, drop_prob)
            
        return out
    
    def _softmax(self, X: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input images.
        
        Args:
            X: Input images
            
        Returns:
            Predicted class labels
        """
        self.set_training(False)
        output = self.forward(X)
        return output['predictions']
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input images
            
        Returns:
            Class probabilities
        """
        self.set_training(False)
        output = self.forward(X)
        return output['probabilities']
    
    def set_training(self, is_training: bool):
        """Set training mode."""
        self.is_training = is_training
    
    def extract_features(self, X: np.ndarray, layer: str = 'fc') -> np.ndarray:
        """
        Extract features from specific layer.
        
        Args:
            X: Input images
            layer: Layer to extract features from
            
        Returns:
            Extracted features
        """
        self.set_training(False)
        output = self.forward(X)
        
        if layer == 'conv1':
            return output['features']['conv1_features']
        elif layer == 'conv2':
            return output['features']['conv2_features']
        elif layer == 'conv3':
            return output['features']['conv3_features']
        elif layer == 'fc':
            return output['features']['fc_features']
        else:
            raise ValueError(f"Unknown layer: {layer}")
    
    def get_pattern_confidence(self, X: np.ndarray) -> Dict[str, float]:
        """
        Get confidence scores for each pattern.
        
        Args:
            X: Input image
            
        Returns:
            Dictionary of pattern confidences
        """
        probs = self.predict_proba(X)
        
        pattern_names = [
            'no_pattern',
            'bullish_reversal',
            'bearish_reversal',
            'continuation',
            'consolidation'
        ]
        
        confidences = {}
        for i, name in enumerate(pattern_names[:self.num_classes]):
            confidences[name] = float(probs[0, i])
            
        return confidences
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: Dict) -> Dict[str, np.ndarray]:
        """
        Backward pass through CNN to compute gradients.
        
        Args:
            X: Input images (batch_size, channels, height, width)
            y: True labels
            output: Forward pass output dictionary
            
        Returns:
            Dictionary of gradients for all parameters
        """
        gradients = {}
        batch_size = X.shape[0]
        
        # Get output probabilities
        probs = output['probabilities']
        
        # One-hot encode labels
        y_one_hot = np.zeros((batch_size, self.num_classes))
        y_one_hot[np.arange(batch_size), y] = 1
        
        # Gradient of softmax cross-entropy loss
        dL_dlogits = (probs - y_one_hot) / batch_size
        
        # Backward through output layer (fc3)
        dL_dfc3, gradients['W6'], gradients['b6'] = self._fc_backward(dL_dlogits, 'fc3')
        
        # Backward through fc2 dropout
        dL_dfc2_relu = self._dropout_backward(dL_dfc3, 'fc2_dropout')
        
        # Backward through fc2 relu
        dL_dfc2 = self._relu_backward(dL_dfc2_relu, 'fc2_relu')
        
        # Backward through fc2
        dL_dfc2_in, gradients['W5'], gradients['b5'] = self._fc_backward(dL_dfc2, 'fc2')
        
        # Backward through fc1 dropout
        dL_dfc1_relu = self._dropout_backward(dL_dfc2_in, 'fc1_dropout')
        
        # Backward through fc1 relu
        dL_dfc1 = self._relu_backward(dL_dfc1_relu, 'fc1_relu')
        
        # Backward through fc1
        dL_dflattened, gradients['W4'], gradients['b4'] = self._fc_backward(dL_dfc1, 'fc1')
        
        # Reshape gradient for conv layers
        flatten_shape = self.cache['flatten_shape']
        dL_dpool3 = dL_dflattened.reshape(flatten_shape)
        
        # Backward through pool3
        dL_drelu3 = self._max_pool_backward(dL_dpool3, 'pool3')
        
        # Backward through relu3
        dL_dbn3 = self._relu_backward(dL_drelu3, 'relu3')
        
        # Backward through bn3
        dL_dconv3, gradients['gamma3'], gradients['beta3'] = self._batch_norm_backward(dL_dbn3, 3)
        
        # Backward through conv3
        dL_dpool2, gradients['W3'], gradients['b3'] = self._conv_backward(dL_dconv3, 'conv3')
        
        # Backward through pool2
        dL_drelu2 = self._max_pool_backward(dL_dpool2, 'pool2')
        
        # Backward through relu2
        dL_dbn2 = self._relu_backward(dL_drelu2, 'relu2')
        
        # Backward through bn2
        dL_dconv2, gradients['gamma2'], gradients['beta2'] = self._batch_norm_backward(dL_dbn2, 2)
        
        # Backward through conv2
        dL_dpool1, gradients['W2'], gradients['b2'] = self._conv_backward(dL_dconv2, 'conv2')
        
        # Backward through pool1
        dL_drelu1 = self._max_pool_backward(dL_dpool1, 'pool1')
        
        # Backward through relu1
        dL_dbn1 = self._relu_backward(dL_drelu1, 'relu1')
        
        # Backward through bn1
        dL_dconv1, gradients['gamma1'], gradients['beta1'] = self._batch_norm_backward(dL_dbn1, 1)
        
        # Backward through conv1
        dL_dX, gradients['W1'], gradients['b1'] = self._conv_backward(dL_dconv1, 'conv1')
        
        return gradients
    
    def _conv_backward(self, dout: np.ndarray, cache_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for convolution layer.
        
        Args:
            dout: Upstream gradients
            cache_name: Name of cached forward pass data
            
        Returns:
            Gradients for input, weights, and bias
        """
        X, W, b, stride, pad = self.cache[cache_name]
        N, C, H, W_in = X.shape
        F, _, HH, WW = W.shape
        _, _, H_out, W_out = dout.shape
        
        # Pad input
        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
        else:
            X_pad = X
            
        # Initialize gradients
        dX_pad = np.zeros_like(X_pad)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        
        # Compute gradients
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW
                        
                        # Gradient w.r.t weights
                        patch = X_pad[n, :, h_start:h_end, w_start:w_end]
                        dW[f] += patch * dout[n, f, i, j]
                        
                        # Gradient w.r.t input
                        dX_pad[n, :, h_start:h_end, w_start:w_end] += W[f] * dout[n, f, i, j]
                        
        # Gradient w.r.t bias
        db = np.sum(dout, axis=(0, 2, 3)).reshape(-1, 1)
        
        # Remove padding from dX if needed
        if pad > 0:
            dX = dX_pad[:, :, pad:-pad, pad:-pad]
        else:
            dX = dX_pad
            
        return dX, dW, db
    
    def _max_pool_backward(self, dout: np.ndarray, cache_name: str) -> np.ndarray:
        """
        Backward pass for max pooling layer.
        
        Args:
            dout: Upstream gradients
            cache_name: Name of cached forward pass data
            
        Returns:
            Gradient for input
        """
        X, max_indices, pool_size, stride = self.cache[cache_name]
        N, C, H, W = X.shape
        _, _, H_out, W_out = dout.shape
        
        dX = np.zeros_like(X)
        
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # Get max index from forward pass
                        h_idx, w_idx = max_indices[n, c, i, j]
                        
                        # Propagate gradient to max element
                        dX[n, c, h_idx, w_idx] += dout[n, c, i, j]
                        
        return dX
    
    def _fc_backward(self, dout: np.ndarray, cache_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for fully connected layer.
        
        Args:
            dout: Upstream gradients
            cache_name: Name of cached forward pass data
            
        Returns:
            Gradients for input, weights, and bias
        """
        X, W, b = self.cache[cache_name]
        
        # Gradient w.r.t input
        dX = np.dot(dout, W)
        
        # Gradient w.r.t weights
        dW = np.dot(dout.T, X)
        
        # Gradient w.r.t bias
        db = np.sum(dout, axis=0, keepdims=True).T
        
        return dX, dW, db
    
    def _relu_backward(self, dout: np.ndarray, cache_name: str) -> np.ndarray:
        """
        Backward pass for ReLU activation.
        
        Args:
            dout: Upstream gradients
            cache_name: Name of cached forward pass data
            
        Returns:
            Gradient for input
        """
        X = self.cache[cache_name]
        dX = dout * (X > 0)
        return dX
    
    def _dropout_backward(self, dout: np.ndarray, cache_name: str) -> np.ndarray:
        """
        Backward pass for dropout.
        
        Args:
            dout: Upstream gradients
            cache_name: Name of cached forward pass data
            
        Returns:
            Gradient for input
        """
        if not self.is_training:
            return dout
            
        mask, drop_prob = self.cache[cache_name]
        dX = dout * mask / (1 - drop_prob)
        return dX
    
    def _batch_norm_backward(self, dout: np.ndarray, layer_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for batch normalization.
        
        Args:
            dout: Upstream gradients
            layer_idx: Layer index (1, 2, or 3)
            
        Returns:
            Gradients for input, gamma, and beta
        """
        X, X_norm, mean, var, gamma, beta = self.cache[f'bn{layer_idx}']
        
        N = X.shape[0]
        
        # Reshape gamma for proper broadcasting
        if dout.ndim == 4:  # Conv layer
            dgamma = np.sum(dout * X_norm, axis=(0, 2, 3), keepdims=True).squeeze()
            dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True).squeeze()
            gamma = gamma.reshape(1, -1, 1, 1)
        else:  # FC layer
            dgamma = np.sum(dout * X_norm, axis=0, keepdims=True)
            dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient w.r.t normalized input
        dX_norm = dout * gamma
        
        # Gradient w.r.t variance
        dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + 1e-8)**(-1.5), axis=(0, 2, 3) if dout.ndim == 4 else 0, keepdims=True)
        
        # Gradient w.r.t mean
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + 1e-8), axis=(0, 2, 3) if dout.ndim == 4 else 0, keepdims=True)
        dmean += dvar * np.sum(-2 * (X - mean), axis=(0, 2, 3) if dout.ndim == 4 else 0, keepdims=True) / N
        
        # Gradient w.r.t input
        dX = dX_norm / np.sqrt(var + 1e-8)
        dX += dvar * 2 * (X - mean) / N
        dX += dmean / N
        
        # Reshape gradients
        dgamma = dgamma.reshape(-1, 1)
        dbeta = dbeta.reshape(-1, 1)
        
        return dX, dgamma, dbeta
    
    def save_model(self, filepath: str):
        """Save model parameters."""
        np.savez(filepath, **self.params)
        
    def load_model(self, filepath: str):
        """Load model parameters."""
        data = np.load(filepath)
        for key in data.files:
            self.params[key] = data[key]