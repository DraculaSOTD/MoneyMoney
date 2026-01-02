"""
Model Conversion Utilities for Hybrid Execution.

Provides converters between NumPy, PyTorch, and TensorFlow models
for flexible deployment and optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import logging
import pickle
import json
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. TF conversions will be disabled.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelFormat:
    """Supported model formats."""
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    CUSTOM = "custom"


class BaseModel(ABC):
    """Abstract base class for unified model interface."""
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights as numpy arrays."""
        pass
    
    @abstractmethod
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights from numpy arrays."""
        pass
    
    @abstractmethod
    def get_architecture(self) -> Dict[str, Any]:
        """Get model architecture description."""
        pass


class NumpyModelWrapper(BaseModel):
    """Wrapper for NumPy-based models."""
    
    def __init__(self, model: Any):
        self.model = model
        self.gpu_manager = get_gpu_manager()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using NumPy model."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif callable(self.model):
            return self.model(X)
        else:
            raise ValueError("Model must have predict method or be callable")
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Extract weights from NumPy model."""
        weights = {}
        
        # Common attribute names for weights
        weight_attrs = ['coef_', 'intercept_', 'weights', 'biases', 'W', 'b',
                       'weight', 'bias', 'kernel', 'params']
        
        for attr in weight_attrs:
            if hasattr(self.model, attr):
                value = getattr(self.model, attr)
                if isinstance(value, (np.ndarray, list)):
                    weights[attr] = np.asarray(value)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (np.ndarray, list)):
                            weights[f"{attr}_{k}"] = np.asarray(v)
        
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights for NumPy model."""
        for key, value in weights.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif '_' in key:
                # Handle nested attributes
                parts = key.split('_', 1)
                if hasattr(self.model, parts[0]):
                    attr = getattr(self.model, parts[0])
                    if isinstance(attr, dict) and parts[1] in attr:
                        attr[parts[1]] = value
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get model architecture."""
        return {
            'type': 'numpy',
            'class': self.model.__class__.__name__,
            'attributes': list(vars(self.model).keys())
        }


class PyTorchModelWrapper(BaseModel):
    """Wrapper for PyTorch models."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gpu_manager = get_gpu_manager()
        self.model.to(self.gpu_manager.device)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using PyTorch model."""
        self.model.eval()
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X).float()
        X_tensor = self.gpu_manager.to_device(X_tensor)
        
        # Add batch dimension if needed
        if X_tensor.dim() == 2 and hasattr(self.model, 'expected_dim') and self.model.expected_dim == 3:
            X_tensor = X_tensor.unsqueeze(0)
        
        with torch.no_grad():
            with self.gpu_manager.autocast():
                output = self.model(X_tensor)
        
        return self.gpu_manager.to_numpy(output)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Extract weights from PyTorch model."""
        weights = {}
        
        for name, param in self.model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        
        # Also get buffers (batch norm stats, etc.)
        for name, buffer in self.model.named_buffers():
            weights[f"buffer_{name}"] = buffer.detach().cpu().numpy()
        
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights for PyTorch model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.data = torch.from_numpy(weights[name]).to(param.device)
            
            for name, buffer in self.model.named_buffers():
                buffer_key = f"buffer_{name}"
                if buffer_key in weights:
                    buffer.data = torch.from_numpy(weights[buffer_key]).to(buffer.device)
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get model architecture."""
        return {
            'type': 'pytorch',
            'class': self.model.__class__.__name__,
            'modules': str(self.model),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }


class TensorFlowModelWrapper(BaseModel):
    """Wrapper for TensorFlow models."""
    
    def __init__(self, model: 'tf.keras.Model'):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        self.model = model
        self.gpu_manager = get_gpu_manager()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using TensorFlow model."""
        return self.model.predict(X, verbose=0)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Extract weights from TensorFlow model."""
        weights = {}
        
        for i, layer in enumerate(self.model.layers):
            layer_weights = layer.get_weights()
            for j, w in enumerate(layer_weights):
                weight_name = f"layer_{i}_{layer.name}_weight_{j}"
                weights[weight_name] = w
        
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights for TensorFlow model."""
        # Group weights by layer
        layer_weights = defaultdict(list)
        
        for key, value in weights.items():
            if key.startswith("layer_"):
                parts = key.split("_")
                layer_idx = int(parts[1])
                layer_weights[layer_idx].append(value)
        
        # Set weights for each layer
        for i, layer in enumerate(self.model.layers):
            if i in layer_weights:
                layer.set_weights(layer_weights[i])
    
    def get_architecture(self) -> Dict[str, Any]:
        """Get model architecture."""
        return {
            'type': 'tensorflow',
            'summary': self.model.to_json() if hasattr(self.model, 'to_json') else str(self.model.summary())
        }


class ModelConverter:
    """
    Universal model converter supporting multiple formats.
    
    Features:
    - NumPy ↔ PyTorch conversion
    - NumPy ↔ TensorFlow conversion
    - PyTorch ↔ TensorFlow conversion
    - ONNX export/import
    - Architecture preservation
    - GPU optimization
    """
    
    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        self.gpu_manager = gpu_manager or get_gpu_manager()
        self.conversion_registry = self._build_conversion_registry()
    
    def _build_conversion_registry(self) -> Dict[Tuple[str, str], Callable]:
        """Build registry of conversion functions."""
        registry = {
            (ModelFormat.NUMPY, ModelFormat.PYTORCH): self._numpy_to_pytorch,
            (ModelFormat.PYTORCH, ModelFormat.NUMPY): self._pytorch_to_numpy,
        }
        
        if TF_AVAILABLE:
            registry.update({
                (ModelFormat.NUMPY, ModelFormat.TENSORFLOW): self._numpy_to_tensorflow,
                (ModelFormat.TENSORFLOW, ModelFormat.NUMPY): self._tensorflow_to_numpy,
                (ModelFormat.PYTORCH, ModelFormat.TENSORFLOW): self._pytorch_to_tensorflow,
                (ModelFormat.TENSORFLOW, ModelFormat.PYTORCH): self._tensorflow_to_pytorch,
            })
        
        return registry
    
    def convert(self, model: Any, source_format: str, target_format: str,
                architecture_hints: Optional[Dict[str, Any]] = None) -> Any:
        """
        Convert model between formats.
        
        Args:
            model: Source model
            source_format: Source format (numpy, pytorch, tensorflow)
            target_format: Target format (numpy, pytorch, tensorflow)
            architecture_hints: Optional hints for architecture reconstruction
            
        Returns:
            Converted model
        """
        if source_format == target_format:
            return model
        
        conversion_key = (source_format, target_format)
        
        if conversion_key not in self.conversion_registry:
            # Try indirect conversion
            intermediate_format = self._find_intermediate_format(source_format, target_format)
            if intermediate_format:
                logger.info(f"Converting via intermediate format: {intermediate_format}")
                intermediate_model = self.convert(model, source_format, intermediate_format, architecture_hints)
                return self.convert(intermediate_model, intermediate_format, target_format, architecture_hints)
            else:
                raise ValueError(f"Conversion from {source_format} to {target_format} not supported")
        
        converter = self.conversion_registry[conversion_key]
        return converter(model, architecture_hints)
    
    def _find_intermediate_format(self, source: str, target: str) -> Optional[str]:
        """Find intermediate format for conversion."""
        # Try PyTorch as intermediate
        if (source, ModelFormat.PYTORCH) in self.conversion_registry and \
           (ModelFormat.PYTORCH, target) in self.conversion_registry:
            return ModelFormat.PYTORCH
        
        # Try NumPy as intermediate
        if (source, ModelFormat.NUMPY) in self.conversion_registry and \
           (ModelFormat.NUMPY, target) in self.conversion_registry:
            return ModelFormat.NUMPY
        
        return None
    
    def _numpy_to_pytorch(self, model: Any, hints: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Convert NumPy model to PyTorch."""
        wrapper = NumpyModelWrapper(model)
        weights = wrapper.get_weights()
        architecture = wrapper.get_architecture()
        
        # Try to reconstruct PyTorch model based on architecture
        if hints and 'pytorch_class' in hints:
            # Use provided PyTorch class
            pytorch_model = hints['pytorch_class'](**hints.get('kwargs', {}))
        else:
            # Create generic feedforward network
            pytorch_model = self._create_pytorch_from_weights(weights, architecture)
        
        # Transfer weights
        pytorch_wrapper = PyTorchModelWrapper(pytorch_model)
        
        # Map numpy weights to PyTorch
        if 'weight_mapping' in (hints or {}):
            mapped_weights = {}
            for torch_name, numpy_name in hints['weight_mapping'].items():
                if numpy_name in weights:
                    mapped_weights[torch_name] = weights[numpy_name]
            pytorch_wrapper.set_weights(mapped_weights)
        else:
            # Attempt automatic mapping
            self._auto_map_weights(weights, pytorch_wrapper)
        
        return pytorch_model
    
    def _pytorch_to_numpy(self, model: nn.Module, hints: Optional[Dict[str, Any]] = None) -> Any:
        """Convert PyTorch model to NumPy."""
        wrapper = PyTorchModelWrapper(model)
        weights = wrapper.get_weights()
        
        if hints and 'numpy_class' in hints:
            # Use provided NumPy class
            numpy_model = hints['numpy_class'](**hints.get('kwargs', {}))
            numpy_wrapper = NumpyModelWrapper(numpy_model)
            
            # Map PyTorch weights to NumPy
            if 'weight_mapping' in hints:
                mapped_weights = {}
                for numpy_name, torch_name in hints['weight_mapping'].items():
                    if torch_name in weights:
                        mapped_weights[numpy_name] = weights[torch_name]
                numpy_wrapper.set_weights(mapped_weights)
            else:
                # Create wrapper function
                numpy_model = self._create_numpy_predictor(model)
        else:
            # Create wrapper function
            numpy_model = self._create_numpy_predictor(model)
        
        return numpy_model
    
    def _numpy_to_tensorflow(self, model: Any, hints: Optional[Dict[str, Any]] = None) -> 'tf.keras.Model':
        """Convert NumPy model to TensorFlow."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        wrapper = NumpyModelWrapper(model)
        weights = wrapper.get_weights()
        
        if hints and 'tensorflow_model' in hints:
            tf_model = hints['tensorflow_model']
        else:
            # Create simple feedforward model
            tf_model = self._create_tensorflow_from_weights(weights)
        
        tf_wrapper = TensorFlowModelWrapper(tf_model)
        
        # Transfer weights
        if 'weight_mapping' in (hints or {}):
            mapped_weights = {}
            for tf_name, numpy_name in hints['weight_mapping'].items():
                if numpy_name in weights:
                    mapped_weights[tf_name] = weights[numpy_name]
            tf_wrapper.set_weights(mapped_weights)
        
        return tf_model
    
    def _tensorflow_to_numpy(self, model: 'tf.keras.Model', hints: Optional[Dict[str, Any]] = None) -> Any:
        """Convert TensorFlow model to NumPy."""
        return self._create_numpy_predictor(model)
    
    def _pytorch_to_tensorflow(self, model: nn.Module, hints: Optional[Dict[str, Any]] = None) -> 'tf.keras.Model':
        """Convert PyTorch to TensorFlow via weights transfer."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        # First convert to numpy
        numpy_model = self._pytorch_to_numpy(model, hints)
        
        # Then to TensorFlow
        return self._numpy_to_tensorflow(numpy_model, hints)
    
    def _tensorflow_to_pytorch(self, model: 'tf.keras.Model', hints: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Convert TensorFlow to PyTorch via weights transfer."""
        # First convert to numpy
        numpy_model = self._tensorflow_to_numpy(model, hints)
        
        # Then to PyTorch
        return self._numpy_to_pytorch(numpy_model, hints)
    
    def _create_pytorch_from_weights(self, weights: Dict[str, np.ndarray], 
                                   architecture: Dict[str, Any]) -> nn.Module:
        """Create a generic PyTorch model from weights."""
        # Analyze weights to determine architecture
        layers = []
        
        # Sort weights by layer order
        weight_items = sorted(weights.items())
        
        i = 0
        while i < len(weight_items):
            key, weight = weight_items[i]
            
            if 'weight' in key or 'W' in key or 'coef' in key:
                # Likely a linear layer
                if weight.ndim == 2:
                    in_features, out_features = weight.shape
                    layer = nn.Linear(in_features, out_features)
                    layer.weight.data = torch.from_numpy(weight.T)  # PyTorch uses transposed convention
                    
                    # Look for bias
                    if i + 1 < len(weight_items):
                        next_key, next_weight = weight_items[i + 1]
                        if 'bias' in next_key or 'b' in next_key or 'intercept' in next_key:
                            layer.bias.data = torch.from_numpy(next_weight)
                            i += 1
                    
                    layers.append(layer)
                    layers.append(nn.ReLU())  # Add activation
            
            i += 1
        
        # Remove last activation
        if layers and isinstance(layers[-1], nn.ReLU):
            layers = layers[:-1]
        
        return nn.Sequential(*layers)
    
    def _create_tensorflow_from_weights(self, weights: Dict[str, np.ndarray]) -> 'tf.keras.Model':
        """Create a generic TensorFlow model from weights."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available")
        
        layers = []
        
        # Analyze weights to build model
        for key, weight in sorted(weights.items()):
            if weight.ndim == 2 and ('weight' in key or 'W' in key):
                layers.append(tf.keras.layers.Dense(weight.shape[1], activation='relu'))
        
        if not layers:
            # Fallback to simple model
            layers = [
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ]
        
        model = tf.keras.Sequential(layers)
        return model
    
    def _create_numpy_predictor(self, model: Union[nn.Module, 'tf.keras.Model']) -> Callable:
        """Create a NumPy-compatible predictor function."""
        
        def numpy_predictor(X: np.ndarray) -> np.ndarray:
            if isinstance(model, nn.Module):
                wrapper = PyTorchModelWrapper(model)
            else:
                wrapper = TensorFlowModelWrapper(model)
            
            return wrapper.predict(X)
        
        # Add attributes for compatibility
        numpy_predictor.original_model = model
        numpy_predictor.predict = numpy_predictor
        
        return numpy_predictor
    
    def _auto_map_weights(self, source_weights: Dict[str, np.ndarray], 
                         target_wrapper: BaseModel):
        """Attempt automatic weight mapping."""
        target_weights = {}
        
        # Get target weight names
        target_arch = target_wrapper.get_architecture()
        
        # Simple heuristic matching
        for source_key, source_value in source_weights.items():
            # Try direct match
            if source_key in target_arch.get('parameters', {}):
                target_weights[source_key] = source_value
            else:
                # Try to find similar names
                for target_key in target_arch.get('parameters', {}).keys():
                    if self._similar_names(source_key, target_key):
                        target_weights[target_key] = source_value
                        break
        
        target_wrapper.set_weights(target_weights)
    
    def _similar_names(self, name1: str, name2: str) -> bool:
        """Check if two parameter names are similar."""
        # Normalize names
        norm1 = name1.lower().replace('_', '').replace('.', '')
        norm2 = name2.lower().replace('_', '').replace('.', '')
        
        # Check for common patterns
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        # Check for common weight/bias patterns
        if ('weight' in norm1 and 'weight' in norm2) or \
           ('bias' in norm1 and 'bias' in norm2):
            return True
        
        return False


class HybridModel:
    """
    Hybrid model that can execute in multiple formats.
    
    Features:
    - Automatic format selection based on input
    - GPU/CPU optimization
    - Format-specific optimizations
    - Seamless switching between backends
    """
    
    def __init__(self, models: Dict[str, Any], preferred_format: Optional[str] = None):
        """
        Initialize hybrid model.
        
        Args:
            models: Dictionary of models in different formats
            preferred_format: Preferred format for execution
        """
        self.models = models
        self.converter = ModelConverter()
        self.gpu_manager = get_gpu_manager()
        
        # Determine preferred format
        if preferred_format and preferred_format in models:
            self.preferred_format = preferred_format
        else:
            # Choose based on available hardware
            if self.gpu_manager.is_gpu and ModelFormat.PYTORCH in models:
                self.preferred_format = ModelFormat.PYTORCH
            elif TF_AVAILABLE and ModelFormat.TENSORFLOW in models:
                self.preferred_format = ModelFormat.TENSORFLOW
            else:
                self.preferred_format = list(models.keys())[0]
    
    def predict(self, X: np.ndarray, format: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using specified or optimal format.
        
        Args:
            X: Input data
            format: Optional format to use
            
        Returns:
            Predictions
        """
        # Select format
        if format and format in self.models:
            use_format = format
        else:
            use_format = self._select_optimal_format(X)
        
        # Get model
        model = self.models[use_format]
        
        # Create wrapper and predict
        if use_format == ModelFormat.NUMPY:
            wrapper = NumpyModelWrapper(model)
        elif use_format == ModelFormat.PYTORCH:
            wrapper = PyTorchModelWrapper(model)
        elif use_format == ModelFormat.TENSORFLOW:
            wrapper = TensorFlowModelWrapper(model)
        else:
            raise ValueError(f"Unknown format: {use_format}")
        
        return wrapper.predict(X)
    
    def _select_optimal_format(self, X: np.ndarray) -> str:
        """Select optimal format based on input characteristics."""
        # For large batches, prefer GPU formats
        if X.shape[0] > 1000 and self.gpu_manager.is_gpu:
            if ModelFormat.PYTORCH in self.models:
                return ModelFormat.PYTORCH
            elif ModelFormat.TENSORFLOW in self.models:
                return ModelFormat.TENSORFLOW
        
        # For small batches, NumPy might be faster
        if X.shape[0] < 10 and ModelFormat.NUMPY in self.models:
            return ModelFormat.NUMPY
        
        return self.preferred_format
    
    def add_format(self, format: str, model: Any):
        """Add a model in a new format."""
        self.models[format] = model
    
    def convert_all_formats(self):
        """Ensure all formats are available via conversion."""
        available_formats = [ModelFormat.NUMPY, ModelFormat.PYTORCH]
        if TF_AVAILABLE:
            available_formats.append(ModelFormat.TENSORFLOW)
        
        # Get one existing model
        source_format = list(self.models.keys())[0]
        source_model = self.models[source_format]
        
        # Convert to missing formats
        for target_format in available_formats:
            if target_format not in self.models:
                try:
                    converted = self.converter.convert(
                        source_model, source_format, target_format
                    )
                    self.models[target_format] = converted
                    logger.info(f"Converted model to {target_format} format")
                except Exception as e:
                    logger.warning(f"Could not convert to {target_format}: {e}")
    
    def benchmark_formats(self, test_data: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark prediction speed in different formats."""
        import time
        
        results = {}
        
        for format_name, model in self.models.items():
            # Warmup
            self.predict(test_data[:10], format=format_name)
            
            # Time runs
            start = time.time()
            for _ in range(num_runs):
                _ = self.predict(test_data, format=format_name)
            
            elapsed = time.time() - start
            results[format_name] = elapsed / num_runs
        
        return results