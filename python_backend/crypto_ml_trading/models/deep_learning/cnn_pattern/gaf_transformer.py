import numpy as np
from typing import Tuple, Optional, List
import warnings


class GAFTransformer:
    """
    Gramian Angular Field transformer for converting time series to images.
    
    GAF transforms 1D time series into 2D images that preserve temporal
    relationships and can be processed by CNNs for pattern recognition.
    
    Features:
    - GASF (Gramian Angular Summation Field)
    - GADF (Gramian Angular Difference Field)
    - Normalization and scaling
    - Multi-scale transformation
    """
    
    def __init__(self, image_size: int = 64, method: str = 'gasf'):
        """
        Initialize GAF transformer.
        
        Args:
            image_size: Size of output image (image_size x image_size)
            method: 'gasf' or 'gadf'
        """
        self.image_size = image_size
        self.method = method.lower()
        
        if self.method not in ['gasf', 'gadf']:
            raise ValueError("Method must be 'gasf' or 'gadf'")
            
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to GAF images.
        
        Args:
            X: Time series data (n_samples, n_timestamps) or (n_timestamps,)
            
        Returns:
            GAF images (n_samples, image_size, image_size) or (image_size, image_size)
        """
        # Handle single time series
        if X.ndim == 1:
            X = X.reshape(1, -1)
            return self._transform_single(X[0])
            
        # Transform multiple time series
        n_samples = X.shape[0]
        images = np.zeros((n_samples, self.image_size, self.image_size))
        
        for i in range(n_samples):
            images[i] = self._transform_single(X[i])
            
        return images
    
    def _transform_single(self, x: np.ndarray) -> np.ndarray:
        """Transform single time series to GAF image."""
        # Interpolate to desired size
        if len(x) != self.image_size:
            x = self._interpolate(x, self.image_size)
            
        # Normalize to [-1, 1]
        x_normalized = self._normalize(x)
        
        # Convert to polar coordinates
        phi = np.arccos(x_normalized)
        
        # Create GAF matrix
        if self.method == 'gasf':
            # Gramian Angular Summation Field
            gaf = np.cos(phi[:, None] + phi[None, :])
        else:  # gadf
            # Gramian Angular Difference Field
            gaf = np.sin(phi[:, None] - phi[None, :])
            
        return gaf
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize time series to [-1, 1]."""
        x_min = np.min(x)
        x_max = np.max(x)
        
        if x_max == x_min:
            return np.zeros_like(x)
            
        # Scale to [0, 1]
        x_scaled = (x - x_min) / (x_max - x_min)
        
        # Scale to [-1, 1]
        x_normalized = 2 * x_scaled - 1
        
        return x_normalized
    
    def _interpolate(self, x: np.ndarray, new_size: int) -> np.ndarray:
        """Interpolate time series to new size."""
        old_size = len(x)
        old_indices = np.linspace(0, old_size - 1, old_size)
        new_indices = np.linspace(0, old_size - 1, new_size)
        
        return np.interp(new_indices, old_indices, x)
    
    def create_multiscale_images(self, x: np.ndarray, 
                               scales: List[int] = [32, 64, 128]) -> List[np.ndarray]:
        """
        Create GAF images at multiple scales.
        
        Args:
            x: Time series data
            scales: List of image sizes
            
        Returns:
            List of GAF images at different scales
        """
        images = []
        
        for scale in scales:
            transformer = GAFTransformer(image_size=scale, method=self.method)
            image = transformer.fit_transform(x)
            images.append(image)
            
        return images
    
    def create_combined_gaf(self, x: np.ndarray) -> np.ndarray:
        """
        Create combined GASF and GADF image.
        
        Args:
            x: Time series data
            
        Returns:
            Combined image (2, image_size, image_size)
        """
        # Create GASF
        gasf_transformer = GAFTransformer(self.image_size, 'gasf')
        gasf = gasf_transformer.fit_transform(x)
        
        # Create GADF
        gadf_transformer = GAFTransformer(self.image_size, 'gadf')
        gadf = gadf_transformer.fit_transform(x)
        
        # Stack as channels
        if x.ndim == 1:
            return np.stack([gasf, gadf], axis=0)
        else:
            return np.stack([gasf, gadf], axis=1)


class RecurrencePlotTransformer:
    """
    Create recurrence plots from time series data.
    
    Recurrence plots visualize the recurrence of states in phase space,
    useful for detecting patterns in nonlinear time series.
    """
    
    def __init__(self, dimension: int = 3, time_delay: int = 1,
                 threshold: Optional[float] = None):
        """
        Initialize recurrence plot transformer.
        
        Args:
            dimension: Embedding dimension
            time_delay: Time delay for embedding
            threshold: Distance threshold (None for adaptive)
        """
        self.dimension = dimension
        self.time_delay = time_delay
        self.threshold = threshold
        
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform time series to recurrence plot.
        
        Args:
            x: Time series data
            
        Returns:
            Recurrence plot matrix
        """
        # Embed time series
        embedded = self._embed_time_series(x)
        
        # Calculate pairwise distances
        n_points = embedded.shape[0]
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                distances[i, j] = np.linalg.norm(embedded[i] - embedded[j])
                
        # Apply threshold
        if self.threshold is None:
            # Adaptive threshold: 10% of max distance
            self.threshold = 0.1 * np.max(distances)
            
        recurrence_plot = (distances < self.threshold).astype(float)
        
        return recurrence_plot
    
    def _embed_time_series(self, x: np.ndarray) -> np.ndarray:
        """Create phase space embedding."""
        n = len(x)
        n_embedded = n - (self.dimension - 1) * self.time_delay
        
        if n_embedded <= 0:
            raise ValueError("Time series too short for given embedding parameters")
            
        embedded = np.zeros((n_embedded, self.dimension))
        
        for i in range(self.dimension):
            start_idx = i * self.time_delay
            end_idx = start_idx + n_embedded
            embedded[:, i] = x[start_idx:end_idx]
            
        return embedded


class MarkovTransitionField:
    """
    Create Markov Transition Field (MTF) from time series.
    
    MTF encodes the transition probabilities between quantized states,
    preserving temporal dependencies in image form.
    """
    
    def __init__(self, image_size: int = 64, n_bins: int = 10):
        """
        Initialize MTF transformer.
        
        Args:
            image_size: Size of output image
            n_bins: Number of quantization bins
        """
        self.image_size = image_size
        self.n_bins = n_bins
        
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform time series to MTF image.
        
        Args:
            x: Time series data
            
        Returns:
            MTF image
        """
        # Interpolate to desired size
        if len(x) != self.image_size:
            x = self._interpolate(x, self.image_size)
            
        # Quantize time series
        quantized = self._quantize(x)
        
        # Create transition matrix
        mtf = np.zeros((self.image_size, self.image_size))
        
        for i in range(self.image_size):
            for j in range(self.image_size):
                # Transition probability from bin at time i to bin at time j
                if quantized[i] == quantized[j]:
                    mtf[i, j] = 1.0
                else:
                    # Weight by temporal distance
                    mtf[i, j] = 1.0 / (1 + abs(i - j))
                    
        return mtf
    
    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize time series into bins."""
        # Create bins
        bins = np.linspace(np.min(x), np.max(x), self.n_bins + 1)
        bins[-1] += 1e-10  # Ensure max value is included
        
        # Quantize
        quantized = np.digitize(x, bins) - 1
        
        return quantized
    
    def _interpolate(self, x: np.ndarray, new_size: int) -> np.ndarray:
        """Interpolate time series to new size."""
        old_size = len(x)
        old_indices = np.linspace(0, old_size - 1, old_size)
        new_indices = np.linspace(0, old_size - 1, new_size)
        
        return np.interp(new_indices, old_indices, x)