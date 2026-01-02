import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from .gaf_transformer import (
    GAFTransformer, RecurrencePlotTransformer, MarkovTransitionField
)


class PatternGenerator:
    """
    Generate various image representations from cryptocurrency time series.
    
    Features:
    - Multiple image generation methods (GAF, Recurrence, MTF)
    - Chart patterns as images (candlestick patterns)
    - Volume and indicator overlays
    - Multi-timeframe analysis
    """
    
    def __init__(self, image_size: int = 64, 
                 methods: List[str] = ['gasf', 'gadf', 'rp']):
        """
        Initialize pattern generator.
        
        Args:
            image_size: Size of generated images
            methods: List of methods to use
        """
        self.image_size = image_size
        self.methods = methods
        self.transformers = self._initialize_transformers()
        
    def _initialize_transformers(self) -> Dict:
        """Initialize image transformers."""
        transformers = {}
        
        if 'gasf' in self.methods:
            transformers['gasf'] = GAFTransformer(self.image_size, 'gasf')
        if 'gadf' in self.methods:
            transformers['gadf'] = GAFTransformer(self.image_size, 'gadf')
        if 'rp' in self.methods:
            transformers['rp'] = RecurrencePlotTransformer()
        if 'mtf' in self.methods:
            transformers['mtf'] = MarkovTransitionField(self.image_size)
            
        return transformers
    
    def generate_pattern_images(self, data: pd.DataFrame, 
                              window_size: int = 60) -> np.ndarray:
        """
        Generate pattern images from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            window_size: Size of sliding window
            
        Returns:
            Array of pattern images (n_samples, n_channels, height, width)
        """
        n_samples = len(data) - window_size + 1
        n_channels = len(self.methods) + 3  # methods + OHLC channels
        images = np.zeros((n_samples, n_channels, self.image_size, self.image_size))
        
        for i in range(n_samples):
            window_data = data.iloc[i:i+window_size]
            
            # Generate images for each method
            channel_idx = 0
            
            # Price-based patterns
            close_prices = window_data['close'].values
            
            for method, transformer in self.transformers.items():
                if method in ['gasf', 'gadf', 'mtf']:
                    images[i, channel_idx] = transformer.fit_transform(close_prices)
                elif method == 'rp':
                    rp = transformer.fit_transform(close_prices)
                    # Resize if needed
                    if rp.shape[0] != self.image_size:
                        rp = self._resize_image(rp, self.image_size)
                    images[i, channel_idx] = rp
                channel_idx += 1
                
            # OHLC channels
            images[i, channel_idx] = self._create_ohlc_image(window_data, 'high')
            images[i, channel_idx + 1] = self._create_ohlc_image(window_data, 'low')
            images[i, channel_idx + 2] = self._create_volume_image(window_data)
            
        return images
    
    def _create_ohlc_image(self, data: pd.DataFrame, column: str) -> np.ndarray:
        """Create image from OHLC data."""
        prices = data[column].values
        
        # Normalize to [0, 1]
        prices_norm = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)
        
        # Create image
        image = np.zeros((self.image_size, self.image_size))
        
        # Map prices to image rows
        price_indices = (prices_norm * (self.image_size - 1)).astype(int)
        time_indices = np.linspace(0, self.image_size - 1, len(prices)).astype(int)
        
        # Draw price line
        for t, p in zip(time_indices, price_indices):
            image[self.image_size - 1 - p, t] = 1.0
            
            # Add thickness
            for offset in [-1, 1]:
                if 0 <= p + offset < self.image_size:
                    image[self.image_size - 1 - (p + offset), t] = 0.5
                    
        return image
    
    def _create_volume_image(self, data: pd.DataFrame) -> np.ndarray:
        """Create volume profile image."""
        if 'volume' not in data:
            return np.zeros((self.image_size, self.image_size))
            
        volumes = data['volume'].values
        prices = data['close'].values
        
        # Create volume profile
        price_bins = np.linspace(prices.min(), prices.max(), self.image_size)
        volume_profile = np.zeros(self.image_size)
        
        for p, v in zip(prices, volumes):
            bin_idx = np.digitize(p, price_bins) - 1
            bin_idx = np.clip(bin_idx, 0, self.image_size - 1)
            volume_profile[bin_idx] += v
            
        # Normalize
        volume_profile = volume_profile / (volume_profile.max() + 1e-10)
        
        # Create image
        image = np.zeros((self.image_size, self.image_size))
        for i, vol in enumerate(volume_profile):
            width = int(vol * self.image_size)
            image[i, :width] = vol
            
        return image
    
    def _resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size using interpolation."""
        if image.shape[0] == target_size and image.shape[1] == target_size:
            return image
            
        # Simple nearest neighbor interpolation
        old_size = image.shape[0]
        scale = old_size / target_size
        
        resized = np.zeros((target_size, target_size))
        for i in range(target_size):
            for j in range(target_size):
                old_i = int(i * scale)
                old_j = int(j * scale)
                old_i = min(old_i, old_size - 1)
                old_j = min(old_j, old_size - 1)
                resized[i, j] = image[old_i, old_j]
                
        return resized
    
    def create_candlestick_patterns(self, data: pd.DataFrame,
                                  pattern_size: int = 20) -> Dict[str, np.ndarray]:
        """
        Detect and create images of candlestick patterns.
        
        Args:
            data: OHLCV data
            pattern_size: Size of pattern window
            
        Returns:
            Dictionary of pattern images
        """
        patterns = {}
        
        # Doji pattern
        doji_mask = self._detect_doji(data)
        patterns['doji'] = self._create_pattern_images(data, doji_mask, pattern_size)
        
        # Hammer pattern
        hammer_mask = self._detect_hammer(data)
        patterns['hammer'] = self._create_pattern_images(data, hammer_mask, pattern_size)
        
        # Engulfing pattern
        engulfing_mask = self._detect_engulfing(data)
        patterns['engulfing'] = self._create_pattern_images(data, engulfing_mask, pattern_size)
        
        # Three white soldiers / Three black crows
        three_soldiers_mask = self._detect_three_soldiers(data)
        patterns['three_soldiers'] = self._create_pattern_images(data, three_soldiers_mask, pattern_size)
        
        return patterns
    
    def _detect_doji(self, data: pd.DataFrame) -> np.ndarray:
        """Detect doji patterns."""
        body = abs(data['close'] - data['open'])
        range_hl = data['high'] - data['low']
        
        # Doji: small body relative to range
        doji = body < 0.1 * range_hl
        
        return doji.values
    
    def _detect_hammer(self, data: pd.DataFrame) -> np.ndarray:
        """Detect hammer patterns."""
        body = abs(data['close'] - data['open'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        
        # Hammer: long lower shadow, small upper shadow
        hammer = (lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)
        
        return hammer.values
    
    def _detect_engulfing(self, data: pd.DataFrame) -> np.ndarray:
        """Detect engulfing patterns."""
        bullish_body = data['close'] > data['open']
        bearish_body = data['close'] < data['open']
        
        # Bullish engulfing
        bullish_engulfing = (
            bearish_body.shift(1) & 
            bullish_body & 
            (data['open'] < data['close'].shift(1)) &
            (data['close'] > data['open'].shift(1))
        )
        
        # Bearish engulfing
        bearish_engulfing = (
            bullish_body.shift(1) & 
            bearish_body & 
            (data['open'] > data['close'].shift(1)) &
            (data['close'] < data['open'].shift(1))
        )
        
        return (bullish_engulfing | bearish_engulfing).values
    
    def _detect_three_soldiers(self, data: pd.DataFrame) -> np.ndarray:
        """Detect three white soldiers or three black crows."""
        bullish = data['close'] > data['open']
        bearish = data['close'] < data['open']
        
        # Three consecutive bullish or bearish candles
        three_bulls = bullish & bullish.shift(1) & bullish.shift(2)
        three_bears = bearish & bearish.shift(1) & bearish.shift(2)
        
        # Each candle closes higher/lower than previous
        ascending = (data['close'] > data['close'].shift(1)) & (data['close'].shift(1) > data['close'].shift(2))
        descending = (data['close'] < data['close'].shift(1)) & (data['close'].shift(1) < data['close'].shift(2))
        
        three_soldiers = (three_bulls & ascending) | (three_bears & descending)
        
        return three_soldiers.values
    
    def _create_pattern_images(self, data: pd.DataFrame, 
                             pattern_mask: np.ndarray,
                             pattern_size: int) -> np.ndarray:
        """Create images around detected patterns."""
        pattern_indices = np.where(pattern_mask)[0]
        
        if len(pattern_indices) == 0:
            return np.array([])
            
        images = []
        
        for idx in pattern_indices:
            # Get window around pattern
            start = max(0, idx - pattern_size // 2)
            end = min(len(data), idx + pattern_size // 2)
            
            if end - start >= pattern_size:
                window = data.iloc[start:end]
                
                # Create OHLC image
                image = self._create_candlestick_image(window)
                images.append(image)
                
        return np.array(images) if images else np.array([])
    
    def _create_candlestick_image(self, data: pd.DataFrame) -> np.ndarray:
        """Create candlestick chart as image."""
        n_candles = len(data)
        image = np.zeros((self.image_size, self.image_size))
        
        # Normalize prices
        price_min = data[['open', 'high', 'low', 'close']].min().min()
        price_max = data[['open', 'high', 'low', 'close']].max().max()
        price_range = price_max - price_min + 1e-10
        
        # Width of each candle
        candle_width = max(1, self.image_size // n_candles)
        
        for i, (_, candle) in enumerate(data.iterrows()):
            x_center = int((i + 0.5) * self.image_size / n_candles)
            
            # Normalize prices to image coordinates
            open_y = int((1 - (candle['open'] - price_min) / price_range) * (self.image_size - 1))
            close_y = int((1 - (candle['close'] - price_min) / price_range) * (self.image_size - 1))
            high_y = int((1 - (candle['high'] - price_min) / price_range) * (self.image_size - 1))
            low_y = int((1 - (candle['low'] - price_min) / price_range) * (self.image_size - 1))
            
            # Draw high-low line
            for y in range(high_y, low_y + 1):
                if 0 <= y < self.image_size:
                    image[y, x_center] = 0.5
                    
            # Draw body
            body_start = min(open_y, close_y)
            body_end = max(open_y, close_y)
            
            for y in range(body_start, body_end + 1):
                for x_offset in range(-candle_width//2, candle_width//2 + 1):
                    x = x_center + x_offset
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        # Bullish candle: 1.0, Bearish: 0.3
                        image[y, x] = 1.0 if candle['close'] > candle['open'] else 0.3
                        
        return image