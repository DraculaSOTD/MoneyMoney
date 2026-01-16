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


class EnhancedPatternGenerator(PatternGenerator):
    """
    Extended pattern generator with multi-timeframe and 12-class pattern detection.

    Pattern Classes (12):
    - 0: no_pattern - No recognizable pattern
    - 1: double_bottom - Bullish reversal (W pattern)
    - 2: double_top - Bearish reversal (M pattern)
    - 3: head_shoulders - Bearish reversal (Head & Shoulders)
    - 4: inv_head_shoulders - Bullish reversal (Inverse H&S)
    - 5: bull_flag - Bullish continuation
    - 6: bear_flag - Bearish continuation
    - 7: ascending_triangle - Bullish (rising lows, flat highs)
    - 8: descending_triangle - Bearish (falling highs, flat lows)
    - 9: symmetrical_triangle - Neutral consolidation
    - 10: cup_handle - Bullish (U-shape with handle)
    - 11: consolidation - Range-bound sideways
    """

    PATTERN_NAMES = [
        'no_pattern', 'double_bottom', 'double_top', 'head_shoulders',
        'inv_head_shoulders', 'bull_flag', 'bear_flag', 'ascending_triangle',
        'descending_triangle', 'symmetrical_triangle', 'cup_handle', 'consolidation'
    ]

    TIMEFRAME_MAP = {
        '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': '1H', '4h': '4H', '1d': '1D', '1D': '1D'
    }

    def __init__(self, image_size: int = 64,
                 methods: List[str] = ['gasf', 'gadf', 'rp']):
        """
        Initialize enhanced pattern generator.

        Args:
            image_size: Size of generated images
            methods: List of image generation methods
        """
        super().__init__(image_size, methods)

    def generate_multi_timeframe_images(self,
                                       data: pd.DataFrame,
                                       timeframes: List[str] = ['1m', '5m', '15m', '1h'],
                                       window_sizes: Dict[str, int] = None) -> Dict[str, np.ndarray]:
        """
        Generate pattern images for multiple timeframes.

        Args:
            data: Raw 1-minute OHLCV data with DatetimeIndex
            timeframes: List of timeframes to generate
            window_sizes: Window size for each timeframe

        Returns:
            Dict mapping timeframe -> image array (n_samples, n_channels, H, W)
        """
        default_windows = {'1m': 60, '5m': 60, '15m': 60, '1h': 48, '4h': 48, '1d': 30}
        window_sizes = window_sizes or default_windows

        images = {}
        for tf in timeframes:
            try:
                if tf == '1m':
                    resampled = data
                else:
                    resampled = self._resample_ohlcv(data, tf)

                if len(resampled) > window_sizes.get(tf, 60):
                    images[tf] = self.generate_pattern_images(
                        resampled,
                        window_size=window_sizes.get(tf, 60)
                    )
            except Exception as e:
                print(f"Warning: Could not generate images for {tf}: {e}")
                continue

        return images

    def _resample_ohlcv(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to specified timeframe.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex
            timeframe: Target timeframe (e.g., '5m', '1h')

        Returns:
            Resampled DataFrame
        """
        rule = self.TIMEFRAME_MAP.get(timeframe, '1T')

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            elif 'date' in data.columns:
                data = data.set_index('date')

        resampled = data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def detect_advanced_patterns(self, data: pd.DataFrame,
                                window_size: int = 60) -> np.ndarray:
        """
        Detect 12 advanced pattern classes for crypto trading.

        Args:
            data: OHLCV DataFrame
            window_size: Size of sliding window

        Returns:
            Array of pattern labels (0-11) for each window
        """
        n_samples = len(data) - window_size + 1
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            window = data.iloc[i:i+window_size].copy()
            labels[i] = self._classify_pattern(window)

        return labels

    def _classify_pattern(self, window: pd.DataFrame) -> int:
        """
        Classify pattern in a price window.

        Priority order: most specific patterns first, then general patterns.

        Args:
            window: OHLCV DataFrame for the window

        Returns:
            Pattern class index (0-11)
        """
        high = window['high'].values
        low = window['low'].values
        close = window['close'].values
        open_prices = window['open'].values

        # Complex patterns first (more specific)
        if self._is_head_shoulders(high, low, close):
            return 3  # head_shoulders
        if self._is_inverse_head_shoulders(high, low, close):
            return 4  # inv_head_shoulders
        if self._is_cup_handle(high, low, close):
            return 10  # cup_handle

        # Double patterns
        if self._is_double_bottom(low, close):
            return 1  # double_bottom
        if self._is_double_top(high, close):
            return 2  # double_top

        # Flag patterns
        if self._is_bull_flag(high, low, close, open_prices):
            return 5  # bull_flag
        if self._is_bear_flag(high, low, close, open_prices):
            return 6  # bear_flag

        # Triangle patterns
        if self._is_ascending_triangle(high, low):
            return 7  # ascending_triangle
        if self._is_descending_triangle(high, low):
            return 8  # descending_triangle
        if self._is_symmetrical_triangle(high, low):
            return 9  # symmetrical_triangle

        # Consolidation
        if self._is_consolidation(close):
            return 11  # consolidation

        return 0  # no_pattern

    def _is_double_bottom(self, low: np.ndarray, close: np.ndarray,
                         tolerance: float = 0.02) -> bool:
        """Detect W-shaped double bottom pattern."""
        n = len(low)
        if n < 20:
            return False

        # Find two significant lows
        first_third = low[:n//3]
        last_third = low[2*n//3:]
        middle = low[n//3:2*n//3]

        first_low_idx = np.argmin(first_third)
        second_low_idx = np.argmin(last_third) + 2*n//3

        first_low = low[first_low_idx]
        second_low = low[second_low_idx]

        # Check if lows are at similar levels
        price_range = np.max(close) - np.min(close)
        if price_range == 0:
            return False

        low_diff = abs(first_low - second_low) / price_range

        # Middle should be higher
        middle_high = np.max(middle)
        neckline_rise = (middle_high - max(first_low, second_low)) / price_range

        return low_diff < tolerance and neckline_rise > 0.03

    def _is_double_top(self, high: np.ndarray, close: np.ndarray,
                      tolerance: float = 0.02) -> bool:
        """Detect M-shaped double top pattern."""
        n = len(high)
        if n < 20:
            return False

        first_third = high[:n//3]
        last_third = high[2*n//3:]
        middle = high[n//3:2*n//3]

        first_high_idx = np.argmax(first_third)
        second_high_idx = np.argmax(last_third) + 2*n//3

        first_high = high[first_high_idx]
        second_high = high[second_high_idx]

        price_range = np.max(close) - np.min(close)
        if price_range == 0:
            return False

        high_diff = abs(first_high - second_high) / price_range

        middle_low = np.min(middle)
        neckline_drop = (min(first_high, second_high) - middle_low) / price_range

        return high_diff < tolerance and neckline_drop > 0.03

    def _is_head_shoulders(self, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray) -> bool:
        """Detect head and shoulders pattern (bearish reversal)."""
        n = len(high)
        if n < 30:
            return False

        # Divide into 5 parts: left shoulder, left neck, head, right neck, right shoulder
        fifth = n // 5

        left_shoulder = np.max(high[:fifth])
        head = np.max(high[2*fifth:3*fifth])
        right_shoulder = np.max(high[4*fifth:])

        left_neck = np.min(low[fifth:2*fifth])
        right_neck = np.min(low[3*fifth:4*fifth])

        price_range = np.max(close) - np.min(close)
        if price_range == 0:
            return False

        # Head should be highest
        head_above_shoulders = head > left_shoulder and head > right_shoulder

        # Shoulders should be at similar levels
        shoulder_diff = abs(left_shoulder - right_shoulder) / price_range

        # Neckline should be roughly horizontal
        neck_diff = abs(left_neck - right_neck) / price_range

        return head_above_shoulders and shoulder_diff < 0.05 and neck_diff < 0.05

    def _is_inverse_head_shoulders(self, high: np.ndarray, low: np.ndarray,
                                   close: np.ndarray) -> bool:
        """Detect inverse head and shoulders pattern (bullish reversal)."""
        n = len(low)
        if n < 30:
            return False

        fifth = n // 5

        left_shoulder = np.min(low[:fifth])
        head = np.min(low[2*fifth:3*fifth])
        right_shoulder = np.min(low[4*fifth:])

        left_neck = np.max(high[fifth:2*fifth])
        right_neck = np.max(high[3*fifth:4*fifth])

        price_range = np.max(close) - np.min(close)
        if price_range == 0:
            return False

        head_below_shoulders = head < left_shoulder and head < right_shoulder
        shoulder_diff = abs(left_shoulder - right_shoulder) / price_range
        neck_diff = abs(left_neck - right_neck) / price_range

        return head_below_shoulders and shoulder_diff < 0.05 and neck_diff < 0.05

    def _is_bull_flag(self, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, open_prices: np.ndarray) -> bool:
        """Detect bull flag pattern (bullish continuation)."""
        n = len(close)
        if n < 15:
            return False

        # First portion should be strong uptrend (pole)
        pole_end = n // 3
        pole_return = (close[pole_end] - close[0]) / close[0] if close[0] != 0 else 0

        if pole_return < 0.03:  # Need at least 3% rise for pole
            return False

        # Flag portion: slight downward consolidation
        flag_start = pole_end
        flag_highs = high[flag_start:]
        flag_lows = low[flag_start:]

        # Flag should slope down slightly
        flag_slope = np.polyfit(range(len(flag_highs)), flag_highs, 1)[0]
        flag_range = (np.max(flag_highs) - np.min(flag_lows)) / close[pole_end] if close[pole_end] != 0 else 1

        return flag_slope < 0 and flag_range < 0.05

    def _is_bear_flag(self, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, open_prices: np.ndarray) -> bool:
        """Detect bear flag pattern (bearish continuation)."""
        n = len(close)
        if n < 15:
            return False

        pole_end = n // 3
        pole_return = (close[pole_end] - close[0]) / close[0] if close[0] != 0 else 0

        if pole_return > -0.03:  # Need at least 3% drop for pole
            return False

        flag_start = pole_end
        flag_highs = high[flag_start:]
        flag_lows = low[flag_start:]

        flag_slope = np.polyfit(range(len(flag_lows)), flag_lows, 1)[0]
        flag_range = (np.max(flag_highs) - np.min(flag_lows)) / abs(close[pole_end]) if close[pole_end] != 0 else 1

        return flag_slope > 0 and flag_range < 0.05

    def _is_ascending_triangle(self, high: np.ndarray, low: np.ndarray,
                               tolerance: float = 0.02) -> bool:
        """Detect ascending triangle (bullish - flat highs, rising lows)."""
        n = len(high)
        if n < 20:
            return False

        # Fit lines to highs and lows
        x = np.arange(n)
        high_slope, high_intercept = np.polyfit(x, high, 1)
        low_slope, low_intercept = np.polyfit(x, low, 1)

        # Highs should be relatively flat, lows should be rising
        price_range = np.max(high) - np.min(low)
        if price_range == 0:
            return False

        high_flat = abs(high_slope * n / price_range) < tolerance
        low_rising = (low_slope * n / price_range) > 0.02

        return high_flat and low_rising

    def _is_descending_triangle(self, high: np.ndarray, low: np.ndarray,
                                tolerance: float = 0.02) -> bool:
        """Detect descending triangle (bearish - falling highs, flat lows)."""
        n = len(high)
        if n < 20:
            return False

        x = np.arange(n)
        high_slope, _ = np.polyfit(x, high, 1)
        low_slope, _ = np.polyfit(x, low, 1)

        price_range = np.max(high) - np.min(low)
        if price_range == 0:
            return False

        high_falling = (high_slope * n / price_range) < -0.02
        low_flat = abs(low_slope * n / price_range) < tolerance

        return high_falling and low_flat

    def _is_symmetrical_triangle(self, high: np.ndarray, low: np.ndarray,
                                 tolerance: float = 0.02) -> bool:
        """Detect symmetrical triangle (converging trendlines)."""
        n = len(high)
        if n < 20:
            return False

        x = np.arange(n)
        high_slope, _ = np.polyfit(x, high, 1)
        low_slope, _ = np.polyfit(x, low, 1)

        price_range = np.max(high) - np.min(low)
        if price_range == 0:
            return False

        high_falling = (high_slope * n / price_range) < -0.01
        low_rising = (low_slope * n / price_range) > 0.01

        # Check convergence
        converging = high_falling and low_rising

        # Check symmetry (slopes should be roughly equal and opposite)
        symmetry = abs(abs(high_slope) - abs(low_slope)) / max(abs(high_slope), abs(low_slope), 1e-10) < 0.5

        return converging and symmetry

    def _is_cup_handle(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray) -> bool:
        """Detect cup and handle pattern (bullish)."""
        n = len(close)
        if n < 30:
            return False

        # Cup should be U-shaped (first 80% of data)
        cup_end = int(n * 0.8)
        cup = close[:cup_end]

        # Find the bottom of the cup
        cup_bottom_idx = np.argmin(cup)

        # Cup should have bottom in the middle third
        if cup_bottom_idx < cup_end // 4 or cup_bottom_idx > 3 * cup_end // 4:
            return False

        # Left and right lips should be at similar levels
        left_lip = np.max(cup[:cup_bottom_idx])
        right_lip = np.max(cup[cup_bottom_idx:])

        price_range = np.max(close) - np.min(close)
        if price_range == 0:
            return False

        lip_diff = abs(left_lip - right_lip) / price_range

        # Handle: small dip at the end
        handle = close[cup_end:]
        if len(handle) < 3:
            return False

        handle_dip = (np.max(handle) - np.min(handle)) / price_range

        return lip_diff < 0.05 and handle_dip < 0.1 and handle_dip > 0.01

    def _is_consolidation(self, close: np.ndarray, threshold: float = 0.03) -> bool:
        """Detect sideways consolidation (range-bound)."""
        n = len(close)
        if n < 10:
            return False

        mean_price = np.mean(close)
        if mean_price == 0:
            return False

        # Calculate price range as percentage
        price_range = (np.max(close) - np.min(close)) / mean_price

        # Check if price stayed within threshold
        return price_range < threshold