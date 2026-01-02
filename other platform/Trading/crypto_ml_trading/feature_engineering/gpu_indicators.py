"""
GPU-Accelerated Technical Indicators using CuPy.

Provides high-performance technical indicator calculations on GPU for
real-time feature engineering in crypto trading.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import lru_cache

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. GPU indicators will fall back to NumPy.")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class GPUTechnicalIndicators:
    """
    GPU-accelerated technical indicators for crypto trading.
    
    Features:
    - CuPy-based GPU computations
    - Automatic CPU fallback
    - Batch processing support
    - Memory-efficient operations
    """
    
    def __init__(self, use_gpu: bool = True, cache_size: int = 128):
        """
        Initialize GPU technical indicators.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            cache_size: LRU cache size for repeated calculations
        """
        self.gpu_manager = get_gpu_manager()
        self.use_gpu = use_gpu and CUPY_AVAILABLE and self.gpu_manager.is_gpu
        
        if self.use_gpu:
            self.xp = cp
            logger.info("GPU indicators initialized with CuPy")
        else:
            self.xp = np
            logger.info("GPU indicators using NumPy (CPU) fallback")
        
        # Configure cache
        self._cache_size = cache_size
        self._configure_caches()
    
    def _configure_caches(self):
        """Configure LRU caches for expensive operations."""
        # Apply caching to expensive methods
        self.sma = lru_cache(maxsize=self._cache_size)(self._sma_uncached)
        self.ema = lru_cache(maxsize=self._cache_size)(self._ema_uncached)
        self.rsi = lru_cache(maxsize=self._cache_size)(self._rsi_uncached)
    
    def to_gpu(self, data: Union[np.ndarray, torch.Tensor]) -> Union[cp.ndarray, np.ndarray]:
        """Convert data to GPU array."""
        if not self.use_gpu:
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            return np.asarray(data)
        
        if isinstance(data, torch.Tensor):
            return cp.asarray(data.cpu().numpy())
        elif isinstance(data, np.ndarray):
            return cp.asarray(data)
        else:
            return cp.asarray(np.array(data))
    
    def to_numpy(self, data: Union[cp.ndarray, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert GPU array back to NumPy."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif self.use_gpu and hasattr(data, 'get'):
            return data.get()
        else:
            return np.asarray(data)
    
    # Simple Moving Average
    def _sma_uncached(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate Simple Moving Average on GPU."""
        gpu_data = self.to_gpu(data)
        
        # Use convolution for efficient SMA
        kernel = self.xp.ones(window) / window
        
        if len(gpu_data.shape) == 1:
            # 1D data
            sma = self.xp.convolve(gpu_data, kernel, mode='valid')
            # Pad to match input length
            pad_width = len(gpu_data) - len(sma)
            sma = self.xp.pad(sma, (pad_width, 0), mode='constant', constant_values=self.xp.nan)
        else:
            # 2D data (batch processing)
            sma = self.xp.apply_along_axis(
                lambda x: self.xp.convolve(x, kernel, mode='valid'), 
                axis=1, arr=gpu_data
            )
            pad_width = gpu_data.shape[1] - sma.shape[1]
            sma = self.xp.pad(sma, ((0, 0), (pad_width, 0)), mode='constant', constant_values=self.xp.nan)
        
        return self.to_numpy(sma)
    
    # Exponential Moving Average
    def _ema_uncached(self, data: np.ndarray, window: int, alpha: Optional[float] = None) -> np.ndarray:
        """Calculate Exponential Moving Average on GPU."""
        gpu_data = self.to_gpu(data)
        
        if alpha is None:
            alpha = 2.0 / (window + 1)
        
        if len(gpu_data.shape) == 1:
            ema = self._ema_1d(gpu_data, alpha)
        else:
            # Batch processing
            ema = self.xp.zeros_like(gpu_data)
            for i in range(gpu_data.shape[0]):
                ema[i] = self._ema_1d(gpu_data[i], alpha)
        
        return self.to_numpy(ema)
    
    def _ema_1d(self, data: Union[cp.ndarray, np.ndarray], alpha: float) -> Union[cp.ndarray, np.ndarray]:
        """Calculate 1D EMA."""
        ema = self.xp.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    # Relative Strength Index
    def _rsi_uncached(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI on GPU."""
        gpu_data = self.to_gpu(data)
        
        # Calculate price changes
        if len(gpu_data.shape) == 1:
            deltas = self.xp.diff(gpu_data)
            gains = self.xp.where(deltas > 0, deltas, 0)
            losses = self.xp.where(deltas < 0, -deltas, 0)
            
            avg_gains = self._rma(gains, period)
            avg_losses = self._rma(losses, period)
            
            rs = avg_gains / self.xp.where(avg_losses != 0, avg_losses, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Pad for alignment
            rsi = self.xp.pad(rsi, (1, 0), mode='constant', constant_values=50)
        else:
            # Batch processing
            rsi = self.xp.zeros_like(gpu_data)
            for i in range(gpu_data.shape[0]):
                rsi[i] = self._rsi_uncached(self.to_numpy(gpu_data[i]), period)
                rsi[i] = self.to_gpu(rsi[i])
        
        return self.to_numpy(rsi)
    
    def _rma(self, data: Union[cp.ndarray, np.ndarray], period: int) -> Union[cp.ndarray, np.ndarray]:
        """Rolling Moving Average (RMA) for RSI calculation."""
        alpha = 1.0 / period
        return self._ema_1d(data, alpha)
    
    # Bollinger Bands
    def bollinger_bands(self, data: np.ndarray, window: int = 20, num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands on GPU.
        
        Returns:
            (upper_band, middle_band, lower_band)
        """
        gpu_data = self.to_gpu(data)
        
        # Middle band (SMA)
        middle = self.to_gpu(self.sma(data, window))
        
        # Calculate rolling standard deviation
        if len(gpu_data.shape) == 1:
            # Efficient rolling std using convolution
            squared_diff = (gpu_data - middle) ** 2
            variance = self.xp.convolve(squared_diff, self.xp.ones(window) / window, mode='same')
            std = self.xp.sqrt(variance)
        else:
            # Batch processing
            std = self.xp.zeros_like(gpu_data)
            for i in range(window, len(gpu_data)):
                window_data = gpu_data[i-window:i]
                std[i] = self.xp.std(window_data)
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        return (self.to_numpy(upper), self.to_numpy(middle), self.to_numpy(lower))
    
    # MACD
    def macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD on GPU.
        
        Returns:
            (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = self.ema(macd_line, signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return (macd_line, signal_line, histogram)
    
    # Stochastic Oscillator
    def stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator on GPU.
        
        Returns:
            (%K, %D)
        """
        high_gpu = self.to_gpu(high)
        low_gpu = self.to_gpu(low)
        close_gpu = self.to_gpu(close)
        
        # Calculate rolling high/low
        k_values = self.xp.zeros_like(close_gpu)
        
        for i in range(k_period, len(close_gpu)):
            period_high = self.xp.max(high_gpu[i-k_period:i])
            period_low = self.xp.min(low_gpu[i-k_period:i])
            
            if period_high != period_low:
                k_values[i] = 100 * (close_gpu[i] - period_low) / (period_high - period_low)
            else:
                k_values[i] = 50
        
        # %D is SMA of %K
        d_values = self.sma(self.to_numpy(k_values), d_period)
        
        return (self.to_numpy(k_values), d_values)
    
    # Average True Range
    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range on GPU."""
        high_gpu = self.to_gpu(high)
        low_gpu = self.to_gpu(low)
        close_gpu = self.to_gpu(close)
        
        # True Range calculation
        high_low = high_gpu - low_gpu
        high_close = self.xp.abs(high_gpu[1:] - close_gpu[:-1])
        low_close = self.xp.abs(low_gpu[1:] - close_gpu[:-1])
        
        # Pad arrays for alignment
        high_close = self.xp.pad(high_close, (1, 0), mode='constant', constant_values=0)
        low_close = self.xp.pad(low_close, (1, 0), mode='constant', constant_values=0)
        
        true_range = self.xp.maximum(high_low, self.xp.maximum(high_close, low_close))
        
        # ATR is RMA of True Range
        atr = self._rma(true_range, period)
        
        return self.to_numpy(atr)
    
    # Volume-Weighted Average Price
    def vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
             volume: np.ndarray) -> np.ndarray:
        """Calculate VWAP on GPU."""
        high_gpu = self.to_gpu(high)
        low_gpu = self.to_gpu(low)
        close_gpu = self.to_gpu(close)
        volume_gpu = self.to_gpu(volume)
        
        # Typical Price
        typical_price = (high_gpu + low_gpu + close_gpu) / 3
        
        # Cumulative calculations
        cum_price_volume = self.xp.cumsum(typical_price * volume_gpu)
        cum_volume = self.xp.cumsum(volume_gpu)
        
        vwap = cum_price_volume / self.xp.where(cum_volume != 0, cum_volume, 1)
        
        return self.to_numpy(vwap)
    
    # Ichimoku Cloud
    def ichimoku_cloud(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, np.ndarray]:
        """Calculate Ichimoku Cloud components on GPU."""
        high_gpu = self.to_gpu(high)
        low_gpu = self.to_gpu(low)
        close_gpu = self.to_gpu(close)
        
        def period_high_low(period: int) -> Tuple[cp.ndarray, cp.ndarray]:
            """Calculate period high/low."""
            period_high = self.xp.zeros_like(high_gpu)
            period_low = self.xp.zeros_like(low_gpu)
            
            for i in range(period, len(high_gpu)):
                period_high[i] = self.xp.max(high_gpu[i-period:i])
                period_low[i] = self.xp.min(low_gpu[i-period:i])
            
            return period_high, period_low
        
        # Tenkan-sen (Conversion Line)
        tenkan_high, tenkan_low = period_high_low(tenkan)
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high, kijun_low = period_high_low(kijun)
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B)
        senkou_b_high, senkou_b_low = period_high_low(senkou_b)
        senkou_span_b = (senkou_b_high + senkou_b_low) / 2
        
        # Chikou Span (Lagging Span)
        chikou_span = close_gpu
        
        return {
            'tenkan_sen': self.to_numpy(tenkan_sen),
            'kijun_sen': self.to_numpy(kijun_sen),
            'senkou_span_a': self.to_numpy(senkou_span_a),
            'senkou_span_b': self.to_numpy(senkou_span_b),
            'chikou_span': self.to_numpy(chikou_span)
        }
    
    # Batch calculate all indicators
    def calculate_all_indicators(self, ohlcv_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate all technical indicators in batch on GPU.
        
        Args:
            ohlcv_data: Dictionary with 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            Dictionary of all calculated indicators
        """
        open_price = ohlcv_data['open']
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close']
        volume = ohlcv_data['volume']
        
        indicators = {}
        
        # Moving averages
        indicators['sma_10'] = self.sma(close, 10)
        indicators['sma_20'] = self.sma(close, 20)
        indicators['sma_50'] = self.sma(close, 50)
        indicators['ema_10'] = self.ema(close, 10)
        indicators['ema_20'] = self.ema(close, 20)
        
        # Oscillators
        indicators['rsi_14'] = self.rsi(close, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.bollinger_bands(close, 20, 2)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # MACD
        macd_line, signal_line, histogram = self.macd(close, 12, 26, 9)
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # Stochastic
        stoch_k, stoch_d = self.stochastic(high, low, close, 14, 3)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        # ATR
        indicators['atr_14'] = self.atr(high, low, close, 14)
        
        # VWAP
        indicators['vwap'] = self.vwap(high, low, close, volume)
        
        # Ichimoku
        ichimoku = self.ichimoku_cloud(high, low, close)
        for key, value in ichimoku.items():
            indicators[f'ichimoku_{key}'] = value
        
        return indicators
    
    def clear_cache(self):
        """Clear LRU caches and GPU memory."""
        # Clear method caches
        if hasattr(self.sma, 'cache_clear'):
            self.sma.cache_clear()
        if hasattr(self.ema, 'cache_clear'):
            self.ema.cache_clear()
        if hasattr(self.rsi, 'cache_clear'):
            self.rsi.cache_clear()
        
        # Clear GPU memory
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
        
        logger.info("Cleared indicator caches and GPU memory")
    
    def benchmark(self, data_size: int = 10000) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance.
        
        Args:
            data_size: Size of test data
            
        Returns:
            Performance comparison metrics
        """
        import time
        
        # Generate test data
        test_data = np.random.randn(data_size)
        
        results = {}
        indicators_to_test = [
            ('SMA_20', lambda d: self.sma(d, 20)),
            ('EMA_20', lambda d: self.ema(d, 20)),
            ('RSI_14', lambda d: self.rsi(d, 14))
        ]
        
        for name, func in indicators_to_test:
            # Warmup
            _ = func(test_data[:100])
            
            # Time execution
            start = time.time()
            for _ in range(10):
                _ = func(test_data)
            gpu_time = (time.time() - start) / 10
            
            results[f"{name}_{'GPU' if self.use_gpu else 'CPU'}"] = gpu_time
        
        return results