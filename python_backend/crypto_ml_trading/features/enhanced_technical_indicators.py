import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict
import warnings
from tqdm import tqdm


class EnhancedTechnicalIndicators:
    """
    Comprehensive technical indicators implementation combining features from multiple projects.
    Includes standard indicators, Japanese techniques, pattern detection, and market microstructure.
    """
    
    # Moving Averages
    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """Simple Moving Average."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], period: int, 
            adjust: bool = True) -> pd.Series:
        """Exponential Moving Average."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.ewm(span=period, adjust=adjust).mean()
    
    @staticmethod
    def wma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """Weighted Moving Average."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def vwma(price: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Volume Weighted Moving Average."""
        pv = price * volume
        return pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    # MACD
    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26,
             signal_period: int = 9) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = EnhancedTechnicalIndicators.ema(data, fast_period)
        ema_slow = EnhancedTechnicalIndicators.ema(data, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = EnhancedTechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal_Line': signal_line,
            'MACD_Histogram': histogram,
            '12_EMA': ema_fast,
            '26_EMA': ema_slow
        })
    
    # RSI
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # Stochastic Oscillator
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator (%K and %D)."""
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - low_min) / (high_max - low_min))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return pd.DataFrame({
            '%K': k_percent,
            '%D': d_percent
        })
    
    # Bollinger Bands
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, 
                       num_std: float = 2) -> pd.DataFrame:
        """Bollinger Bands."""
        middle_band = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        return pd.DataFrame({
            'Middle_Band': middle_band,
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'BB_Width': upper_band - lower_band,
            'BB_Percent': (data - lower_band) / (upper_band - lower_band)
        })
    
    # ATR (Average True Range)
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average True Range."""
        high_minus_low = high - low
        high_minus_prev_close = (high - close.shift()).abs()
        low_minus_prev_close = (low - close.shift()).abs()
        
        true_range = pd.concat([high_minus_low, high_minus_prev_close, 
                               low_minus_prev_close], axis=1).max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    # Parabolic SAR
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, 
                      start_af: float = 0.02, increment_af: float = 0.02, 
                      max_af: float = 0.20) -> pd.Series:
        """Parabolic SAR."""
        data_length = len(high)
        sar = low.iloc[0]
        ep = high.iloc[0]
        af = start_af
        uptrend = True
        sar_values = [sar]
        
        for i in range(1, data_length):
            prev_sar = sar
            
            if uptrend:
                sar = prev_sar + af * (ep - prev_sar)
                if low.iloc[i] <= sar:
                    uptrend = False
                    sar = ep
                    af = start_af
                    ep = low.iloc[i]
            else:
                sar = prev_sar - af * (prev_sar - ep)
                if high.iloc[i] >= sar:
                    uptrend = True
                    sar = ep
                    af = start_af
                    ep = high.iloc[i]
            
            sar_values.append(sar)
            
            if uptrend:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(max_af, af + increment_af)
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(max_af, af + increment_af)
        
        return pd.Series(sar_values, index=high.index)
    
    # ADX (Average Directional Index)
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.DataFrame:
        """Average Directional Index with +DI and -DI."""
        # Calculate True Range
        high_minus_low = high - low
        high_minus_prev_close = (high - close.shift()).abs()
        low_minus_prev_close = (low - close.shift()).abs()
        true_range = pd.concat([high_minus_low, high_minus_prev_close, 
                               low_minus_prev_close], axis=1).max(axis=1)
        
        # Calculate +DM and -DM
        pos_dm = high.diff().where(high.diff() > low.diff(), 0).clip(lower=0)
        neg_dm = low.diff().where(low.diff() > high.diff(), 0).clip(lower=0).abs()
        
        # Smooth the TR, +DM, and -DM
        smooth_tr = true_range.rolling(window=period).sum()
        smooth_pos_dm = pos_dm.rolling(window=period).sum()
        smooth_neg_dm = neg_dm.rolling(window=period).sum()
        
        # Calculate +DI and -DI
        pos_di = (smooth_pos_dm / smooth_tr) * 100
        neg_di = (smooth_neg_dm / smooth_tr) * 100
        
        # Calculate DX and ADX
        dx = ((pos_di - neg_di).abs() / (pos_di + neg_di)) * 100
        adx = dx.rolling(window=period).mean()
        
        return pd.DataFrame({
            f'ADX_{period}': adx,
            f'+DI_{period}': pos_di,
            f'-DI_{period}': neg_di
        })
    
    # Ichimoku Cloud
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                      conversion_period: int = 9, base_period: int = 26,
                      lead_span_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
        """Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=conversion_period).max() + 
                     low.rolling(window=conversion_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=base_period).max() + 
                    low.rolling(window=base_period).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=lead_span_b_period).max() + 
                         low.rolling(window=lead_span_b_period).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return pd.DataFrame({
            'Tenkan_sen': tenkan_sen,
            'Kijun_sen': kijun_sen,
            'Senkou_Span_A': senkou_span_a,
            'Senkou_Span_B': senkou_span_b,
            'Chikou_Span': chikou_span
        })
    
    # Momentum
    @staticmethod
    def momentum(data: pd.Series, period: int = 14) -> pd.Series:
        """Momentum indicator."""
        return data - data.shift(period)
    
    # Chaikin Money Flow
    @staticmethod
    def cmf(high: pd.Series, low: pd.Series, close: pd.Series, 
            volume: pd.Series, period: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Handle cases where high equals low
        mfv = mfm * volume
        
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    # Pivot Points
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """Standard Pivot Points with support and resistance levels."""
        pivot = (high + low + close) / 3
        
        support_1 = 2 * pivot - high
        support_2 = pivot - (high - low)
        resistance_1 = 2 * pivot - low
        resistance_2 = pivot + (high - low)
        
        return pd.DataFrame({
            'Pivot_Point': pivot,
            'Support_1': support_1,
            'Support_2': support_2,
            'Resistance_1': resistance_1,
            'Resistance_2': resistance_2
        })
    
    # Support and Resistance Detection
    @staticmethod
    def identify_support_resistance(high: pd.Series, low: pd.Series, 
                                   window: int = 20) -> pd.DataFrame:
        """Identify dynamic support and resistance levels."""
        # Detect local maxima and minima
        resistance = (high > high.shift(-window)) & (high > high.shift(window))
        support = (low < low.shift(-window)) & (low < low.shift(window))
        
        # Assign levels, setting to 0 where not found
        resistance_level = high.where(resistance, 0)
        support_level = low.where(support, 0)
        
        return pd.DataFrame({
            'Resistance_Level': resistance_level,
            'Support_Level': support_level
        })
    
    # Divergence Detection
    @staticmethod
    def identify_divergences(price: pd.Series, indicator: pd.Series, 
                           window: int = 5) -> pd.DataFrame:
        """Identify bullish and bearish divergences."""
        # Detect local maxima and minima
        price_maxima = (price > price.shift(-window)) & (price > price.shift(window))
        price_minima = (price < price.shift(-window)) & (price < price.shift(window))
        indicator_maxima = (indicator > indicator.shift(-window)) & (indicator > indicator.shift(window))
        indicator_minima = (indicator < indicator.shift(-window)) & (indicator < indicator.shift(window))
        
        # Identify divergences
        bullish_divergence = price_minima & ~indicator_minima
        bearish_divergence = price_maxima & ~indicator_maxima
        
        return pd.DataFrame({
            'Bullish_Divergence': bullish_divergence.astype(int),
            'Bearish_Divergence': bearish_divergence.astype(int)
        })
    
    # Elliott Wave Detection (Basic)
    @staticmethod
    def detect_elliott_waves_basic(close: pd.Series, window: int = 5) -> pd.DataFrame:
        """Basic Elliott Wave pattern detection."""
        # Detect peaks and troughs
        peaks = (close > close.shift(-window)) & (close > close.shift(window))
        troughs = (close < close.shift(-window)) & (close < close.shift(window))
        
        # Create sequence (1 for peak, -1 for trough, 0 otherwise)
        sequence = peaks.astype(int) - troughs.astype(int)
        
        # Placeholder for Elliott Wave sequence identification
        elliott_wave_sequence = pd.Series(0, index=close.index)
        
        # Simple pattern detection (can be enhanced)
        for i in range(len(sequence) - 8):
            sub_seq = sequence.iloc[i:i+8].values
            # Check for basic 5-3 pattern
            if np.array_equal(sub_seq[:5], [1, 0, 1, 0, 1]) and np.array_equal(sub_seq[5:], [-1, 0, -1]):
                elliott_wave_sequence.iloc[i:i+8] = 1
            elif np.array_equal(sub_seq[:5], [-1, 0, -1, 0, -1]) and np.array_equal(sub_seq[5:], [1, 0, 1]):
                elliott_wave_sequence.iloc[i:i+8] = -1
        
        return pd.DataFrame({
            'Peak': peaks.astype(int),
            'Trough': troughs.astype(int),
            'Elliott_Wave_Sequence': elliott_wave_sequence
        })
    
    # Comprehensive Feature Engineering
    @staticmethod
    def compute_all_indicators(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compute all technical indicators for OHLCV data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            config: Optional configuration dict for indicator parameters
            
        Returns:
            DataFrame with all computed indicators
        """
        if config is None:
            config = {}
        
        # Create a copy to avoid modifying original
        result = df.copy()
        
        # Extract price series
        open_price = result['open']
        high = result['high']
        low = result['low']
        close = result['close']
        volume = result['volume']
        
        # Progress bar for indicator computation
        indicators = [
            ('Moving Averages', lambda: {
                'SMA_12': EnhancedTechnicalIndicators.sma(close, 12),
                'SMA_26': EnhancedTechnicalIndicators.sma(close, 26),
                'SMA_50': EnhancedTechnicalIndicators.sma(close, 50),
                'SMA_100': EnhancedTechnicalIndicators.sma(close, 100),
                'SMA_200': EnhancedTechnicalIndicators.sma(close, 200),
                'EMA_12': EnhancedTechnicalIndicators.ema(close, 12),
                'EMA_26': EnhancedTechnicalIndicators.ema(close, 26),
                'VWMA_20': EnhancedTechnicalIndicators.vwma(close, volume, 20)
            }),
            ('MACD', lambda: EnhancedTechnicalIndicators.macd(close)),
            ('RSI', lambda: {'RSI_14': EnhancedTechnicalIndicators.rsi(close, 14)}),
            ('Stochastic', lambda: EnhancedTechnicalIndicators.stochastic(high, low, close)),
            ('Bollinger Bands', lambda: EnhancedTechnicalIndicators.bollinger_bands(close)),
            ('ATR', lambda: {
                'ATR_14': EnhancedTechnicalIndicators.atr(high, low, close, 14),
                'ATR_14_prev': EnhancedTechnicalIndicators.atr(high, low, close, 14).shift(1)
            }),
            ('Parabolic SAR', lambda: {'Parabolic_SAR': EnhancedTechnicalIndicators.parabolic_sar(high, low)}),
            ('ADX', lambda: EnhancedTechnicalIndicators.adx(high, low, close)),
            ('Ichimoku', lambda: EnhancedTechnicalIndicators.ichimoku_cloud(high, low, close)),
            ('Momentum', lambda: {'Momentum_14': EnhancedTechnicalIndicators.momentum(close, 14)}),
            ('CMF', lambda: {'CMF_20': EnhancedTechnicalIndicators.cmf(high, low, close, volume, 20)}),
            ('Pivot Points', lambda: EnhancedTechnicalIndicators.pivot_points(high, low, close)),
            ('Support/Resistance', lambda: EnhancedTechnicalIndicators.identify_support_resistance(high, low)),
        ]
        
        # Compute indicators with progress bar
        for name, func in tqdm(indicators, desc="Computing Technical Indicators"):
            try:
                indicator_data = func()
                if isinstance(indicator_data, pd.DataFrame):
                    for col in indicator_data.columns:
                        result[col] = indicator_data[col]
                elif isinstance(indicator_data, pd.Series):
                    result[name] = indicator_data
                elif isinstance(indicator_data, dict):
                    for col, data in indicator_data.items():
                        result[col] = data
            except Exception as e:
                warnings.warn(f"Error computing {name}: {str(e)}")
        
        # Add lagged features
        result['Close_prev'] = close.shift(1)
        result['Prev_Volume'] = volume.shift(1)
        
        # Compute divergences for MACD and RSI
        if 'MACD' in result.columns and 'RSI_14' in result.columns:
            macd_div = EnhancedTechnicalIndicators.identify_divergences(close, result['MACD'])
            rsi_div = EnhancedTechnicalIndicators.identify_divergences(close, result['RSI_14'])
            
            result['Bullish_MACD_Divergence'] = macd_div['Bullish_Divergence']
            result['Bearish_MACD_Divergence'] = macd_div['Bearish_Divergence']
            result['Bullish_RSI_Divergence'] = rsi_div['Bullish_Divergence']
            result['Bearish_RSI_Divergence'] = rsi_div['Bearish_Divergence']
        
        # Elliott Wave detection
        elliott = EnhancedTechnicalIndicators.detect_elliott_waves_basic(close)
        for col in elliott.columns:
            result[col] = elliott[col]
        
        return result