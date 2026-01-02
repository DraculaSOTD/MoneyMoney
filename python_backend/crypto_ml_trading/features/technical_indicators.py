import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
import warnings


class TechnicalIndicators:
    """Custom implementation of technical indicators without external TA libraries."""
    
    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            data: Price series
            period: Number of periods
            
        Returns:
            SMA series
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], period: int, 
            adjust: bool = True) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            data: Price series
            period: Number of periods
            adjust: Whether to use adjust parameter in pandas
            
        Returns:
            EMA series
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.ewm(span=period, adjust=adjust).mean()
    
    @staticmethod
    def wma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """
        Weighted Moving Average.
        
        Args:
            data: Price series
            period: Number of periods
            
        Returns:
            WMA series
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        weights = np.arange(1, period + 1)
        wma = data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return wma
    
    @staticmethod
    def vwma(price: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """
        Volume Weighted Moving Average.
        
        Args:
            price: Price series
            volume: Volume series
            period: Number of periods
            
        Returns:
            VWMA series
        """
        pv = price * volume
        return pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    
    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26,
             signal_period: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with MACD, signal, and histogram
        """
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            data: Price series
            period: Number of periods
            
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period (SMA of %K)
            
        Returns:
            DataFrame with %K and %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'k': k_percent,
            'd': d_percent
        })
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 14) -> pd.Series:
        """
        Williams %R.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods
            
        Returns:
            Williams %R series
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 20) -> pd.Series:
        """
        Commodity Channel Index.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods
            
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20,
                       num_std: float = 2) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Args:
            data: Price series
            period: Number of periods for SMA
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with upper, middle, and lower bands
        """
        middle = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': (upper - lower) / middle,
            'percent_b': (data - lower) / (upper - lower)
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods
            
        Returns:
            ATR series
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        ema_period: int = 20, atr_period: int = 10,
                        multiplier: float = 2) -> pd.DataFrame:
        """
        Keltner Channels.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            ema_period: EMA period for middle line
            atr_period: ATR period
            multiplier: ATR multiplier for bands
            
        Returns:
            DataFrame with upper, middle, and lower channels
        """
        middle = TechnicalIndicators.ema(close, ema_period)
        atr = TechnicalIndicators.atr(high, low, close, atr_period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series,
                     initial_af: float = 0.02,
                     step_af: float = 0.02,
                     max_af: float = 0.2) -> pd.Series:
        """
        Parabolic SAR (Stop and Reverse).
        
        Args:
            high: High price series
            low: Low price series
            initial_af: Initial acceleration factor
            step_af: Step for acceleration factor
            max_af: Maximum acceleration factor
            
        Returns:
            SAR series
        """
        n = len(high)
        sar = np.zeros(n)
        ep = np.zeros(n)  # Extreme point
        af = np.zeros(n)  # Acceleration factor
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = initial_af
        trend[0] = 1
        
        for i in range(1, n):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # Check for reversal
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = initial_af
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + step_af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                        
            else:  # Downtrend
                sar[i] = sar[i-1] - af[i-1] * (sar[i-1] - ep[i-1])
                
                # Check for reversal
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = initial_af
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + step_af, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(sar, index=high.index)
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                      tenkan_period: int = 9, kijun_period: int = 26,
                      senkou_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
        """
        Ichimoku Cloud indicator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            tenkan_period: Conversion line period
            kijun_period: Base line period
            senkou_b_period: Leading span B period
            displacement: Cloud displacement
            
        Returns:
            DataFrame with all Ichimoku components
        """
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume.
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        price_diff = close.diff()
        volume_direction = np.where(price_diff > 0, volume,
                                   np.where(price_diff < 0, -volume, 0))
        obv = volume_direction.cumsum()
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Volume Weighted Average Price.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Rolling period (None for cumulative)
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        
        if period is None:
            # Cumulative VWAP (resets each day in practice)
            cumulative_pv = (typical_price * volume).cumsum()
            cumulative_volume = volume.cumsum()
            vwap = cumulative_pv / cumulative_volume
        else:
            # Rolling VWAP
            pv = typical_price * volume
            vwap = pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            
        return vwap
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Money Flow Index.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Number of periods
            
        Returns:
            MFI series
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        price_diff = typical_price.diff()
        positive_flow = money_flow.where(price_diff > 0, 0)
        negative_flow = money_flow.where(price_diff < 0, 0)
        
        # Money flow ratio
        positive_sum = positive_flow.rolling(window=period).sum()
        negative_sum = negative_flow.rolling(window=period).sum()
        
        mfr = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    @staticmethod
    def accumulation_distribution(high: pd.Series, low: pd.Series,
                                 close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            A/D line series
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        
        ad = (clv * volume).cumsum()
        return ad
    
    @staticmethod
    def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Number of periods
            
        Returns:
            CMF series
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        
        money_flow_volume = clv * volume
        cmf = (money_flow_volume.rolling(window=period).sum() / 
               volume.rolling(window=period).sum())
        
        return cmf
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series,
                         period: int = 20) -> pd.DataFrame:
        """
        Donchian Channels.
        
        Args:
            high: High price series
            low: Low price series
            period: Number of periods
            
        Returns:
            DataFrame with upper, middle, and lower channels
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.DataFrame:
        """
        Classic Pivot Points.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            DataFrame with pivot points and support/resistance levels
        """
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return pd.DataFrame({
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        })
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for OHLCV data.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all indicators added
        """
        result = df.copy()
        
        # Basic price-derived features
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Trend indicators
        result['sma_10'] = TechnicalIndicators.sma(df['close'], 10)
        result['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        result['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        result['ema_10'] = TechnicalIndicators.ema(df['close'], 10)
        result['ema_20'] = TechnicalIndicators.ema(df['close'], 20)
        
        # MACD
        macd = TechnicalIndicators.macd(df['close'])
        result['macd'] = macd['macd']
        result['macd_signal'] = macd['signal']
        result['macd_histogram'] = macd['histogram']
        
        # Momentum indicators
        result['rsi'] = TechnicalIndicators.rsi(df['close'])
        
        stoch = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        result['stoch_k'] = stoch['k']
        result['stoch_d'] = stoch['d']
        
        result['williams_r'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
        result['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        bb = TechnicalIndicators.bollinger_bands(df['close'])
        result['bb_upper'] = bb['upper']
        result['bb_middle'] = bb['middle']
        result['bb_lower'] = bb['lower']
        result['bb_bandwidth'] = bb['bandwidth']
        result['bb_percent'] = bb['percent_b']
        
        result['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        kc = TechnicalIndicators.keltner_channels(df['high'], df['low'], df['close'])
        result['kc_upper'] = kc['upper']
        result['kc_middle'] = kc['middle']
        result['kc_lower'] = kc['lower']
        
        # Volume indicators
        if 'volume' in df.columns:
            result['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
            result['mfi'] = TechnicalIndicators.mfi(df['high'], df['low'], df['close'], df['volume'])
            result['ad_line'] = TechnicalIndicators.accumulation_distribution(
                df['high'], df['low'], df['close'], df['volume']
            )
            result['cmf'] = TechnicalIndicators.chaikin_money_flow(
                df['high'], df['low'], df['close'], df['volume']
            )
            
        return result