import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import warnings
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_time: datetime, end_time: datetime,
                  interval: str = '1m') -> pd.DataFrame:
        """Fetch data from source."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass


class FileDataSource(DataSource):
    """Load data from local files (CSV, JSON, Parquet)."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(self, symbol: str, start_time: datetime, end_time: datetime,
                  interval: str = '1m') -> pd.DataFrame:
        """
        Load data from file for specified symbol and time range.
        
        Expected file format: symbol_interval.csv (e.g., BTCUSDT_1m.csv)
        """
        file_path = self.data_dir / f"{symbol}_{interval}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        
        # Filter by time range
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        df = df[mask].copy()
        
        # Ensure required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from file names."""
        csv_files = list(self.data_dir.glob("*_*.csv"))
        symbols = list(set([f.stem.split('_')[0] for f in csv_files]))
        return sorted(symbols)
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval: str = '1m'):
        """Save data to file."""
        file_path = self.data_dir / f"{symbol}_{interval}.csv"
        df.to_csv(file_path, index=False)


class DataLoader:
    """Main data loader with caching and preprocessing capabilities."""
    
    def __init__(self, data_source: DataSource, cache_dir: Optional[str] = None):
        self.data_source = data_source
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        
    def load_data(self, symbol: str, start_time: Union[str, datetime],
                 end_time: Union[str, datetime], interval: str = '1m',
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Load OHLCV data for specified symbol and time range.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_time: Start time for data
            end_time: End time for data
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert string times to datetime
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        # Check cache
        cache_key = f"{symbol}_{interval}_{start_time}_{end_time}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Load from source
        df = self.data_source.fetch_data(symbol, start_time, end_time, interval)
        
        # Basic validation
        df = self._validate_data(df)
        
        # Add derived features
        df = self._add_basic_features(df)
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = df.copy()
            
        return df
    
    def load_multiple_symbols(self, symbols: List[str], start_time: Union[str, datetime],
                            end_time: Union[str, datetime], interval: str = '1m') -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.load_data(symbol, start_time, end_time, interval)
            except Exception as e:
                warnings.warn(f"Failed to load data for {symbol}: {str(e)}")
        return data
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Check for missing values
        if df.isnull().any().any():
            warnings.warn("Missing values detected in data")
            # Forward fill for price data
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].fillna(method='ffill')
            # Zero fill for volume
            df['volume'] = df['volume'].fillna(0)
        
        # Validate OHLC relationships
        invalid_candles = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_candles.any():
            warnings.warn(f"Found {invalid_candles.sum()} invalid candles")
            # Fix invalid candles
            df.loc[invalid_candles, 'high'] = df.loc[invalid_candles, ['open', 'close', 'high']].max(axis=1)
            df.loc[invalid_candles, 'low'] = df.loc[invalid_candles, ['open', 'close', 'low']].min(axis=1)
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] <= 0).any().any():
            warnings.warn("Zero or negative prices detected")
            # Remove rows with invalid prices
            df = df[(df[price_cols] > 0).all(axis=1)]
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features to OHLCV data."""
        # Price-based features
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['weighted_price'] = (df['high'] + df['low'] + 2 * df['close']) / 4
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = df['price_change'] / df['open']
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume features
        df['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def resample_data(self, df: pd.DataFrame, source_interval: str,
                     target_interval: str) -> pd.DataFrame:
        """
        Resample data from one interval to another.
        
        Args:
            df: DataFrame with OHLCV data
            source_interval: Source interval (e.g., '1m')
            target_interval: Target interval (e.g., '5m')
            
        Returns:
            Resampled DataFrame
        """
        # Parse intervals
        interval_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        
        if source_interval not in interval_map or target_interval not in interval_map:
            raise ValueError(f"Invalid interval. Supported: {list(interval_map.keys())}")
        
        # Set timestamp as index
        df_resampled = df.set_index('timestamp')
        
        # Resample OHLCV
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_resampled = df_resampled.resample(interval_map[target_interval]).agg(agg_rules)
        
        # Remove incomplete candles
        df_resampled = df_resampled.dropna()
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        # Recalculate derived features
        df_resampled = self._add_basic_features(df_resampled)
        
        return df_resampled
    
    def get_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2,
                           gap_periods: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets with optional gap.
        
        Args:
            df: DataFrame with time series data
            test_size: Fraction of data for testing
            gap_periods: Number of periods to skip between train and test
            
        Returns:
            Tuple of (train_df, test_df)
        """
        n = len(df)
        test_periods = int(n * test_size)
        train_periods = n - test_periods - gap_periods
        
        if train_periods <= 0:
            raise ValueError("Not enough data for train/test split with specified gap")
        
        train_df = df.iloc[:train_periods].copy()
        test_df = df.iloc[train_periods + gap_periods:].copy()
        
        return train_df, test_df
    
    def create_sliding_windows(self, df: pd.DataFrame, window_size: int,
                             prediction_horizon: int = 1,
                             stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for time series prediction.
        
        Args:
            df: DataFrame with features
            window_size: Size of input window
            prediction_horizon: Number of steps to predict ahead
            stride: Step size between windows
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Select feature columns (exclude metadata)
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'symbol', 'interval']]
        
        data = df[feature_cols].values
        
        X, y = [], []
        
        for i in range(0, len(data) - window_size - prediction_horizon + 1, stride):
            X.append(data[i:i + window_size])
            # Predict close price
            close_idx = feature_cols.index('close')
            y.append(data[i + window_size + prediction_horizon - 1, close_idx])
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax',
                      feature_range: Tuple[float, float] = (0, 1)) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize data using specified method.
        
        Args:
            df: DataFrame to normalize
            method: 'minmax', 'standard', or 'robust'
            feature_range: Range for minmax scaling
            
        Returns:
            Tuple of (normalized_df, scaler_params)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'timestamp']
        
        df_norm = df.copy()
        scaler_params = {}
        
        for col in numeric_cols:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                    df_norm[col] = df_norm[col] * (feature_range[1] - feature_range[0]) + feature_range[0]
                scaler_params[col] = {'min': min_val, 'max': max_val, 'feature_range': feature_range}
                
            elif method == 'standard':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df_norm[col] = (df[col] - mean) / std
                scaler_params[col] = {'mean': mean, 'std': std}
                
            elif method == 'robust':
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    df_norm[col] = (df[col] - median) / iqr
                scaler_params[col] = {'median': median, 'iqr': iqr}
        
        scaler_params['method'] = method
        scaler_params['columns'] = list(numeric_cols)
        
        return df_norm, scaler_params
    
    def denormalize_data(self, df: pd.DataFrame, scaler_params: Dict) -> pd.DataFrame:
        """Reverse normalization using saved parameters."""
        df_denorm = df.copy()
        method = scaler_params['method']
        
        for col in scaler_params['columns']:
            if col not in df.columns:
                continue
                
            params = scaler_params[col]
            
            if method == 'minmax':
                feature_range = params['feature_range']
                df_denorm[col] = (df[col] - feature_range[0]) / (feature_range[1] - feature_range[0])
                df_denorm[col] = df_denorm[col] * (params['max'] - params['min']) + params['min']
                
            elif method == 'standard':
                df_denorm[col] = df[col] * params['std'] + params['mean']
                
            elif method == 'robust':
                df_denorm[col] = df[col] * params['iqr'] + params['median']
        
        return df_denorm
    
    def clear_cache(self):
        """Clear in-memory cache."""
        self._cache.clear()
        
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        return {
            'num_cached': len(self._cache),
            'cache_keys': list(self._cache.keys()),
            'memory_usage_mb': sum(df.memory_usage().sum() for df in self._cache.values()) / 1e6
        }


def create_synthetic_data(symbol: str = 'BTCUSDT', 
                         start_time: datetime = datetime(2023, 1, 1),
                         end_time: datetime = datetime(2023, 12, 31),
                         interval: str = '1m',
                         initial_price: float = 30000.0,
                         volatility: float = 0.02,
                         trend: float = 0.0001) -> pd.DataFrame:
    """
    Create synthetic OHLCV data for testing.
    
    Args:
        symbol: Symbol name
        start_time: Start time
        end_time: End time
        interval: Time interval
        initial_price: Starting price
        volatility: Price volatility (std dev of returns)
        trend: Drift trend
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    # Generate timestamps
    freq_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': 'H', '4h': '4H', '1d': 'D'
    }
    
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq_map[interval])
    n = len(timestamps)
    
    # Generate price series using geometric Brownian motion
    dt = 1 / (60 * 24)  # 1 minute as fraction of day
    returns = np.random.normal(trend * dt, volatility * np.sqrt(dt), n)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from price series
    data = []
    for i in range(n):
        # Add intrabar volatility
        intrabar_vol = volatility * 0.5
        high_offset = abs(np.random.normal(0, intrabar_vol))
        low_offset = -abs(np.random.normal(0, intrabar_vol))
        
        open_price = prices[i] * (1 + np.random.normal(0, intrabar_vol * 0.1))
        close_price = prices[i]
        high_price = max(open_price, close_price) * (1 + high_offset)
        low_price = min(open_price, close_price) * (1 + low_offset)
        
        # Generate volume (correlated with price changes)
        base_volume = 1000000
        volume_multiplier = 1 + abs(returns[i]) * 50  # Higher volume on larger moves
        volume = base_volume * volume_multiplier * (1 + np.random.normal(0, 0.3))
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': max(0, volume)
        })
    
    df = pd.DataFrame(data)
    return df