import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import warnings
from abc import ABC, abstractmethod
from .data_loader import DataLoader, DataSource, FileDataSource


class BinanceDataSource(DataSource):
    """Load data from Binance CSV format files."""
    
    BINANCE_COLUMNS = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(self, symbol: str, start_time: datetime, end_time: datetime,
                  interval: str = '1m') -> pd.DataFrame:
        """
        Load data from Binance format file for specified symbol and time range.
        
        Expected file format: symbol_interval.csv (e.g., ETHUSDT_15m.csv)
        """
        file_path = self.data_dir / f"{symbol}_{interval}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Check if it's Binance format
        if all(col in df.columns for col in self.BINANCE_COLUMNS[:6]):
            # Convert Binance format to standard format
            df = self._convert_binance_to_standard(df)
        elif 'timestamp' in df.columns:
            # Already in standard format
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("Unknown data format. Expected Binance format or standard OHLCV format.")
        
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
    
    def _convert_binance_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Binance format to standard OHLCV format."""
        # Create new dataframe with standard columns
        standard_df = pd.DataFrame()
        
        # Convert timestamp
        if df['Open time'].dtype == 'object':
            standard_df['timestamp'] = pd.to_datetime(df['Open time'])
        else:
            # Assume it's Unix timestamp in milliseconds
            standard_df['timestamp'] = pd.to_datetime(df['Open time'], unit='ms')
        
        # Map price and volume columns
        standard_df['open'] = df['Open'].astype(float)
        standard_df['high'] = df['High'].astype(float)
        standard_df['low'] = df['Low'].astype(float)
        standard_df['close'] = df['Close'].astype(float)
        standard_df['volume'] = df['Volume'].astype(float)
        
        # Add additional Binance-specific columns if needed
        if 'Quote asset volume' in df.columns:
            standard_df['quote_volume'] = df['Quote asset volume'].astype(float)
        if 'Number of trades' in df.columns:
            standard_df['num_trades'] = df['Number of trades'].astype(int)
        if 'Taker buy base asset volume' in df.columns:
            standard_df['taker_buy_volume'] = df['Taker buy base asset volume'].astype(float)
        if 'Taker buy quote asset volume' in df.columns:
            standard_df['taker_buy_quote_volume'] = df['Taker buy quote asset volume'].astype(float)
        
        return standard_df
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from file names."""
        csv_files = list(self.data_dir.glob("*_*.csv"))
        symbols = list(set([f.stem.split('_')[0] for f in csv_files]))
        return sorted(symbols)


class UniversalDataSource(DataSource):
    """Universal data source that auto-detects format (Binance or standard)."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.file_source = FileDataSource(data_dir)
        self.binance_source = BinanceDataSource(data_dir)
        
    def fetch_data(self, symbol: str, start_time: datetime, end_time: datetime,
                  interval: str = '1m') -> pd.DataFrame:
        """Auto-detect format and load data."""
        file_path = self.data_dir / f"{symbol}_{interval}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Read first row to detect format
        df_sample = pd.read_csv(file_path, nrows=1)
        
        # Check if it's Binance format
        if 'Open time' in df_sample.columns and 'Close time' in df_sample.columns:
            return self.binance_source.fetch_data(symbol, start_time, end_time, interval)
        else:
            return self.file_source.fetch_data(symbol, start_time, end_time, interval)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.file_source.get_available_symbols()


class EnhancedDataLoader(DataLoader):
    """
    Enhanced data loader with support for multiple data formats and 
    comprehensive feature engineering.
    """
    
    def __init__(self, data_source: Optional[DataSource] = None, 
                 data_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize enhanced data loader.
        
        Args:
            data_source: Custom data source
            data_dir: Directory containing data files (auto-detect format)
            cache_dir: Directory for caching processed data
        """
        if data_source is None and data_dir is not None:
            data_source = UniversalDataSource(data_dir)
        elif data_source is None:
            raise ValueError("Either data_source or data_dir must be provided")
            
        super().__init__(data_source, cache_dir)
        
    def load_data_with_indicators(self, symbol: str, start_time: Union[str, datetime],
                                 end_time: Union[str, datetime], interval: str = '1m',
                                 indicators: Optional[List[str]] = None,
                                 use_cache: bool = True) -> pd.DataFrame:
        """
        Load data with technical indicators computed.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time for data
            end_time: End time for data  
            interval: Time interval
            indicators: List of indicators to compute (None = all)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        # Load base data
        df = self.load_data(symbol, start_time, end_time, interval, use_cache)
        
        # Import enhanced indicators
        from ..features.enhanced_technical_indicators import EnhancedTechnicalIndicators
        
        # Compute indicators
        if indicators is None:
            # Compute all indicators
            df = EnhancedTechnicalIndicators.compute_all_indicators(df)
        else:
            # Compute selected indicators
            df = self._compute_selected_indicators(df, indicators)
        
        return df
    
    def _compute_selected_indicators(self, df: pd.DataFrame, 
                                   indicators: List[str]) -> pd.DataFrame:
        """Compute only selected indicators."""
        from ..features.enhanced_technical_indicators import EnhancedTechnicalIndicators
        
        result = df.copy()
        eti = EnhancedTechnicalIndicators
        
        # Extract price series
        open_price = result['open']
        high = result['high']
        low = result['low']  
        close = result['close']
        volume = result['volume']
        
        # Map indicator names to computation functions
        indicator_map = {
            'sma': lambda: {
                'SMA_12': eti.sma(close, 12),
                'SMA_26': eti.sma(close, 26),
                'SMA_50': eti.sma(close, 50),
                'SMA_100': eti.sma(close, 100),
                'SMA_200': eti.sma(close, 200)
            },
            'ema': lambda: {
                'EMA_12': eti.ema(close, 12),
                'EMA_26': eti.ema(close, 26)
            },
            'macd': lambda: eti.macd(close),
            'rsi': lambda: {'RSI_14': eti.rsi(close, 14)},
            'stochastic': lambda: eti.stochastic(high, low, close),
            'bollinger_bands': lambda: eti.bollinger_bands(close),
            'atr': lambda: {
                'ATR_14': eti.atr(high, low, close, 14),
                'ATR_14_prev': eti.atr(high, low, close, 14).shift(1)
            },
            'parabolic_sar': lambda: {'Parabolic_SAR': eti.parabolic_sar(high, low)},
            'adx': lambda: eti.adx(high, low, close),
            'ichimoku': lambda: eti.ichimoku_cloud(high, low, close),
            'momentum': lambda: {'Momentum_14': eti.momentum(close, 14)},
            'cmf': lambda: {'CMF_20': eti.cmf(high, low, close, volume, 20)},
            'pivot_points': lambda: eti.pivot_points(high, low, close),
            'support_resistance': lambda: eti.identify_support_resistance(high, low),
            'divergences': lambda: {
                **eti.identify_divergences(close, result.get('MACD', close)),
                **eti.identify_divergences(close, result.get('RSI_14', close))
            },
            'elliott_waves': lambda: eti.detect_elliott_waves_basic(close)
        }
        
        # Compute requested indicators
        for indicator in indicators:
            if indicator in indicator_map:
                try:
                    data = indicator_map[indicator]()
                    if isinstance(data, pd.DataFrame):
                        for col in data.columns:
                            result[col] = data[col]
                    elif isinstance(data, dict):
                        for col, series in data.items():
                            result[col] = series
                except Exception as e:
                    warnings.warn(f"Error computing {indicator}: {str(e)}")
        
        return result
    
    def prepare_ml_features(self, df: pd.DataFrame, 
                          feature_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepare features for machine learning models.
        
        Args:
            df: DataFrame with OHLCV and indicators
            feature_config: Configuration for feature engineering
            
        Returns:
            DataFrame with ML-ready features
        """
        if feature_config is None:
            feature_config = {
                'price_features': True,
                'volume_features': True,
                'technical_features': True,
                'time_features': True,
                'lag_features': True,
                'interaction_features': False
            }
        
        result = df.copy()
        
        # Price-based features
        if feature_config.get('price_features', True):
            result['returns'] = result['close'].pct_change()
            result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
            result['price_range'] = result['high'] - result['low']
            result['price_change'] = result['close'] - result['open']
            result['price_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])
            
        # Volume features
        if feature_config.get('volume_features', True):
            result['volume_change'] = result['volume'].pct_change()
            result['volume_ma_ratio'] = result['volume'] / result['volume'].rolling(20).mean()
            
        # Time features
        if feature_config.get('time_features', True) and 'timestamp' in result.columns:
            result['hour'] = result['timestamp'].dt.hour
            result['day_of_week'] = result['timestamp'].dt.dayofweek
            result['day_of_month'] = result['timestamp'].dt.day
            result['month'] = result['timestamp'].dt.month
            result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
            
        # Lag features
        if feature_config.get('lag_features', True):
            for lag in [1, 2, 3, 5, 10]:
                result[f'returns_lag_{lag}'] = result['returns'].shift(lag)
                result[f'volume_lag_{lag}'] = result['volume'].shift(lag)
                
        # Interaction features
        if feature_config.get('interaction_features', False):
            # Price-volume interaction
            result['price_volume'] = result['close'] * result['volume']
            result['price_volume_ma'] = result['price_volume'].rolling(20).mean()
            
            # Indicator interactions
            if 'RSI_14' in result.columns and 'MACD' in result.columns:
                result['rsi_macd_interaction'] = result['RSI_14'] * result['MACD']
                
        return result
    
    def create_dataset_for_ml(self, df: pd.DataFrame,
                            target_column: str = 'returns',
                            feature_columns: Optional[List[str]] = None,
                            lookback_window: int = 100,
                            forecast_horizon: int = 1,
                            train_test_split: float = 0.8) -> Dict:
        """
        Create dataset ready for machine learning models.
        
        Args:
            df: DataFrame with features
            target_column: Column to predict
            feature_columns: List of feature columns (None = auto-select)
            lookback_window: Number of historical periods for each sample
            forecast_horizon: Number of periods to forecast
            train_test_split: Fraction of data for training
            
        Returns:
            Dictionary with train/test data and metadata
        """
        # Auto-select features if not specified
        if feature_columns is None:
            exclude_cols = ['timestamp', 'symbol', target_column]
            feature_columns = [col for col in df.columns if col not in exclude_cols
                             and not col.startswith('future_')]
        
        # Create sliding windows
        X, y = self.create_sliding_windows(
            df[feature_columns + [target_column]], 
            lookback_window, 
            forecast_horizon
        )
        
        # Train/test split
        split_idx = int(len(X) * train_test_split)
        
        dataset = {
            'X_train': X[:split_idx],
            'y_train': y[:split_idx],
            'X_test': X[split_idx:],
            'y_test': y[split_idx:],
            'feature_columns': feature_columns,
            'target_column': target_column,
            'lookback_window': lookback_window,
            'forecast_horizon': forecast_horizon,
            'train_size': split_idx,
            'test_size': len(X) - split_idx
        }
        
        return dataset