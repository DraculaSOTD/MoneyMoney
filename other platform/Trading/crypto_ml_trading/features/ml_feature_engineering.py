import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings


class MLFeatureEngineering:
    """
    Advanced feature engineering for machine learning models.
    Includes percentage changes, lagged features, and transformations
    inspired by the ML project.
    """
    
    def __init__(self):
        """Initialize feature engineering module."""
        self.scalers = {}
        self.feature_names = []
        
    def create_percentage_features(self, df: pd.DataFrame, 
                                 price_cols: Optional[List[str]] = None,
                                 volume_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert price and volume data to percentage changes for stationarity.
        
        Args:
            df: DataFrame with OHLCV data
            price_cols: Price columns to convert (default: OHLC)
            volume_cols: Volume columns to convert
            
        Returns:
            DataFrame with percentage change features
        """
        result = df.copy()
        
        # Default price columns
        if price_cols is None:
            price_cols = ['open', 'high', 'low', 'close']
        
        # Default volume columns
        if volume_cols is None:
            volume_cols = ['volume']
        
        # Price percentage changes
        for col in price_cols:
            if col in result.columns:
                # Percentage change
                result[f'{col}_pct'] = result[col].pct_change()
                
                # Log returns for close price
                if col == 'close':
                    result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
                    
                # Price ratios
                if col != 'close':
                    result[f'{col}_to_close'] = result[col] / result['close']
        
        # Volume percentage changes
        for col in volume_cols:
            if col in result.columns:
                result[f'{col}_pct'] = result[col].pct_change()
                
        # OHLC specific features
        if all(col in result.columns for col in ['open', 'high', 'low', 'close']):
            # Garman-Klass volatility estimator
            result['gk_volatility'] = np.sqrt(
                0.5 * np.log(result['high'] / result['low'])**2 - 
                (2 * np.log(2) - 1) * np.log(result['close'] / result['open'])**2
            )
            
            # Parkinson volatility
            result['parkinson_vol'] = np.sqrt(
                np.log(result['high'] / result['low'])**2 / (4 * np.log(2))
            )
            
            # Price position in range
            result['price_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])
            
        return result
    
    def create_lagged_features(self, df: pd.DataFrame,
                             feature_cols: Optional[List[str]] = None,
                             lags: List[int] = [1, 2, 3, 5, 10],
                             include_diff: bool = True) -> pd.DataFrame:
        """
        Create lagged features for time series modeling.
        
        Args:
            df: DataFrame with features
            feature_cols: Columns to create lags for
            lags: List of lag periods
            include_diff: Include differenced features
            
        Returns:
            DataFrame with lagged features
        """
        result = df.copy()
        
        # Default features to lag
        if feature_cols is None:
            feature_cols = []
            # Price-based
            if 'close_pct' in result.columns:
                feature_cols.append('close_pct')
            elif 'returns' in result.columns:
                feature_cols.append('returns')
            
            # Volume
            if 'volume_pct' in result.columns:
                feature_cols.append('volume_pct')
            elif 'volume' in result.columns:
                feature_cols.append('volume')
            
            # Indicators
            for col in ['RSI_14', 'MACD', 'ATR_14']:
                if col in result.columns:
                    feature_cols.append(col)
        
        # Create lags
        for col in feature_cols:
            if col not in result.columns:
                continue
                
            for lag in lags:
                # Lagged value
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
                
                # Differenced value
                if include_diff:
                    result[f'{col}_diff_{lag}'] = result[col] - result[col].shift(lag)
        
        # Special time-based lags (T1, T2 from ML project)
        if 'close' in result.columns:
            result['T1'] = result['close'].shift(1)  # Previous close
            result['T2'] = result['close'].shift(2)  # 2 periods ago
            
        return result
    
    def create_rolling_features(self, df: pd.DataFrame,
                              windows: List[int] = [5, 10, 20, 50],
                              features: List[str] = ['returns']) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            df: DataFrame with base features
            windows: Rolling window sizes
            features: Features to compute rolling stats for
            
        Returns:
            DataFrame with rolling features
        """
        result = df.copy()
        
        # Ensure we have returns
        if 'returns' not in result.columns and 'close' in result.columns:
            result['returns'] = result['close'].pct_change()
        
        for feat in features:
            if feat not in result.columns:
                continue
                
            for window in windows:
                # Rolling statistics
                result[f'{feat}_mean_{window}'] = result[feat].rolling(window).mean()
                result[f'{feat}_std_{window}'] = result[feat].rolling(window).std()
                result[f'{feat}_skew_{window}'] = result[feat].rolling(window).skew()
                result[f'{feat}_kurt_{window}'] = result[feat].rolling(window).kurt()
                
                # Min/Max
                result[f'{feat}_min_{window}'] = result[feat].rolling(window).min()
                result[f'{feat}_max_{window}'] = result[feat].rolling(window).max()
                
                # Cumulative return for returns feature
                if feat == 'returns':
                    result[f'cum_return_{window}'] = (1 + result[feat]).rolling(window).apply(
                        lambda x: x.prod() - 1, raw=True
                    )
        
        # 30-day cumulative return (from ML project)
        if 'returns' in result.columns:
            result['30_Ret'] = (1 + result['returns']).rolling(30).apply(
                lambda x: x.prod() - 1, raw=True
            )
            
        # Average range
        if all(col in result.columns for col in ['high', 'low']):
            result['avg_range_10'] = (result['high'] - result['low']).rolling(10).mean()
            
        # Additional features from ML project
        # Rolling cumulative returns (Roll_Rets)
        if 'returns' in result.columns:
            result['Roll_Rets'] = result['returns'].rolling(30).sum()
            
        # Average range over 30 days (Avg_Range)
        if 'price_range' in result.columns:
            result['Avg_Range'] = result['price_range'].rolling(30).mean()
        elif all(col in result.columns for col in ['high', 'low']):
            result['Avg_Range'] = (result['high'] - result['low']).rolling(30).mean()
            
        return result
    
    def create_time_features(self, df: pd.DataFrame,
                           timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Create time-based features including Unix timestamp.
        
        Args:
            df: DataFrame with timestamp
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with time features
        """
        result = df.copy()
        
        if timestamp_col not in result.columns:
            warnings.warn(f"Timestamp column '{timestamp_col}' not found")
            return result
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
            result[timestamp_col] = pd.to_datetime(result[timestamp_col])
        
        # Unix timestamp (as in ML project)
        result['unix_timestamp'] = result[timestamp_col].astype('int64') / 10**9
        
        # Time features
        result['hour'] = result[timestamp_col].dt.hour
        result['day_of_week'] = result[timestamp_col].dt.dayofweek
        result['day_of_month'] = result[timestamp_col].dt.day
        result['month'] = result[timestamp_col].dt.month
        result['quarter'] = result[timestamp_col].dt.quarter
        
        # Day of week encoding (DOW)
        result['DOW'] = result['day_of_week']
        
        # Binary features
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        result['is_month_start'] = result[timestamp_col].dt.is_month_start.astype(int)
        result['is_month_end'] = result[timestamp_col].dt.is_month_end.astype(int)
        
        # Trading sessions
        result['session'] = pd.cut(
            result['hour'],
            bins=[-1, 8, 16, 24],
            labels=['asian', 'european', 'american']
        )
        
        # One-hot encode session
        session_dummies = pd.get_dummies(result['session'], prefix='session')
        result = pd.concat([result, session_dummies], axis=1)
        
        return result
    
    def create_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with derived features
        """
        result = df.copy()
        
        # RSI-based features
        if 'RSI_14' in result.columns:
            # RSI returns (RSI_Ret from ML project)
            result['RSI_Ret'] = result['RSI_14'].pct_change()
            
            # RSI categories
            result['RSI_oversold'] = (result['RSI_14'] < 30).astype(int)
            result['RSI_overbought'] = (result['RSI_14'] > 70).astype(int)
            
        # MACD features
        if all(col in result.columns for col in ['MACD', 'Signal_Line']):
            # MACD histogram
            result['MACD_histogram'] = result['MACD'] - result['Signal_Line']
            
            # MACD signal
            result['MACD_signal'] = (result['MACD'] > result['Signal_Line']).astype(int)
            result['MACD_signal_change'] = result['MACD_signal'].diff()
            
        # Bollinger Band features
        if all(col in result.columns for col in ['close', 'Upper_Band', 'Lower_Band']):
            # Position in bands
            bb_width = result['Upper_Band'] - result['Lower_Band']
            result['BB_position'] = (result['close'] - result['Lower_Band']) / bb_width
            
            # Band squeeze
            result['BB_squeeze'] = bb_width / result['close']
            
        # Moving average features
        ma_pairs = [
            ('SMA_12', 'SMA_26'),
            ('SMA_50', 'SMA_200'),
            ('EMA_12', 'EMA_26')
        ]
        
        for short_ma, long_ma in ma_pairs:
            if short_ma in result.columns and long_ma in result.columns:
                # Crossover signal
                result[f'{short_ma}_{long_ma}_signal'] = (
                    result[short_ma] > result[long_ma]
                ).astype(int)
                
                # Distance between MAs
                result[f'{short_ma}_{long_ma}_spread'] = (
                    result[short_ma] - result[long_ma]
                ) / result[long_ma]
        
        # Volume indicators
        if 'volume' in result.columns:
            # Volume Oscillator (from ML project)
            result['Vol_Osc'] = (
                result['volume'].rolling(12).mean() - 
                result['volume'].rolling(26).mean()
            )
            
            # Volume rate of change
            result['Volume_ROC'] = result['volume'].pct_change(10)
            
        # Fibonacci retracement levels (from ML project)
        if all(col in result.columns for col in ['high', 'low', 'close']):
            # Calculate over rolling window
            window = 50
            max_price = result['high'].rolling(window).max()
            min_price = result['low'].rolling(window).min()
            price_range = max_price - min_price
            
            result['Fib_236'] = min_price + 0.236 * price_range
            result['Fib_382'] = min_price + 0.382 * price_range
            result['Fib_618'] = min_price + 0.618 * price_range
            
            # Price position relative to Fib levels
            result['Price_to_Fib236'] = (result['close'] - result['Fib_236']) / result['close']
            result['Price_to_Fib382'] = (result['close'] - result['Fib_382']) / result['close']
            result['Price_to_Fib618'] = (result['close'] - result['Fib_618']) / result['close']
            
        # Close rate of change (Close_Rt from RL project)
        if 'close' in result.columns:
            result['Close_Rt'] = result['close'].pct_change()
            
        # Benchmark returns (if benchmark data available)
        # This would need to be passed in or calculated separately
        # result['Bench_C_Rets'] = benchmark_returns
        
        return result
    
    def prepare_ml_features(self, df: pd.DataFrame,
                          feature_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Complete ML feature preparation pipeline.
        
        Args:
            df: DataFrame with OHLCV and indicators
            feature_config: Configuration for feature engineering
            
        Returns:
            DataFrame with all ML features
        """
        if feature_config is None:
            feature_config = {
                'percentage_features': True,
                'lagged_features': True,
                'rolling_features': True,
                'time_features': True,
                'indicator_features': True
            }
        
        result = df.copy()
        
        # 1. Percentage features (for stationarity)
        if feature_config.get('percentage_features', True):
            result = self.create_percentage_features(result)
        
        # 2. Indicator features
        if feature_config.get('indicator_features', True):
            result = self.create_indicator_features(result)
        
        # 3. Lagged features
        if feature_config.get('lagged_features', True):
            result = self.create_lagged_features(result)
        
        # 4. Rolling features
        if feature_config.get('rolling_features', True):
            result = self.create_rolling_features(result)
        
        # 5. Time features
        if feature_config.get('time_features', True):
            result = self.create_time_features(result)
        
        # Store feature names
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.feature_names = [col for col in result.columns if col not in exclude_cols]
        
        return result
    
    def scale_features(self, df: pd.DataFrame,
                      method: str = 'standard',
                      feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Scale features with different methods for different feature types.
        
        Args:
            df: DataFrame with features
            method: Default scaling method
            feature_groups: Dict mapping scaling method to feature list
            
        Returns:
            Scaled DataFrame
        """
        result = df.copy()
        
        if feature_groups is None:
            # Default grouping
            feature_groups = {
                'minmax': [col for col in result.columns if 'RSI' in col or 'position' in col],
                'robust': [col for col in result.columns if 'volume' in col],
                'standard': [col for col in result.columns if col not in 
                           feature_groups.get('minmax', []) + feature_groups.get('robust', [])]
            }
        
        # Apply scaling by group
        for scale_method, features in feature_groups.items():
            features = [f for f in features if f in result.columns]
            if not features:
                continue
            
            if scale_method == 'standard':
                scaler = StandardScaler()
            elif scale_method == 'minmax':
                scaler = MinMaxScaler()
            elif scale_method == 'robust':
                scaler = RobustScaler()
            else:
                continue
            
            # Fit and transform
            result[features] = scaler.fit_transform(result[features])
            
            # Store scaler
            self.scalers[scale_method] = {
                'scaler': scaler,
                'features': features
            }
        
        return result
    
    def create_sequences(self, df: pd.DataFrame,
                        sequence_length: int = 100,
                        target_col: str = 'label',
                        feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/GRU models.
        
        Args:
            df: DataFrame with features and labels
            sequence_length: Length of input sequences
            target_col: Target column name
            feature_cols: Feature columns to use
            
        Returns:
            Tuple of (X, y) arrays with shape (samples, sequence_length, features)
        """
        if feature_cols is None:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Ensure features exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Extract arrays
        features = df[feature_cols].values
        targets = df[target_col].values if target_col in df.columns else None
        
        # Create sequences
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            if targets is not None:
                y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y) if targets is not None else None
        
        return X, y