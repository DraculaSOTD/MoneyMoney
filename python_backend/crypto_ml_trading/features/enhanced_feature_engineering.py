"""
Enhanced feature engineering with stationarity analysis.

Combines technical indicators with stationarity testing and
appropriate transformations for better model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from .stationarity_analyzer import StationarityAnalyzer
from .enhanced_technical_indicators import EnhancedTechnicalIndicators

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineering:
    """
    Advanced feature engineering with stationarity considerations.
    
    Features:
    - Automatic stationarity testing
    - Feature transformations
    - Lagged features with stationarity
    - Return-based features
    - Log transformations where appropriate
    """
    
    def __init__(self, 
                 stationarity_threshold: float = 0.05,
                 max_lag: int = 30,
                 return_periods: List[int] = [1, 5, 10, 20, 30]):
        """
        Initialize enhanced feature engineering.
        
        Args:
            stationarity_threshold: Significance level for stationarity tests
            max_lag: Maximum lag for features
            return_periods: Periods for return calculations
        """
        self.stationarity_analyzer = StationarityAnalyzer(stationarity_threshold)
        self.indicator_calculator = EnhancedTechnicalIndicators()
        self.max_lag = max_lag
        self.return_periods = return_periods
        self.feature_info = {}
        
    def create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create various return-based features (naturally stationary).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with return features
        """
        features = pd.DataFrame(index=df.index)
        
        # Simple returns
        for period in self.return_periods:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Volatility (rolling std of returns)
        returns = df['close'].pct_change()
        for window in [10, 20, 30]:
            features[f'volatility_{window}'] = returns.rolling(window).std()
        
        # Price relative to moving averages (stationary ratios)
        for period in [10, 20, 50]:
            ma = df['close'].rolling(period).mean()
            features[f'price_ma_ratio_{period}'] = df['close'] / ma - 1
        
        # Volume-based returns
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean() - 1
        
        # High-low spread (volatility proxy)
        features['hl_spread'] = (df['high'] - df['low']) / df['close']
        features['hl_spread_ma'] = features['hl_spread'].rolling(10).mean()
        
        # Close-open spread
        features['co_spread'] = (df['close'] - df['open']) / df['open']
        
        return features
    
    def create_stationary_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and ensure stationarity.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with stationary indicators
        """
        features = pd.DataFrame(index=df.index)
        
        # RSI (bounded, typically stationary)
        features['rsi_14'] = self.indicator_calculator.rsi(df['close'], 14)
        features['rsi_30'] = self.indicator_calculator.rsi(df['close'], 30)
        
        # Stochastic (bounded, typically stationary)
        stoch = self.indicator_calculator.stochastic(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch['k']
        features['stoch_d'] = stoch['d']
        
        # MACD components (differenced, more stationary)
        macd_result = self.indicator_calculator.macd(df['close'])
        features['macd_line'] = macd_result['macd']
        features['macd_signal'] = macd_result['signal']
        features['macd_histogram'] = macd_result['histogram']
        
        # Bollinger Bands (normalized)
        bb = self.indicator_calculator.bollinger_bands(df['close'])
        features['bb_width'] = (bb['upper'] - bb['lower']) / bb['middle']
        features['bb_position'] = (df['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
        
        # ATR (normalized by price)
        features['atr_norm'] = self.indicator_calculator.atr(df['high'], df['low'], df['close']) / df['close']
        
        # Money Flow Index (bounded)
        features['mfi'] = self.indicator_calculator.money_flow_index(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # Williams %R (bounded)
        features['williams_r'] = self.indicator_calculator.williams_r(
            df['high'], df['low'], df['close']
        )
        
        return features
    
    def create_lagged_features(self, features: pd.DataFrame, 
                              stationary_only: bool = True) -> pd.DataFrame:
        """
        Create lagged features with stationarity check.
        
        Args:
            features: DataFrame with features
            stationary_only: Only create lags for stationary features
            
        Returns:
            DataFrame with lagged features
        """
        lagged_features = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            # Check stationarity if required
            if stationary_only:
                test_result = self.stationarity_analyzer.test_stationarity(
                    features[col].dropna()
                )
                if not test_result.get('overall_stationary', False):
                    logger.info(f"Skipping lags for non-stationary feature: {col}")
                    continue
            
            # Create lags
            for lag in [1, 2, 3, 5, 10]:
                if lag <= self.max_lag:
                    lagged_features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        return lagged_features
    
    def apply_transformations(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply transformations to achieve stationarity.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Transformed DataFrame and transformation info
        """
        transformed = pd.DataFrame(index=df.index)
        transform_info = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            # Test stationarity
            test_result = self.stationarity_analyzer.test_stationarity(series)
            
            if test_result.get('overall_stationary', False):
                # Already stationary
                transformed[col] = df[col]
                transform_info[col] = {'stationary': True, 'transform': 'none'}
            else:
                # Apply transformation
                trans_series, info = self.stationarity_analyzer.make_stationary(
                    series, method='auto'
                )
                
                # Align with original index
                transformed[col] = trans_series.reindex(df.index)
                transform_info[col] = info
                transform_info[col]['stationary'] = False
                
                logger.info(f"Transformed {col}: {info['transformations']}")
        
        return transformed, transform_info
    
    def create_all_features(self, df: pd.DataFrame, 
                           apply_stationarity: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive feature set with stationarity considerations.
        
        Args:
            df: DataFrame with OHLCV data
            apply_stationarity: Whether to apply stationarity transformations
            
        Returns:
            Dictionary with features and metadata
        """
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()
        
        for col in required_cols:
            if col not in df_lower.columns:
                raise ValueError(f"Required column '{col}' not found")
        
        # Create return features (naturally stationary)
        logger.info("Creating return features...")
        return_features = self.create_return_features(df_lower)
        
        # Create technical indicators
        logger.info("Creating technical indicators...")
        indicator_features = self.create_stationary_indicators(df_lower)
        
        # Combine features
        all_features = pd.concat([return_features, indicator_features], axis=1)
        
        # Create lagged features
        logger.info("Creating lagged features...")
        lagged_features = self.create_lagged_features(all_features, stationary_only=True)
        
        # Add lagged features
        all_features = pd.concat([all_features, lagged_features], axis=1)
        
        # Apply transformations if needed
        transform_info = None
        if apply_stationarity:
            logger.info("Applying stationarity transformations...")
            # Only transform non-return features that need it
            non_return_cols = [col for col in all_features.columns 
                              if not col.startswith(('return_', 'log_return_'))]
            
            if non_return_cols:
                features_to_transform = all_features[non_return_cols]
                transformed_features, transform_info = self.apply_transformations(
                    features_to_transform
                )
                
                # Replace with transformed versions
                all_features[non_return_cols] = transformed_features
        
        # Handle NaN values
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        # Store feature info
        self.feature_info = {
            'n_features': all_features.shape[1],
            'feature_names': all_features.columns.tolist(),
            'transform_info': transform_info,
            'stationary_features': [col for col in all_features.columns 
                                   if transform_info is None or 
                                   transform_info.get(col, {}).get('stationary', True)]
        }
        
        logger.info(f"Created {self.feature_info['n_features']} features")
        logger.info(f"Stationary features: {len(self.feature_info['stationary_features'])}")
        
        return {
            'features': all_features,
            'info': self.feature_info,
            'transform_info': transform_info
        }
    
    def prepare_for_modeling(self, df: pd.DataFrame, 
                            target_col: str = 'close',
                            target_type: str = 'classification',
                            lookahead: int = 1) -> Dict[str, Any]:
        """
        Prepare features and targets for modeling.
        
        Args:
            df: DataFrame with OHLCV data
            target_col: Column to create target from
            target_type: 'classification' or 'regression'
            lookahead: Steps ahead to predict
            
        Returns:
            Dictionary with features, targets, and metadata
        """
        # Create features
        feature_result = self.create_all_features(df, apply_stationarity=True)
        features = feature_result['features']
        
        # Create target
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()
        
        if target_type == 'classification':
            # Create classification target (0: down, 1: up, 2: neutral)
            returns = df_lower[target_col].pct_change(lookahead).shift(-lookahead)
            
            # Define thresholds
            up_threshold = 0.001  # 0.1%
            down_threshold = -0.001
            
            target = pd.Series(1, index=returns.index)  # Default neutral
            target[returns > up_threshold] = 0  # Buy signal
            target[returns < down_threshold] = 2  # Sell signal
            
        else:  # regression
            # Predict future returns
            target = df_lower[target_col].pct_change(lookahead).shift(-lookahead)
        
        # Align features and target
        valid_idx = features.index.intersection(target.dropna().index)
        features_aligned = features.loc[valid_idx]
        target_aligned = target.loc[valid_idx]
        
        # Remove any remaining NaN
        mask = ~(features_aligned.isna().any(axis=1) | target_aligned.isna())
        features_final = features_aligned[mask]
        target_final = target_aligned[mask]
        
        result = {
            'features': features_final,
            'target': target_final,
            'feature_info': feature_result['info'],
            'transform_info': feature_result['transform_info'],
            'target_type': target_type,
            'lookahead': lookahead,
            'n_samples': len(features_final),
            'n_features': features_final.shape[1]
        }
        
        logger.info(f"Prepared {result['n_samples']} samples with "
                   f"{result['n_features']} features")
        
        return result