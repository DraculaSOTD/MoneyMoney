import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.technical_indicators import TechnicalIndicators
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
from features.market_microstructure import MarketMicrostructureFeatures


class FeaturePipeline:
    """
    Comprehensive feature engineering pipeline for cryptocurrency trading.
    
    Combines:
    - Technical indicators
    - Market microstructure features
    - Time-based features
    - Cross-asset features
    - Statistical features
    - Pattern-based features
    
    Designed for real-time feature generation with efficient computation.
    """
    
    def __init__(self, feature_config: Optional[Union[Dict, str]] = None):
        """
        Initialize feature pipeline.
        
        Args:
            feature_config: Configuration dict or path to config file
        """
        if isinstance(feature_config, str):
            self.feature_config = self._load_config(feature_config)
        else:
            self.feature_config = feature_config or self._default_config()
        
        self.feature_names = []
        self.scaler_params = None
        self.enhanced_indicators = EnhancedTechnicalIndicators()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Default feature configuration."""
        # Try to load from default config file
        default_config_path = Path(__file__).parent.parent / 'config' / 'feature_config.yaml'
        if default_config_path.exists():
            return self._load_config(str(default_config_path))
        
        # Fallback to hardcoded config
        return {
            'technical_indicators': {
                'enabled': True,
                'indicators': ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr']
            },
            'microstructure': {
                'enabled': True,
                'features': ['ofi', 'spread', 'kyle_lambda', 'vpin']
            },
            'time_features': {
                'enabled': True,
                'cyclical': True
            },
            'statistical_features': {
                'enabled': True,
                'windows': [5, 10, 30, 60]
            },
            'interaction_features': {
                'enabled': False,  # Can be expensive
                'max_interactions': 10
            }
        }
    
    def generate_features(self, df: pd.DataFrame, 
                         training: bool = True) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            training: Whether this is training data (for fitting scalers)
            
        Returns:
            DataFrame with all features
        """
        result = df.copy()
        
        # Technical indicators
        if self.feature_config['technical_indicators']['enabled']:
            result = self._add_technical_indicators(result)
            
        # Market microstructure
        if self.feature_config['microstructure']['enabled']:
            result = self._add_microstructure_features(result)
            
        # Time features
        if self.feature_config['time_features']['enabled']:
            result = self._add_time_features(result)
            
        # Statistical features
        if self.feature_config['statistical_features']['enabled']:
            result = self._add_statistical_features(result)
            
        # Price action features
        result = self._add_price_action_features(result)
        
        # Interaction features
        if self.feature_config['interaction_features']['enabled']:
            result = self._add_interaction_features(result)
            
        # Store feature names
        if training:
            self.feature_names = [col for col in result.columns 
                                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        return result
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # Use existing technical indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Add additional custom indicators
        # Normalized indicators
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Scale to [-1, 1]
        df['cci_normalized'] = df['cci'] / 100
        
        # Indicator divergences
        df['price_sma_divergence'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_ema_divergence'] = (df['close'] - df['ema_20']) / df['ema_20']
        
        # Bollinger Band position
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        df = MarketMicrostructureFeatures.calculate_all_microstructure_features(df)
        
        # Add rolling statistics of microstructure features
        if 'kyle_lambda' in df.columns:
            df['kyle_lambda_ma'] = df['kyle_lambda'].rolling(20).mean()
            df['kyle_lambda_std'] = df['kyle_lambda'].rolling(20).std()
            
        if 'vpin' in df.columns:
            df['vpin_ma'] = df['vpin'].rolling(20).mean()
            df['vpin_spike'] = df['vpin'] / (df['vpin_ma'] + 1e-10)
            
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in df.columns:
            return df
            
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        if self.feature_config['time_features']['cyclical']:
            # Hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Day of week
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Day of month
            df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
            df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30)
            
        # Trading session features (crypto-specific)
        # Asian session: 00:00 - 08:00 UTC
        # European session: 08:00 - 16:00 UTC
        # American session: 16:00 - 00:00 UTC
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        windows = self.feature_config['statistical_features']['windows']
        
        for window in windows:
            # Return statistics
            df[f'return_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'return_std_{window}'] = df['returns'].rolling(window).std()
            df[f'return_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'return_kurt_{window}'] = df['returns'].rolling(window).kurt()
            
            # Price statistics
            df[f'price_min_{window}'] = df['close'].rolling(window).min()
            df[f'price_max_{window}'] = df['close'].rolling(window).max()
            df[f'price_range_{window}'] = df[f'price_max_{window}'] - df[f'price_min_{window}']
            
            # Volume statistics
            if 'volume' in df.columns:
                df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
                df[f'volume_trend_{window}'] = df['volume'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0]
                )
                
        # Autocorrelation features
        for lag in [1, 5, 10]:
            df[f'return_autocorr_{lag}'] = df['returns'].rolling(30).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            
        return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action based features."""
        # Candlestick patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        # Doji detection
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)
        
        # Hammer/Shooting star
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                          (df['upper_shadow'] < df['body_size'])).astype(int)
        df['is_shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                 (df['lower_shadow'] < df['body_size'])).astype(int)
        
        # Price levels
        df['distance_from_high'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['distance_from_low'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        # Support/Resistance
        df['near_resistance'] = (df['distance_from_high'] < 0.01).astype(int)
        df['near_support'] = (df['distance_from_low'] < 0.01).astype(int)
        
        # Momentum
        df['momentum_1'] = df['close'] / df['close'].shift(1) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between indicators."""
        # Volume-Price interactions
        if 'volume' in df.columns:
            df['volume_price_corr'] = df['close'].rolling(20).corr(df['volume'])
            df['volume_return_corr'] = df['returns'].rolling(20).corr(df['volume'])
            
        # Indicator interactions
        if 'rsi' in df.columns and 'macd' in df.columns:
            df['rsi_macd_interaction'] = df['rsi'] * df['macd']
            df['rsi_macd_divergence'] = (df['rsi'] - 50) * np.sign(df['macd'])
            
        # Volatility interactions
        if 'atr' in df.columns and 'bb_bandwidth' in df.columns:
            df['volatility_ratio'] = df['atr'] / (df['bb_bandwidth'] + 1e-10)
            
        # Trend-Volume interaction
        if 'sma_20' in df.columns and 'volume' in df.columns:
            df['trend_volume'] = np.sign(df['close'] - df['sma_20']) * df['volume']
            
        return df
    
    def select_features(self, df: pd.DataFrame, 
                       method: str = 'importance',
                       n_features: int = 50) -> List[str]:
        """
        Select most important features.
        
        Args:
            df: DataFrame with all features
            method: Selection method ('importance', 'correlation', 'variance')
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Get numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and metadata columns
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'returns', 'log_returns']
        features = [col for col in numeric_features if col not in exclude_cols]
        
        if method == 'variance':
            # Select features with highest variance
            variances = df[features].var()
            selected = variances.nlargest(n_features).index.tolist()
            
        elif method == 'correlation':
            # Select features with low correlation to each other
            corr_matrix = df[features].corr().abs()
            
            # Start with feature with highest variance
            selected = [variances.idxmax()]
            
            while len(selected) < n_features and len(selected) < len(features):
                # Find feature with lowest correlation to selected features
                min_corr = 1.0
                next_feature = None
                
                for feature in features:
                    if feature not in selected:
                        max_corr_to_selected = corr_matrix.loc[feature, selected].max()
                        if max_corr_to_selected < min_corr:
                            min_corr = max_corr_to_selected
                            next_feature = feature
                            
                if next_feature:
                    selected.append(next_feature)
                else:
                    break
                    
        else:  # importance
            # Use correlation with returns as proxy for importance
            if 'returns' in df.columns:
                importance = df[features].corrwith(df['returns']).abs()
                selected = importance.nlargest(n_features).index.tolist()
            else:
                # Fallback to variance
                return self.select_features(df, method='variance', n_features=n_features)
                
        return selected
    
    def create_lagged_features(self, df: pd.DataFrame,
                             features: List[str],
                             lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged versions of features.
        
        Args:
            df: DataFrame with features
            features: Features to lag
            lags: Lag values
            
        Returns:
            DataFrame with lagged features added
        """
        result = df.copy()
        
        for feature in features:
            if feature in df.columns:
                for lag in lags:
                    result[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
                    
        return result
    
    def create_target_variables(self, df: pd.DataFrame,
                              horizons: List[int] = [1, 5, 10, 30]) -> pd.DataFrame:
        """
        Create target variables for prediction.
        
        Args:
            df: DataFrame with price data
            horizons: Prediction horizons
            
        Returns:
            DataFrame with target variables
        """
        result = df.copy()
        
        for horizon in horizons:
            # Future returns
            result[f'target_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
            
            # Direction (classification)
            result[f'target_direction_{horizon}'] = (result[f'target_return_{horizon}'] > 0).astype(int)
            
            # Volatility target
            if 'returns' in df.columns:
                result[f'target_volatility_{horizon}'] = df['returns'].rolling(horizon).std().shift(-horizon)
                
        return result
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with feature statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = pd.DataFrame({
            'mean': df[numeric_cols].mean(),
            'std': df[numeric_cols].std(),
            'min': df[numeric_cols].min(),
            'max': df[numeric_cols].max(),
            'nulls': df[numeric_cols].isnull().sum(),
            'null_pct': df[numeric_cols].isnull().sum() / len(df) * 100,
            'unique': df[numeric_cols].nunique(),
            'skewness': df[numeric_cols].skew(),
            'kurtosis': df[numeric_cols].kurt()
        })
        
        return summary.round(4)