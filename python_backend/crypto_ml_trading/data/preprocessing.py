import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
from .data_validator import DataValidator


class AdvancedPreprocessor:
    """
    Advanced data preprocessing for ML models.
    Includes stationarity correction, feature-specific scaling, and data cleaning.
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scalers = {}
        self.preprocessing_params = {}
        self.validator = DataValidator()
        
    def preprocess(self, df: pd.DataFrame,
                  target_col: Optional[str] = None,
                  config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: DataFrame to preprocess
            target_col: Target column name
            config: Preprocessing configuration
            
        Returns:
            Preprocessed DataFrame
        """
        if config is None:
            config = self._default_config()
        
        result = df.copy()
        
        # 1. Validate data
        validation_report = self.validator.validate(result, target_col)
        if not validation_report['is_ml_ready']:
            warnings.warn("Data has quality issues. Applying corrections...")
        
        # 2. Handle missing values
        if config['handle_missing']['enabled']:
            result = self.handle_missing_values(
                result, 
                method=config['handle_missing']['method']
            )
        
        # 3. Handle infinite values
        if config['handle_infinite']['enabled']:
            result = self.handle_infinite_values(
                result,
                method=config['handle_infinite']['method']
            )
        
        # 4. Remove duplicates
        if config['remove_duplicates']['enabled']:
            result = self.remove_duplicates(
                result,
                subset=config['remove_duplicates']['subset']
            )
        
        # 5. Correct for stationarity
        if config['stationarity']['enabled']:
            result = self.correct_stationarity(
                result,
                columns=config['stationarity']['columns'],
                method=config['stationarity']['method']
            )
        
        # 6. Handle outliers
        if config['outliers']['enabled']:
            result = self.handle_outliers(
                result,
                method=config['outliers']['method'],
                threshold=config['outliers']['threshold']
            )
        
        # 7. Feature scaling
        if config['scaling']['enabled']:
            result = self.scale_features(
                result,
                method=config['scaling']['method'],
                feature_groups=config['scaling']['feature_groups']
            )
        
        return result
    
    def _default_config(self) -> Dict:
        """Default preprocessing configuration."""
        return {
            'handle_missing': {
                'enabled': True,
                'method': 'forward_fill'  # forward_fill, interpolate, mean, drop
            },
            'handle_infinite': {
                'enabled': True,
                'method': 'clip'  # clip, remove, replace
            },
            'remove_duplicates': {
                'enabled': True,
                'subset': None  # None for all columns
            },
            'stationarity': {
                'enabled': True,
                'columns': 'auto',  # auto, list of columns, or None
                'method': 'pct_change'  # pct_change, diff, log_diff
            },
            'outliers': {
                'enabled': True,
                'method': 'iqr',  # iqr, zscore, isolation_forest
                'threshold': 3.0
            },
            'scaling': {
                'enabled': True,
                'method': 'standard',  # standard, minmax, robust
                'feature_groups': None  # Custom scaling by feature type
            }
        }
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with missing values
            method: Method to handle missing values
            
        Returns:
            DataFrame with missing values handled
        """
        result = df.copy()
        
        if method == 'forward_fill':
            # Forward fill for time series
            result = result.fillna(method='ffill')
            # Backward fill remaining
            result = result.fillna(method='bfill')
            
        elif method == 'interpolate':
            # Interpolate numeric columns
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].interpolate(method='linear')
            
        elif method == 'mean':
            # Fill with mean for numeric columns
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                result[col].fillna(result[col].mean(), inplace=True)
                
        elif method == 'drop':
            # Drop rows with missing values
            result = result.dropna()
            
        elif method == 'smart':
            # Smart filling based on column type
            for col in result.columns:
                if result[col].dtype in ['float64', 'int64']:
                    # For price data, use forward fill
                    if any(keyword in col.lower() for keyword in ['price', 'open', 'high', 'low', 'close']):
                        result[col].fillna(method='ffill', inplace=True)
                    # For volume, use 0
                    elif 'volume' in col.lower():
                        result[col].fillna(0, inplace=True)
                    # For indicators, use interpolation
                    else:
                        result[col].interpolate(method='linear', inplace=True)
        
        # Store method used
        self.preprocessing_params['missing_value_method'] = method
        
        return result
    
    def handle_infinite_values(self, df: pd.DataFrame,
                             method: str = 'clip') -> pd.DataFrame:
        """
        Handle infinite values in numeric columns.
        
        Args:
            df: DataFrame with infinite values
            method: Method to handle infinite values
            
        Returns:
            DataFrame with infinite values handled
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        if method == 'clip':
            # Replace inf with max/min finite values
            for col in numeric_cols:
                if np.isinf(result[col]).any():
                    finite_vals = result[col][np.isfinite(result[col])]
                    if len(finite_vals) > 0:
                        max_val = finite_vals.max()
                        min_val = finite_vals.min()
                        result[col] = result[col].replace([np.inf, -np.inf], [max_val, min_val])
                        
        elif method == 'remove':
            # Remove rows with infinite values
            result = result[~result[numeric_cols].isin([np.inf, -np.inf]).any(axis=1)]
            
        elif method == 'replace':
            # Replace with NaN and then handle as missing values
            result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
            result = self.handle_missing_values(result, method='interpolate')
        
        return result
    
    def remove_duplicates(self, df: pd.DataFrame,
                         subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: DataFrame with potential duplicates
            subset: Columns to consider for duplicates
            
        Returns:
            DataFrame without duplicates
        """
        result = df.copy()
        
        # Remove duplicates
        result = result.drop_duplicates(subset=subset, keep='last')
        
        # If timestamp column exists, ensure no duplicate timestamps
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
        for col in timestamp_cols:
            if col in result.columns:
                result = result.drop_duplicates(subset=[col], keep='last')
                break
        
        return result
    
    def correct_stationarity(self, df: pd.DataFrame,
                           columns: Union[str, List[str]] = 'auto',
                           method: str = 'pct_change') -> pd.DataFrame:
        """
        Correct for non-stationarity in time series data.
        
        Args:
            df: DataFrame with non-stationary data
            columns: Columns to transform ('auto' to detect)
            method: Transformation method
            
        Returns:
            DataFrame with stationary transformations
        """
        result = df.copy()
        
        # Auto-detect columns if needed
        if columns == 'auto':
            # Transform price columns to returns
            price_cols = [col for col in result.columns if any(
                keyword in col.lower() for keyword in ['open', 'high', 'low', 'close', 'price']
            ) and 'pct' not in col.lower() and 'return' not in col.lower()]
            
            # Transform volume columns
            volume_cols = [col for col in result.columns if 'volume' in col.lower() 
                         and 'pct' not in col.lower()]
            
            columns = price_cols + volume_cols
        
        # Apply transformation
        for col in columns:
            if col not in result.columns:
                continue
                
            if method == 'pct_change':
                # Percentage change
                result[f'{col}_pct'] = result[col].pct_change()
                # Keep original for reference
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    pass  # Keep original OHLCV
                else:
                    result.drop(columns=[col], inplace=True)
                    
            elif method == 'diff':
                # First difference
                result[f'{col}_diff'] = result[col].diff()
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    result.drop(columns=[col], inplace=True)
                    
            elif method == 'log_diff':
                # Log difference
                result[f'{col}_log_diff'] = np.log(result[col]).diff()
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    result.drop(columns=[col], inplace=True)
        
        # Store transformed columns
        self.preprocessing_params['stationary_columns'] = columns
        self.preprocessing_params['stationary_method'] = method
        
        return result
    
    def handle_outliers(self, df: pd.DataFrame,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in the dataset.
        
        Args:
            df: DataFrame with potential outliers
            method: Method to detect outliers
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        # Skip percentage and return columns (they can have extreme values)
        cols_to_check = [col for col in numeric_cols if not any(
            keyword in col.lower() for keyword in ['pct', 'return', 'ret', 'diff']
        )]
        
        if method == 'iqr':
            # IQR method
            for col in cols_to_check:
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Clip outliers
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
                
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            for col in cols_to_check:
                z_scores = np.abs(stats.zscore(result[col].dropna()))
                result.loc[z_scores > threshold, col] = result[col].median()
                
        elif method == 'isolation_forest':
            # Isolation Forest for multivariate outlier detection
            from sklearn.ensemble import IsolationForest
            
            # Prepare data
            X = result[cols_to_check].fillna(result[cols_to_check].median())
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            outlier_mask = iso_forest.fit_predict(X) == -1
            
            # Remove outliers
            result = result[~outlier_mask]
        
        return result
    
    def scale_features(self, df: pd.DataFrame,
                      method: str = 'standard',
                      feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Scale features with different methods for different feature types.
        
        Args:
            df: DataFrame with features
            method: Default scaling method
            feature_groups: Custom scaling by feature group
            
        Returns:
            Scaled DataFrame
        """
        result = df.copy()
        
        if feature_groups is None:
            # Default grouping based on feature types
            feature_groups = self._create_feature_groups(result)
        
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
            elif scale_method == 'none':
                continue  # Skip scaling
            else:
                scaler = StandardScaler()  # Default
            
            # Fit and transform
            result[features] = scaler.fit_transform(result[features].fillna(0))
            
            # Store scaler
            self.scalers[scale_method] = {
                'scaler': scaler,
                'features': features
            }
        
        return result
    
    def _create_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Create feature groups for scaling."""
        groups = {
            'none': [],      # No scaling
            'minmax': [],    # MinMax scaling (0-1)
            'standard': [],  # Standard scaling
            'robust': []     # Robust scaling
        }
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Skip target and metadata
            if col in ['label', 'decision', 'timestamp', 'unix_timestamp']:
                continue
                
            # RSI and other bounded indicators (0-100)
            if 'rsi' in col.lower() or any(ind in col.lower() for ind in ['%k', '%d']):
                groups['none'].append(col)  # Already bounded
                
            # Percentage changes and returns
            elif any(keyword in col.lower() for keyword in ['pct', 'return', 'ret', '_rt']):
                groups['standard'].append(col)
                
            # Volume features
            elif 'volume' in col.lower():
                groups['robust'].append(col)  # Robust to volume spikes
                
            # Price-based features
            elif any(keyword in col.lower() for keyword in ['price', 'open', 'high', 'low', 'close']):
                groups['minmax'].append(col)
                
            # Default to standard scaling
            else:
                groups['standard'].append(col)
        
        return groups
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted parameters.
        
        Args:
            df: New DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        result = df.copy()
        
        # Apply scaling using fitted scalers
        for scale_method, scaler_info in self.scalers.items():
            scaler = scaler_info['scaler']
            features = scaler_info['features']
            
            # Only transform features that exist
            features_to_transform = [f for f in features if f in result.columns]
            if features_to_transform:
                result[features_to_transform] = scaler.transform(
                    result[features_to_transform].fillna(0)
                )
        
        return result