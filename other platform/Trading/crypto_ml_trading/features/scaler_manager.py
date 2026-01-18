"""
Feature Scaler Manager
Manages feature-specific scaling strategies for ML models.
"""

import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer
)

logger = logging.getLogger(__name__)


class LogStandardScaler:
    """
    Custom scaler that applies log transform followed by StandardScaler.
    Used for highly skewed features like volume.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.standard_scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'LogStandardScaler':
        """Fit the scaler on training data."""
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
        X_log = np.log1p(np.clip(X, self.epsilon, None))
        self.standard_scaler.fit(X_log)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
        X_log = np.log1p(np.clip(X, self.epsilon, None))
        return self.standard_scaler.transform(X_log)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform to original scale."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
        X_unscaled = self.standard_scaler.inverse_transform(X)
        return np.expm1(X_unscaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


# Scaler type registry
SCALER_TYPES = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
    'power': PowerTransformer,
    'log_standard': LogStandardScaler,
}


# Feature to scaler mapping
# Maps feature names to (scaler_type, scaler_kwargs)
SCALER_REGISTRY: Dict[str, Tuple[str, Dict]] = {
    # Price features -> RobustScaler (handles outliers)
    'open_price': ('robust', {}),
    'high_price': ('robust', {}),
    'low_price': ('robust', {}),
    'close_price': ('robust', {}),

    # Volume -> Log + StandardScaler (highly skewed)
    'volume': ('log_standard', {}),
    'quote_asset_volume': ('log_standard', {}),
    'taker_buy_base_volume': ('log_standard', {}),
    'taker_buy_quote_volume': ('log_standard', {}),
    'obv': ('log_standard', {}),

    # Bounded oscillators [0-100] -> MinMaxScaler [0-1]
    'rsi_14': ('minmax', {'feature_range': (0, 1)}),
    'rsi_7': ('minmax', {'feature_range': (0, 1)}),
    'rsi_21': ('minmax', {'feature_range': (0, 1)}),
    'stoch_k': ('minmax', {'feature_range': (0, 1)}),
    'stoch_d': ('minmax', {'feature_range': (0, 1)}),
    'mfi': ('minmax', {'feature_range': (0, 1)}),
    'williams_r': ('minmax', {'feature_range': (0, 1)}),
    'adx': ('minmax', {'feature_range': (0, 1)}),
    'plus_di': ('minmax', {'feature_range': (0, 1)}),
    'minus_di': ('minmax', {'feature_range': (0, 1)}),

    # Unbounded indicators -> RobustScaler
    'macd': ('robust', {}),
    'macd_signal': ('robust', {}),
    'macd_histogram': ('robust', {}),
    'cci': ('robust', {}),
    'momentum': ('robust', {}),
    'roc': ('robust', {}),

    # Bollinger Bands -> RobustScaler
    'bb_upper': ('robust', {}),
    'bb_middle': ('robust', {}),
    'bb_lower': ('robust', {}),
    'bb_width': ('log_standard', {}),
    'bb_percent': ('minmax', {'feature_range': (0, 1)}),

    # Keltner Channels -> RobustScaler
    'kc_upper': ('robust', {}),
    'kc_middle': ('robust', {}),
    'kc_lower': ('robust', {}),

    # Percentage changes -> StandardScaler
    'close_pct': ('standard', {}),
    'log_returns': ('standard', {}),
    'volume_pct_change': ('standard', {}),
    'range_pct': ('standard', {}),
    'body_pct': ('standard', {}),

    # Volatility measures -> Log + StandardScaler
    'atr_14': ('log_standard', {}),
    'atr_7': ('log_standard', {}),
    'parkinson_vol': ('log_standard', {}),
    'gk_volatility': ('log_standard', {}),
    'yang_zhang_vol': ('log_standard', {}),

    # Moving averages -> RobustScaler
    'sma_10': ('robust', {}),
    'sma_20': ('robust', {}),
    'sma_50': ('robust', {}),
    'sma_200': ('robust', {}),
    'ema_12': ('robust', {}),
    'ema_26': ('robust', {}),
    'ema_50': ('robust', {}),
    'vwap': ('robust', {}),

    # Support/Resistance -> RobustScaler
    'pivot': ('robust', {}),
    'r1': ('robust', {}),
    'r2': ('robust', {}),
    'r3': ('robust', {}),
    's1': ('robust', {}),
    's2': ('robust', {}),
    's3': ('robust', {}),

    # Microstructure -> StandardScaler
    'bid_ask_spread': ('standard', {}),
    'trade_intensity': ('standard', {}),
    'order_imbalance': ('standard', {}),
}


class FeatureScalerManager:
    """Manages feature-specific scaling strategies."""

    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.fitted = False
        self._feature_order: List[str] = []

    def _get_scaler_for_feature(self, feature_name: str) -> Any:
        """
        Get the appropriate scaler for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Scaler instance
        """
        # Check if feature is in registry
        if feature_name in SCALER_REGISTRY:
            scaler_type, kwargs = SCALER_REGISTRY[feature_name]
        else:
            # Default to StandardScaler for unknown features
            scaler_type, kwargs = 'standard', {}
            logger.debug(f"Using default StandardScaler for unknown feature: {feature_name}")

        # Create scaler instance
        scaler_class = SCALER_TYPES.get(scaler_type)
        if scaler_class is None:
            logger.warning(f"Unknown scaler type {scaler_type}, using StandardScaler")
            scaler_class = StandardScaler
            kwargs = {}

        return scaler_class(**kwargs)

    def fit(self, train_data: pd.DataFrame) -> 'FeatureScalerManager':
        """
        Fit scalers on training data only.

        IMPORTANT: This should only be called with training data to prevent data leakage.

        Args:
            train_data: Training DataFrame

        Returns:
            self for method chaining
        """
        logger.info(f"Fitting scalers on {len(train_data)} training samples")

        self.scalers = {}
        self._feature_order = list(train_data.columns)

        for column in train_data.columns:
            # Skip non-numeric columns
            if not np.issubdtype(train_data[column].dtype, np.number):
                logger.debug(f"Skipping non-numeric column: {column}")
                continue

            # Get appropriate scaler
            scaler = self._get_scaler_for_feature(column)

            # Fit on training data
            values = train_data[column].values.reshape(-1, 1)

            # Handle NaN values
            mask = ~np.isnan(values.flatten())
            if mask.sum() == 0:
                logger.warning(f"Column {column} has all NaN values, skipping")
                continue

            clean_values = values[mask].reshape(-1, 1)
            scaler.fit(clean_values)
            self.scalers[column] = scaler

        self.fitted = True
        logger.info(f"Fitted {len(self.scalers)} scalers")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scalers.

        Args:
            data: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise RuntimeError("Scalers not fitted. Call fit() first with training data.")

        result = data.copy()

        for column in data.columns:
            if column not in self.scalers:
                logger.debug(f"No scaler for column {column}, keeping original values")
                continue

            scaler = self.scalers[column]
            values = data[column].values.reshape(-1, 1)

            # Handle NaN values - transform non-NaN, preserve NaN
            mask = ~np.isnan(values.flatten())
            if mask.sum() == 0:
                continue

            transformed = np.full_like(values, np.nan)
            transformed[mask] = scaler.transform(values[mask].reshape(-1, 1)).flatten()
            result[column] = transformed.flatten()

        return result

    def fit_transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scalers and transform training data.

        Args:
            train_data: Training DataFrame

        Returns:
            Transformed training DataFrame
        """
        return self.fit(train_data).transform(train_data)

    def inverse_transform(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Inverse transform specific columns back to original scale.

        Args:
            data: Transformed DataFrame
            columns: List of columns to inverse transform (default: all)

        Returns:
            DataFrame with inverse transformed columns
        """
        if not self.fitted:
            raise RuntimeError("Scalers not fitted. Call fit() first.")

        result = data.copy()
        columns = columns or list(self.scalers.keys())

        for column in columns:
            if column not in self.scalers:
                continue
            if column not in data.columns:
                continue

            scaler = self.scalers[column]
            values = data[column].values.reshape(-1, 1)

            mask = ~np.isnan(values.flatten())
            if mask.sum() == 0:
                continue

            inverse = np.full_like(values, np.nan)
            inverse[mask] = scaler.inverse_transform(values[mask].reshape(-1, 1)).flatten()
            result[column] = inverse.flatten()

        return result

    def save(self, path: str) -> None:
        """
        Save fitted scalers to file.

        Args:
            path: Path to save file
        """
        if not self.fitted:
            raise RuntimeError("Scalers not fitted. Call fit() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'scalers': self.scalers,
            'feature_order': self._feature_order,
            'fitted': self.fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved {len(self.scalers)} scalers to {path}")

    def load(self, path: str) -> 'FeatureScalerManager':
        """
        Load fitted scalers from file.

        Args:
            path: Path to load file

        Returns:
            self for method chaining
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.scalers = state['scalers']
        self._feature_order = state['feature_order']
        self.fitted = state['fitted']

        logger.info(f"Loaded {len(self.scalers)} scalers from {path}")

        return self

    def get_scaler_info(self) -> Dict[str, str]:
        """
        Get information about which scaler is used for each feature.

        Returns:
            Dictionary mapping feature name to scaler type
        """
        info = {}
        for feature in self._feature_order:
            if feature in SCALER_REGISTRY:
                info[feature] = SCALER_REGISTRY[feature][0]
            else:
                info[feature] = 'standard (default)'
        return info


def create_scaler_manager() -> FeatureScalerManager:
    """Factory function to create a new scaler manager."""
    return FeatureScalerManager()


def get_recommended_scaler(feature_name: str) -> str:
    """
    Get the recommended scaler type for a feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Scaler type string
    """
    if feature_name in SCALER_REGISTRY:
        return SCALER_REGISTRY[feature_name][0]
    return 'standard'
