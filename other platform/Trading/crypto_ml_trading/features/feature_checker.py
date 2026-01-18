"""
Feature Completeness Checker
Verifies all required features are calculated for model training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from database.models import TradingProfile
from crypto_ml_trading.training.config import (
    FEATURE_GROUPS,
    MODEL_FEATURE_REQUIREMENTS,
    get_features_for_model
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureReport:
    """Report of feature completeness."""
    profile_id: int
    total_features_required: int
    features_present: int
    features_missing: List[str]
    feature_coverage_percent: float
    low_variance_features: List[str]
    high_correlation_pairs: List[tuple]
    nan_percentages: Dict[str, float]
    is_complete: bool
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'profile_id': self.profile_id,
            'total_features_required': self.total_features_required,
            'features_present': self.features_present,
            'features_missing': self.features_missing,
            'feature_coverage_percent': self.feature_coverage_percent,
            'low_variance_features': self.low_variance_features,
            'high_correlation_pairs': [
                {'feature1': p[0], 'feature2': p[1], 'correlation': p[2]}
                for p in self.high_correlation_pairs
            ],
            'nan_percentages': self.nan_percentages,
            'is_complete': self.is_complete,
            'details': self.details
        }


# Required features by category
REQUIRED_FEATURES: Dict[str, List[str]] = {
    'ohlcv': [
        'open_price', 'high_price', 'low_price', 'close_price', 'volume'
    ],
    'moving_averages': [
        'sma_10', 'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26'
    ],
    'momentum': [
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi', 'momentum'
    ],
    'volatility': [
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr_14', 'kc_upper', 'kc_middle', 'kc_lower',
        'parkinson_vol', 'gk_volatility'
    ],
    'trend': [
        'adx', 'plus_di', 'minus_di', 'parabolic_sar'
    ],
    'volume_indicators': [
        'obv', 'vwap', 'cmf', 'ad_line'
    ],
    'support_resistance': [
        'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3'
    ],
}


class FeatureCompletenessChecker:
    """Verifies all required features are calculated."""

    # Thresholds
    LOW_VARIANCE_THRESHOLD = 0.01
    HIGH_CORRELATION_THRESHOLD = 0.95
    MAX_NAN_PERCENT = 5.0

    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session

    def check_completeness(
        self,
        profile_id: int,
        data: Optional[pd.DataFrame] = None,
        model_type: Optional[str] = None
    ) -> FeatureReport:
        """
        Check feature completeness for a profile.

        Args:
            profile_id: ID of the trading profile
            data: Optional DataFrame with features (if not provided, loads from DB)
            model_type: Optional model type to check specific feature requirements

        Returns:
            FeatureReport with completeness analysis
        """
        logger.info(f"Checking feature completeness for profile {profile_id}")

        # Load data if not provided
        if data is None:
            data = self._load_profile_features(profile_id)

        if data.empty:
            return FeatureReport(
                profile_id=profile_id,
                total_features_required=0,
                features_present=0,
                features_missing=[],
                feature_coverage_percent=0.0,
                low_variance_features=[],
                high_correlation_pairs=[],
                nan_percentages={},
                is_complete=False
            )

        # Get required features
        if model_type:
            required_features = set(get_features_for_model(model_type))
        else:
            # Get all features from all categories
            required_features = set()
            for features in REQUIRED_FEATURES.values():
                required_features.update(features)

        # Check which features are present
        present_features = set(data.columns)
        missing_features = list(required_features - present_features)
        matched_features = required_features & present_features

        # Calculate coverage
        total_required = len(required_features)
        features_present = len(matched_features)
        coverage_percent = (features_present / total_required * 100) if total_required > 0 else 0.0

        # Check for low variance features
        low_variance = self._find_low_variance_features(
            data[list(matched_features)] if matched_features else pd.DataFrame()
        )

        # Check for highly correlated features
        high_corr_pairs = self._find_high_correlations(
            data[list(matched_features)] if matched_features else pd.DataFrame()
        )

        # Check NaN percentages
        nan_percentages = self._calculate_nan_percentages(data[list(matched_features)])

        # Check if complete
        is_complete = (
            len(missing_features) == 0 and
            coverage_percent >= 100.0 and
            all(pct < self.MAX_NAN_PERCENT for pct in nan_percentages.values())
        )

        # Build details
        details = {
            'categories_checked': list(REQUIRED_FEATURES.keys()),
            'model_type': model_type,
            'required_features': list(required_features),
            'present_features': list(matched_features)
        }

        report = FeatureReport(
            profile_id=profile_id,
            total_features_required=total_required,
            features_present=features_present,
            features_missing=missing_features,
            feature_coverage_percent=coverage_percent,
            low_variance_features=low_variance,
            high_correlation_pairs=high_corr_pairs,
            nan_percentages=nan_percentages,
            is_complete=is_complete,
            details=details
        )

        logger.info(
            f"Feature check complete: {features_present}/{total_required} "
            f"({coverage_percent:.1f}%), missing: {len(missing_features)}"
        )

        return report

    def _load_profile_features(self, profile_id: int) -> pd.DataFrame:
        """Load feature data for a profile from database."""
        if not self.db_session:
            logger.warning("No database session, returning empty DataFrame")
            return pd.DataFrame()

        # For now, just check if the profile has indicators calculated
        profile = self.db_session.query(TradingProfile).filter(
            TradingProfile.id == profile_id
        ).first()

        if not profile:
            return pd.DataFrame()

        # In production, would load actual feature data from a features table
        # For now, return empty DataFrame
        return pd.DataFrame()

    def _find_low_variance_features(
        self,
        data: pd.DataFrame,
        threshold: float = None
    ) -> List[str]:
        """
        Find features with variance below threshold.

        Args:
            data: DataFrame with features
            threshold: Variance threshold (default 0.01)

        Returns:
            List of low variance feature names
        """
        if threshold is None:
            threshold = self.LOW_VARIANCE_THRESHOLD

        if data.empty:
            return []

        low_variance = []
        for col in data.columns:
            if not np.issubdtype(data[col].dtype, np.number):
                continue

            variance = data[col].var()
            if variance < threshold:
                low_variance.append(col)

        return low_variance

    def _find_high_correlations(
        self,
        data: pd.DataFrame,
        threshold: float = None
    ) -> List[tuple]:
        """
        Find highly correlated feature pairs.

        Args:
            data: DataFrame with features
            threshold: Correlation threshold (default 0.95)

        Returns:
            List of tuples (feature1, feature2, correlation)
        """
        if threshold is None:
            threshold = self.HIGH_CORRELATION_THRESHOLD

        if data.empty or len(data.columns) < 2:
            return []

        # Calculate correlation matrix
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return []

        corr_matrix = data[numeric_cols].corr()

        # Find pairs above threshold
        high_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    high_corr.append((
                        numeric_cols[i],
                        numeric_cols[j],
                        round(corr, 4)
                    ))

        return high_corr

    def _calculate_nan_percentages(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate NaN percentage for each column.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary mapping column name to NaN percentage
        """
        if data.empty:
            return {}

        total_rows = len(data)
        nan_pcts = {}

        for col in data.columns:
            nan_count = data[col].isna().sum()
            nan_pcts[col] = round(nan_count / total_rows * 100, 2)

        return nan_pcts

    def get_feature_manifest(self, model_type: str) -> List[str]:
        """
        Get the complete list of features required for a model type.

        Args:
            model_type: Model type identifier

        Returns:
            List of required feature names
        """
        return get_features_for_model(model_type)

    def remove_low_variance_features(
        self,
        data: pd.DataFrame,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove features with low variance.

        Args:
            data: DataFrame with features
            threshold: Variance threshold

        Returns:
            DataFrame with low variance features removed
        """
        low_var = self._find_low_variance_features(data, threshold)

        if low_var:
            logger.info(f"Removing {len(low_var)} low variance features: {low_var}")
            return data.drop(columns=low_var, errors='ignore')

        return data

    def remove_correlated_features(
        self,
        data: pd.DataFrame,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove one feature from each highly correlated pair.

        Keeps the first feature alphabetically from each pair.

        Args:
            data: DataFrame with features
            threshold: Correlation threshold

        Returns:
            DataFrame with redundant features removed
        """
        high_corr = self._find_high_correlations(data, threshold)

        if not high_corr:
            return data

        # Collect features to remove (keep first alphabetically)
        to_remove = set()
        for feat1, feat2, _ in high_corr:
            # Remove the one that comes later alphabetically
            to_remove.add(max(feat1, feat2))

        if to_remove:
            logger.info(f"Removing {len(to_remove)} correlated features: {to_remove}")
            return data.drop(columns=list(to_remove), errors='ignore')

        return data

    def get_category_completeness(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Get completeness breakdown by feature category.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary mapping category to completeness info
        """
        present_features = set(data.columns)
        result = {}

        for category, features in REQUIRED_FEATURES.items():
            required = set(features)
            present = required & present_features
            missing = required - present_features

            result[category] = {
                'required': len(required),
                'present': len(present),
                'missing': list(missing),
                'coverage_percent': (len(present) / len(required) * 100) if required else 100.0
            }

        return result


def check_features_for_training(
    data: pd.DataFrame,
    model_type: str = None
) -> bool:
    """
    Quick check if data has required features for training.

    Args:
        data: DataFrame with features
        model_type: Optional model type

    Returns:
        True if data has all required features
    """
    checker = FeatureCompletenessChecker()
    report = checker.check_completeness(
        profile_id=0,
        data=data,
        model_type=model_type
    )
    return report.is_complete
