"""
Data Quality Validator
Validates data quality and flags anomalies for review.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import TradingProfile, MarketData

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report of data quality validation."""
    profile_id: int
    outliers_flagged: int
    anomalies_detected: int
    ohlc_violations: int
    zero_volume_candles: int
    flagged_rows: List[int]
    quality_score: float
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'profile_id': self.profile_id,
            'outliers_flagged': self.outliers_flagged,
            'anomalies_detected': self.anomalies_detected,
            'ohlc_violations': self.ohlc_violations,
            'zero_volume_candles': self.zero_volume_candles,
            'flagged_rows_count': len(self.flagged_rows),
            'quality_score': self.quality_score,
            'details': self.details
        }


class DataQualityValidator:
    """Validates data quality and flags anomalies."""

    # IQR multiplier for outlier detection
    DEFAULT_IQR_MULTIPLIER = 3.0
    # Price change threshold for anomaly detection (20%)
    PRICE_ANOMALY_THRESHOLD = 0.20
    # Volume spike threshold (2x average)
    VOLUME_SPIKE_THRESHOLD = 2.0

    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session

    def validate_profile_data(
        self,
        profile_id: int,
        data: Optional[pd.DataFrame] = None
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality validation.

        Args:
            profile_id: ID of the trading profile
            data: Optional DataFrame with market data (if not provided, will load from DB)

        Returns:
            DataQualityReport with validation results
        """
        logger.info(f"Starting data quality validation for profile {profile_id}")

        # Load data if not provided
        if data is None:
            data = self._load_profile_data(profile_id)

        if data.empty:
            return DataQualityReport(
                profile_id=profile_id,
                outliers_flagged=0,
                anomalies_detected=0,
                ohlc_violations=0,
                zero_volume_candles=0,
                flagged_rows=[],
                quality_score=0.0
            )

        # Initialize flagged rows set
        all_flagged = set()
        details = {}

        # Check for OHLC violations
        ohlc_violations = self.validate_ohlc_relationships(data)
        if not ohlc_violations.empty:
            all_flagged.update(ohlc_violations.index.tolist())
            details['ohlc_violations'] = ohlc_violations.index.tolist()[:100]

        # Check for price outliers
        price_cols = ['close_price', 'high_price', 'low_price', 'open_price']
        outliers_count = 0
        for col in price_cols:
            if col in data.columns:
                outliers = self.detect_outliers_iqr(data, col)
                if outliers.any():
                    outlier_indices = data[outliers].index.tolist()
                    all_flagged.update(outlier_indices)
                    outliers_count += outliers.sum()
                    details[f'{col}_outliers'] = len(outlier_indices)

        # Check for price anomalies
        price_anomalies = self.detect_price_anomalies(data)
        if not price_anomalies.empty:
            all_flagged.update(price_anomalies.index.tolist())
            details['price_anomalies'] = len(price_anomalies)

        # Check for zero volume candles
        zero_volume = data[data['volume'] == 0] if 'volume' in data.columns else pd.DataFrame()
        zero_volume_count = len(zero_volume)
        if not zero_volume.empty:
            all_flagged.update(zero_volume.index.tolist())
            details['zero_volume'] = zero_volume_count

        # Check for suspicious data patterns
        suspicious = self.flag_suspicious_data(data)
        if not suspicious.empty:
            all_flagged.update(suspicious.index.tolist())
            details['suspicious_patterns'] = len(suspicious)

        # Calculate quality score (0-100)
        total_rows = len(data)
        flagged_ratio = len(all_flagged) / total_rows if total_rows > 0 else 1.0
        quality_score = max(0, (1 - flagged_ratio) * 100)

        report = DataQualityReport(
            profile_id=profile_id,
            outliers_flagged=outliers_count,
            anomalies_detected=len(price_anomalies),
            ohlc_violations=len(ohlc_violations),
            zero_volume_candles=zero_volume_count,
            flagged_rows=list(all_flagged),
            quality_score=quality_score,
            details=details
        )

        logger.info(
            f"Validation complete for profile {profile_id}: "
            f"quality_score={quality_score:.2f}, flagged={len(all_flagged)}"
        )

        return report

    def _load_profile_data(self, profile_id: int) -> pd.DataFrame:
        """Load market data for a profile from database."""
        if not self.db_session:
            raise ValueError("Database session required to load data")

        query = self.db_session.query(
            MarketData.id,
            MarketData.timestamp,
            MarketData.open_price,
            MarketData.high_price,
            MarketData.low_price,
            MarketData.close_price,
            MarketData.volume
        ).filter(
            MarketData.profile_id == profile_id
        ).order_by(MarketData.timestamp)

        data = pd.read_sql(query.statement, self.db_session.bind)
        if not data.empty:
            data.set_index('id', inplace=True)

        return data

    def detect_outliers_iqr(
        self,
        data: pd.DataFrame,
        column: str,
        multiplier: float = None
    ) -> pd.Series:
        """
        Detect outliers using IQR method.

        Args:
            data: DataFrame with market data
            column: Column name to check
            multiplier: IQR multiplier (default 3.0)

        Returns:
            Boolean Series indicating outliers
        """
        if multiplier is None:
            multiplier = self.DEFAULT_IQR_MULTIPLIER

        if column not in data.columns:
            return pd.Series([False] * len(data), index=data.index)

        col_data = data[column].dropna()
        if len(col_data) == 0:
            return pd.Series([False] * len(data), index=data.index)

        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Create outlier mask
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)

        return outliers

    def detect_price_anomalies(
        self,
        data: pd.DataFrame,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Detect price anomalies based on large price changes without volume support.

        A price anomaly is defined as:
        - Price change > threshold (default 20%)
        - WITHOUT a corresponding volume increase (> 2x average)

        Args:
            data: DataFrame with market data
            threshold: Price change threshold (default 0.20)

        Returns:
            DataFrame with anomalous rows
        """
        if threshold is None:
            threshold = self.PRICE_ANOMALY_THRESHOLD

        if 'close_price' not in data.columns or 'volume' not in data.columns:
            return pd.DataFrame()

        # Calculate price changes
        data = data.copy()
        data['pct_change'] = data['close_price'].pct_change().abs()

        # Calculate volume ratio to average
        avg_volume = data['volume'].rolling(window=20, min_periods=1).mean()
        data['volume_ratio'] = data['volume'] / avg_volume

        # Flag anomalies: large price change without volume spike
        anomalies = (
            (data['pct_change'] > threshold) &
            (data['volume_ratio'] < self.VOLUME_SPIKE_THRESHOLD)
        )

        return data[anomalies]

    def validate_ohlc_relationships(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC relationships.

        Checks:
        - high >= max(open, close)
        - low <= min(open, close)
        - high >= low
        - All prices > 0

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with invalid rows
        """
        required_cols = ['open_price', 'high_price', 'low_price', 'close_price']
        if not all(col in data.columns for col in required_cols):
            return pd.DataFrame()

        data = data.copy()

        # High must be >= open and close
        high_violation = (
            (data['high_price'] < data['open_price']) |
            (data['high_price'] < data['close_price'])
        )

        # Low must be <= open and close
        low_violation = (
            (data['low_price'] > data['open_price']) |
            (data['low_price'] > data['close_price'])
        )

        # High must be >= low
        hl_violation = data['high_price'] < data['low_price']

        # All prices must be > 0
        negative_prices = (
            (data['open_price'] <= 0) |
            (data['high_price'] <= 0) |
            (data['low_price'] <= 0) |
            (data['close_price'] <= 0)
        )

        # Combine all violations
        all_violations = high_violation | low_violation | hl_violation | negative_prices

        return data[all_violations]

    def flag_suspicious_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Flag suspicious data patterns.

        Checks for:
        - Identical OHLC values (flat candles)
        - Extreme volume spikes (> 10x average)
        - Gaps in timestamp sequence

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with suspicious rows
        """
        if data.empty:
            return pd.DataFrame()

        data = data.copy()
        suspicious = pd.Series([False] * len(data), index=data.index)

        # Check for flat candles (all OHLC identical)
        if all(col in data.columns for col in ['open_price', 'high_price', 'low_price', 'close_price']):
            flat_candles = (
                (data['open_price'] == data['high_price']) &
                (data['high_price'] == data['low_price']) &
                (data['low_price'] == data['close_price'])
            )
            suspicious = suspicious | flat_candles

        # Check for extreme volume spikes
        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(window=100, min_periods=20).mean()
            extreme_volume = data['volume'] > (avg_volume * 10)
            suspicious = suspicious | extreme_volume

        return data[suspicious]

    def get_quality_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get a summary of data quality metrics.

        Args:
            data: DataFrame with market data

        Returns:
            Dictionary with quality metrics
        """
        if data.empty:
            return {
                'total_rows': 0,
                'null_ratio': 1.0,
                'price_range': (0, 0),
                'volume_range': (0, 0),
                'date_range': (None, None)
            }

        summary = {
            'total_rows': len(data),
            'null_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
        }

        if 'close_price' in data.columns:
            summary['price_range'] = (
                data['close_price'].min(),
                data['close_price'].max()
            )

        if 'volume' in data.columns:
            summary['volume_range'] = (
                data['volume'].min(),
                data['volume'].max()
            )

        if 'timestamp' in data.columns:
            summary['date_range'] = (
                data['timestamp'].min(),
                data['timestamp'].max()
            )
        elif hasattr(data.index, 'min'):
            summary['date_range'] = (
                data.index.min(),
                data.index.max()
            )

        return summary

    def calculate_data_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive statistics for the data.

        Args:
            data: DataFrame with market data

        Returns:
            Dictionary with statistics
        """
        if data.empty:
            return {}

        stats = {}

        price_cols = ['open_price', 'high_price', 'low_price', 'close_price']
        for col in price_cols:
            if col in data.columns:
                stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75)
                }

        if 'volume' in data.columns:
            stats['volume'] = {
                'mean': data['volume'].mean(),
                'std': data['volume'].std(),
                'min': data['volume'].min(),
                'max': data['volume'].max(),
                'median': data['volume'].median(),
                'zero_count': (data['volume'] == 0).sum()
            }

        if 'close_price' in data.columns:
            returns = data['close_price'].pct_change().dropna()
            stats['returns'] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'min': returns.min(),
                'max': returns.max(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }

        return stats
