"""
Profile Data Auditor
Audits profile data completeness before training.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from database.models import TradingProfile, MarketData

logger = logging.getLogger(__name__)


@dataclass
class DataGap:
    """Represents a gap in market data."""
    start: datetime
    end: datetime
    missing_minutes: int

    def __post_init__(self):
        if self.missing_minutes <= 0:
            self.missing_minutes = int((self.end - self.start).total_seconds() / 60)


@dataclass
class DataAuditReport:
    """Comprehensive report of profile data completeness."""
    profile_id: int
    symbol: str
    total_candles: int
    expected_candles: int
    coverage_percent: float
    date_range: Tuple[Optional[datetime], Optional[datetime]]
    gaps: List[DataGap]
    null_statistics: Dict[str, int]
    invalid_data_count: int
    indicator_coverage_percent: float
    is_training_ready: bool
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            'profile_id': self.profile_id,
            'symbol': self.symbol,
            'total_candles': self.total_candles,
            'expected_candles': self.expected_candles,
            'coverage_percent': self.coverage_percent,
            'date_range': {
                'start': self.date_range[0].isoformat() if self.date_range[0] else None,
                'end': self.date_range[1].isoformat() if self.date_range[1] else None
            },
            'gaps_count': len(self.gaps),
            'total_missing_minutes': sum(g.missing_minutes for g in self.gaps),
            'gaps': [
                {
                    'start': g.start.isoformat(),
                    'end': g.end.isoformat(),
                    'missing_minutes': g.missing_minutes
                }
                for g in self.gaps[:10]  # Limit to first 10 gaps in output
            ],
            'null_statistics': self.null_statistics,
            'invalid_data_count': self.invalid_data_count,
            'indicator_coverage_percent': self.indicator_coverage_percent,
            'is_training_ready': self.is_training_ready,
            'issues': self.issues
        }


class ProfileDataAuditor:
    """Audits profile data completeness before training."""

    # Required coverage threshold for training
    MIN_COVERAGE_PERCENT = 95.0
    MIN_INDICATOR_COVERAGE = 90.0
    MIN_CANDLES_FOR_TRAINING = 100000  # ~70 days of 1-minute data
    MAX_GAP_MINUTES = 60  # Maximum acceptable single gap

    def __init__(self, db_session: Session):
        self.db_session = db_session

    def audit_profile(self, profile_id: int) -> DataAuditReport:
        """
        Perform comprehensive data audit for a profile.

        Args:
            profile_id: ID of the trading profile to audit

        Returns:
            DataAuditReport with complete audit results
        """
        logger.info(f"Starting audit for profile {profile_id}")

        # Get profile info
        profile = self.db_session.query(TradingProfile).filter(
            TradingProfile.id == profile_id
        ).first()

        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        symbol = profile.symbol

        # Get date range and total candles
        date_range = self._get_date_range(profile_id)
        total_candles = self._get_total_candles(profile_id)

        # Calculate expected candles based on date range
        if date_range[0] and date_range[1]:
            expected_candles = self._calculate_expected_candles(date_range[0], date_range[1])
        else:
            expected_candles = 0

        # Calculate coverage
        coverage_percent = (total_candles / expected_candles * 100) if expected_candles > 0 else 0.0

        # Find gaps
        gaps = self.find_gaps(profile_id, min_gap_minutes=2)

        # Check null values
        null_statistics = self.check_null_values(profile_id)

        # Check invalid data (OHLC violations)
        invalid_data_count = self.check_invalid_data(profile_id)

        # Check indicator coverage
        indicator_coverage = self.check_indicator_coverage(profile_id)

        # Compile issues
        issues = []

        if total_candles < self.MIN_CANDLES_FOR_TRAINING:
            issues.append(f"Insufficient data: {total_candles} candles (need {self.MIN_CANDLES_FOR_TRAINING})")

        if coverage_percent < self.MIN_COVERAGE_PERCENT:
            issues.append(f"Low coverage: {coverage_percent:.2f}% (need {self.MIN_COVERAGE_PERCENT}%)")

        if len(gaps) > 0:
            large_gaps = [g for g in gaps if g.missing_minutes > self.MAX_GAP_MINUTES]
            if large_gaps:
                issues.append(f"Found {len(large_gaps)} gaps larger than {self.MAX_GAP_MINUTES} minutes")
            total_missing = sum(g.missing_minutes for g in gaps)
            issues.append(f"Total {len(gaps)} gaps, {total_missing} minutes missing")

        total_nulls = sum(null_statistics.values())
        if total_nulls > 0:
            issues.append(f"Found {total_nulls} null values across columns")

        if invalid_data_count > 0:
            issues.append(f"Found {invalid_data_count} OHLC validation errors")

        if indicator_coverage < self.MIN_INDICATOR_COVERAGE:
            issues.append(f"Low indicator coverage: {indicator_coverage:.2f}% (need {self.MIN_INDICATOR_COVERAGE}%)")

        # Determine if ready for training
        is_ready = self.is_training_ready_check(
            total_candles=total_candles,
            coverage_percent=coverage_percent,
            invalid_data_count=invalid_data_count,
            null_statistics=null_statistics,
            indicator_coverage=indicator_coverage
        )

        report = DataAuditReport(
            profile_id=profile_id,
            symbol=symbol,
            total_candles=total_candles,
            expected_candles=expected_candles,
            coverage_percent=coverage_percent,
            date_range=date_range,
            gaps=gaps,
            null_statistics=null_statistics,
            invalid_data_count=invalid_data_count,
            indicator_coverage_percent=indicator_coverage,
            is_training_ready=is_ready,
            issues=issues
        )

        logger.info(f"Audit complete for profile {profile_id}: ready={is_ready}, coverage={coverage_percent:.2f}%")

        return report

    def _get_date_range(self, profile_id: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the first and last timestamp for a profile's market data."""
        result = self.db_session.query(
            func.min(MarketData.timestamp),
            func.max(MarketData.timestamp)
        ).filter(MarketData.profile_id == profile_id).first()

        return (result[0], result[1]) if result else (None, None)

    def _get_total_candles(self, profile_id: int) -> int:
        """Get total number of candles for a profile."""
        return self.db_session.query(func.count(MarketData.id)).filter(
            MarketData.profile_id == profile_id
        ).scalar() or 0

    def _calculate_expected_candles(self, start: datetime, end: datetime) -> int:
        """Calculate expected number of 1-minute candles in a date range."""
        total_minutes = int((end - start).total_seconds() / 60)
        return total_minutes + 1  # +1 to include both endpoints

    def check_candle_coverage(
        self,
        profile_id: int,
        start: datetime,
        end: datetime
    ) -> float:
        """
        Check candle coverage for a specific date range.

        Args:
            profile_id: Profile ID
            start: Start datetime
            end: End datetime

        Returns:
            Coverage percentage (0-100)
        """
        actual_count = self.db_session.query(func.count(MarketData.id)).filter(
            MarketData.profile_id == profile_id,
            MarketData.timestamp >= start,
            MarketData.timestamp <= end
        ).scalar() or 0

        expected_count = self._calculate_expected_candles(start, end)

        return (actual_count / expected_count * 100) if expected_count > 0 else 0.0

    def find_gaps(
        self,
        profile_id: int,
        min_gap_minutes: int = 5
    ) -> List[DataGap]:
        """
        Find gaps in market data where consecutive timestamps differ by more than expected.

        Args:
            profile_id: Profile ID
            min_gap_minutes: Minimum gap size to report (default 5 minutes)

        Returns:
            List of DataGap objects
        """
        # PostgreSQL-specific query using LAG window function
        query = text("""
            WITH consecutive AS (
                SELECT
                    timestamp,
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                FROM market_data
                WHERE profile_id = :profile_id
            )
            SELECT
                prev_timestamp as gap_start,
                timestamp as gap_end,
                EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) / 60 as gap_minutes
            FROM consecutive
            WHERE prev_timestamp IS NOT NULL
              AND EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) / 60 >= :min_gap
            ORDER BY prev_timestamp
        """)

        result = self.db_session.execute(
            query,
            {'profile_id': profile_id, 'min_gap': min_gap_minutes}
        )

        gaps = []
        for row in result:
            gaps.append(DataGap(
                start=row.gap_start,
                end=row.gap_end,
                missing_minutes=int(row.gap_minutes)
            ))

        logger.info(f"Found {len(gaps)} gaps for profile {profile_id}")
        return gaps

    def check_null_values(self, profile_id: int) -> Dict[str, int]:
        """
        Check for null values in critical columns.

        Args:
            profile_id: Profile ID

        Returns:
            Dictionary mapping column name to null count
        """
        columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        null_stats = {}

        for col in columns:
            count = self.db_session.query(func.count(MarketData.id)).filter(
                MarketData.profile_id == profile_id,
                getattr(MarketData, col) == None
            ).scalar() or 0

            if count > 0:
                null_stats[col] = count

        return null_stats

    def check_invalid_data(self, profile_id: int) -> int:
        """
        Check for OHLC validation errors.

        Validates:
        - high >= max(open, close)
        - low <= min(open, close)
        - high >= low
        - All prices > 0
        - volume >= 0

        Args:
            profile_id: Profile ID

        Returns:
            Count of invalid rows
        """
        query = text("""
            SELECT COUNT(*) as invalid_count
            FROM market_data
            WHERE profile_id = :profile_id
              AND (
                high_price < open_price OR
                high_price < close_price OR
                low_price > open_price OR
                low_price > close_price OR
                high_price < low_price OR
                open_price <= 0 OR
                high_price <= 0 OR
                low_price <= 0 OR
                close_price <= 0 OR
                volume < 0
              )
        """)

        result = self.db_session.execute(query, {'profile_id': profile_id}).scalar()
        return result or 0

    def check_indicator_coverage(self, profile_id: int) -> float:
        """
        Check what percentage of data has indicators calculated.

        This checks if the profile has been through the indicator calculation pipeline.

        Args:
            profile_id: Profile ID

        Returns:
            Indicator coverage percentage (0-100)
        """
        # Check profile's indicator status
        profile = self.db_session.query(TradingProfile).filter(
            TradingProfile.id == profile_id
        ).first()

        if not profile:
            return 0.0

        # Check if indicators have been calculated
        if not profile.has_indicators:
            return 0.0

        # For now, if has_indicators is True, assume 100% coverage
        # In a more sophisticated implementation, we could check specific indicator columns
        return 100.0 if profile.has_indicators else 0.0

    def is_training_ready_check(
        self,
        total_candles: int,
        coverage_percent: float,
        invalid_data_count: int,
        null_statistics: Dict[str, int],
        indicator_coverage: float
    ) -> bool:
        """
        Determine if a profile is ready for model training.

        Args:
            total_candles: Total number of candles
            coverage_percent: Data coverage percentage
            invalid_data_count: Number of invalid OHLC rows
            null_statistics: Null value counts by column
            indicator_coverage: Indicator coverage percentage

        Returns:
            True if ready for training, False otherwise
        """
        # Check minimum data requirements
        if total_candles < self.MIN_CANDLES_FOR_TRAINING:
            return False

        # Check coverage
        if coverage_percent < self.MIN_COVERAGE_PERCENT:
            return False

        # Check for invalid data (allow up to 0.1% invalid)
        invalid_ratio = invalid_data_count / total_candles if total_candles > 0 else 1.0
        if invalid_ratio > 0.001:
            return False

        # Check for nulls in critical columns
        total_nulls = sum(null_statistics.values())
        null_ratio = total_nulls / total_candles if total_candles > 0 else 1.0
        if null_ratio > 0.001:
            return False

        # Check indicator coverage
        if indicator_coverage < self.MIN_INDICATOR_COVERAGE:
            return False

        return True

    def is_training_ready(self, report: DataAuditReport) -> bool:
        """
        Check if a profile is training-ready based on an audit report.

        Args:
            report: DataAuditReport from audit_profile()

        Returns:
            True if ready for training
        """
        return report.is_training_ready

    def get_gaps_summary(self, gaps: List[DataGap]) -> Dict:
        """
        Get a summary of data gaps.

        Args:
            gaps: List of DataGap objects

        Returns:
            Dictionary with gap statistics
        """
        if not gaps:
            return {
                'total_gaps': 0,
                'total_missing_minutes': 0,
                'largest_gap_minutes': 0,
                'average_gap_minutes': 0
            }

        missing_minutes = [g.missing_minutes for g in gaps]

        return {
            'total_gaps': len(gaps),
            'total_missing_minutes': sum(missing_minutes),
            'largest_gap_minutes': max(missing_minutes),
            'average_gap_minutes': sum(missing_minutes) / len(missing_minutes),
            'gaps_over_1_hour': sum(1 for m in missing_minutes if m > 60),
            'gaps_over_24_hours': sum(1 for m in missing_minutes if m > 1440)
        }
