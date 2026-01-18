"""
Data Gap Filler
Fills gaps in market data by fetching from Binance API.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from database.models import TradingProfile, MarketData
from exchanges.binance_connector import BinanceConnector
from crypto_ml_trading.training.data_auditor import DataGap

logger = logging.getLogger(__name__)


@dataclass
class GapFillReport:
    """Report of gap filling operation."""
    total_gaps: int
    gaps_filled: int
    gaps_unfillable: int
    candles_inserted: int
    candles_skipped_duplicate: int
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_gaps': self.total_gaps,
            'gaps_filled': self.gaps_filled,
            'gaps_unfillable': self.gaps_unfillable,
            'candles_inserted': self.candles_inserted,
            'candles_skipped_duplicate': self.candles_skipped_duplicate,
            'errors': self.errors
        }


class DataGapFiller:
    """Fills gaps in market data by fetching from Binance API."""

    # Maximum retries for API calls
    MAX_RETRIES = 5
    # Base delay for exponential backoff (seconds)
    BASE_DELAY = 1.0
    # Maximum delay cap (seconds)
    MAX_DELAY = 60.0
    # Batch size for bulk inserts
    BATCH_INSERT_SIZE = 1000

    def __init__(
        self,
        binance_connector: BinanceConnector,
        db_session: Session
    ):
        self.binance_connector = binance_connector
        self.db_session = db_session

    async def fill_gaps(
        self,
        profile_id: int,
        gaps: List[DataGap],
        progress_callback: Optional[callable] = None
    ) -> GapFillReport:
        """
        Fill all gaps in market data for a profile.

        Args:
            profile_id: ID of the trading profile
            gaps: List of DataGap objects to fill
            progress_callback: Optional async callback(current, total, message) for progress

        Returns:
            GapFillReport with results
        """
        if not gaps:
            return GapFillReport(
                total_gaps=0,
                gaps_filled=0,
                gaps_unfillable=0,
                candles_inserted=0,
                candles_skipped_duplicate=0
            )

        # Get profile info
        profile = self.db_session.query(TradingProfile).filter(
            TradingProfile.id == profile_id
        ).first()

        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        symbol = profile.symbol

        logger.info(f"Starting gap fill for profile {profile_id} ({symbol}): {len(gaps)} gaps")

        total_gaps = len(gaps)
        gaps_filled = 0
        gaps_unfillable = 0
        total_inserted = 0
        total_duplicates = 0
        errors = []

        for i, gap in enumerate(gaps):
            try:
                if progress_callback:
                    await progress_callback(
                        i + 1,
                        total_gaps,
                        f"Filling gap {i+1}/{total_gaps}: {gap.missing_minutes} minutes"
                    )

                logger.info(f"Filling gap {i+1}/{total_gaps}: {gap.start} to {gap.end} ({gap.missing_minutes} minutes)")

                # Fetch candles for this gap
                candles = await self.fetch_candles_for_gap(symbol, gap)

                if candles:
                    # Insert candles with duplicate handling
                    inserted, duplicates = self.insert_candles_batch(profile_id, symbol, candles)
                    total_inserted += inserted
                    total_duplicates += duplicates
                    gaps_filled += 1
                    logger.info(f"Gap {i+1} filled: {inserted} inserted, {duplicates} duplicates")
                else:
                    gaps_unfillable += 1
                    errors.append(f"Gap {i+1}: No candles returned from API")
                    logger.warning(f"Gap {i+1} unfillable: no data from API")

            except Exception as e:
                gaps_unfillable += 1
                error_msg = f"Gap {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Error filling gap {i+1}: {e}")

        # Commit all changes
        self.db_session.commit()

        # Update profile data count
        self._update_profile_data_count(profile_id)

        report = GapFillReport(
            total_gaps=total_gaps,
            gaps_filled=gaps_filled,
            gaps_unfillable=gaps_unfillable,
            candles_inserted=total_inserted,
            candles_skipped_duplicate=total_duplicates,
            errors=errors
        )

        logger.info(
            f"Gap fill complete for profile {profile_id}: "
            f"{gaps_filled}/{total_gaps} filled, {total_inserted} candles inserted"
        )

        return report

    async def fetch_candles_for_gap(
        self,
        symbol: str,
        gap: DataGap
    ) -> List[Dict]:
        """
        Fetch candles from Binance for a specific gap.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            gap: DataGap object defining the gap to fill

        Returns:
            List of candle dictionaries
        """
        retry_count = 0
        last_error = None

        while retry_count < self.MAX_RETRIES:
            try:
                # Use the fill_gap method from BinanceConnector
                df = await self.binance_connector.fill_gap(
                    symbol=symbol,
                    gap_start=gap.start,
                    gap_end=gap.end
                )

                if df.empty:
                    return []

                # Convert DataFrame to list of dicts
                df = df.reset_index()
                candles = df.to_dict('records')

                return candles

            except Exception as e:
                last_error = e
                retry_count += 1
                delay = self.handle_rate_limit(retry_count)
                logger.warning(
                    f"Error fetching candles for gap, retry {retry_count}/{self.MAX_RETRIES}: {e}"
                )
                await asyncio.sleep(delay)

        logger.error(f"Failed to fetch candles after {self.MAX_RETRIES} retries: {last_error}")
        raise last_error

    def insert_candles_batch(
        self,
        profile_id: int,
        symbol: str,
        candles: List[Dict]
    ) -> tuple:
        """
        Insert candles in batches with duplicate handling.

        Uses PostgreSQL ON CONFLICT DO NOTHING for efficient duplicate handling.

        Args:
            profile_id: Profile ID
            symbol: Trading pair symbol
            candles: List of candle dictionaries

        Returns:
            Tuple of (inserted_count, duplicate_count)
        """
        if not candles:
            return 0, 0

        total_inserted = 0
        total_duplicates = 0

        # Process in batches
        for batch_start in range(0, len(candles), self.BATCH_INSERT_SIZE):
            batch = candles[batch_start:batch_start + self.BATCH_INSERT_SIZE]

            # Prepare records for insert
            records = []
            for candle in batch:
                record = {
                    'profile_id': profile_id,
                    'symbol': symbol,
                    'timestamp': candle['timestamp'],
                    'open_price': candle['open'],
                    'high_price': candle['high'],
                    'low_price': candle['low'],
                    'close_price': candle['close'],
                    'volume': candle['volume'],
                    'quote_asset_volume': candle.get('quote_volume'),
                    'number_of_trades': candle.get('trades')
                }
                records.append(record)

            # Use PostgreSQL upsert with ON CONFLICT DO NOTHING
            stmt = insert(MarketData).values(records)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=['profile_id', 'timestamp']
            )

            result = self.db_session.execute(stmt)

            # Count inserted vs duplicates
            batch_inserted = result.rowcount if result.rowcount >= 0 else len(batch)
            batch_duplicates = len(batch) - batch_inserted

            total_inserted += batch_inserted
            total_duplicates += batch_duplicates

        return total_inserted, total_duplicates

    def handle_rate_limit(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            retry_count: Current retry attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.BASE_DELAY * (2 ** retry_count),
            self.MAX_DELAY
        )
        return delay

    def _update_profile_data_count(self, profile_id: int) -> None:
        """Update the profile's total_data_points count."""
        from sqlalchemy import func

        count = self.db_session.query(func.count(MarketData.id)).filter(
            MarketData.profile_id == profile_id
        ).scalar() or 0

        profile = self.db_session.query(TradingProfile).filter(
            TradingProfile.id == profile_id
        ).first()

        if profile:
            profile.total_data_points = count
            profile.has_data = count > 0
            profile.data_updated_at = datetime.utcnow()

    async def fill_gaps_to_target_coverage(
        self,
        profile_id: int,
        target_coverage: float = 95.0,
        max_iterations: int = 3
    ) -> GapFillReport:
        """
        Iteratively fill gaps until target coverage is reached.

        Args:
            profile_id: Profile ID
            target_coverage: Target coverage percentage (default 95%)
            max_iterations: Maximum fill iterations

        Returns:
            Combined GapFillReport from all iterations
        """
        from crypto_ml_trading.training.data_auditor import ProfileDataAuditor

        auditor = ProfileDataAuditor(self.db_session)
        combined_report = GapFillReport(
            total_gaps=0,
            gaps_filled=0,
            gaps_unfillable=0,
            candles_inserted=0,
            candles_skipped_duplicate=0
        )

        for iteration in range(max_iterations):
            logger.info(f"Gap fill iteration {iteration + 1}/{max_iterations}")

            # Audit current state
            audit_report = auditor.audit_profile(profile_id)

            if audit_report.coverage_percent >= target_coverage:
                logger.info(f"Target coverage reached: {audit_report.coverage_percent:.2f}%")
                break

            if not audit_report.gaps:
                logger.info("No more gaps to fill")
                break

            # Fill gaps
            report = await self.fill_gaps(profile_id, audit_report.gaps)

            # Accumulate results
            combined_report.total_gaps += report.total_gaps
            combined_report.gaps_filled += report.gaps_filled
            combined_report.gaps_unfillable += report.gaps_unfillable
            combined_report.candles_inserted += report.candles_inserted
            combined_report.candles_skipped_duplicate += report.candles_skipped_duplicate
            combined_report.errors.extend(report.errors)

            # If no gaps were filled in this iteration, stop
            if report.gaps_filled == 0:
                logger.info("No gaps filled in this iteration, stopping")
                break

        return combined_report
