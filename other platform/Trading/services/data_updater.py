"""
Automatic Data Updater Service
================================

This service runs every minute to automatically update trading data
for all active profiles with auto-updates enabled.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import pytz

from sqlalchemy.orm import Session
from sqlalchemy import desc
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from database.models import SessionLocal, TradingProfile, MarketData
from exchanges.binance_connector import BinanceConnector
from services.prediction_service import prediction_service
from services.accuracy_tracker import accuracy_tracker
from services.model_retrainer import model_retrainer

logger = logging.getLogger(__name__)

# Database timezone (South Africa Standard Time)
DB_TIMEZONE = pytz.timezone('Africa/Johannesburg')
UTC_TIMEZONE = pytz.UTC


class DataUpdaterService:
    """Service for automatic data updates"""

    def __init__(self):
        self.binance = BinanceConnector(testnet=False)  # Use production Binance API
        self.scheduler = None
        self._session_initialized = False

    def start_scheduler(self):
        """Initialize and start the scheduler"""
        self.scheduler = AsyncIOScheduler()

        # Job 1: Data updates + predictions (every minute)
        self.scheduler.add_job(
            self.update_all_active_profiles,
            'cron',
            minute='*',  # Every minute
            id='auto_data_update',
            replace_existing=True
        )

        # Job 2: Accuracy tracking (every hour)
        self.scheduler.add_job(
            accuracy_tracker.track_accuracy_all_models,
            'cron',
            minute='5',  # Run at 5 minutes past each hour
            id='accuracy_tracking',
            replace_existing=True
        )

        # Job 3: Monthly model retraining (1st of month at 2 AM)
        self.scheduler.add_job(
            model_retrainer.retrain_all_models,
            'cron',
            day='1',  # 1st of month
            hour='2',  # 2 AM
            minute='0',
            id='monthly_retraining',
            replace_existing=True
        )

        self.scheduler.start()
        logger.info("✅ Auto-update scheduler started")
        logger.info("  - Data updates + predictions: Every minute")
        logger.info("  - Accuracy tracking: Every hour at :05")
        logger.info("  - Monthly retraining: 1st of month at 2:00 AM")

    async def update_all_active_profiles(self):
        """
        Main update function - called every minute by scheduler
        Updates all profiles where is_active=True and has_data=True
        """
        # Initialize Binance session if not already done
        if not self._session_initialized:
            await self.binance.connect()
            self._session_initialized = True

        db = SessionLocal()
        try:
            # Get all active profiles with data
            active_profiles = db.query(TradingProfile).filter(
                TradingProfile.is_active == True,
                TradingProfile.has_data == True
            ).all()

            if not active_profiles:
                logger.debug("No active profiles to update")
                return

            logger.info(f"Updating {len(active_profiles)} active profiles")

            # Update each profile
            for profile in active_profiles:
                try:
                    await self._update_single_profile(profile)
                except Exception as e:
                    logger.error(f"Error updating {profile.symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in update_all_active_profiles: {e}", exc_info=True)
        finally:
            db.close()

    async def _update_single_profile(self, profile: TradingProfile) -> bool:
        """Update a single trading profile"""
        db = SessionLocal()
        try:
            symbol = profile.symbol

            # Re-fetch profile in this session to allow updates
            profile = db.query(TradingProfile).filter(
                TradingProfile.symbol == symbol
            ).first()

            if not profile:
                logger.error(f"{symbol}: Profile not found")
                return False

            # Get latest timestamp from database
            latest_data = db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(desc(MarketData.timestamp)).first()

            if not latest_data:
                logger.debug(f"{symbol}: No existing data, skipping")
                return False

            # Convert database timestamp (stored in SAST) to UTC
            latest_timestamp_sast = DB_TIMEZONE.localize(latest_data.timestamp)
            latest_timestamp_utc = latest_timestamp_sast.astimezone(UTC_TIMEZONE)

            # Get current UTC time (timezone-aware)
            now_utc = datetime.now(UTC_TIMEZONE)

            # Check if enough time has passed (at least 1 minute)
            time_diff = (now_utc - latest_timestamp_utc).total_seconds()
            if time_diff < 60:
                logger.debug(f"{symbol}: Too soon to update ({int(time_diff)}s)")
                return True

            logger.info(f"{symbol}: Gap of {int(time_diff/60)} minutes detected, fetching new candles...")

            # Fetch new candles (convert to naive UTC for API)
            new_candles = await self._fetch_new_candles(
                profile,
                start_time=(latest_timestamp_utc + timedelta(seconds=60)).replace(tzinfo=None),
                end_time=now_utc.replace(tzinfo=None)
            )

            if new_candles is None or len(new_candles) == 0:
                logger.debug(f"{symbol}: No new candles available")
                return True

            # Insert new candles
            inserted_count = self._insert_candles(db, profile.id, symbol, new_candles)

            if inserted_count > 0:
                # Update profile metadata (convert UTC back to SAST for database)
                now_sast = now_utc.astimezone(DB_TIMEZONE).replace(tzinfo=None)
                profile.data_updated_at = now_sast
                profile.total_data_points = db.query(MarketData).filter(
                    MarketData.symbol == symbol
                ).count()
                db.commit()

                logger.info(f"✅ {symbol}: Added {inserted_count} new candles (total: {profile.total_data_points})")

                # Generate predictions with updated data
                try:
                    prediction_result = await prediction_service.generate_predictions_for_profile(profile, db)
                    if prediction_result['success']:
                        logger.info(f"✅ {symbol}: Generated {prediction_result['predictions_generated']} predictions")
                    else:
                        logger.debug(f"{symbol}: Prediction generation skipped - {prediction_result.get('reason', 'unknown')}")
                except Exception as e:
                    logger.error(f"{symbol}: Error generating predictions: {e}")

            return True

        except Exception as e:
            logger.error(f"Error updating {profile.symbol}: {e}", exc_info=True)
            db.rollback()
            return False
        finally:
            db.close()

    async def _fetch_new_candles(self, profile: TradingProfile, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch new candles from exchange"""
        try:
            if profile.data_source == 'binance':
                logger.info(f"{profile.symbol}: Fetching klines from {start_time} to {end_time}")

                # Use date range to fetch candles from start_time to end_time
                # This handles gaps from server downtime automatically
                klines = await self.binance.get_klines(
                    symbol=profile.symbol,
                    interval=profile.data_interval or '1m',
                    limit=1000,  # Increased from 10 to handle longer gaps
                    start_time=start_time,
                    end_time=end_time
                )

                logger.info(f"{profile.symbol}: Binance returned {len(klines)} klines")

                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': k.open_time,
                    'open': float(k.open),
                    'high': float(k.high),
                    'low': float(k.low),
                    'close': float(k.close),
                    'volume': float(k.volume),
                    'quote_asset_volume': float(k.quote_volume),
                    'number_of_trades': k.trades
                } for k in klines])

                logger.info(f"{profile.symbol}: DataFrame created with {len(df)} rows")

                # Filter to only candles newer than what we have
                if not df.empty:
                    logger.info(f"{profile.symbol}: Before filter - timestamps range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    df = df[df['timestamp'] > start_time]
                    logger.info(f"{profile.symbol}: After filtering (timestamp > {start_time}): {len(df)} rows")

                return df if not df.empty else None
            else:
                logger.warning(f"Unsupported data source: {profile.data_source}")
                return None

        except Exception as e:
            logger.error(f"Error fetching candles: {e}", exc_info=True)
            return None

    async def backfill_missing_data(self, symbol: str) -> dict:
        """
        Backfill missing historical data for a symbol.
        Fetches data in batches of 1000 candles until the gap is filled.

        Returns:
            dict with status, candles_added, and time_range
        """
        # Initialize session if needed
        if not self._session_initialized:
            await self.binance.connect()
            self._session_initialized = True

        db = SessionLocal()
        try:
            # Get profile
            profile = db.query(TradingProfile).filter(
                TradingProfile.symbol == symbol
            ).first()

            if not profile:
                return {
                    'success': False,
                    'error': f'Profile not found for symbol {symbol}'
                }

            # Get latest timestamp from database
            latest_data = db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(desc(MarketData.timestamp)).first()

            if not latest_data:
                return {
                    'success': False,
                    'error': f'No existing data found for {symbol}'
                }

            latest_timestamp = latest_data.timestamp
            now = datetime.now(timezone.utc).replace(tzinfo=None)  # Get actual UTC time

            logger.info(f"Starting backfill for {symbol} from {latest_timestamp} to {now}")

            total_inserted = 0
            batch_count = 0
            max_batches = 100  # Increased to handle larger gaps (100 batches = 100,000 candles)

            # Start from the next minute after latest data
            current_start = latest_timestamp + timedelta(minutes=1)

            # Keep fetching batches until we're caught up
            while batch_count < max_batches and current_start < now:
                batch_count += 1

                # Calculate end time for this batch (1000 minutes ahead or until now)
                batch_end = min(current_start + timedelta(minutes=1000), now)

                # Fetch batch of candles with date range
                klines = await self.binance.get_klines(
                    symbol=profile.symbol,
                    interval=profile.data_interval or '1m',
                    limit=1000,
                    start_time=current_start,
                    end_time=batch_end
                )

                if not klines:
                    logger.info(f"No candles returned for {symbol} from {current_start} to {batch_end}")
                    break

                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': k.open_time,
                    'open': float(k.open),
                    'high': float(k.high),
                    'low': float(k.low),
                    'close': float(k.close),
                    'volume': float(k.volume),
                    'quote_asset_volume': float(k.quote_volume),
                    'number_of_trades': k.trades
                } for k in klines])

                if df.empty:
                    logger.info(f"Caught up! No more new candles for {symbol}")
                    break

                # Insert batch
                inserted = self._insert_candles(db, profile.id, symbol, df)
                total_inserted += inserted

                logger.info(f"Batch {batch_count}: Added {inserted} candles for {symbol} (total: {total_inserted}) [{current_start} to {batch_end}]")

                # Update current_start for next iteration
                current_start = df['timestamp'].max() + timedelta(minutes=1)

                # Check if we've caught up
                if current_start >= now:
                    logger.info(f"Caught up to current time for {symbol}")
                    break

                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)

            # Update profile metadata
            profile.data_updated_at = now
            profile.total_data_points = db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).count()
            db.commit()

            return {
                'success': True,
                'symbol': symbol,
                'candles_added': total_inserted,
                'batches_processed': batch_count,
                'total_candles': profile.total_data_points,
                'time_range': {
                    'from': str(latest_data.timestamp),
                    'to': str(latest_timestamp)
                }
            }

        except Exception as e:
            logger.error(f"Error during backfill for {symbol}: {e}", exc_info=True)
            db.rollback()
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            db.close()

    def _insert_candles(self, db: Session, profile_id: int, symbol: str, candles: pd.DataFrame) -> int:
        """Insert new candles into database"""
        inserted_count = 0

        for _, row in candles.iterrows():
            try:
                # Check if already exists
                existing = db.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp == row['timestamp']
                ).first()

                if not existing:
                    # Insert new candle
                    market_data = MarketData(
                        symbol=symbol,
                        profile_id=profile_id,
                        timestamp=row['timestamp'],
                        open_price=float(row.get('open', row.get('Open', 0))),
                        high_price=float(row.get('high', row.get('High', 0))),
                        low_price=float(row.get('low', row.get('Low', 0))),
                        close_price=float(row.get('close', row.get('Close', 0))),
                        volume=float(row.get('volume', row.get('Volume', 0))),
                        number_of_trades=int(row.get('number_of_trades', 0)),
                        quote_asset_volume=float(row.get('quote_asset_volume', 0))
                    )
                    db.add(market_data)
                    inserted_count += 1

                    # Commit in batches
                    if inserted_count % 100 == 0:
                        db.commit()

            except Exception as e:
                logger.error(f"Error inserting candle: {e}")
                continue

        db.commit()
        return inserted_count


# Singleton instance
data_updater = DataUpdaterService()
