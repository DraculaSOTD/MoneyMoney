"""
Automatic Data Updater Service
================================

This service runs every minute to automatically update trading data
for all active profiles. It fetches the latest candles and calculates
indicators incrementally.

Features:
- Fetches new candles every minute for active profiles
- Calculates indicators incrementally (only for new data)
- Broadcasts updates via WebSocket
- Error handling and retry logic
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd

from sqlalchemy.orm import Session
from sqlalchemy import desc

from python_backend.database.models import SessionLocal, TradingProfile, MarketData, IndicatorData
from python_backend.exchanges.binance_connector import BinanceConnector
from python_backend.exchanges.yahoo_connector import YahooConnector
from python_backend.services.websocket_manager import connection_manager
from python_backend.utils.enhanced_technical_indicators import EnhancedTechnicalIndicators

logger = logging.getLogger(__name__)


class DataUpdaterService:
    """Service for automatic data updates"""

    def __init__(self):
        self.binance = BinanceConnector()
        self.yahoo = YahooConnector()
        self.indicators_calculator = EnhancedTechnicalIndicators()

    async def update_all_active_profiles(self):
        """
        Main update function - called every minute by scheduler
        Updates all profiles where is_active=True and auto_update_enabled=True
        """
        db = SessionLocal()
        try:
            # Get all active profiles
            active_profiles = db.query(TradingProfile).filter(
                TradingProfile.is_active == True,
                TradingProfile.has_data == True,
                TradingProfile.auto_update_enabled == True
            ).all()

            if not active_profiles:
                logger.debug("No active profiles to update")
                return

            logger.info(f"Updating {len(active_profiles)} active profiles")

            # Update each profile
            tasks = []
            for profile in active_profiles:
                task = self._update_single_profile(profile)
                tasks.append(task)

            # Run updates in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            success_count = sum(1 for r in results if r is True)
            error_count = sum(1 for r in results if isinstance(r, Exception))

            logger.info(f"Update complete: {success_count} successful, {error_count} errors")

        except Exception as e:
            logger.error(f"Error in update_all_active_profiles: {e}", exc_info=True)
        finally:
            db.close()

    async def _update_single_profile(self, profile: TradingProfile) -> bool:
        """
        Update a single trading profile

        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            symbol = profile.symbol
            logger.debug(f"Updating profile: {symbol}")

            # Step 1: Get latest timestamp from database
            latest_data = db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(desc(MarketData.timestamp)).first()

            if not latest_data:
                logger.warning(f"No existing data for {symbol}, skipping update")
                return False

            latest_timestamp = latest_data.timestamp
            now = datetime.utcnow()

            # Step 2: Check if enough time has passed (at least 1 minute)
            time_diff = (now - latest_timestamp).total_seconds()
            if time_diff < 60:
                logger.debug(f"{symbol}: Too soon to update ({time_diff}s since last)")
                return True  # Not an error, just skipping

            # Step 3: Fetch new candles
            try:
                new_candles = await self._fetch_new_candles(
                    profile,
                    start_time=latest_timestamp + timedelta(seconds=60),
                    end_time=now
                )
            except Exception as e:
                logger.error(f"Error fetching candles for {symbol}: {e}")
                return False

            if new_candles is None or len(new_candles) == 0:
                logger.debug(f"No new candles for {symbol}")
                return True

            logger.info(f"{symbol}: Found {len(new_candles)} new candles")

            # Step 4: Insert new candles into database (UPSERT)
            inserted_count = await self._insert_candles(db, profile.id, symbol, new_candles)

            # Step 5: Calculate indicators incrementally
            if inserted_count > 0:
                await self._calculate_indicators_incremental(db, profile, inserted_count)

            # Step 6: Update profile metadata
            profile.data_updated_at = now
            profile.total_data_points = db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).count()
            db.commit()

            # Step 7: Broadcast update via WebSocket
            await connection_manager.broadcast_preprocessing_progress(
                job_id=f"auto_update_{symbol}_{int(now.timestamp())}",
                symbol=symbol,
                progress=100,
                status="completed",
                stage=f"Updated {inserted_count} new candles",
                records_processed=inserted_count,
                total_records=profile.total_data_points
            )

            logger.info(f"✓ {symbol}: Successfully updated with {inserted_count} new candles")
            return True

        except Exception as e:
            logger.error(f"Error updating profile {profile.symbol}: {e}", exc_info=True)
            db.rollback()
            return False
        finally:
            db.close()

    async def _fetch_new_candles(self, profile: TradingProfile, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch new candles from exchange

        Args:
            profile: Trading profile
            start_time: Start time for fetch
            end_time: End time for fetch

        Returns:
            DataFrame with new candles or None
        """
        try:
            if profile.data_source == 'binance':
                candles = await self.binance.get_all_historical_data(
                    symbol=profile.symbol,
                    interval=profile.data_interval or '1m',
                    start_date=start_time,
                    end_date=end_time
                )
            elif profile.data_source in ['yahoo', 'yfinance']:
                candles = await self.yahoo.get_historical_data(
                    symbol=profile.symbol,
                    interval=profile.data_interval or '1m',
                    start_date=start_time,
                    end_date=end_time
                )
            else:
                logger.error(f"Unknown data source: {profile.data_source}")
                return None

            return candles

        except Exception as e:
            logger.error(f"Error fetching candles: {e}", exc_info=True)
            raise

    async def _insert_candles(self, db: Session, profile_id: int, symbol: str, candles: pd.DataFrame) -> int:
        """
        Insert new candles into database using UPSERT

        Returns:
            Number of candles inserted
        """
        inserted_count = 0

        for _, row in candles.iterrows():
            try:
                # Check if already exists
                existing = db.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp == row['timestamp']
                ).first()

                if existing:
                    # Update existing (in case data was revised)
                    existing.open_price = float(row.get('open', row.get('Open', 0)))
                    existing.high_price = float(row.get('high', row.get('High', 0)))
                    existing.low_price = float(row.get('low', row.get('Low', 0)))
                    existing.close_price = float(row.get('close', row.get('Close', 0)))
                    existing.volume = float(row.get('volume', row.get('Volume', 0)))
                else:
                    # Insert new
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
                logger.error(f"Error inserting candle: {e}", exc_info=True)
                continue

        db.commit()
        return inserted_count

    async def _calculate_indicators_incremental(self, db: Session, profile: TradingProfile, new_candles_count: int):
        """
        Calculate indicators incrementally for new candles only

        Uses a rolling window approach to avoid recalculating all history
        """
        try:
            symbol = profile.symbol

            # For indicators like SMA-200, we need at least 200 candles of history
            # Fetch last 250 candles to have enough context
            window_size = 250

            market_data = db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(desc(MarketData.timestamp)).limit(window_size).all()

            if len(market_data) < 50:  # Need minimum data
                logger.warning(f"{symbol}: Not enough data for indicators ({len(market_data)} candles)")
                return

            # Reverse to chronological order
            market_data = list(reversed(market_data))

            # Convert to DataFrame
            df = pd.DataFrame([{
                'id': md.id,
                'timestamp': md.timestamp,
                'open': md.open_price,
                'high': md.high_price,
                'low': md.low_price,
                'close': md.close_price,
                'volume': md.volume
            } for md in market_data])

            # Calculate indicators for the window
            df_with_indicators = self.indicators_calculator.compute_all_indicators(df)

            # Only store indicators for the NEW candles (last new_candles_count rows)
            new_rows = df_with_indicators.tail(new_candles_count)

            for _, row in new_rows.iterrows():
                try:
                    market_data_id = int(row['id'])

                    # Check if indicators already exist
                    existing_indicator = db.query(IndicatorData).filter(
                        IndicatorData.market_data_id == market_data_id
                    ).first()

                    indicator_values = {
                        'market_data_id': market_data_id,
                        'profile_id': profile.id,
                        # Moving averages
                        'sma_10': float(row.get('sma_10', 0)) if pd.notna(row.get('sma_10')) else None,
                        'sma_20': float(row.get('sma_20', 0)) if pd.notna(row.get('sma_20')) else None,
                        'sma_50': float(row.get('sma_50', 0)) if pd.notna(row.get('sma_50')) else None,
                        'sma_200': float(row.get('sma_200', 0)) if pd.notna(row.get('sma_200')) else None,
                        'ema_12': float(row.get('ema_12', 0)) if pd.notna(row.get('ema_12')) else None,
                        'ema_26': float(row.get('ema_26', 0)) if pd.notna(row.get('ema_26')) else None,
                        # MACD
                        'macd': float(row.get('macd', 0)) if pd.notna(row.get('macd')) else None,
                        'macd_signal': float(row.get('macd_signal', 0)) if pd.notna(row.get('macd_signal')) else None,
                        'macd_histogram': float(row.get('macd_histogram', 0)) if pd.notna(row.get('macd_histogram')) else None,
                        # RSI
                        'rsi_14': float(row.get('rsi_14', 0)) if pd.notna(row.get('rsi_14')) else None,
                        # Bollinger Bands
                        'bb_upper': float(row.get('bb_upper', 0)) if pd.notna(row.get('bb_upper')) else None,
                        'bb_middle': float(row.get('bb_middle', 0)) if pd.notna(row.get('bb_middle')) else None,
                        'bb_lower': float(row.get('bb_lower', 0)) if pd.notna(row.get('bb_lower')) else None,
                        'bb_width': float(row.get('bb_width', 0)) if pd.notna(row.get('bb_width')) else None,
                        # ATR
                        'atr_14': float(row.get('atr_14', 0)) if pd.notna(row.get('atr_14')) else None,
                        # Stochastic
                        'stoch_k': float(row.get('stoch_k', 0)) if pd.notna(row.get('stoch_k')) else None,
                        'stoch_d': float(row.get('stoch_d', 0)) if pd.notna(row.get('stoch_d')) else None,
                        # ADX
                        'adx': float(row.get('adx', 0)) if pd.notna(row.get('adx')) else None,
                        'plus_di': float(row.get('plus_di', 0)) if pd.notna(row.get('plus_di')) else None,
                        'minus_di': float(row.get('minus_di', 0)) if pd.notna(row.get('minus_di')) else None,
                        # OBV
                        'obv': float(row.get('obv', 0)) if pd.notna(row.get('obv')) else None,
                        # Metadata
                        'calculated_at': datetime.utcnow(),
                        'config_version': 1
                    }

                    if existing_indicator:
                        # Update existing
                        for key, value in indicator_values.items():
                            if key != 'market_data_id':  # Don't update PK
                                setattr(existing_indicator, key, value)
                    else:
                        # Insert new
                        indicator = IndicatorData(**indicator_values)
                        db.add(indicator)

                except Exception as e:
                    logger.error(f"Error storing indicators: {e}", exc_info=True)
                    continue

            db.commit()

            # Update profile
            profile.has_indicators = True
            profile.indicators_updated_at = datetime.utcnow()
            db.commit()

            logger.info(f"✓ {symbol}: Calculated indicators for {new_candles_count} new candles")

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            db.rollback()


# Singleton instance
data_updater = DataUpdaterService()
