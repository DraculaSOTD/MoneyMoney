"""
Data Aggregation Service
========================

Aggregates 1-minute OHLCV data into different timeframes (5m, 1h, 1D, 1M).
All base data is stored at 1-minute intervals, and this service provides
on-demand aggregation with intelligent caching.

Supported timeframes:
- 1m: Raw 1-minute data (no aggregation)
- 5m: 5-minute candles (aggregate 5 x 1m candles)
- 1h: 1-hour candles (aggregate 60 x 1m candles)
- 1D: 1-day candles (aggregate 1440 x 1m candles)
- 1M: 1-month candles (aggregate ~43,200 x 1m candles, varies by month)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from functools import lru_cache
import hashlib
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle

from python_backend.database.models import MarketData

logger = logging.getLogger(__name__)

# Performance optimization settings
MAX_WORKERS = min(4, mp.cpu_count())  # Max parallel workers
CHUNK_SIZE = 10000  # Process data in chunks for large datasets
ENABLE_PARALLEL = True  # Enable parallel processing for large datasets


class DataAggregator:
    """
    Aggregates 1-minute OHLCV data into various timeframes with caching.
    """

    # Timeframe definitions in minutes
    TIMEFRAME_MINUTES = {
        '1m': 1,
        '5m': 5,
        '1h': 60,
        '1D': 1440,
        '1M': 43200  # Approximate, actual varies by month
    }

    def __init__(self, db: Session, cache_ttl: int = 300):
        """
        Initialize aggregator.

        Args:
            db: Database session
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.db = db
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_cache_key(self, symbol: str, timeframe: str,
                       start_time: datetime, end_time: datetime) -> str:
        """Generate cache key for aggregation request."""
        key_str = f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False

        cached_time = self._cache[cache_key].get('timestamp')
        if not cached_time:
            return False

        age = (datetime.utcnow() - cached_time).total_seconds()
        return age < self.cache_ttl

    def _set_cache(self, cache_key: str, data: pd.DataFrame):
        """Store data in cache with timestamp."""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': datetime.utcnow()
        }

    def _get_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data'].copy()
        return None

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache for a symbol or all symbols.

        Args:
            symbol: Symbol to clear cache for (None = clear all)
        """
        if symbol is None:
            self._cache.clear()
            logger.info("Cleared all aggregation cache")
        else:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for {symbol}")

    def get_raw_data(self, symbol: str, profile_id: int,
                     start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Fetch raw 1-minute data from python_backend.database.

        Args:
            symbol: Trading symbol
            profile_id: Trading profile ID
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Query database for 1-minute data
            data = self.db.query(MarketData).filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.profile_id == profile_id,
                    MarketData.timestamp >= start_time,
                    MarketData.timestamp <= end_time
                )
            ).order_by(MarketData.timestamp).all()

            if not data:
                logger.warning(f"No data found for {symbol} between {start_time} and {end_time}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': d.timestamp,
                'open': float(d.open_price),
                'high': float(d.high_price),
                'low': float(d.low_price),
                'close': float(d.close_price),
                'volume': float(d.volume)
            } for d in data])

            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            logger.info(f"Fetched {len(df)} 1-minute candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching raw data for {symbol}: {e}")
            raise

    def aggregate_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Aggregate 1-minute data to specified timeframe.

        Args:
            df: DataFrame with 1-minute OHLCV data
            timeframe: Target timeframe ('5m', '1h', '1D', '1M')

        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df

        if timeframe == '1m':
            # No aggregation needed
            return df.copy()

        if timeframe not in self.TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        try:
            # Set timestamp as index for resampling
            df = df.set_index('timestamp')

            # Determine resampling frequency
            if timeframe == '5m':
                freq = '5T'  # 5 minutes
            elif timeframe == '1h':
                freq = '1H'  # 1 hour
            elif timeframe == '1D':
                freq = '1D'  # 1 day
            elif timeframe == '1M':
                freq = '1M'  # 1 month (calendar month)

            # Resample and aggregate
            # OHLC: first, max, min, last
            # Volume: sum
            aggregated = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Remove rows with NaN (incomplete periods)
            aggregated = aggregated.dropna()

            # Reset index to get timestamp back as column
            aggregated = aggregated.reset_index()

            logger.info(f"Aggregated {len(df)} candles to {len(aggregated)} {timeframe} candles")
            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating to {timeframe}: {e}")
            raise

    def get_aggregated_data(self, symbol: str, profile_id: int,
                           timeframe: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: Optional[int] = None,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Get aggregated data for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            profile_id: Trading profile ID
            timeframe: Target timeframe ('1m', '5m', '1h', '1D', '1M')
            start_time: Start timestamp (default: 30 days ago)
            end_time: End timestamp (default: now)
            limit: Maximum number of candles to return
            use_cache: Whether to use cached data

        Returns:
            DataFrame with aggregated OHLCV data
        """
        # Set default time range
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, start_time, end_time)
        if use_cache:
            cached_data = self._get_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Cache hit for {symbol} {timeframe}")
                if limit:
                    return cached_data.tail(limit)
                return cached_data

        # Fetch raw 1-minute data
        raw_data = self.get_raw_data(symbol, profile_id, start_time, end_time)

        if raw_data.empty:
            return pd.DataFrame()

        # Aggregate to target timeframe
        aggregated = self.aggregate_to_timeframe(raw_data, timeframe)

        # Cache the result
        if use_cache:
            self._set_cache(cache_key, aggregated)

        # Apply limit if specified
        if limit:
            aggregated = aggregated.tail(limit)

        return aggregated

    def get_latest_candle(self, symbol: str, profile_id: int,
                         timeframe: str = '1m') -> Optional[Dict[str, Any]]:
        """
        Get the most recent candle for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            profile_id: Trading profile ID
            timeframe: Timeframe ('1m', '5m', '1h', '1D', '1M')

        Returns:
            Dictionary with OHLCV data or None
        """
        # For real-time data, get last hour of data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        # Don't use cache for latest candle (need real-time data)
        data = self.get_aggregated_data(
            symbol=symbol,
            profile_id=profile_id,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=1,
            use_cache=False
        )

        if data.empty:
            return None

        # Convert to dictionary
        latest = data.iloc[-1]
        return {
            'timestamp': latest['timestamp'].isoformat(),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'close': float(latest['close']),
            'volume': float(latest['volume'])
        }

    def get_multiple_timeframes(self, symbol: str, profile_id: int,
                               timeframes: List[str],
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes efficiently.

        Args:
            symbol: Trading symbol
            profile_id: Trading profile ID
            timeframes: List of timeframes to fetch
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum candles per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        result = {}

        # Fetch raw data once
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)

        raw_data = self.get_raw_data(symbol, profile_id, start_time, end_time)

        if raw_data.empty:
            return {tf: pd.DataFrame() for tf in timeframes}

        # Aggregate to each timeframe
        for timeframe in timeframes:
            try:
                aggregated = self.aggregate_to_timeframe(raw_data, timeframe)

                if limit:
                    aggregated = aggregated.tail(limit)

                result[timeframe] = aggregated

            except Exception as e:
                logger.error(f"Error aggregating to {timeframe}: {e}")
                result[timeframe] = pd.DataFrame()

        return result

    def get_stats_for_timeframe(self, symbol: str, profile_id: int,
                               timeframe: str,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get statistical summary for a timeframe.

        Args:
            symbol: Trading symbol
            profile_id: Trading profile ID
            timeframe: Timeframe to analyze
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Dictionary with statistics
        """
        data = self.get_aggregated_data(
            symbol=symbol,
            profile_id=profile_id,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        if data.empty:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'candle_count': 0,
                'error': 'No data available'
            }

        # Calculate statistics
        stats = {
            'symbol': symbol,
            'timeframe': timeframe,
            'candle_count': len(data),
            'start_time': data['timestamp'].min().isoformat(),
            'end_time': data['timestamp'].max().isoformat(),
            'price_high': float(data['high'].max()),
            'price_low': float(data['low'].min()),
            'price_open': float(data['open'].iloc[0]),
            'price_close': float(data['close'].iloc[-1]),
            'price_change': float(data['close'].iloc[-1] - data['open'].iloc[0]),
            'price_change_pct': float((data['close'].iloc[-1] - data['open'].iloc[0]) / data['open'].iloc[0] * 100),
            'total_volume': float(data['volume'].sum()),
            'avg_volume': float(data['volume'].mean()),
            'volatility': float(data['close'].pct_change().std() * 100)  # Standard deviation of returns
        }

        return stats

    def get_raw_data_chunked(self, symbol: str, profile_id: int,
                            start_time: datetime, end_time: datetime,
                            chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
        """
        Fetch raw data in chunks for large datasets (performance optimization).

        Args:
            symbol: Trading symbol
            profile_id: Trading profile ID
            start_time: Start timestamp
            end_time: End timestamp
            chunk_size: Number of rows per chunk

        Returns:
            DataFrame with all data
        """
        try:
            # Get total count first
            total_count = self.db.query(MarketData).filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.profile_id == profile_id,
                    MarketData.timestamp >= start_time,
                    MarketData.timestamp <= end_time
                )
            ).count()

            if total_count == 0:
                return pd.DataFrame()

            # If dataset is small, use regular method
            if total_count <= chunk_size:
                return self.get_raw_data(symbol, profile_id, start_time, end_time)

            logger.info(f"Large dataset detected ({total_count} rows). Using chunked loading.")

            # Load in chunks
            chunks = []
            offset = 0

            while offset < total_count:
                chunk_data = self.db.query(MarketData).filter(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.profile_id == profile_id,
                        MarketData.timestamp >= start_time,
                        MarketData.timestamp <= end_time
                    )
                ).order_by(MarketData.timestamp).offset(offset).limit(chunk_size).all()

                if not chunk_data:
                    break

                chunk_df = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'open': float(d.open_price),
                    'high': float(d.high_price),
                    'low': float(d.low_price),
                    'close': float(d.close_price),
                    'volume': float(d.volume)
                } for d in chunk_data])

                chunks.append(chunk_df)
                offset += chunk_size

                logger.debug(f"Loaded chunk {len(chunks)} ({len(chunk_df)} rows)")

            # Combine all chunks
            if not chunks:
                return pd.DataFrame()

            df = pd.concat(chunks, ignore_index=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            logger.info(f"Loaded {len(df)} rows in {len(chunks)} chunks")
            return df

        except Exception as e:
            logger.error(f"Error in chunked data loading: {e}")
            raise

    def aggregate_with_parallel(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Aggregate data using parallel processing for large datasets.

        Args:
            df: DataFrame with 1-minute data
            timeframe: Target timeframe

        Returns:
            Aggregated DataFrame
        """
        # For small datasets, use regular aggregation
        if len(df) < CHUNK_SIZE or not ENABLE_PARALLEL:
            return self.aggregate_to_timeframe(df, timeframe)

        logger.info(f"Using parallel aggregation for {len(df)} rows")

        try:
            # Split data into chunks by date
            df = df.set_index('timestamp')
            df_sorted = df.sort_index()

            # Calculate chunk boundaries (split by days)
            date_range = pd.date_range(
                start=df_sorted.index.min().date(),
                end=df_sorted.index.max().date(),
                freq='D'
            )

            # Process each day's data
            daily_chunks = []
            for i in range(len(date_range) - 1):
                day_data = df_sorted[
                    (df_sorted.index >= date_range[i]) &
                    (df_sorted.index < date_range[i + 1])
                ]
                if not day_data.empty:
                    daily_chunks.append(day_data)

            # Add last day
            last_day = df_sorted[df_sorted.index >= date_range[-1]]
            if not last_day.empty:
                daily_chunks.append(last_day)

            # Aggregate each chunk in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Reset index for each chunk
                chunk_dfs = [chunk.reset_index() for chunk in daily_chunks]

                # Aggregate in parallel
                aggregated_chunks = list(executor.map(
                    lambda chunk: self.aggregate_to_timeframe(chunk, timeframe),
                    chunk_dfs
                ))

            # Combine results
            if not aggregated_chunks:
                return pd.DataFrame()

            result = pd.concat(aggregated_chunks, ignore_index=True)
            result = result.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Parallel aggregation complete: {len(df)} -> {len(result)} candles")
            return result

        except Exception as e:
            logger.error(f"Error in parallel aggregation, falling back to sequential: {e}")
            # Fallback to regular aggregation
            return self.aggregate_to_timeframe(df, timeframe)

    def optimize_cache_memory(self):
        """
        Optimize cache memory usage by removing old/unused entries.
        """
        if not self._cache:
            return

        current_time = datetime.utcnow()
        keys_to_remove = []

        for key, value in self._cache.items():
            cache_age = (current_time - value['timestamp']).total_seconds()

            # Remove entries older than 2x TTL
            if cache_age > (self.cache_ttl * 2):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        if keys_to_remove:
            logger.info(f"Removed {len(keys_to_remove)} expired cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        if not self._cache:
            return {
                'total_entries': 0,
                'total_size_mb': 0,
                'oldest_entry_age_seconds': None,
                'newest_entry_age_seconds': None
            }

        current_time = datetime.utcnow()
        ages = [
            (current_time - entry['timestamp']).total_seconds()
            for entry in self._cache.values()
        ]

        # Estimate cache size
        try:
            cache_size_bytes = len(pickle.dumps(self._cache))
            cache_size_mb = cache_size_bytes / (1024 * 1024)
        except:
            cache_size_mb = 0

        return {
            'total_entries': len(self._cache),
            'total_size_mb': round(cache_size_mb, 2),
            'oldest_entry_age_seconds': int(max(ages)) if ages else None,
            'newest_entry_age_seconds': int(min(ages)) if ages else None,
            'avg_entry_age_seconds': int(sum(ages) / len(ages)) if ages else None
        }


# Utility functions for quick access

def aggregate_1m_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Quick helper: Aggregate 1m data to 5m."""
    aggregator = DataAggregator(db=None)  # No DB needed for pure aggregation
    return aggregator.aggregate_to_timeframe(df, '5m')


def aggregate_1m_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Quick helper: Aggregate 1m data to 1h."""
    aggregator = DataAggregator(db=None)
    return aggregator.aggregate_to_timeframe(df, '1h')


def aggregate_1m_to_1d(df: pd.DataFrame) -> pd.DataFrame:
    """Quick helper: Aggregate 1m data to 1D."""
    aggregator = DataAggregator(db=None)
    return aggregator.aggregate_to_timeframe(df, '1D')


def aggregate_1m_to_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Quick helper: Aggregate 1m data to 1M (monthly)."""
    aggregator = DataAggregator(db=None)
    return aggregator.aggregate_to_timeframe(df, '1M')
