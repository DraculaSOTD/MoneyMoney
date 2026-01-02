"""
Data Loading Service
====================

Loads OHLCV data from CSV or database and prepares it for ML model inference.
Computes all technical indicators using the EnhancedTechnicalIndicators class.

This service bridges raw market data and ML-ready feature sets.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_

# Import technical indicators
import sys
sys.path.append(str(Path(__file__).parent.parent / 'crypto_ml_trading'))
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators

from database.models import MarketData, TradingProfile

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and prepares data for ML model inference.
    """

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize data loader.

        Args:
            db: Database session (optional)
        """
        self.db = db
        self.indicators = EnhancedTechnicalIndicators()
        self.data_cache = {}  # Cache for computed indicators

    def load_from_csv(
        self,
        symbol: str,
        csv_path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            csv_path: Path to CSV file (auto-detects if None)
            limit: Maximum number of rows to load (most recent)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Auto-detect CSV path if not provided
            if csv_path is None:
                base_path = Path(__file__).parent.parent / 'crypto_ml_trading' / 'data' / 'historical'
                csv_path = base_path / f'{symbol}_1m.csv'

            if not Path(csv_path).exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Load CSV
            df = pd.read_csv(csv_path)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Apply limit if specified
            if limit:
                df = df.tail(limit)

            logger.info(f"Loaded {len(df)} rows from {csv_path}")

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error loading CSV for {symbol}: {e}")
            raise

    def load_from_database(
        self,
        symbol: str,
        profile_id: int,
        limit: int = 100,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.

        Args:
            symbol: Trading symbol
            profile_id: Trading profile ID
            limit: Number of candles to load (most recent)
            end_time: End timestamp (default: now)

        Returns:
            DataFrame with OHLCV data
        """
        if self.db is None:
            raise ValueError("Database session not provided")

        try:
            if end_time is None:
                end_time = datetime.utcnow()

            # Query database
            query = self.db.query(MarketData).filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.profile_id == profile_id,
                    MarketData.timestamp <= end_time
                )
            ).order_by(MarketData.timestamp.desc()).limit(limit)

            data = query.all()

            if not data:
                logger.warning(f"No data found for {symbol} (profile_id: {profile_id})")
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

            # Sort by timestamp (ascending)
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Loaded {len(df)} candles from database for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error loading data from database for {symbol}: {e}")
            raise

    def compute_indicators(
        self,
        df: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compute all technical indicators on OHLCV data.

        Uses EnhancedTechnicalIndicators to add 70+ features:
        - Moving Averages (SMA, EMA, VWMA)
        - MACD and components
        - RSI
        - Stochastic Oscillator
        - Bollinger Bands
        - ATR
        - Parabolic SAR
        - ADX
        - Ichimoku Cloud
        - Momentum
        - CMF
        - Pivot Points
        - Support/Resistance
        - Divergences
        - Elliott Waves

        Args:
            df: DataFrame with OHLCV columns
            config: Optional configuration for indicators

        Returns:
            DataFrame with OHLCV + all indicators
        """
        try:
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            if len(df) < 200:
                logger.warning(f"Only {len(df)} candles available. Some indicators may have NaN values.")

            # Compute all indicators
            logger.info("Computing technical indicators...")
            df_with_indicators = self.indicators.compute_all_indicators(df, config)

            # Count total features
            num_features = len(df_with_indicators.columns)
            num_indicators = num_features - len(required_cols) - 1  # -1 for timestamp if present

            logger.info(f"Computed {num_indicators} indicators. Total features: {num_features}")

            return df_with_indicators

        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            raise

    def prepare_for_inference(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for ML model inference.

        Handles:
        - Removing NaN values
        - Creating sequences for time-series models
        - Feature selection

        Args:
            df: DataFrame with OHLCV + indicators
            sequence_length: Number of timesteps for sequence models

        Returns:
            Tuple of (feature_array, feature_names)
        """
        try:
            # Remove timestamp column if present
            if 'timestamp' in df.columns:
                df = df.drop('timestamp', axis=1)

            # Get feature names
            feature_names = df.columns.tolist()

            # Drop rows with NaN (from indicator calculation)
            df_clean = df.dropna()

            if len(df_clean) == 0:
                raise ValueError("All rows contain NaN values after indicator computation")

            logger.info(f"Data shape after cleaning: {df_clean.shape}")
            logger.info(f"Dropped {len(df) - len(df_clean)} rows with NaN values")

            # Convert to numpy array
            feature_array = df_clean.values

            return feature_array, feature_names

        except Exception as e:
            logger.error(f"Error preparing data for inference: {e}")
            raise

    def get_latest_candles(
        self,
        symbol: str,
        profile_id: Optional[int] = None,
        limit: int = 100,
        with_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Get latest candles with optional indicator computation.

        Args:
            symbol: Trading symbol
            profile_id: Profile ID (required if loading from database)
            limit: Number of candles
            with_indicators: Whether to compute indicators

        Returns:
            DataFrame with OHLCV data (+ indicators if requested)
        """
        try:
            # Try loading from database first
            if self.db and profile_id:
                df = self.load_from_database(symbol, profile_id, limit)

                if df.empty:
                    logger.info(f"No data in database, trying CSV fallback for {symbol}")
                    df = self.load_from_csv(symbol, limit=limit)
            else:
                # Load from CSV
                df = self.load_from_csv(symbol, limit=limit)

            if df.empty:
                raise ValueError(f"No data available for {symbol}")

            # Compute indicators if requested
            if with_indicators:
                df = self.compute_indicators(df)

            return df

        except Exception as e:
            logger.error(f"Error getting latest candles for {symbol}: {e}")
            raise

    def get_feature_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about features in the dataset.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with feature statistics
        """
        try:
            # Remove timestamp if present
            if 'timestamp' in df.columns:
                df = df.drop('timestamp', axis=1)

            total_features = len(df.columns)
            total_rows = len(df)

            # Count NaN values
            nan_counts = df.isna().sum()
            features_with_nan = (nan_counts > 0).sum()

            # Get data types
            dtypes = df.dtypes.value_counts().to_dict()

            info = {
                'total_features': total_features,
                'total_rows': total_rows,
                'features_with_nan': int(features_with_nan),
                'max_nan_count': int(nan_counts.max()),
                'data_types': {str(k): int(v) for k, v in dtypes.items()},
                'feature_list': df.columns.tolist(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }

            return info

        except Exception as e:
            logger.error(f"Error getting feature info: {e}")
            return {}


# Utility functions for quick access
def load_btcusdt_data(limit: int = 100, with_indicators: bool = True) -> pd.DataFrame:
    """
    Quick helper to load BTCUSDT data from CSV.

    Args:
        limit: Number of candles
        with_indicators: Whether to compute indicators

    Returns:
        DataFrame with OHLCV data (+ indicators)
    """
    loader = DataLoader()
    df = loader.load_from_csv('BTCUSDT', limit=limit)

    if with_indicators:
        df = loader.compute_indicators(df)

    return df


def get_available_indicators() -> List[str]:
    """
    Get list of all available technical indicators.

    Returns:
        List of indicator names
    """
    # Create sample data to get indicator names
    sample_df = pd.DataFrame({
        'open': [100] * 250,
        'high': [101] * 250,
        'low': [99] * 250,
        'close': [100.5] * 250,
        'volume': [1000] * 250
    })

    loader = DataLoader()
    df_with_indicators = loader.compute_indicators(sample_df)

    # Get indicator names (exclude OHLCV)
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    indicators = [col for col in df_with_indicators.columns if col not in base_cols]

    return indicators
