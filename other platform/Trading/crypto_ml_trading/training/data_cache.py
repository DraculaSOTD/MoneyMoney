"""
Preprocessed Data Cache System for Large-Scale ML Training.

This module enables training on ALL available data (4.4M+ samples) by:
1. Preprocessing data once and saving to HDF5 cache
2. Loading batches on-demand during training (memory efficient)
3. Caching scalers and metadata for consistent preprocessing

Cache Structure:
    cache/preprocessed/{symbol}/
        features.h5       # (N samples, F features) ~2-3 GB
        metadata.json     # Column names, preprocessing config
        scaler.pkl        # Fitted scaler for inference
"""

import numpy as np
import pandas as pd
import h5py
import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


# Model-specific cache configuration
# use_cache: Whether to use HDF5 cache (False for statistical models)
# max_samples: None = ALL data, integer = limit samples
MODEL_CACHE_CONFIG = {
    # Statistical models - don't use cache, use recent limited data
    'ARIMA': {'use_cache': False, 'max_samples': 50000},
    'GARCH': {'use_cache': False, 'max_samples': 50000},

    # Deep learning - use ALL cached data
    'GRU_Attention': {'use_cache': True, 'max_samples': None},
    'LSTM': {'use_cache': True, 'max_samples': None},
    'Transformer': {'use_cache': True, 'max_samples': None},
    'CNN_Pattern': {'use_cache': True, 'max_samples': None},

    # Tree models - use ALL cached data (very efficient)
    'XGBoost': {'use_cache': True, 'max_samples': None},
    'Random_Forest': {'use_cache': True, 'max_samples': None},
    'LightGBM': {'use_cache': True, 'max_samples': None},

    # Others
    'Prophet': {'use_cache': False, 'max_samples': 100000},
    'Sentiment': {'use_cache': False, 'max_samples': 50000},
}

DEFAULT_CACHE_CONFIG = {'use_cache': True, 'max_samples': None}


class PreprocessedDataCache:
    """
    Manages preprocessed data caching for large-scale ML training.

    Workflow:
    1. First training run: preprocess_and_save() - takes 10-15 min
    2. Subsequent runs: get_cached_dataset() - starts in 10-15 sec

    Example:
        cache = PreprocessedDataCache()

        if not cache.is_cached('BTCUSDT'):
            await cache.preprocess_and_save('BTCUSDT', profile_id, db)

        # Get PyTorch Dataset that reads from cache
        dataset = cache.get_cached_dataset('BTCUSDT', sequence_length=60)
    """

    def __init__(self, cache_dir: str = 'cache/preprocessed'):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_symbol_dir(self, symbol: str) -> Path:
        """Get cache directory for a specific symbol."""
        return self.cache_dir / symbol

    def _get_cache_paths(self, symbol: str) -> Dict[str, Path]:
        """Get all cache file paths for a symbol."""
        symbol_dir = self._get_symbol_dir(symbol)
        return {
            'features': symbol_dir / 'features.h5',
            'metadata': symbol_dir / 'metadata.json',
            'scaler': symbol_dir / 'scaler.pkl'
        }

    def is_cached(self, symbol: str) -> bool:
        """
        Check if preprocessed cache exists for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            True if valid cache exists
        """
        paths = self._get_cache_paths(symbol)

        # All files must exist
        if not all(p.exists() for p in paths.values()):
            return False

        # Verify HDF5 file is valid
        try:
            with h5py.File(paths['features'], 'r') as f:
                if 'features' not in f or 'labels' not in f:
                    return False
                # Sanity check: must have data
                if len(f['features']) == 0:
                    return False
        except Exception as e:
            logger.warning(f"Cache validation failed for {symbol}: {e}")
            return False

        return True

    def get_cache_info(self, symbol: str) -> Optional[Dict]:
        """
        Get information about cached data.

        Args:
            symbol: Trading symbol

        Returns:
            Cache info dict or None if not cached
        """
        if not self.is_cached(symbol):
            return None

        paths = self._get_cache_paths(symbol)

        # Load metadata
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        # Get HDF5 stats
        with h5py.File(paths['features'], 'r') as f:
            n_samples = len(f['features'])
            n_features = f['features'].shape[1]

        # Get file size
        file_size_mb = paths['features'].stat().st_size / (1024 * 1024)

        return {
            'symbol': symbol,
            'n_samples': n_samples,
            'n_features': n_features,
            'file_size_mb': file_size_mb,
            'created_at': metadata.get('created_at'),
            'feature_columns': metadata.get('feature_columns', []),
            'preprocessing_config': metadata.get('preprocessing_config', {}),
        }

    async def preprocess_and_save(
        self,
        symbol: str,
        profile_id: int,
        db,
        chunk_size: int = 100_000,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Preprocess ALL data and save to HDF5 cache (one-time operation).

        This processes data in chunks to avoid memory issues:
        1. Load 100K records at a time from database
        2. Compute technical indicators and features
        3. Write to HDF5 file incrementally

        Args:
            symbol: Trading symbol
            profile_id: Profile ID for database query
            db: Database session
            chunk_size: Records to process per chunk
            progress_callback: Optional async callback(percent, message)

        Returns:
            Dict with preprocessing stats
        """
        from database.models import MarketData
        from sqlalchemy import func
        from crypto_ml_trading.features import EnhancedTechnicalIndicators, FeaturePipeline
        from sklearn.preprocessing import RobustScaler
        import asyncio

        symbol_dir = self._get_symbol_dir(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        paths = self._get_cache_paths(symbol)

        # Get total count
        total_count = db.query(func.count(MarketData.id)).filter(
            MarketData.profile_id == profile_id
        ).scalar()

        if total_count < 100:
            raise ValueError(f"Insufficient data: only {total_count} records available")

        logger.info(f"Preprocessing {total_count:,} samples for {symbol} (chunk_size={chunk_size:,})")

        if progress_callback:
            await progress_callback(0, f"Starting preprocessing of {total_count:,} samples...")

        # Initialize feature computation
        indicators = EnhancedTechnicalIndicators()
        feature_config = {
            'technical_indicators': {'enabled': True, 'indicators': ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr']},
            'microstructure': {'enabled': True, 'features': ['ofi', 'spread', 'kyle_lambda', 'vpin']},
            'time_features': {'enabled': True, 'cyclical': True},
            'statistical_features': {'enabled': True, 'windows': [5, 10, 30, 60]},
            'interaction_features': {'enabled': False, 'max_interactions': 10}
        }
        feature_pipeline = FeaturePipeline(feature_config=feature_config)

        # Phase 1: Process first chunk to determine feature count
        logger.info("Processing first chunk to determine feature dimensions...")

        first_chunk = db.query(MarketData).filter(
            MarketData.profile_id == profile_id
        ).order_by(MarketData.timestamp.asc()).limit(chunk_size).all()

        # Convert to DataFrame
        first_df = self._records_to_dataframe(first_chunk)

        # Compute features on first chunk
        first_df = indicators.compute_all_indicators(first_df, config={})
        first_df = feature_pipeline.generate_features(first_df, training=True)
        first_df = self._create_trading_signals(first_df)
        first_df = first_df.dropna()

        # Separate features and labels
        feature_cols = [col for col in first_df.columns if col != 'signal']
        n_features = len(feature_cols)

        logger.info(f"Feature count: {n_features}, columns: {feature_cols[:10]}...")

        if progress_callback:
            await progress_callback(5, f"Determined {n_features} features, creating HDF5 file...")

        # Initialize scaler with first chunk
        scaler = RobustScaler()
        first_features = first_df[feature_cols].values
        scaler.fit(first_features)

        # Estimate total samples after preprocessing (account for NaN removal)
        # Typically lose ~200-500 rows due to indicator warmup
        estimated_total = total_count - 500

        # Create HDF5 file with resizable datasets
        temp_path = paths['features'].with_suffix('.tmp')

        with h5py.File(temp_path, 'w') as f:
            # Create resizable datasets
            features_ds = f.create_dataset(
                'features',
                shape=(0, n_features),
                maxshape=(None, n_features),
                dtype='float32',
                chunks=(min(10000, chunk_size), n_features),
                compression='gzip',
                compression_opts=4
            )

            labels_ds = f.create_dataset(
                'labels',
                shape=(0,),
                maxshape=(None,),
                dtype='int64',
                chunks=(min(10000, chunk_size),),
                compression='gzip',
                compression_opts=4
            )

            # Write first chunk
            scaled_features = scaler.transform(first_features).astype(np.float32)
            labels = first_df['signal'].values.astype(np.int64)

            features_ds.resize((len(scaled_features), n_features))
            labels_ds.resize((len(labels),))

            features_ds[:] = scaled_features
            labels_ds[:] = labels

            total_written = len(scaled_features)
            logger.info(f"Wrote first chunk: {total_written:,} samples")

            # Process remaining chunks
            offset = chunk_size
            chunk_num = 1

            while offset < total_count:
                # Yield to event loop periodically
                await asyncio.sleep(0)

                # Load next chunk with overlap for indicator continuity
                overlap = 500  # Rows for indicator warmup
                query_offset = max(0, offset - overlap)

                chunk = db.query(MarketData).filter(
                    MarketData.profile_id == profile_id
                ).order_by(
                    MarketData.timestamp.asc()
                ).offset(query_offset).limit(chunk_size + overlap).all()

                if not chunk:
                    break

                # Convert to DataFrame
                chunk_df = self._records_to_dataframe(chunk)

                # Compute features
                chunk_df = indicators.compute_all_indicators(chunk_df, config={})
                chunk_df = feature_pipeline.generate_features(chunk_df, training=True)
                chunk_df = self._create_trading_signals(chunk_df)
                chunk_df = chunk_df.dropna()

                # Remove overlap rows (only keep new data)
                if query_offset != offset:
                    chunk_df = chunk_df.iloc[overlap:]

                if len(chunk_df) == 0:
                    offset += chunk_size
                    continue

                # Extract and scale features
                chunk_features = chunk_df[feature_cols].values
                scaled_chunk = scaler.transform(chunk_features).astype(np.float32)
                chunk_labels = chunk_df['signal'].values.astype(np.int64)

                # Append to HDF5
                old_size = total_written
                new_size = total_written + len(scaled_chunk)

                features_ds.resize((new_size, n_features))
                labels_ds.resize((new_size,))

                features_ds[old_size:new_size] = scaled_chunk
                labels_ds[old_size:new_size] = chunk_labels

                total_written = new_size
                offset += chunk_size
                chunk_num += 1

                # Progress update
                progress = min(95, int(10 + (offset / total_count) * 85))
                if progress_callback and chunk_num % 5 == 0:
                    await progress_callback(
                        progress,
                        f"Processed {total_written:,} / ~{estimated_total:,} samples..."
                    )

                logger.info(f"Chunk {chunk_num}: {total_written:,} total samples written")

        # Move temp file to final location
        temp_path.rename(paths['features'])

        # Save scaler
        with open(paths['scaler'], 'wb') as f:
            pickle.dump(scaler, f)

        # Save metadata
        metadata = {
            'symbol': symbol,
            'profile_id': profile_id,
            'n_samples': total_written,
            'n_features': n_features,
            'feature_columns': feature_cols,
            'created_at': datetime.utcnow().isoformat(),
            'preprocessing_config': feature_config,
            'scaler_type': 'RobustScaler',
        }

        with open(paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)

        if progress_callback:
            await progress_callback(100, f"Cache complete: {total_written:,} samples")

        logger.info(f"Preprocessing complete: {total_written:,} samples saved to {paths['features']}")

        return {
            'n_samples': total_written,
            'n_features': n_features,
            'cache_path': str(paths['features']),
            'file_size_mb': paths['features'].stat().st_size / (1024 * 1024),
        }

    def _records_to_dataframe(self, records) -> pd.DataFrame:
        """Convert database records to DataFrame."""
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'open': float(d.open_price),
            'high': float(d.high_price),
            'low': float(d.low_price),
            'close': float(d.close_price),
            'volume': float(d.volume)
        } for d in records])

        df.set_index('timestamp', inplace=True)
        return df

    def _create_trading_signals(
        self,
        df: pd.DataFrame,
        lookforward: int = 5,
        threshold: float = 0.002
    ) -> pd.DataFrame:
        """Create trading signals based on future returns."""
        df = df.copy()

        # Calculate future returns
        df['future_return'] = df['close'].pct_change(periods=lookforward).shift(-lookforward)

        # Create signals
        df['signal'] = 0  # Hold
        df.loc[df['future_return'] > threshold, 'signal'] = 1  # Buy
        df.loc[df['future_return'] < -threshold, 'signal'] = 2  # Sell

        # Remove future_return column
        df = df.drop(columns=['future_return'])

        return df

    def get_cached_dataset(
        self,
        symbol: str,
        sequence_length: int = 60,
        model_name: Optional[str] = None
    ):
        """
        Get PyTorch Dataset that reads from HDF5 cache.

        Args:
            symbol: Trading symbol
            sequence_length: Sequence length for time series models
            model_name: Optional model name to check if cache should be used

        Returns:
            HDF5SequenceDataset instance
        """
        # Check if model should use cache
        if model_name:
            config = MODEL_CACHE_CONFIG.get(model_name, DEFAULT_CACHE_CONFIG)
            if not config.get('use_cache', True):
                raise ValueError(
                    f"Model {model_name} is configured to NOT use cache. "
                    "Use load_market_data_for_model() instead."
                )

        if not self.is_cached(symbol):
            raise ValueError(
                f"No cache found for {symbol}. "
                "Call preprocess_and_save() first."
            )

        paths = self._get_cache_paths(symbol)

        # Import here to avoid circular imports
        from .hdf5_dataset import HDF5SequenceDataset

        return HDF5SequenceDataset(
            h5_path=str(paths['features']),
            sequence_length=sequence_length
        )

    def get_scaler(self, symbol: str):
        """
        Get fitted scaler for a symbol (for inference preprocessing).

        Args:
            symbol: Trading symbol

        Returns:
            Fitted scaler object
        """
        if not self.is_cached(symbol):
            raise ValueError(f"No cache found for {symbol}")

        paths = self._get_cache_paths(symbol)

        with open(paths['scaler'], 'rb') as f:
            return pickle.load(f)

    def get_feature_columns(self, symbol: str) -> List[str]:
        """Get list of feature column names."""
        if not self.is_cached(symbol):
            raise ValueError(f"No cache found for {symbol}")

        paths = self._get_cache_paths(symbol)

        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)

        return metadata.get('feature_columns', [])

    def invalidate_cache(self, symbol: str) -> bool:
        """
        Delete cache for a symbol (e.g., when data is updated).

        Args:
            symbol: Trading symbol

        Returns:
            True if cache was deleted
        """
        symbol_dir = self._get_symbol_dir(symbol)

        if not symbol_dir.exists():
            return False

        import shutil
        shutil.rmtree(symbol_dir)

        logger.info(f"Invalidated cache for {symbol}")
        return True

    def should_use_cache(self, model_name: str) -> bool:
        """
        Check if a model should use cached data.

        Args:
            model_name: Model type name

        Returns:
            True if model should use HDF5 cache
        """
        config = MODEL_CACHE_CONFIG.get(model_name, DEFAULT_CACHE_CONFIG)
        return config.get('use_cache', True)


def get_data_cache(cache_dir: str = 'cache/preprocessed') -> PreprocessedDataCache:
    """Factory function to get cache instance."""
    return PreprocessedDataCache(cache_dir=cache_dir)
