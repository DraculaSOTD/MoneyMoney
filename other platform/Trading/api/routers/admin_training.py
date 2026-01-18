# ==================== Model Training Task Functions ====================
# This file contains the ML model training functions that will be imported into admin.py

# Model-specific data loading configuration
# max_samples: None = use ALL data, integer = limit to that many samples
# strategy: 'recent' = use most recent data, 'all' = use all available data
MODEL_DATA_CONFIG = {
    # Statistical models - use recent data (they can't handle millions anyway)
    'ARIMA': {'max_samples': 50000, 'strategy': 'recent'},
    'GARCH': {'max_samples': 50000, 'strategy': 'recent'},

    # Deep learning - can train on large datasets via batched loading
    'GRU_Attention': {'max_samples': 500000, 'strategy': 'recent'},  # 500K samples
    'LSTM': {'max_samples': 500000, 'strategy': 'recent'},
    'Transformer': {'max_samples': 500000, 'strategy': 'recent'},
    'CNN_Pattern': {'max_samples': 200000, 'strategy': 'recent'},  # Image gen is slow

    # Tree models - very efficient, can handle large datasets
    'XGBoost': {'max_samples': 1000000, 'strategy': 'recent'},  # 1M samples
    'Random_Forest': {'max_samples': 500000, 'strategy': 'recent'},
    'LightGBM': {'max_samples': 1000000, 'strategy': 'recent'},  # Most efficient

    # Others
    'Prophet': {'max_samples': 100000, 'strategy': 'recent'},
    'Sentiment': {'max_samples': 50000, 'strategy': 'recent'},
}

# Default config for unknown models
DEFAULT_DATA_CONFIG = {'max_samples': 100000, 'strategy': 'recent'}

# GARCH training configuration for memory optimization
# Set options to reduce memory usage during training
GARCH_TRAINING_CONFIG = {
    'skip_order_selection': True,      # Skip grid search, use fixed order (saves ~4 models)
    'fixed_order': (1, 1),             # Default GARCH(1,1) - most common in practice
    'simplified_ensemble': True,       # Use simplified ensemble (fewer models)
    'ensemble_models': ['garch'],      # Models to use when simplified (skip EGARCH/GJR)
    'enable_gc': True,                 # Enable explicit garbage collection between fits
}

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import asyncio
import gc

from database.models import SessionLocal, ModelTrainingJob, MarketData, JobStatus, ProfileModel, ModelTrainingHistory, ModelStatus
from sqlalchemy import func
import uuid
from services.websocket_manager import connection_manager
from crypto_ml_trading.features import EnhancedTechnicalIndicators, FeaturePipeline
from crypto_ml_trading.models import AutoARIMA
from crypto_ml_trading.training.data_cache import PreprocessedDataCache, MODEL_CACHE_CONFIG
from crypto_ml_trading.training.hdf5_dataset import HDF5DataModule, HDF5SequenceDataset, HDF5FlatDataset
from sklearn.model_selection import train_test_split
from crypto_ml_trading.models.statistical.arima.arima_model import ARIMA, AutoARIMA
from crypto_ml_trading.models.statistical.garch.garch_model import GARCH, EGARCH, GJRGARCH
from crypto_ml_trading.models.statistical.garch.utils import GARCHUtils
from crypto_ml_trading.models.statistical.garch.volatility_forecaster import VolatilityForecaster, RealizedVolatilityCalculator
from crypto_ml_trading.models.deep_learning.cnn_pattern.cnn_model import CNNPatternRecognizer
from crypto_ml_trading.models.deep_learning.cnn_pattern.enhanced_trainer import EnhancedCNNPatternTrainer
from crypto_ml_trading.models.deep_learning.cnn_pattern.pattern_generator import PatternGenerator, EnhancedPatternGenerator

# Import PyTorch CNN model if available
try:
    from crypto_ml_trading.models.deep_learning.cnn_pattern.cnn_pattern_gpu import CNNPatternGPU, create_cnn_pattern_gpu
    from crypto_ml_trading.models.deep_learning.cnn_pattern.pytorch_trainer import CNNPatternPyTorchTrainer
    CNN_PYTORCH_AVAILABLE = True
except ImportError:
    CNN_PYTORCH_AVAILABLE = False
    CNNPatternGPU = None
    CNNPatternPyTorchTrainer = None

# New model imports for LSTM, Transformer, XGBoost, Random Forest, LightGBM, Prophet
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    from crypto_ml_trading.models.deep_learning.gru_attention_gpu import GRUAttentionGPU
except ImportError:
    TORCH_AVAILABLE = False
    GRUAttentionGPU = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import logging

# Import sentiment service
try:
    from services.sentiment_service import get_sentiment_service, SentimentService
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    SentimentService = None

logger = logging.getLogger(__name__)


def load_market_data_for_model(db, profile_id: int, model_name: str, chunk_size: int = 100000):
    """
    Load market data efficiently based on model requirements.

    Different models can handle different amounts of data:
    - Statistical models (ARIMA, GARCH): Limited to ~50K samples
    - Deep learning (GRU, LSTM, etc.): Can handle 500K+ via batched training
    - Tree models (XGBoost, LightGBM): Very efficient, can handle 1M+

    Args:
        db: Database session
        profile_id: Profile ID to load data for
        model_name: Model type to get appropriate config
        chunk_size: Size of chunks for loading (for progress logging)

    Returns:
        List of MarketData records in chronological order
    """
    config = MODEL_DATA_CONFIG.get(model_name, DEFAULT_DATA_CONFIG)
    max_samples = config['max_samples']

    # Get total count
    total_count = db.query(func.count(MarketData.id)).filter(
        MarketData.profile_id == profile_id
    ).scalar()

    logger.info(f"Total available samples in database: {total_count:,}")

    if max_samples is None:
        # Use ALL data
        samples_to_use = total_count
    else:
        samples_to_use = min(max_samples, total_count)

    logger.info(f"Training {model_name} on {samples_to_use:,} samples (config: max={max_samples})")

    # Load data - use most recent samples
    if samples_to_use <= chunk_size:
        # Small enough to load in one query
        market_data = db.query(MarketData).filter(
            MarketData.profile_id == profile_id
        ).order_by(MarketData.timestamp.desc()).limit(samples_to_use).all()
        return market_data[::-1]  # Reverse to chronological order
    else:
        # Load in chunks for large datasets
        all_data = []
        loaded = 0

        # Calculate offset to get the most recent samples_to_use records
        start_offset = max(0, total_count - samples_to_use)

        while loaded < samples_to_use:
            current_offset = start_offset + loaded
            current_limit = min(chunk_size, samples_to_use - loaded)

            chunk = db.query(MarketData).filter(
                MarketData.profile_id == profile_id
            ).order_by(MarketData.timestamp.asc()).offset(current_offset).limit(current_limit).all()

            if not chunk:
                break

            all_data.extend(chunk)
            loaded += len(chunk)

            # Progress logging for large loads
            if samples_to_use > chunk_size:
                logger.info(f"Loaded {loaded:,} / {samples_to_use:,} samples ({100*loaded/samples_to_use:.1f}%)")

        return all_data


def create_trading_signals(df: pd.DataFrame, lookforward: int = 5, threshold: float = 0.002) -> pd.DataFrame:
    """
    Create trading signals based on future returns.
    Signal column will be added as the LAST column.

    Args:
        df: DataFrame with OHLCV and features
        lookforward: Number of periods to look forward
        threshold: Threshold for Buy/Sell signals

    Returns:
        DataFrame with 'signal' column at the end (0=Hold, 1=Buy, 2=Sell)
    """
    df = df.copy()

    # Calculate future returns
    df['future_return'] = df['close'].pct_change(periods=lookforward).shift(-lookforward)

    # Create signals
    df['signal'] = 0  # Hold
    df.loc[df['future_return'] > threshold, 'signal'] = 1  # Buy
    df.loc[df['future_return'] < -threshold, 'signal'] = 2  # Sell

    # Remove rows with NaN signals
    df = df.dropna(subset=['signal'])

    # Remove future_return column (was just for calculation)
    df = df.drop(columns=['future_return'])

    # Ensure signal is the LAST column
    signal_col = df.pop('signal')
    df['signal'] = signal_col

    logger.info(f"Created signals: Buy={sum(df['signal']==1)}, Hold={sum(df['signal']==0)}, Sell={sum(df['signal']==2)}")

    return df


def create_sequences(df: pd.DataFrame, sequence_length: int = 100) -> tuple:
    """
    Create sequences for time series models.

    Args:
        df: DataFrame with features and signal column (signal must be last column)
        sequence_length: Length of input sequences

    Returns:
        X (sequences), y (labels)
    """
    # Get feature columns (all except signal)
    feature_cols = [col for col in df.columns if col != 'signal']

    # Extract features and labels
    features = df[feature_cols].values
    labels = df['signal'].values

    X, y = [], []

    for i in range(len(df) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(labels[i+sequence_length])

    return np.array(X), np.array(y)


async def train_model_task(
    job_id: str,
    symbol: str,
    profile_id: int,
    model_name: str
):
    """
    Background task to train a single model.

    Args:
        job_id: Unique job identifier
        symbol: Trading symbol
        profile_id: Profile ID
        model_name: Model type (ARIMA, GARCH, GRU_Attention, CNN_Pattern)
    """
    logger.info(f"{'='*80}")
    logger.info(f"TRAIN_MODEL_TASK CALLED - job_id={job_id}, symbol={symbol}, profile_id={profile_id}, model_name={model_name}")
    logger.info(f"{'='*80}")

    logger.info(f"[{job_id}] Creating database session...")
    db = SessionLocal()
    logger.info(f"[{job_id}] Database session created successfully")
    try:
        # Get job
        logger.info(f"[{job_id}] Querying database for job record...")
        job = db.query(ModelTrainingJob).filter(ModelTrainingJob.job_id == job_id).first()
        if not job:
            logger.error(f"[{job_id}] Job not found in database")
            return
        logger.info(f"[{job_id}] Job record found successfully")

        # Update job status to RUNNING
        logger.info(f"[{job_id}] Updating job status to RUNNING...")
        job.status = JobStatus.RUNNING
        job.progress = 0
        db.commit()
        logger.info(f"[{job_id}] Job status updated to RUNNING")

        logger.info(f"[{job_id}] Starting training for {model_name} on {symbol}")

        # Broadcast training start
        try:
            logger.info(f"[{job_id}] Broadcasting training start to WebSocket...")
            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, model_name, 0, "starting"
            )
            logger.info(f"[{job_id}] WebSocket broadcast successful (training start)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] WebSocket broadcast failed (non-critical): {ws_error}", exc_info=True)

        # Check if this model should use preprocessed data cache
        cache = PreprocessedDataCache()
        cache_config = MODEL_CACHE_CONFIG.get(model_name, {'use_cache': False})
        use_cache = cache_config.get('use_cache', False)

        if use_cache:
            logger.info(f"[{job_id}] Model {model_name} uses preprocessed cache - checking cache status...")

            if not cache.is_cached(symbol):
                logger.info(f"[{job_id}] Cache not found for {symbol}, preprocessing ALL data (one-time, may take 10-15 min)...")

                try:
                    await connection_manager.broadcast_model_training_progress(
                        job_id, symbol, model_name, 1, "preprocessing_all_data"
                    )
                except Exception:
                    pass

                # Define progress callback for preprocessing
                async def preprocess_progress(percent, message):
                    try:
                        # Scale to 0-35% of total progress
                        scaled_progress = int(percent * 0.35)
                        await connection_manager.broadcast_model_training_progress(
                            job_id, symbol, model_name, scaled_progress, f"Caching: {message}"
                        )
                    except Exception:
                        pass
                    job.progress = int(percent * 0.35)
                    db.commit()

                # Preprocess and cache ALL data
                cache_result = await cache.preprocess_and_save(
                    symbol=symbol,
                    profile_id=profile_id,
                    db=db,
                    chunk_size=100_000,
                    progress_callback=preprocess_progress
                )

                logger.info(f"[{job_id}] Cache created: {cache_result['n_samples']:,} samples, {cache_result['file_size_mb']:.1f} MB")

            else:
                cache_info = cache.get_cache_info(symbol)
                logger.info(f"[{job_id}] Using existing cache: {cache_info['n_samples']:,} samples")

            # For models using cache, load from HDF5 instead of database
            # The cache contains preprocessed features - no need for indicator computation
            job.progress = 40
            db.commit()

            try:
                await connection_manager.broadcast_model_training_progress(
                    job_id, symbol, model_name, 40, f"Loaded {cache.get_cache_info(symbol)['n_samples']:,} samples from cache"
                )
            except Exception:
                pass

        # Load market data using model-specific configuration
        logger.info(f"[{job_id}] Loading market data for profile {profile_id}...")
        market_data = load_market_data_for_model(db, profile_id, model_name)

        if not market_data or len(market_data) < 100:
            raise ValueError(f"Insufficient data: only {len(market_data) if market_data else 0} records")

        logger.info(f"[{job_id}] Converting {len(market_data)} records to DataFrame...")
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'open': float(d.open_price),
            'high': float(d.high_price),
            'low': float(d.low_price),
            'close': float(d.close_price),
            'volume': float(d.volume)
        } for d in market_data])

        df.set_index('timestamp', inplace=True)

        logger.info(f"[{job_id}] Loaded {len(df)} data points, updating progress to 5%")
        job.progress = 5
        db.commit()

        # Add technical indicators
        try:
            logger.info(f"[{job_id}] Broadcasting preprocessing status (10%)...")
            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, model_name, 10, "preprocessing"
            )
            logger.info(f"[{job_id}] WebSocket broadcast successful (preprocessing)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] WebSocket broadcast failed (non-critical): {ws_error}", exc_info=True)

        logger.info(f"[{job_id}] Computing technical indicators...")
        indicators = EnhancedTechnicalIndicators()
        df = indicators.compute_all_indicators(df, config={})

        logger.info(f"[{job_id}] Added indicators, shape: {df.shape}, updating progress to 20%")
        job.progress = 20
        db.commit()

        # Engineer features
        try:
            logger.info(f"[{job_id}] Broadcasting preprocessing status (25%)...")
            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, model_name, 25, "preprocessing"
            )
            logger.info(f"[{job_id}] WebSocket broadcast successful (feature engineering)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] WebSocket broadcast failed (non-critical): {ws_error}", exc_info=True)

        logger.info(f"[{job_id}] Generating features...")
        # Use explicit config to avoid YAML structure mismatch
        feature_config = {
            'technical_indicators': {'enabled': True, 'indicators': ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr']},
            'microstructure': {'enabled': True, 'features': ['ofi', 'spread', 'kyle_lambda', 'vpin']},
            'time_features': {'enabled': True, 'cyclical': True},
            'statistical_features': {'enabled': True, 'windows': [5, 10, 30, 60]},
            'interaction_features': {'enabled': False, 'max_interactions': 10}
        }
        feature_pipeline = FeaturePipeline(feature_config=feature_config)
        df = feature_pipeline.generate_features(df, training=True)

        logger.info(f"[{job_id}] Engineered features, shape: {df.shape}, updating progress to 30%")
        job.progress = 30
        db.commit()

        # Create labels/signals - SIGNAL COLUMN AT END
        try:
            logger.info(f"[{job_id}] Broadcasting preprocessing status (35%)...")
            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, model_name, 35, "preprocessing"
            )
            logger.info(f"[{job_id}] WebSocket broadcast successful (signal creation)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] WebSocket broadcast failed (non-critical): {ws_error}", exc_info=True)

        logger.info(f"[{job_id}] Creating trading signals...")
        df = create_trading_signals(df, lookforward=5, threshold=0.002)

        logger.info(f"[{job_id}] Created signals, final shape: {df.shape}, updating progress to 40%")
        job.progress = 40
        db.commit()

        # Train model based on type
        try:
            logger.info(f"[{job_id}] Broadcasting training status (45%)...")
            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, model_name, 45, "training"
            )
            logger.info(f"[{job_id}] WebSocket broadcast successful (training)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] WebSocket broadcast failed (non-critical): {ws_error}", exc_info=True)

        results = {}
        logger.info(f"[{job_id}] Entering model-specific training function for {model_name}...")

        if model_name == "ARIMA":
            logger.info(f"[{job_id}] Calling train_arima_model...")
            results = await train_arima_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_arima_model completed, results: {results}")
        elif model_name == "GARCH":
            logger.info(f"[{job_id}] Calling train_garch_model...")
            results = await train_garch_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_garch_model completed, results: {results}")
        elif model_name == "GRU_Attention":
            logger.info(f"[{job_id}] Calling train_gru_model...")
            # Use cached data if available for much faster training on ALL data
            results = await train_gru_model(df, job_id, symbol, job, db, cache=cache if use_cache else None)
            logger.info(f"[{job_id}] train_gru_model completed, results: {results}")
        elif model_name == "CNN_Pattern":
            logger.info(f"[{job_id}] Calling train_cnn_model...")
            results = await train_cnn_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_cnn_model completed, results: {results}")
        elif model_name == "LSTM":
            logger.info(f"[{job_id}] Calling train_lstm_model...")
            # Use cached data if available for much faster training on ALL data
            results = await train_lstm_model(df, job_id, symbol, job, db, cache=cache if use_cache else None)
            logger.info(f"[{job_id}] train_lstm_model completed, results: {results}")
        elif model_name == "Transformer":
            logger.info(f"[{job_id}] Calling train_transformer_model...")
            # Use cached data if available for much faster training on ALL data
            results = await train_transformer_model(df, job_id, symbol, job, db, cache=cache if use_cache else None)
            logger.info(f"[{job_id}] train_transformer_model completed, results: {results}")
        elif model_name == "XGBoost":
            logger.info(f"[{job_id}] Calling train_xgboost_model...")
            # Use cached data if available for much faster training on ALL data
            results = await train_xgboost_model(df, job_id, symbol, job, db, cache=cache if use_cache else None)
            logger.info(f"[{job_id}] train_xgboost_model completed, results: {results}")
        elif model_name == "Random_Forest":
            logger.info(f"[{job_id}] Calling train_random_forest_model...")
            # Use cached data if available for much faster training on ALL data
            results = await train_random_forest_model(df, job_id, symbol, job, db, cache=cache if use_cache else None)
            logger.info(f"[{job_id}] train_random_forest_model completed, results: {results}")
        elif model_name == "LightGBM":
            logger.info(f"[{job_id}] Calling train_lightgbm_model...")
            # Use cached data if available for much faster training on ALL data
            results = await train_lightgbm_model(df, job_id, symbol, job, db, cache=cache if use_cache else None)
            logger.info(f"[{job_id}] train_lightgbm_model completed, results: {results}")
        elif model_name == "Prophet":
            logger.info(f"[{job_id}] Calling train_prophet_model...")
            results = await train_prophet_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_prophet_model completed, results: {results}")
        elif model_name == "Sentiment":
            logger.info(f"[{job_id}] Calling train_sentiment_model...")
            results = await train_sentiment_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_sentiment_model completed, results: {results}")
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Save model
        try:
            logger.info(f"[{job_id}] Broadcasting saving status (95%)...")
            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, model_name, 95, "saving"
            )
            logger.info(f"[{job_id}] WebSocket broadcast successful (saving)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] WebSocket broadcast failed (non-critical): {ws_error}", exc_info=True)

        logger.info(f"[{job_id}] Creating model directory...")
        model_dir = Path("models") / symbol / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model files (model-specific logic in train functions)
        logger.info(f"[{job_id}] Model saved to {model_dir}, updating progress to 98%")
        job.progress = 98
        db.commit()

        # Update job as completed
        logger.info(f"[{job_id}] Updating job status to COMPLETED...")
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.accuracy = float(results.get('accuracy', 0.0))
        job.completed_at = datetime.utcnow()
        db.commit()

        # Create or update ProfileModel record
        profile_model = db.query(ProfileModel).filter(
            ProfileModel.profile_id == profile_id,
            ProfileModel.model_name == model_name
        ).first()

        if not profile_model:
            profile_model = ProfileModel(
                profile_id=profile_id,
                model_name=model_name,
                model_type=model_name,
                status=ModelStatus.TRAINED,
                validation_accuracy=float(results.get('accuracy', 0.0)),
                test_accuracy=float(results.get('accuracy', 0.0)),
                last_trained=datetime.utcnow()
            )
            db.add(profile_model)
        else:
            profile_model.status = ModelStatus.TRAINED
            profile_model.validation_accuracy = float(results.get('accuracy', 0.0))
            profile_model.test_accuracy = float(results.get('accuracy', 0.0))
            profile_model.last_trained = datetime.utcnow()

        db.commit()
        db.refresh(profile_model)

        # Create ModelTrainingHistory record
        training_history = ModelTrainingHistory(
            profile_id=profile_id,
            model_id=profile_model.id,
            run_id=str(uuid.uuid4()),
            started_at=job.started_at,
            completed_at=datetime.utcnow(),
            duration=(datetime.utcnow() - job.started_at).total_seconds() if job.started_at else 0,
            status='completed',
            val_accuracy=float(results.get('accuracy', 0.0)),
            test_accuracy=float(results.get('accuracy', 0.0))
        )
        db.add(training_history)
        db.commit()

        logger.info(f"[{job_id}] Training completed for {model_name} on {symbol} with accuracy {results.get('accuracy', 0.0)}")

        # Broadcast completion
        try:
            logger.info(f"[{job_id}] Broadcasting training completion...")
            await connection_manager.broadcast_model_training_completed(
                job_id, symbol, model_name, float(results.get('accuracy', 0.0))
            )
            logger.info(f"[{job_id}] WebSocket broadcast successful (completion)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] WebSocket broadcast failed (non-critical): {ws_error}", exc_info=True)

    except Exception as e:
        logger.error(f"{'='*80}")
        logger.error(f"TRAINING FAILED FOR {model_name} - job_id={job_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"{'='*80}", exc_info=True)

        try:
            logger.info(f"[{job_id}] Attempting to update job status to FAILED in database...")
            job = db.query(ModelTrainingJob).filter(ModelTrainingJob.job_id == job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.utcnow()
                db.commit()
                logger.info(f"[{job_id}] Job status updated to FAILED in database")
            else:
                logger.error(f"[{job_id}] Could not find job to update status")
        except Exception as db_error:
            logger.error(f"[{job_id}] Failed to update job status in database: {db_error}", exc_info=True)

        try:
            logger.info(f"[{job_id}] Attempting to broadcast training failure via WebSocket...")
            await connection_manager.broadcast_model_training_failed(job_id, symbol, model_name, str(e))
            logger.info(f"[{job_id}] WebSocket broadcast successful (failure notification)")
        except Exception as ws_error:
            logger.error(f"[{job_id}] Failed to broadcast failure via WebSocket: {ws_error}", exc_info=True)

    finally:
        logger.info(f"[{job_id}] Closing database session...")
        db.close()
        logger.info(f"[{job_id}] Database session closed, train_model_task exiting")


async def train_arima_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train ARIMA model for time series forecasting."""
    try:
        logger.info("Training ARIMA model")

        # Use closing prices for ARIMA
        prices = df['close'].values

        # Proper time series train/test split (80% train, 20% test)
        # CRITICAL: Never train on test data to avoid data leakage
        split_idx = int(len(prices) * 0.8)
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]

        logger.info(f"Train/test split: {len(train_prices)} train, {len(test_prices)} test samples")

        # Find best ARIMA order and fit model on TRAINING DATA ONLY
        # Reduced grid search (18 combinations instead of 108) for faster training
        auto_arima = AutoARIMA(max_p=2, max_d=1, max_q=2)
        # Run blocking fit in thread pool to prevent event loop blocking
        model = await asyncio.to_thread(auto_arima.fit, train_prices)

        logger.info(f"Best ARIMA order: {auto_arima.best_params}")
        job.progress = 60
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "ARIMA", 60, f"Best order selected: {auto_arima.best_params}, model fitted..."
        )

        # Model is already fitted from auto_arima.fit()
        logger.info("ARIMA model fitted on training data")
        job.progress = 80
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "ARIMA", 80, "Model fitted, evaluating on held-out test data..."
        )

        # Make predictions on held-out TEST data (data model has never seen)
        forecast_steps = min(len(test_prices), 50)
        predictions = model.predict(steps=forecast_steps)

        # Evaluate on TEST data only - this gives honest out-of-sample performance
        actual = test_prices[:forecast_steps]

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        # Calculate directional accuracy (more meaningful for trading)
        # Compare if predicted direction matches actual direction
        actual_returns = np.diff(actual)
        pred_returns = np.diff(predictions)
        if len(actual_returns) > 0:
            direction_correct = np.sum(np.sign(actual_returns) == np.sign(pred_returns))
            directional_accuracy = (direction_correct / len(actual_returns)) * 100
        else:
            directional_accuracy = 50.0  # Default to random chance

        # Use directional accuracy as the primary metric (more honest for trading)
        accuracy = directional_accuracy

        logger.info(f"ARIMA out-of-sample evaluation: MAPE={mape:.2f}%, Directional Accuracy={directional_accuracy:.2f}%")

        # Now retrain on full dataset for deployment (but metrics are from test set)
        model = await asyncio.to_thread(auto_arima.fit, prices)

        # Save model
        model_path = Path("models") / symbol / "ARIMA" / "model.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        return {
            'model': model,
            'accuracy': accuracy,
            'order': auto_arima.best_params,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'train_samples': len(train_prices),
            'test_samples': len(test_prices)
        }

    except Exception as e:
        logger.error(f"ARIMA training failed: {e}")
        raise


async def train_garch_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """
    Train GARCH model for volatility forecasting with advanced features.

    Improvements:
    - Auto order selection (p,q) using BIC
    - Ensemble with EGARCH for asymmetric volatility
    - Realized volatility (Garman-Klass) for better evaluation
    - Model diagnostics tracking
    """
    try:
        logger.info("Training GARCH model with advanced features")

        # Calculate returns
        returns = df['close'].pct_change().dropna().values

        # Proper time series train/test split (80% train, 20% test)
        split_idx = int(len(returns) * 0.8)
        train_returns = returns[:split_idx]
        test_returns = returns[split_idx:]

        logger.info(f"Train/test split: {len(train_returns)} train, {len(test_returns)} test samples")

        job.progress = 50
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GARCH", 50, "Returns calculated, selecting optimal order..."
        )

        # Phase 1.1: Order selection (conditional based on config to save memory)
        if not GARCH_TRAINING_CONFIG.get('skip_order_selection', False):
            logger.info("Selecting optimal GARCH order using BIC...")
            try:
                best_order = await asyncio.to_thread(
                    GARCHUtils.select_garch_order, train_returns, 2, 2, 'bic'
                )
                best_p, best_q = best_order
                logger.info(f"Selected optimal order: GARCH({best_p},{best_q})")
            except Exception as e:
                logger.warning(f"Order selection failed, using default (1,1): {e}")
                best_p, best_q = 1, 1
        else:
            # Use fixed order to save memory (skips fitting 4 models during grid search)
            best_p, best_q = GARCH_TRAINING_CONFIG.get('fixed_order', (1, 1))
            logger.info(f"Using fixed order: GARCH({best_p},{best_q}) (order selection skipped to save memory)")

        # Garbage collection after order selection
        if GARCH_TRAINING_CONFIG.get('enable_gc', True):
            gc.collect()
            logger.info("Garbage collection completed after order selection")

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GARCH", 60, f"Using GARCH({best_p},{best_q}), fitting ensemble models..."
        )

        # Phase 1.2 & 1.3: Train ensemble models (conditional based on config to save memory)
        if GARCH_TRAINING_CONFIG.get('simplified_ensemble', False):
            ensemble_models = GARCH_TRAINING_CONFIG.get('ensemble_models', ['garch'])
            logger.info(f"Training simplified ensemble with {ensemble_models} (saves memory)...")
        else:
            ensemble_models = ['garch', 'egarch', 'gjr']
            logger.info("Training full ensemble models (GARCH + EGARCH + GJR-GARCH)...")

        forecaster = VolatilityForecaster(models=ensemble_models)
        await asyncio.to_thread(forecaster.fit_models, train_returns, False)

        # Reuse GARCH from ensemble if available, otherwise create new one
        if 'garch' in forecaster.models:
            garch_model = forecaster.models['garch']
            logger.info("Reusing GARCH model from ensemble")
        else:
            garch_model = GARCH(p=best_p, q=best_q)
            await asyncio.to_thread(garch_model.fit, train_returns)

        # Garbage collection after ensemble fitting
        if GARCH_TRAINING_CONFIG.get('enable_gc', True):
            gc.collect()
            logger.info("Garbage collection completed after ensemble fitting")

        logger.info("Ensemble models fitted on training data")
        job.progress = 75
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GARCH", 75, "Models fitted, evaluating on held-out test data..."
        )

        # Phase 1.4: Evaluate using realized volatility (Garman-Klass)
        forecast_steps = min(len(test_returns), 50)

        # Get ensemble forecast
        ensemble_forecast = forecaster.forecast_ensemble(steps=forecast_steps)
        volatility_forecast = ensemble_forecast['volatility']

        # Calculate realized volatility using Garman-Klass (better than squared returns)
        # Need OHLC data from the test period
        test_high = df['high'].values[split_idx:split_idx + forecast_steps]
        test_low = df['low'].values[split_idx:split_idx + forecast_steps]
        test_open = df['open'].values[split_idx:split_idx + forecast_steps]
        test_close = df['close'].values[split_idx:split_idx + forecast_steps]

        # Calculate per-period realized volatility
        if len(test_high) >= forecast_steps:
            # Use Garman-Klass for realized volatility estimation
            log_hl = np.log(test_high / test_low)
            log_co = np.log(test_close / test_open)
            realized_variance = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            realized_vol = np.sqrt(np.abs(realized_variance))
        else:
            # Fallback to squared returns if OHLC data insufficient
            realized_vol = np.abs(test_returns[:forecast_steps])

        # Calculate accuracy using correlation
        if len(volatility_forecast) > 1 and len(realized_vol) > 1:
            correlation = np.corrcoef(volatility_forecast[:len(realized_vol)], realized_vol)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            accuracy = (correlation + 1) * 50
        else:
            accuracy = 50.0

        logger.info(f"GARCH ensemble out-of-sample evaluation: Accuracy={accuracy:.2f}%")

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GARCH", 85, "Evaluation complete, retraining on full data..."
        )

        # Phase 1.5: Compute diagnostics
        diagnostics = {}
        if garch_model.alpha is not None and garch_model.beta is not None:
            persistence = float(np.sum(garch_model.alpha) + np.sum(garch_model.beta))
            diagnostics = {
                'selected_order': f"GARCH({best_p},{best_q})",
                'persistence': persistence,
                'half_life': np.log(0.5) / np.log(persistence) if persistence < 1 and persistence > 0 else float('inf'),
                'aic': float(garch_model.aic) if garch_model.aic else None,
                'bic': float(garch_model.bic) if garch_model.bic else None,
                'omega': float(garch_model.omega) if garch_model.omega else None,
                'alpha': [float(a) for a in garch_model.alpha] if garch_model.alpha is not None else None,
                'beta': [float(b) for b in garch_model.beta] if garch_model.beta is not None else None,
                'ensemble_models': list(forecaster.models.keys()),
                'ensemble_weights': ensemble_forecast.get('weights', {})
            }
            logger.info(f"GARCH diagnostics: persistence={persistence:.4f}, AIC={garch_model.aic:.2f}")

        # Retrain on full dataset for deployment
        final_model = GARCH(p=best_p, q=best_q)
        await asyncio.to_thread(final_model.fit, returns)

        # Also retrain ensemble on full data (use same config as earlier)
        if GARCH_TRAINING_CONFIG.get('simplified_ensemble', False):
            final_ensemble_models = GARCH_TRAINING_CONFIG.get('ensemble_models', ['garch'])
        else:
            final_ensemble_models = ['garch', 'egarch', 'gjr']

        final_forecaster = VolatilityForecaster(models=final_ensemble_models)
        await asyncio.to_thread(final_forecaster.fit_models, returns, False)

        # Garbage collection after final training
        if GARCH_TRAINING_CONFIG.get('enable_gc', True):
            gc.collect()
            logger.info("Garbage collection completed after final training")

        # Forecast volatility for future using ensemble
        final_ensemble_forecast = final_forecaster.forecast_ensemble(steps=10)
        volatility_forecast = final_ensemble_forecast['volatility']

        # Save main model
        model_path = Path("models") / symbol / "GARCH" / "model.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(str(model_path))

        # Save EGARCH model if available
        if 'egarch' in final_forecaster.models:
            egarch_path = Path("models") / symbol / "EGARCH" / "model.npz"
            egarch_path.parent.mkdir(parents=True, exist_ok=True)
            final_forecaster.models['egarch'].save(str(egarch_path))
            logger.info(f"EGARCH model saved to {egarch_path}")

        logger.info(f"GARCH model saved to {model_path}")

        return {
            'model': final_model,
            'accuracy': accuracy,
            'volatility_forecast': volatility_forecast.tolist(),
            'train_samples': len(train_returns),
            'test_samples': len(test_returns),
            'diagnostics': diagnostics,
            'ensemble_models': list(final_forecaster.models.keys())
        }

    except Exception as e:
        logger.error(f"GARCH training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


async def train_gru_model(df: pd.DataFrame, job_id: str, symbol: str, job, db, cache: PreprocessedDataCache = None) -> Dict:
    """Train GRU-Attention model (GPU) for trading signals.

    Args:
        df: DataFrame with features (used if cache is None)
        job_id: Training job ID
        symbol: Trading symbol
        job: Job record
        db: Database session
        cache: Optional PreprocessedDataCache for training on ALL cached data
    """
    try:
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch is not available. Please install torch.")

        logger.info("Training GRU-Attention model (GPU)")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"GRU-Attention training on device: {device}")

        sequence_length = 50

        # Check if we should use cached data
        if cache is not None and cache.is_cached(symbol):
            # Use HDF5DataModule for memory-efficient training on ALL data
            cache_info = cache.get_cache_info(symbol)
            logger.info(f"Using cached data: {cache_info['n_samples']:,} samples, {cache_info['n_features']} features")

            h5_path = cache._get_cache_paths(symbol)['features']
            data_module = HDF5DataModule(
                h5_path=str(h5_path),
                sequence_length=sequence_length,
                train_ratio=0.7,
                val_ratio=0.15,
                batch_size=64,  # Larger batch for cached data
                num_workers=4
            )

            # Get data loaders
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()

            n_features = cache_info['n_features']
            n_samples = cache_info['n_samples']

            logger.info(f"Created HDF5 data loaders: train={len(data_module.train_dataset()):,}, val={len(data_module.val_dataset()):,}")

            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "GRU_Attention", 55, f"Loaded {n_samples:,} samples from cache, preparing model..."
            )

            # For validation, we'll sample from the validation loader
            val_dataset = data_module.val_dataset()
            # Get a sample to determine feature shape
            sample_x, sample_y = val_dataset[0]
            n_features = sample_x.shape[1]

        else:
            # Fallback to traditional DataFrame-based approach
            # Create sequences
            X, y = create_sequences(df, sequence_length=sequence_length)

            logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "GRU_Attention", 55, "Sequences created, preparing data..."
            )

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train.astype(int))
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val.astype(int))

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = None  # Will use X_val_t directly

            n_features = X.shape[2]

        model = GRUAttentionGPU(
            input_dim=n_features,       # Number of features
            hidden_dim=128,
            n_heads=4,
            n_layers=3,
            output_dim=3,               # 3 classes: buy, hold, sell
            dropout=0.2,
            bidirectional=True,
            use_layer_norm=True
        ).to(device)

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GRU_Attention", 60, "Training GRU-Attention model on GPU..."
        )

        # Run training based on data source
        if cache is not None and cache.is_cached(symbol):
            # Train with HDF5 data loaders (memory efficient for large datasets)
            best_val_acc, epochs_trained, best_state = await asyncio.to_thread(
                _train_gru_with_loaders,
                model, train_loader, val_loader, device
            )
        else:
            # Traditional training with in-memory tensors
            best_val_acc, epochs_trained, best_state = await asyncio.to_thread(
                _train_gru_sync,
                model, train_loader, X_val_t, y_val_t, y_val, device
            )

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # GPU cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GRU_Attention", 85, "Model trained, evaluating..."
        )

        accuracy = best_val_acc * 100
        logger.info(f"GRU-Attention accuracy: {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "GRU_Attention"
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': X.shape[2],
            'hidden_dim': 128,
            'n_heads': 4,
            'n_layers': 3,
            'output_dim': 3
        }, str(model_path / "model.pt"))

        return {
            'model': model,
            'accuracy': accuracy,
            'epochs_trained': epochs_trained
        }

    except Exception as e:
        logger.error(f"GRU training failed: {e}")
        raise


async def train_cnn_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train CNN-Pattern model for pattern recognition with PyTorch GPU acceleration."""
    try:
        # Check if PyTorch GPU version should be used
        use_pytorch = CNN_PYTORCH_AVAILABLE and TORCH_AVAILABLE and torch.cuda.is_available()

        if use_pytorch:
            logger.info("Training CNN-Pattern model (PyTorch GPU with CBAM attention)")
        else:
            logger.info("Training CNN-Pattern model (NumPy fallback)")

        # Use EnhancedPatternGenerator for 12-class detection when using PyTorch
        if use_pytorch:
            pattern_gen = EnhancedPatternGenerator(image_size=64, methods=['gasf', 'gadf', 'rp'])
            window_size = 50

            # Generate pattern images
            images = pattern_gen.generate_pattern_images(df, window_size=window_size)

            # Detect advanced patterns (12 classes)
            labels = pattern_gen.detect_advanced_patterns(df, window_size=window_size)

            # Align lengths (generate_pattern_images: n_samples = len(df) - window_size + 1)
            min_len = min(len(images), len(labels))
            images = images[:min_len]
            labels = labels[:min_len]

            unique_classes = np.unique(labels)
            num_classes = len(unique_classes)
            logger.info(f"Generated {len(images)} pattern images with {num_classes} unique pattern classes")
            logger.info(f"Pattern distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        else:
            # Fallback to original pattern generator
            pattern_gen = PatternGenerator(image_size=64)
            window_size = 50
            images = pattern_gen.generate_pattern_images(df, window_size=window_size)
            labels = df.iloc[window_size-1:window_size-1+len(images)]['signal'].values
            labels = np.array(labels, dtype=np.int64)
            num_classes = 3
            logger.info(f"Generated {len(images)} pattern images")

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "CNN_Pattern", 60,
            f"Pattern images generated ({len(images)} samples), training {'PyTorch GPU' if use_pytorch else 'NumPy'} model..."
        )

        # Split data with stratification if possible
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError:
            # If stratification fails (e.g., too few samples per class), skip it
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.2, random_state=42
            )

        if use_pytorch:
            # Initialize PyTorch model with CBAM attention
            model = CNNPatternGPU(
                input_channels=images.shape[1],
                num_classes=12,  # Always use 12 classes for the model
                image_size=64,
                use_cbam=True,
                dropout=0.3
            )

            # Initialize PyTorch trainer
            trainer = CNNPatternPyTorchTrainer(
                model=model,
                learning_rate=0.001,
                batch_size=32,
                use_mixed_precision=True,
                use_focal_loss=True  # Better for imbalanced classes
            )

            # Train with progress callback
            async def progress_cb(progress, epoch, train_acc, val_acc):
                scaled_progress = 60 + int(progress * 0.25)  # 60-85%
                await connection_manager.broadcast_model_training_progress(
                    job_id, symbol, "CNN_Pattern", scaled_progress,
                    f"Epoch {epoch}: Train {train_acc:.1%}, Val {val_acc:.1%}"
                )

            # Run training
            train_results = await asyncio.to_thread(
                trainer.train,
                X_train, y_train,
                X_val, y_val,
                50,   # epochs
                10    # early_stopping_patience
            )

            accuracy = train_results.get('val_accuracy', 0.0) * 100

            logger.info("PyTorch CNN model trained")
            job.progress = 85
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "CNN_Pattern", 85, "Model trained, saving..."
            )

            # Save PyTorch model
            model_path = Path("models") / symbol / "CNN_Pattern"
            model_path.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path / "model.pt"))

            # Log per-class metrics if available
            class_metrics = train_results.get('class_metrics', {})
            if class_metrics:
                logger.info("Per-class metrics:")
                for class_name, metrics in class_metrics.items():
                    logger.info(f"  {class_name}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")

        else:
            # Fallback to NumPy implementation
            model = CNNPatternRecognizer(image_size=64, num_classes=3)
            trainer = EnhancedCNNPatternTrainer(
                model=model,
                optimizer='adam',
                learning_rate=0.001,
                batch_size=32
            )

            train_results = await asyncio.to_thread(
                trainer.train,
                X_train, y_train,
                X_val, y_val,
                20,  # epochs
                5    # early_stopping_patience
            )

            accuracy = train_results.get('val_accuracy', 0.0) * 100

            logger.info("NumPy CNN model trained")
            job.progress = 85
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "CNN_Pattern", 85, "Model trained, saving..."
            )

            model_path = Path("models") / symbol / "CNN_Pattern"
            model_path.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path / "model.npz"))

        logger.info(f"CNN accuracy: {accuracy:.2f}%")

        # Final GPU cleanup
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        return {
            'model': model,
            'accuracy': accuracy,
            'train_results': train_results,
            'pytorch': use_pytorch,
            'num_classes': num_classes if use_pytorch else 3
        }

    except Exception as e:
        logger.error(f"CNN training failed: {e}")
        raise


# ==================== LSTM Model Training ====================

# Only define PyTorch classes if torch is available
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM model for time series classification."""

        def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                     num_classes: int = 3, dropout: float = 0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )

        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use the last hidden state
            out = self.fc(h_n[-1])
            return out

        def save(self, filepath: str):
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_size': self.lstm.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers
            }, filepath)

        @classmethod
        def load(cls, filepath: str):
            checkpoint = torch.load(filepath)
            model = cls(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            return model


def _train_lstm_sync(model, train_loader, X_val_t, y_val_t, y_val, device, epochs=30, patience=5):
    """
    Synchronous LSTM training function to run in thread pool.
    Returns (best_val_acc, epochs_trained, best_state).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    best_state = None
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = np.mean(val_preds == y_val)

        scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        epochs_trained = epoch + 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Clear GPU cache periodically to prevent memory buildup
        if device.type == 'cuda' and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return best_val_acc, epochs_trained, best_state


def _train_gru_sync(model, train_loader, X_val_t, y_val_t, y_val, device, epochs=10, patience=5):
    """
    Synchronous GRU training function to run in thread pool.
    Returns (best_val_acc, epochs_trained, best_state).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    best_state = None
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = np.mean(val_preds == y_val)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        epochs_trained = epoch + 1

        if patience_counter >= patience:
            logger.info(f"GRU early stopping at epoch {epoch + 1}")
            break

        if device.type == 'cuda' and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return best_val_acc, epochs_trained, best_state


def _train_gru_with_loaders(model, train_loader, val_loader, device, epochs=20, patience=5):
    """
    Training function that uses DataLoaders for both train and validation.
    Used for memory-efficient training on large HDF5 datasets.

    Returns (best_val_acc, epochs_trained, best_state).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    patience_counter = 0
    best_state = None
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation using DataLoader
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        epochs_trained = epoch + 1

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if device.type == 'cuda' and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return best_val_acc, epochs_trained, best_state


async def train_lstm_model(df: pd.DataFrame, job_id: str, symbol: str, job, db, cache: PreprocessedDataCache = None) -> Dict:
    """Train LSTM model for trading signal classification."""
    try:
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch is not available. Please install torch.")

        logger.info("Training LSTM model")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"LSTM training on device: {device}")

        sequence_length = 100

        # Check if we should use cached data
        if cache is not None and cache.is_cached(symbol):
            # Use HDF5DataModule for memory-efficient training on ALL data
            cache_info = cache.get_cache_info(symbol)
            logger.info(f"Using cached data: {cache_info['n_samples']:,} samples, {cache_info['n_features']} features")

            h5_path = cache._get_cache_paths(symbol)['features']
            data_module = HDF5DataModule(
                h5_path=str(h5_path),
                sequence_length=sequence_length,
                train_ratio=0.7,
                val_ratio=0.15,
                batch_size=64,
                num_workers=4
            )

            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()

            n_features = cache_info['n_features']

            logger.info(f"Created HDF5 data loaders: train={len(data_module.train_dataset()):,}, val={len(data_module.val_dataset()):,}")

            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "LSTM", 55, f"Loaded {cache_info['n_samples']:,} samples from cache..."
            )
        else:
            # Fallback to traditional DataFrame-based approach
            X, y = create_sequences(df, sequence_length=sequence_length)

            logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "LSTM", 55, "Sequences created, preparing data..."
            )

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train.astype(int))
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val.astype(int))

            train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = None

            n_features = X.shape[2]

        model = LSTMModel(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            num_classes=3
        ).to(device)

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "LSTM", 60, "Training LSTM model..."
        )

        # Run training based on data source
        if cache is not None and cache.is_cached(symbol):
            best_val_acc, epochs_trained, best_state = await asyncio.to_thread(
                _train_gru_with_loaders,  # Reuse the same training function
                model, train_loader, val_loader, device, 30, 5
            )
        else:
            best_val_acc, epochs_trained, best_state = await asyncio.to_thread(
                _train_lstm_sync,
                model, train_loader, X_val_t, y_val_t, y_val, device
            )

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final GPU cleanup after training
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "LSTM", 85, "Model trained, evaluating..."
        )

        accuracy = best_val_acc * 100
        logger.info(f"LSTM accuracy: {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "LSTM"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path / "model.pt"))

        return {
            'model': model,
            'accuracy': accuracy,
            'epochs_trained': epochs_trained
        }

    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        raise


# ==================== Transformer Model Training ====================

# Only define TransformerClassifier if torch is available
if TORCH_AVAILABLE:
    class TransformerClassifier(nn.Module):
        """Transformer model for time series classification."""

        def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4,
                     num_layers: int = 2, num_classes: int = 3, dropout: float = 0.1):
            super(TransformerClassifier, self).__init__()

            self.input_size = input_size
            self.d_model = d_model

            # Input projection
            self.input_projection = nn.Linear(input_size, d_model)

            # Positional encoding
            self.pos_encoder = nn.Parameter(torch.randn(1, 200, d_model) * 0.1)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            batch_size, seq_len, _ = x.shape

            # Project input
            x = self.input_projection(x)

            # Add positional encoding
            x = x + self.pos_encoder[:, :seq_len, :]

            # Transformer encoding
            x = self.transformer(x)

            # Global average pooling
            x = x.mean(dim=1)

            # Classification
            return self.classifier(x)

        def save(self, filepath: str):
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_size': self.input_size,
                'd_model': self.d_model
            }, filepath)

        @classmethod
        def load(cls, filepath: str):
            checkpoint = torch.load(filepath)
            model = cls(
                input_size=checkpoint['input_size'],
                d_model=checkpoint['d_model']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            return model


def _train_transformer_sync(model, train_loader, X_val_t, y_val_t, y_val, device, epochs=30, patience=5):
    """
    Synchronous Transformer training function to run in thread pool.
    Returns (best_val_acc, epochs_trained, best_state).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience_counter = 0
    best_state = None
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = np.mean(val_preds == y_val)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        epochs_trained = epoch + 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Clear GPU cache periodically to prevent memory buildup
        if device.type == 'cuda' and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return best_val_acc, epochs_trained, best_state


async def train_transformer_model(df: pd.DataFrame, job_id: str, symbol: str, job, db, cache: PreprocessedDataCache = None) -> Dict:
    """Train Transformer model for trading signal classification."""
    try:
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch is not available. Please install torch.")

        logger.info("Training Transformer model")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Transformer training on device: {device}")

        sequence_length = 100

        # Check if we should use cached data
        if cache is not None and cache.is_cached(symbol):
            cache_info = cache.get_cache_info(symbol)
            logger.info(f"Using cached data: {cache_info['n_samples']:,} samples, {cache_info['n_features']} features")

            h5_path = cache._get_cache_paths(symbol)['features']
            data_module = HDF5DataModule(
                h5_path=str(h5_path),
                sequence_length=sequence_length,
                train_ratio=0.7,
                val_ratio=0.15,
                batch_size=32,
                num_workers=4
            )

            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()

            n_features = cache_info['n_features']

            logger.info(f"Created HDF5 data loaders: train={len(data_module.train_dataset()):,}, val={len(data_module.val_dataset()):,}")

            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "Transformer", 55, f"Loaded {cache_info['n_samples']:,} samples from cache..."
            )
        else:
            X, y = create_sequences(df, sequence_length=sequence_length)

            logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "Transformer", 55, "Sequences created, preparing data..."
            )

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train.astype(int))
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val.astype(int))

            train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = None

            n_features = X.shape[2]

        model = TransformerClassifier(
            input_size=n_features,
            d_model=128,
            nhead=4,
            num_layers=2,
            num_classes=3
        ).to(device)

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Transformer", 60, "Training Transformer model..."
        )

        # Run training based on data source
        if cache is not None and cache.is_cached(symbol):
            best_val_acc, epochs_trained, best_state = await asyncio.to_thread(
                _train_gru_with_loaders,
                model, train_loader, val_loader, device, 30, 5
            )
        else:
            best_val_acc, epochs_trained, best_state = await asyncio.to_thread(
                _train_transformer_sync,
                model, train_loader, X_val_t, y_val_t, y_val, device
            )

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final GPU cleanup after training
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Transformer", 85, "Model trained, evaluating..."
        )

        accuracy = best_val_acc * 100
        logger.info(f"Transformer accuracy: {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "Transformer"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path / "model.pt"))

        return {
            'model': model,
            'accuracy': accuracy,
            'epochs_trained': epochs_trained
        }

    except Exception as e:
        logger.error(f"Transformer training failed: {e}")
        raise


# ==================== XGBoost Model Training ====================

def prepare_features_for_ensemble(df: pd.DataFrame, sequence_length: int = 100) -> tuple:
    """Prepare flattened features for ensemble models."""
    # Get feature columns (all except signal)
    feature_cols = [col for col in df.columns if col != 'signal']

    X_list = []
    y_list = []

    for i in range(sequence_length, len(df)):
        # Use rolling statistics instead of full sequence
        window = df[feature_cols].iloc[i-sequence_length:i]

        # Create feature vector with rolling statistics
        features = []
        for col in feature_cols:
            features.extend([
                window[col].iloc[-1],  # Current value
                window[col].mean(),    # Mean
                window[col].std(),     # Std
                window[col].min(),     # Min
                window[col].max(),     # Max
                window[col].iloc[-1] - window[col].iloc[0],  # Change
            ])

        X_list.append(features)
        y_list.append(df['signal'].iloc[i])

    return np.array(X_list), np.array(y_list)


async def train_xgboost_model(df: pd.DataFrame, job_id: str, symbol: str, job, db, cache: PreprocessedDataCache = None) -> Dict:
    """Train XGBoost model for trading signal classification."""
    try:
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not available. Please install xgboost.")

        logger.info("Training XGBoost model")

        # Check if we should use cached data
        if cache is not None and cache.is_cached(symbol):
            cache_info = cache.get_cache_info(symbol)
            logger.info(f"Using cached data: {cache_info['n_samples']:,} samples, {cache_info['n_features']} features")

            h5_path = cache._get_cache_paths(symbol)['features']
            flat_dataset = HDF5FlatDataset(str(h5_path))

            # Load data (already preprocessed and scaled in cache)
            X, y = flat_dataset.get_numpy_arrays()

            logger.info(f"Loaded from cache: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "XGBoost", 55, f"Loaded {cache_info['n_samples']:,} samples from cache..."
            )

            # Data is already preprocessed in cache, no need to scale again
            X_scaled = X
            scaler = cache.get_scaler(symbol)
        else:
            # Fallback to traditional DataFrame-based approach
            X, y = prepare_features_for_ensemble(df, sequence_length=50)

            logger.info(f"Prepared features: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "XGBoost", 55, "Features prepared, training model..."
            )

            # Handle NaN/Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y.astype(int), test_size=0.2, random_state=42
        )

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "XGBoost", 60, "Training XGBoost classifier..."
        )

        # Train XGBoost with GPU support if available
        use_gpu = torch.cuda.is_available() if TORCH_AVAILABLE else False
        logger.info(f"XGBoost training on {'GPU' if use_gpu else 'CPU'}")

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            n_jobs=4,
            early_stopping_rounds=10,
            eval_metric='mlogloss',
            tree_method='hist' if not use_gpu else 'hist',  # XGBoost 2.0+ auto-detects GPU
            device='cuda' if use_gpu else 'cpu'
        )

        # Run blocking fit in thread pool to prevent event loop blocking
        def _fit_xgboost():
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            return model

        await asyncio.to_thread(_fit_xgboost)

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "XGBoost", 85, "Model trained, evaluating..."
        )

        # Evaluate
        val_preds = model.predict(X_val)
        accuracy = np.mean(val_preds == y_val) * 100

        logger.info(f"XGBoost accuracy: {accuracy:.2f}%")

        # Save model and scaler
        model_path = Path("models") / symbol / "XGBoost"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path / "model.json"))

        with open(model_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        return {
            'accuracy': accuracy,
            'n_estimators': model.n_estimators,
            'feature_importance': model.feature_importances_.tolist()
        }

    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        raise


# ==================== Random Forest Model Training ====================

async def train_random_forest_model(df: pd.DataFrame, job_id: str, symbol: str, job, db, cache: PreprocessedDataCache = None) -> Dict:
    """Train Random Forest model for trading signal classification."""
    try:
        logger.info("Training Random Forest model")

        # Check if we should use cached data
        if cache is not None and cache.is_cached(symbol):
            cache_info = cache.get_cache_info(symbol)
            logger.info(f"Using cached data: {cache_info['n_samples']:,} samples, {cache_info['n_features']} features")

            h5_path = cache._get_cache_paths(symbol)['features']
            flat_dataset = HDF5FlatDataset(str(h5_path))

            X, y = flat_dataset.get_numpy_arrays()

            logger.info(f"Loaded from cache: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "Random_Forest", 55, f"Loaded {cache_info['n_samples']:,} samples from cache..."
            )

            X_scaled = X
            scaler = cache.get_scaler(symbol)
        else:
            X, y = prepare_features_for_ensemble(df, sequence_length=50)

            logger.info(f"Prepared features: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "Random_Forest", 55, "Features prepared, training model..."
            )

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y.astype(int), test_size=0.2, random_state=42
        )

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Random_Forest", 60, "Training Random Forest classifier..."
        )

        # Train Random Forest
        # n_jobs=4 limits CPU cores to prevent system overload when training multiple models
        logger.info("Random Forest training on CPU (scikit-learn does not support GPU)")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=4
        )

        # Run blocking fit in thread pool to prevent event loop blocking
        await asyncio.to_thread(model.fit, X_train, y_train)

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Random_Forest", 85, "Model trained, evaluating..."
        )

        # Evaluate
        val_preds = model.predict(X_val)
        accuracy = np.mean(val_preds == y_val) * 100

        logger.info(f"Random Forest accuracy: {accuracy:.2f}%")

        # Save model and scaler
        model_path = Path("models") / symbol / "Random_Forest"
        model_path.mkdir(parents=True, exist_ok=True)

        with open(model_path / "model.pkl", 'wb') as f:
            pickle.dump(model, f)

        with open(model_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        return {
            'accuracy': accuracy,
            'n_estimators': model.n_estimators,
            'feature_importance': model.feature_importances_.tolist()
        }

    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        raise


# ==================== LightGBM Model Training ====================

async def train_lightgbm_model(df: pd.DataFrame, job_id: str, symbol: str, job, db, cache: PreprocessedDataCache = None) -> Dict:
    """Train LightGBM model for trading signal classification."""
    try:
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not available. Please install lightgbm.")

        logger.info("Training LightGBM model")

        # Check if we should use cached data
        if cache is not None and cache.is_cached(symbol):
            cache_info = cache.get_cache_info(symbol)
            logger.info(f"Using cached data: {cache_info['n_samples']:,} samples, {cache_info['n_features']} features")

            h5_path = cache._get_cache_paths(symbol)['features']
            flat_dataset = HDF5FlatDataset(str(h5_path))

            X, y = flat_dataset.get_numpy_arrays()

            logger.info(f"Loaded from cache: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "LightGBM", 55, f"Loaded {cache_info['n_samples']:,} samples from cache..."
            )

            X_scaled = X
            scaler = cache.get_scaler(symbol)
        else:
            X, y = prepare_features_for_ensemble(df, sequence_length=50)

            logger.info(f"Prepared features: X={X.shape}, y={y.shape}")
            job.progress = 55
            db.commit()

            await connection_manager.broadcast_model_training_progress(
                job_id, symbol, "LightGBM", 55, "Features prepared, training model..."
            )

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y.astype(int), test_size=0.2, random_state=42
        )

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "LightGBM", 60, "Training LightGBM classifier..."
        )

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train LightGBM with GPU support if available
        use_gpu = torch.cuda.is_available() if TORCH_AVAILABLE else False
        logger.info(f"LightGBM training on {'GPU' if use_gpu else 'CPU'}")

        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'num_threads': 4,
            'device': 'gpu' if use_gpu else 'cpu'
        }

        # Run blocking train in thread pool to prevent event loop blocking
        def _train_lightgbm():
            return lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )

        model = await asyncio.to_thread(_train_lightgbm)

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "LightGBM", 85, "Model trained, evaluating..."
        )

        # Evaluate
        val_preds = np.argmax(model.predict(X_val), axis=1)
        accuracy = np.mean(val_preds == y_val) * 100

        logger.info(f"LightGBM accuracy: {accuracy:.2f}%")

        # Save model and scaler
        model_path = Path("models") / symbol / "LightGBM"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path / "model.txt"))

        with open(model_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        return {
            'accuracy': accuracy,
            'num_iterations': model.num_trees(),
            'feature_importance': model.feature_importance().tolist()
        }

    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        raise


# ==================== Prophet Model Training ====================

async def train_prophet_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train Prophet model for price forecasting."""
    try:
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet is not available. Please install prophet.")

        logger.info("Training Prophet model")

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['close'].values
        })

        logger.info(f"Prepared Prophet data: {len(prophet_df)} records")
        job.progress = 55
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Prophet", 55, "Data prepared, fitting Prophet model..."
        )

        # Split data for evaluation
        train_size = int(len(prophet_df) * 0.8)
        train_df = prophet_df[:train_size]
        val_df = prophet_df[train_size:]

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Prophet", 60, "Training Prophet model..."
        )

        # Initialize and train Prophet
        logger.info("Prophet training on CPU (Prophet does not support GPU)")
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )

        # Suppress Prophet output
        import logging as lg
        lg.getLogger('prophet').setLevel(lg.WARNING)
        lg.getLogger('cmdstanpy').setLevel(lg.WARNING)

        # Run blocking fit in thread pool to prevent event loop blocking
        await asyncio.to_thread(model.fit, train_df)

        job.progress = 80
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Prophet", 80, "Model trained, evaluating forecasts..."
        )

        # Make predictions on validation set
        future = model.make_future_dataframe(periods=len(val_df), freq='T')  # T for minute
        forecast = model.predict(future)

        # Get predictions for validation period
        val_predictions = forecast[forecast['ds'].isin(val_df['ds'])]['yhat'].values
        val_actuals = val_df['y'].values[:len(val_predictions)]

        # Calculate MAPE
        if len(val_predictions) > 0:
            mape = np.mean(np.abs((val_actuals - val_predictions) / val_actuals)) * 100
            accuracy = max(0, 100 - mape)
        else:
            accuracy = 50.0  # Default if no validation data

        job.progress = 85
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Prophet", 85, "Saving model..."
        )

        logger.info(f"Prophet accuracy (100-MAPE): {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "Prophet"
        model_path.mkdir(parents=True, exist_ok=True)

        with open(model_path / "model.pkl", 'wb') as f:
            pickle.dump(model, f)

        return {
            'accuracy': accuracy,
            'mape': 100 - accuracy if accuracy < 100 else 0,
            'training_samples': len(train_df)
        }

    except Exception as e:
        logger.error(f"Prophet training failed: {e}")
        raise


# ==================== Sentiment Model ====================

async def train_sentiment_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """
    Train Sentiment analysis model combining Fear & Greed Index and Reddit sentiment.

    This model uses:
    1. Fear & Greed Index from Alternative.me API
    2. Reddit sentiment analysis using FinBERT
    3. XGBoost classifier to combine sentiment with price action

    Args:
        df: DataFrame with OHLCV data and technical indicators
        job_id: Job identifier
        symbol: Trading symbol
        job: Training job record
        db: Database session

    Returns:
        Dict with model, accuracy, and training info
    """
    try:
        logger.info(f"Training Sentiment model for {symbol}")

        if not SENTIMENT_AVAILABLE:
            raise ImportError("Sentiment service not available. Install: pip install praw transformers")

        # Initialize sentiment service
        sentiment_service = get_sentiment_service()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Sentiment", 50,
            "Fetching sentiment data from Fear & Greed Index and Reddit..."
        )

        # Get current sentiment data
        sentiment_data = await sentiment_service.get_combined_sentiment(symbol)

        logger.info(f"Sentiment data: Fear & Greed={sentiment_data['fear_greed']['current_value']}, "
                   f"Reddit score={sentiment_data['reddit']['sentiment_score']:.3f}")

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Sentiment", 60,
            "Preparing features for sentiment model..."
        )

        # Prepare features: combine technical indicators with sentiment
        # Get sentiment features
        sentiment_features = sentiment_service.get_sentiment_features(sentiment_data)

        # Prepare price-based features
        X, y = prepare_features_for_ensemble(df, sequence_length=50)

        if len(X) < 100:
            raise ValueError(f"Insufficient data for training: {len(X)} samples")

        # Add sentiment features to each sample (broadcast to all samples)
        # Note: In production, you'd want historical sentiment data
        # For now, we use current sentiment as a static feature
        n_samples = len(X)
        sentiment_broadcast = np.tile(sentiment_features, (n_samples, 1))
        X_with_sentiment = np.hstack([X, sentiment_broadcast])

        logger.info(f"Features shape: {X_with_sentiment.shape} (added {len(sentiment_features)} sentiment features)")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_with_sentiment, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Handle NaN/Inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Sentiment", 70,
            "Training XGBoost classifier with sentiment features..."
        )

        # Use XGBoost for the classifier
        if XGBOOST_AVAILABLE:
            # Check if GPU is available
            use_gpu = False
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except:
                pass

            params = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'tree_method': 'hist',
                'device': 'cuda' if use_gpu else 'cpu'
            }

            model = xgb.XGBClassifier(**params)

            # Train
            def _train_xgb():
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
                return model

            model = await asyncio.to_thread(_train_xgb)

        else:
            # Fallback to Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            def _train_rf():
                model.fit(X_train_scaled, y_train)
                return model

            model = await asyncio.to_thread(_train_rf)

        # Evaluate
        val_preds = model.predict(X_val_scaled)
        accuracy = np.mean(val_preds == y_val) * 100

        logger.info(f"Sentiment model accuracy: {accuracy:.2f}%")

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Sentiment", 85,
            f"Model trained with {accuracy:.1f}% accuracy, saving..."
        )

        # Save model
        model_path = Path("models") / symbol / "Sentiment"
        model_path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        if XGBOOST_AVAILABLE:
            model.save_model(str(model_path / "model.json"))
        else:
            with open(model_path / "model.pkl", 'wb') as f:
                pickle.dump(model, f)

        # Save scaler
        with open(model_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        # Save sentiment config
        import json
        sentiment_config = {
            'symbol': symbol,
            'n_sentiment_features': len(sentiment_features),
            'fear_greed_weight': 0.6,
            'reddit_weight': 0.4,
            'trained_at': datetime.utcnow().isoformat(),
            'last_sentiment': {
                'fear_greed': sentiment_data['fear_greed']['current_value'],
                'reddit_score': sentiment_data['reddit']['sentiment_score'],
                'combined_score': sentiment_data['combined_score'],
                'signal': sentiment_data['signal']
            }
        }
        with open(model_path / "sentiment_config.json", 'w') as f:
            json.dump(sentiment_config, f, indent=2)

        logger.info(f"Sentiment model saved to {model_path}")

        return {
            'model': model,
            'accuracy': accuracy,
            'sentiment_data': sentiment_data,
            'training_samples': len(X_train)
        }

    except Exception as e:
        logger.error(f"Sentiment training failed: {e}")
        raise
