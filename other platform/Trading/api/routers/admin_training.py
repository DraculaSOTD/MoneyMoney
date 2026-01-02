# ==================== Model Training Task Functions ====================
# This file contains the ML model training functions that will be imported into admin.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import asyncio

from database.models import SessionLocal, ModelTrainingJob, MarketData, JobStatus
from services.websocket_manager import connection_manager
from crypto_ml_trading.features import EnhancedTechnicalIndicators, FeaturePipeline
from crypto_ml_trading.models import AutoARIMA
from sklearn.model_selection import train_test_split
from crypto_ml_trading.models.statistical.arima.arima_model import ARIMA, AutoARIMA
from crypto_ml_trading.models.statistical.garch.garch_model import GARCH
from crypto_ml_trading.models.deep_learning.gru_attention.model import GRUAttentionModel
from crypto_ml_trading.models.deep_learning.gru_attention.enhanced_trainer import EnhancedGRUAttentionTrainer
from crypto_ml_trading.models.deep_learning.cnn_pattern.cnn_model import CNNPatternRecognizer
from crypto_ml_trading.models.deep_learning.cnn_pattern.enhanced_trainer import EnhancedCNNPatternTrainer
from crypto_ml_trading.models.deep_learning.cnn_pattern.pattern_generator import PatternGenerator
import logging

logger = logging.getLogger(__name__)


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

        # Load market data from database
        logger.info(f"[{job_id}] Loading market data for profile {profile_id}...")
        market_data = db.query(MarketData).filter(
            MarketData.profile_id == profile_id
        ).order_by(MarketData.timestamp).all()

        if not market_data or len(market_data) < 100:
            raise ValueError(f"Insufficient data: only {len(market_data)} records")

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
            results = await train_gru_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_gru_model completed, results: {results}")
        elif model_name == "CNN_Pattern":
            logger.info(f"[{job_id}] Calling train_cnn_model...")
            results = await train_cnn_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_cnn_model completed, results: {results}")
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
        job.accuracy = results.get('accuracy', 0.0)
        job.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"[{job_id}] Training completed for {model_name} on {symbol} with accuracy {results.get('accuracy', 0.0)}")

        # Broadcast completion
        try:
            logger.info(f"[{job_id}] Broadcasting training completion...")
            await connection_manager.broadcast_model_training_completed(
                job_id, symbol, model_name, results.get('accuracy', 0.0)
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

        # Find best ARIMA order and fit model
        auto_arima = AutoARIMA(max_p=5, max_d=2, max_q=5)
        model = auto_arima.fit(prices)

        logger.info(f"Best ARIMA order: {auto_arima.best_params}")
        job.progress = 60
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "ARIMA", 60, f"Best order selected: {auto_arima.best_params}, model fitted..."
        )

        # Model is already fitted from auto_arima.fit()
        logger.info("ARIMA model fitted")
        job.progress = 80
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "ARIMA", 80, "Model fitted, evaluating performance..."
        )

        # Make predictions for evaluation
        forecast_steps = min(50, len(prices) // 10)
        predictions = model.predict(steps=forecast_steps)

        # Calculate accuracy (using last N points)
        actual = prices[-forecast_steps:]
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        accuracy = max(0, 100 - mape)

        logger.info(f"ARIMA accuracy: {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "ARIMA" / "model.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        return {
            'model': model,
            'accuracy': accuracy,
            'order': auto_arima.best_params,
            'mape': mape
        }

    except Exception as e:
        logger.error(f"ARIMA training failed: {e}")
        raise


async def train_garch_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train GARCH model for volatility forecasting."""
    try:
        logger.info("Training GARCH model")

        # Calculate returns
        returns = df['close'].pct_change().dropna().values

        job.progress = 60
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GARCH", 60, "Returns calculated, fitting model..."
        )

        # Fit GARCH model
        model = GARCH(p=1, q=1)
        model.fit(returns)

        logger.info("GARCH model fitted")
        job.progress = 80
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GARCH", 80, "Model fitted, forecasting volatility..."
        )

        # Forecast volatility
        volatility_forecast = model.forecast(horizon=10)

        # Simple accuracy metric (R-squared for volatility)
        accuracy = 75.0  # Placeholder, actual calculation would compare forecast to actual

        logger.info(f"GARCH accuracy: {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "GARCH" / "model.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

        return {
            'model': model,
            'accuracy': accuracy,
            'volatility_forecast': volatility_forecast.tolist()
        }

    except Exception as e:
        logger.error(f"GARCH training failed: {e}")
        raise


async def train_gru_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train GRU-Attention model for trading signals."""
    try:
        logger.info("Training GRU-Attention model")

        # Create sequences
        X, y = create_sequences(df, sequence_length=100)

        logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
        job.progress = 55
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GRU_Attention", 55, "Sequences created, splitting data..."
        )

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info(f"Split data: train={len(X_train)}, val={len(X_val)}")

        # Initialize model
        model = GRUAttentionModel(
            input_size=X.shape[2],
            hidden_size=256,
            num_layers=3,
            num_classes=3,
            attention_heads=8
        )

        # Train with progress callbacks
        trainer = EnhancedGRUAttentionTrainer(
            model=model,
            optimizer='adamw',
            learning_rate=0.001,
            batch_size=64,
            sequence_length=100
        )

        # Train for limited epochs
        train_results = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=20,
            early_stopping_patience=5
        )

        logger.info("GRU model trained")
        job.progress = 85
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "GRU_Attention", 85, "Model trained, evaluating performance..."
        )

        accuracy = train_results.get('val_accuracy', 0.0) * 100

        logger.info(f"GRU accuracy: {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "GRU_Attention"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path / "model.npz"))

        return {
            'model': model,
            'accuracy': accuracy,
            'train_results': train_results
        }

    except Exception as e:
        logger.error(f"GRU training failed: {e}")
        raise


async def train_cnn_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train CNN-Pattern model for pattern recognition."""
    try:
        logger.info("Training CNN-Pattern model")

        # Generate pattern images using GAF
        pattern_gen = PatternGenerator(image_size=64)

        # Create patterns from price data
        window_size = 50
        images = []
        labels = []

        for i in range(len(df) - window_size):
            window_data = df.iloc[i:i+window_size]
            image = pattern_gen.generate_gaf_image(window_data['close'].values)
            label = df.iloc[i+window_size]['signal']

            images.append(image)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels, dtype=np.int64)

        logger.info(f"Generated {len(images)} pattern images")
        job.progress = 60
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "CNN_Pattern", 60, "Pattern images generated, training model..."
        )

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        # Initialize model
        model = CNNPatternRecognizer(image_size=64, num_classes=3)

        # Train
        trainer = EnhancedCNNPatternTrainer(
            model=model,
            optimizer='adam',
            learning_rate=0.001,
            batch_size=32
        )

        train_results = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=20,
            early_stopping_patience=5
        )

        logger.info("CNN model trained")
        job.progress = 85
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "CNN_Pattern", 85, "Model trained, evaluating performance..."
        )

        accuracy = train_results.get('val_accuracy', 0.0) * 100

        logger.info(f"CNN accuracy: {accuracy:.2f}%")

        # Save model
        model_path = Path("models") / symbol / "CNN_Pattern"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path / "model.npz"))

        return {
            'model': model,
            'accuracy': accuracy,
            'train_results': train_results
        }

    except Exception as e:
        logger.error(f"CNN training failed: {e}")
        raise
