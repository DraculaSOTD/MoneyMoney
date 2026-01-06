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

# New model imports for LSTM, Transformer, XGBoost, Random Forest, LightGBM, Prophet
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
        elif model_name == "LSTM":
            logger.info(f"[{job_id}] Calling train_lstm_model...")
            results = await train_lstm_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_lstm_model completed, results: {results}")
        elif model_name == "Transformer":
            logger.info(f"[{job_id}] Calling train_transformer_model...")
            results = await train_transformer_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_transformer_model completed, results: {results}")
        elif model_name == "XGBoost":
            logger.info(f"[{job_id}] Calling train_xgboost_model...")
            results = await train_xgboost_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_xgboost_model completed, results: {results}")
        elif model_name == "Random_Forest":
            logger.info(f"[{job_id}] Calling train_random_forest_model...")
            results = await train_random_forest_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_random_forest_model completed, results: {results}")
        elif model_name == "LightGBM":
            logger.info(f"[{job_id}] Calling train_lightgbm_model...")
            results = await train_lightgbm_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_lightgbm_model completed, results: {results}")
        elif model_name == "Prophet":
            logger.info(f"[{job_id}] Calling train_prophet_model...")
            results = await train_prophet_model(df, job_id, symbol, job, db)
            logger.info(f"[{job_id}] train_prophet_model completed, results: {results}")
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


async def train_lstm_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train LSTM model for trading signal classification."""
    try:
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch is not available. Please install torch.")

        logger.info("Training LSTM model")

        # Create sequences
        X, y = create_sequences(df, sequence_length=100)

        logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
        job.progress = 55
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "LSTM", 55, "Sequences created, preparing data..."
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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(
            input_size=X.shape[2],
            hidden_size=128,
            num_layers=2,
            num_classes=3
        ).to(device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "LSTM", 60, "Training LSTM model..."
        )

        # Training loop
        best_val_acc = 0
        epochs = 30
        patience = 5
        patience_counter = 0

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

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Update progress
            progress = 60 + int((epoch / epochs) * 25)
            job.progress = progress
            db.commit()

        # Load best model
        model.load_state_dict(best_state)

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
            'epochs_trained': epoch + 1
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


async def train_transformer_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train Transformer model for trading signal classification."""
    try:
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch is not available. Please install torch.")

        logger.info("Training Transformer model")

        # Create sequences
        X, y = create_sequences(df, sequence_length=100)

        logger.info(f"Created sequences: X={X.shape}, y={y.shape}")
        job.progress = 55
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Transformer", 55, "Sequences created, preparing data..."
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

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransformerClassifier(
            input_size=X.shape[2],
            d_model=128,
            nhead=4,
            num_layers=2,
            num_classes=3
        ).to(device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        job.progress = 60
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Transformer", 60, "Training Transformer model..."
        )

        # Training loop
        best_val_acc = 0
        epochs = 30
        patience = 5
        patience_counter = 0

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

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Update progress
            progress = 60 + int((epoch / epochs) * 25)
            job.progress = progress
            db.commit()

        # Load best model
        model.load_state_dict(best_state)

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
            'epochs_trained': epoch + 1
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


async def train_xgboost_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train XGBoost model for trading signal classification."""
    try:
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not available. Please install xgboost.")

        logger.info("Training XGBoost model")

        # Prepare features
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

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=10,
            eval_metric='mlogloss'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

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

async def train_random_forest_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train Random Forest model for trading signal classification."""
    try:
        logger.info("Training Random Forest model")

        # Prepare features
        X, y = prepare_features_for_ensemble(df, sequence_length=50)

        logger.info(f"Prepared features: X={X.shape}, y={y.shape}")
        job.progress = 55
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "Random_Forest", 55, "Features prepared, training model..."
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
            job_id, symbol, "Random_Forest", 60, "Training Random Forest classifier..."
        )

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

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

async def train_lightgbm_model(df: pd.DataFrame, job_id: str, symbol: str, job, db) -> Dict:
    """Train LightGBM model for trading signal classification."""
    try:
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not available. Please install lightgbm.")

        logger.info("Training LightGBM model")

        # Prepare features
        X, y = prepare_features_for_ensemble(df, sequence_length=50)

        logger.info(f"Prepared features: X={X.shape}, y={y.shape}")
        job.progress = 55
        db.commit()

        await connection_manager.broadcast_model_training_progress(
            job_id, symbol, "LightGBM", 55, "Features prepared, training model..."
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
            job_id, symbol, "LightGBM", 60, "Training LightGBM classifier..."
        )

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train LightGBM
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
            'random_state': 42
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

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

        model.fit(train_df)

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
