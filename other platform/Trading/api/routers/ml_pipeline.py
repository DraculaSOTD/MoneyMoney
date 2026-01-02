from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from pathlib import Path
import sys

# Add crypto_ml_trading to path
sys.path.append(str(Path(__file__).parent.parent.parent / "crypto_ml_trading"))

from exchanges.binance_connector import BinanceConnector
from crypto_ml_trading.data.preprocessing import AdvancedPreprocessor
from crypto_ml_trading.data.enhanced_data_loader import EnhancedDataLoader
from crypto_ml_trading.models.unified_interface import UnifiedDeepLearningModel
from database.repository import Repository
from database.models import SessionLocal

router = APIRouter()

# In-memory storage for job progress (in production, use Redis or similar)
job_progress = {}
# WebSocket connections for real-time updates
websocket_connections: Set[WebSocket] = set()

class DataFetchRequest(BaseModel):
    symbol: str
    interval: str = "1m"  # ENFORCED: Always use 1-minute intervals
    days_back: int = 30
    profile_id: int

class PreprocessRequest(BaseModel):
    profile_id: int
    data_id: str
    scaling_method: str = "standard"  # standard, minmax, robust
    handle_missing: str = "forward_fill"  # forward_fill, interpolate, drop
    features: List[str]

class ModelCatalogResponse(BaseModel):
    model_type: str
    display_name: str
    description: str
    category: str  # deep_learning, statistical, reinforcement, ensemble
    supported_features: List[str]
    default_parameters: Dict[str, Any]

class TrainingRequest(BaseModel):
    profile_id: int
    model_id: int
    data_id: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    current_step: str
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/data/fetch", response_model=JobStatus)
async def fetch_data(
    request: DataFetchRequest,
    background_tasks: BackgroundTasks,
    db: SessionLocal = Depends(get_db)
):
    """Fetch historical data from Binance API."""
    job_id = str(uuid.uuid4())
    
    # Initialize job progress
    job_progress[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "current_step": "Initializing",
        "message": "Starting data fetch...",
        "result": None,
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(
        fetch_data_task,
        job_id,
        request.symbol,
        request.interval,
        request.days_back,
        request.profile_id,
        db
    )
    
    return JobStatus(job_id=job_id, **job_progress[job_id])

async def fetch_data_task(
    job_id: str,
    symbol: str,
    interval: str,
    days_back: int,
    profile_id: int,
    db: SessionLocal
):
    """Background task to fetch data."""
    try:
        # Update progress
        job_progress[job_id].update({
            "status": "running",
            "progress": 0.1,
            "current_step": "Connecting to Binance",
            "message": f"Fetching {symbol} data..."
        })
        await broadcast_job_update(job_id)
        
        # Initialize Binance connector
        async with BinanceConnector(testnet=False) as connector:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            job_progress[job_id].update({
                "progress": 0.3,
                "current_step": "Downloading data",
                "message": f"Downloading {days_back} days of {interval} data..."
            })
            await broadcast_job_update(job_id)
            
            # Fetch historical data
            df = await connector.get_historical_data(symbol, interval, days_back)
            
            job_progress[job_id].update({
                "progress": 0.7,
                "current_step": "Processing data",
                "message": "Processing and storing data..."
            })
            await broadcast_job_update(job_id)
            
            # Store data (simplified - in production, save to database)
            data_id = f"data_{job_id}"
            data_path = Path(f"/tmp/{data_id}.json")
            df.to_json(str(data_path), orient='records', date_format='iso')
            
            job_progress[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "current_step": "Complete",
                "message": f"Successfully fetched {len(df)} data points",
                "result": {
                    "data_id": data_id,
                    "rows": len(df),
                    "start_date": df.index[0].isoformat(),
                    "end_date": df.index[-1].isoformat(),
                    "columns": list(df.columns)
                }
            })
            await broadcast_job_update(job_id)
            
    except Exception as e:
        job_progress[job_id].update({
            "status": "failed",
            "progress": job_progress[job_id]["progress"],
            "error": str(e),
            "message": f"Failed to fetch data: {str(e)}"
        })
        await broadcast_job_update(job_id)

@router.post("/data/preprocess", response_model=JobStatus)
async def preprocess_data(
    request: PreprocessRequest,
    background_tasks: BackgroundTasks,
    db: SessionLocal = Depends(get_db)
):
    """Preprocess fetched data."""
    job_id = str(uuid.uuid4())
    
    job_progress[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "current_step": "Initializing",
        "message": "Starting preprocessing...",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(
        preprocess_data_task,
        job_id,
        request.data_id,
        request.scaling_method,
        request.handle_missing,
        request.features,
        request.profile_id,
        db
    )
    
    return JobStatus(job_id=job_id, **job_progress[job_id])

async def preprocess_data_task(
    job_id: str,
    data_id: str,
    scaling_method: str,
    handle_missing: str,
    features: List[str],
    profile_id: int,
    db: SessionLocal
):
    """Background task to preprocess data."""
    try:
        import pandas as pd
        
        job_progress[job_id].update({
            "status": "running",
            "progress": 0.1,
            "current_step": "Loading data",
            "message": "Loading raw data..."
        })
        await broadcast_job_update(job_id)
        
        # Load data
        data_path = Path(f"/tmp/{data_id}.json")
        if not data_path.exists():
            raise FileNotFoundError(f"Data {data_id} not found")
        
        df = pd.read_json(str(data_path), orient='records')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        job_progress[job_id].update({
            "progress": 0.3,
            "current_step": "Feature engineering",
            "message": "Creating technical indicators..."
        })
        await broadcast_job_update(job_id)
        
        # Add technical indicators
        from crypto_ml_trading.features.enhanced_technical_indicators import EnhancedTechnicalIndicators
        
        indicators = EnhancedTechnicalIndicators()
        enhanced_df = indicators.add_all_indicators(df)
        
        job_progress[job_id].update({
            "progress": 0.6,
            "current_step": "Preprocessing",
            "message": f"Applying {scaling_method} scaling..."
        })
        await broadcast_job_update(job_id)
        
        # Initialize preprocessor
        preprocessor = AdvancedPreprocessor()
        
        # Configure preprocessing
        config = {
            "handle_missing": {
                "enabled": True,
                "method": handle_missing
            },
            "scaling": {
                "method": scaling_method,
                "features": features
            }
        }
        
        # Preprocess data
        processed_df = preprocessor.preprocess(enhanced_df, config=config)
        
        job_progress[job_id].update({
            "progress": 0.9,
            "current_step": "Saving results",
            "message": "Saving preprocessed data..."
        })
        await broadcast_job_update(job_id)
        
        # Save processed data
        processed_id = f"processed_{job_id}"
        processed_path = Path(f"/tmp/{processed_id}.json")
        processed_df.to_json(str(processed_path), orient='records', date_format='iso')
        
        job_progress[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "current_step": "Complete",
            "message": "Preprocessing completed successfully",
            "result": {
                "processed_id": processed_id,
                "rows": len(processed_df),
                "features": list(processed_df.columns),
                "scaling_method": scaling_method,
                "missing_handled": handle_missing
            }
        })
        await broadcast_job_update(job_id)
        
    except Exception as e:
        job_progress[job_id].update({
            "status": "failed",
            "progress": job_progress[job_id]["progress"],
            "error": str(e),
            "message": f"Preprocessing failed: {str(e)}"
        })
        await broadcast_job_update(job_id)

@router.get("/models/catalog", response_model=List[ModelCatalogResponse])
async def get_model_catalog():
    """Get available ML models catalog."""
    catalog = [
        {
            "model_type": "lstm",
            "display_name": "LSTM (Long Short-Term Memory)",
            "description": "Recurrent neural network for sequential data",
            "category": "deep_learning",
            "supported_features": ["price", "volume", "technical_indicators"],
            "default_parameters": {
                "lstm_units": [50, 50],
                "dropout_rate": 0.2,
                "batch_size": 512,
                "epochs": 100
            }
        },
        {
            "model_type": "gru_attention",
            "display_name": "GRU with Attention",
            "description": "Gated Recurrent Unit with attention mechanism",
            "category": "deep_learning",
            "supported_features": ["price", "volume", "technical_indicators", "sentiment"],
            "default_parameters": {
                "gru_units": [64, 32],
                "attention_units": 32,
                "dropout_rate": 0.3,
                "batch_size": 256,
                "epochs": 150
            }
        },
        {
            "model_type": "transformer",
            "display_name": "Transformer",
            "description": "Self-attention based model for time series",
            "category": "deep_learning",
            "supported_features": ["price", "volume", "technical_indicators", "market_microstructure"],
            "default_parameters": {
                "d_model": 128,
                "n_heads": 8,
                "n_layers": 4,
                "dropout_rate": 0.1,
                "batch_size": 128,
                "epochs": 200
            }
        },
        {
            "model_type": "tcn",
            "display_name": "TCN (Temporal Convolutional Network)",
            "description": "1D convolutional network for sequences",
            "category": "deep_learning",
            "supported_features": ["price", "volume", "technical_indicators"],
            "default_parameters": {
                "num_channels": [32, 32, 32],
                "kernel_size": 3,
                "dropout_rate": 0.2,
                "batch_size": 256,
                "epochs": 100
            }
        },
        {
            "model_type": "arima",
            "display_name": "ARIMA",
            "description": "Statistical model for time series forecasting",
            "category": "statistical",
            "supported_features": ["price"],
            "default_parameters": {
                "p": 5,
                "d": 1,
                "q": 1,
                "seasonal": False
            }
        },
        {
            "model_type": "garch",
            "display_name": "GARCH",
            "description": "Volatility modeling and forecasting",
            "category": "statistical",
            "supported_features": ["returns", "volatility"],
            "default_parameters": {
                "p": 1,
                "q": 1,
                "mean_model": "AR"
            }
        },
        {
            "model_type": "ppo",
            "display_name": "PPO (Proximal Policy Optimization)",
            "description": "Reinforcement learning for trading decisions",
            "category": "reinforcement",
            "supported_features": ["price", "volume", "portfolio_state"],
            "default_parameters": {
                "hidden_size": 256,
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "n_epochs": 10
            }
        },
        {
            "model_type": "drqn",
            "display_name": "DRQN (Deep Recurrent Q-Network)",
            "description": "Deep RL with memory for sequential decisions",
            "category": "reinforcement",
            "supported_features": ["price", "volume", "portfolio_state", "order_book"],
            "default_parameters": {
                "hidden_size": 128,
                "sequence_length": 20,
                "learning_rate": 1e-3,
                "batch_size": 32
            }
        },
        {
            "model_type": "ensemble_voting",
            "display_name": "Voting Ensemble",
            "description": "Combine predictions from multiple models",
            "category": "ensemble",
            "supported_features": ["all"],
            "default_parameters": {
                "voting": "soft",
                "weights": "uniform"
            }
        },
        {
            "model_type": "ensemble_stacking",
            "display_name": "Stacking Ensemble",
            "description": "Meta-learner combining base models",
            "category": "ensemble",
            "supported_features": ["all"],
            "default_parameters": {
                "meta_learner": "ridge",
                "cv_folds": 5
            }
        }
    ]
    
    return catalog

@router.post("/training/start", response_model=JobStatus)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: SessionLocal = Depends(get_db)
):
    """Start model training."""
    job_id = str(uuid.uuid4())
    
    job_progress[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "current_step": "Initializing",
        "message": "Preparing training...",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(
        train_model_task,
        job_id,
        request,
        db
    )
    
    return JobStatus(job_id=job_id, **job_progress[job_id])

async def train_model_task(job_id: str, request: TrainingRequest, db: SessionLocal):
    """Background task to train model with ALL indicators and real ML training."""
    import pandas as pd
    import numpy as np
    import pickle
    import os
    from sqlalchemy import desc
    from database.models import TradingProfile, ProfileModel, MarketData, ModelStatus, ModelTrainingHistory
    from crypto_ml_trading.features.technical_indicators import EnhancedTechnicalIndicators
    from crypto_ml_trading.data.preprocessing import AdvancedPreprocessor
    from crypto_ml_trading.models.unified_interface import UnifiedDeepLearningModel
    import time

    start_time = time.time()

    try:
        # Step 1: Load profile and model info
        job_progress[job_id].update({
            "status": "running",
            "progress": 0.05,
            "current_step": "Loading profile",
            "message": "Loading trading profile..."
        })
        await broadcast_job_update(job_id)

        profile = db.query(TradingProfile).filter(TradingProfile.id == request.profile_id).first()
        if not profile:
            raise ValueError(f"Profile {request.profile_id} not found")

        model_info = db.query(ProfileModel).filter(ProfileModel.id == request.model_id).first()
        if not model_info:
            raise ValueError(f"Model {request.model_id} not found")

        symbol = profile.symbol
        model_type = model_info.model_type

        # Step 2: Load ALL market data from database
        job_progress[job_id].update({
            "progress": 0.1,
            "current_step": "Loading data",
            "message": f"Loading all market data for {symbol}..."
        })
        await broadcast_job_update(job_id)

        market_data_records = db.query(MarketData).filter(
            MarketData.profile_id == request.profile_id
        ).order_by(MarketData.timestamp).all()

        if not market_data_records or len(market_data_records) < 1000:
            raise ValueError(f"Insufficient data: {len(market_data_records)} candles. Need at least 1000.")

        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'open': d.open_price,
            'high': d.high_price,
            'low': d.low_price,
            'close': d.close_price,
            'volume': d.volume
        } for d in market_data_records])

        # Step 3: Calculate ALL 50+ technical indicators
        job_progress[job_id].update({
            "progress": 0.2,
            "current_step": "Calculating indicators",
            "message": "Calculating all 50+ technical indicators..."
        })
        await broadcast_job_update(job_id)

        indicator_calculator = EnhancedTechnicalIndicators()
        df_with_indicators = indicator_calculator.calculate_all_indicators(df)

        if df_with_indicators is None or df_with_indicators.empty:
            raise ValueError("Failed to calculate technical indicators")

        # Remove rows with NaN values (from indicator calculation)
        df_with_indicators = df_with_indicators.dropna()

        if len(df_with_indicators) < 500:
            raise ValueError(f"Insufficient clean data after indicators: {len(df_with_indicators)} rows")

        # Step 4: Preprocess data
        job_progress[job_id].update({
            "progress": 0.3,
            "current_step": "Preprocessing",
            "message": "Preprocessing and scaling features..."
        })
        await broadcast_job_update(job_id)

        preprocessor = AdvancedPreprocessor()

        # Prepare features (all columns except timestamp and OHLCV)
        feature_columns = [col for col in df_with_indicators.columns
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Create target: predict next candle's close price
        df_with_indicators['target'] = df_with_indicators['close'].shift(-1)
        df_with_indicators = df_with_indicators.dropna()

        X = df_with_indicators[feature_columns].values
        y = df_with_indicators['target'].values

        # Train/val/test split
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Step 5: Initialize and train the model
        job_progress[job_id].update({
            "progress": 0.4,
            "current_step": "Initializing model",
            "message": f"Initializing {model_type.upper()} model..."
        })
        await broadcast_job_update(job_id)

        # Get model parameters or use defaults
        model_params = model_info.parameters or {}

        # Initialize model
        model = UnifiedDeepLearningModel(
            model_type=model_type,
            input_shape=(X_train_scaled.shape[1],),
            **model_params
        )

        # Training with progress updates
        job_progress[job_id].update({
            "progress": 0.5,
            "current_step": "Training",
            "message": f"Training {model_type.upper()} model..."
        })
        await broadcast_job_update(job_id)

        # Train the model
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=request.epochs,
            batch_size=request.batch_size,
            verbose=0  # Suppress output
        )

        # Step 6: Evaluate model
        job_progress[job_id].update({
            "progress": 0.9,
            "current_step": "Evaluating",
            "message": "Evaluating model performance..."
        })
        await broadcast_job_update(job_id)

        # Get predictions for validation and test sets
        val_predictions = model.predict(X_val_scaled)
        test_predictions = model.predict(X_test_scaled)

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        val_r2 = r2_score(y_val, val_predictions)

        test_mae = mean_absolute_error(y_test, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)

        # Calculate MAPE
        val_mape = np.mean(np.abs((y_val - val_predictions.flatten()) / y_val)) * 100
        test_mape = np.mean(np.abs((y_test - test_predictions.flatten()) / y_test)) * 100

        # Step 7: Save model and scaler to disk
        job_progress[job_id].update({
            "progress": 0.95,
            "current_step": "Saving",
            "message": "Saving model files..."
        })
        await broadcast_job_update(job_id)

        # Create directories
        models_dir = Path(__file__).parent.parent / "models" / "trained"
        scalers_dir = Path(__file__).parent.parent / "models" / "scalers"
        models_dir.mkdir(parents=True, exist_ok=True)
        scalers_dir.mkdir(parents=True, exist_ok=True)

        # Generate version number
        version = model_info.model_version or "1.0.0"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        model_filename = f"{symbol}_{model_type}_v{version}_{timestamp}.pkl"
        scaler_filename = f"{symbol}_{model_type}_v{version}_{timestamp}_scaler.pkl"

        model_path = models_dir / model_filename
        scaler_path = scalers_dir / scaler_filename

        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Step 8: Update database
        training_duration = time.time() - start_time

        model_info.status = ModelStatus.TRAINED
        model_info.last_trained = datetime.utcnow()
        model_info.training_duration = training_duration
        model_info.training_samples = len(X_train)
        model_info.features = feature_columns
        model_info.model_path = str(model_path)
        model_info.scaler_path = str(scaler_path)
        model_info.validation_accuracy = float(val_r2)
        model_info.validation_loss = float(val_rmse)
        model_info.test_accuracy = float(test_r2)
        model_info.preprocessing_config = {
            'scaler': 'StandardScaler',
            'train_size': train_size,
            'val_size': val_size,
            'test_size': len(X_test)
        }

        db.commit()

        # Complete
        job_progress[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "current_step": "Complete",
            "message": f"Training completed successfully in {training_duration:.1f}s",
            "result": {
                "model_id": model_info.id,
                "model_type": model_type,
                "version": version,
                "training_duration": training_duration,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "features_count": len(feature_columns),
                "validation_metrics": {
                    "mae": float(val_mae),
                    "rmse": float(val_rmse),
                    "mape": float(val_mape),
                    "r2_score": float(val_r2)
                },
                "test_metrics": {
                    "mae": float(test_mae),
                    "rmse": float(test_rmse),
                    "mape": float(test_mape),
                    "r2_score": float(test_r2)
                }
            }
        })
        await broadcast_job_update(job_id)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        job_progress[job_id].update({
            "status": "failed",
            "progress": job_progress[job_id].get("progress", 0.0),
            "error": str(e),
            "message": f"Training failed: {str(e)}"
        })
        await broadcast_job_update(job_id)

        # Log detailed error
        print(f"Training failed for job {job_id}:")
        print(error_details)

@router.get("/training/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a running job."""
    if job_id not in job_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(job_id=job_id, **job_progress[job_id])

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in job_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_progress[job_id]["status"] == "running":
        job_progress[job_id]["status"] = "cancelled"
        job_progress[job_id]["message"] = "Job cancelled by user"
    
    return {"message": "Job cancelled"}

# WebSocket endpoint for real-time updates
@router.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates."""
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connected for training updates"
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client (ping/pong or job subscriptions)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "subscribe":
                    job_id = message.get("job_id")
                    if job_id and job_id in job_progress:
                        # Send current job status
                        await websocket.send_json({
                            "type": "job_update",
                            "job_id": job_id,
                            **job_progress[job_id]
                        })
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                
    finally:
        websocket_connections.remove(websocket)
        
async def broadcast_job_update(job_id: str):
    """Broadcast job updates to all connected WebSocket clients."""
    if job_id not in job_progress:
        return

    message = {
        "type": "job_update",
        "job_id": job_id,
        **job_progress[job_id]
    }

    # Send to all connected clients
    disconnected = set()
    for websocket in websocket_connections:
        try:
            await websocket.send_json(message)
        except:
            disconnected.add(websocket)

    # Remove disconnected clients
    for ws in disconnected:
        websocket_connections.discard(ws)


# ===================================================================
# MODEL MANAGEMENT ENDPOINTS
# ===================================================================

class ModelVersionResponse(BaseModel):
    """Response model for model version info"""
    id: int
    model_name: str
    model_version: str
    status: str
    is_deployed: bool
    created_at: datetime
    validation_accuracy: Optional[float] = None
    test_sharpe: Optional[float] = None

    class Config:
        from_attributes = True


@router.get("/api/profiles/{profile_id}/models/{model_id}/versions")
async def get_model_versions(
    profile_id: int,
    model_id: int,
    db: SessionLocal = Depends(get_db)
):
    """
    Get all versions of a model for rollback
    Returns models ordered by creation date (newest first)
    """
    try:
        from database.models import ProfileModel

        # Get the current model to find its name
        current_model = db.query(ProfileModel).filter(
            ProfileModel.id == model_id,
            ProfileModel.profile_id == profile_id
        ).first()

        if not current_model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get all versions of this model (by model_name)
        versions = db.query(ProfileModel).filter(
            ProfileModel.profile_id == profile_id,
            ProfileModel.model_name == current_model.model_name
        ).order_by(
            ProfileModel.created_at.desc()
        ).all()

        return {
            "success": True,
            "model_name": current_model.model_name,
            "current_version": current_model.model_version,
            "total_versions": len(versions),
            "versions": [
                {
                    "id": v.id,
                    "model_name": v.model_name,
                    "model_version": v.model_version,
                    "status": v.status.value if hasattr(v.status, 'value') else str(v.status),
                    "is_deployed": v.is_deployed,
                    "created_at": v.created_at.isoformat(),
                    "validation_accuracy": v.validation_accuracy,
                    "test_sharpe": v.test_sharpe,
                    "last_trained": v.last_trained.isoformat() if v.last_trained else None
                }
                for v in versions
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/profiles/{profile_id}/models/{old_model_id}/rollback/{target_model_id}")
async def rollback_model(
    profile_id: int,
    old_model_id: int,
    target_model_id: int,
    db: SessionLocal = Depends(get_db)
):
    """
    Rollback to a previous model version

    Args:
        profile_id: Profile ID
        old_model_id: Current deployed model ID
        target_model_id: Target model version ID to rollback to
    """
    try:
        from database.models import ProfileModel, ModelStatus

        # Get both models
        old_model = db.query(ProfileModel).filter(
            ProfileModel.id == old_model_id,
            ProfileModel.profile_id == profile_id
        ).first()

        target_model = db.query(ProfileModel).filter(
            ProfileModel.id == target_model_id,
            ProfileModel.profile_id == profile_id
        ).first()

        if not old_model:
            raise HTTPException(status_code=404, detail="Current model not found")

        if not target_model:
            raise HTTPException(status_code=404, detail="Target model version not found")

        # Verify they're the same model type
        if old_model.model_name != target_model.model_name:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot rollback between different models: {old_model.model_name} != {target_model.model_name}"
            )

        # Verify target model is trainedand not already deployed
        if target_model.status not in [ModelStatus.TRAINED, ModelStatus.DEPLOYED]:
            raise HTTPException(
                status_code=400,
                detail=f"Target model must be trained (status: {target_model.status})"
            )

        if target_model.is_deployed:
            raise HTTPException(
                status_code=400,
                detail="Target model is already deployed"
            )

        # Perform rollback
        # 1. Undeploy old model
        old_model.is_deployed = False
        old_model.is_primary = False
        old_model.status = ModelStatus.TRAINED

        # 2. Deploy target model
        target_model.is_deployed = True
        target_model.is_primary = True
        target_model.status = ModelStatus.DEPLOYED
        target_model.deployed_at = datetime.utcnow()

        db.commit()

        return {
            "success": True,
            "message": f"Successfully rolled back from v{old_model.model_version} to v{target_model.model_version}",
            "old_version": {
                "id": old_model.id,
                "version": old_model.model_version,
                "status": "archived"
            },
            "new_version": {
                "id": target_model.id,
                "version": target_model.model_version,
                "status": "deployed",
                "deployed_at": target_model.deployed_at.isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Helper function for model retrainer
async def train_model_for_profile(
    profile_id: int,
    model_type: str,
    db: SessionLocal,
    epochs: int = 50,
    batch_size: int = 32
) -> Dict:
    """
    Train a new model for a profile (called by monthly retrainer).

    Args:
        profile_id: Trading profile ID
        model_type: Type of model ('lstm', 'gru', 'transformer', etc.)
        db: Database session
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        dict with training results
    """
    from database.models import TradingProfile, ProfileModel, ModelStatus
    import uuid

    try:
        # Get profile
        profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()
        if not profile:
            return {
                'success': False,
                'error': f'Profile {profile_id} not found'
            }

        # Get current model of this type to get parameters
        existing_model = db.query(ProfileModel).filter(
            ProfileModel.profile_id == profile_id,
            ProfileModel.model_type == model_type
        ).order_by(ProfileModel.created_at.desc()).first()

        # Generate new version number
        if existing_model:
            version_parts = existing_model.model_version.split('.')
            major = int(version_parts[0])
            new_version = f"{major + 1}.0.0"
            model_params = existing_model.parameters or {}
        else:
            new_version = "1.0.0"
            model_params = {}

        # Create new model record
        new_model = ProfileModel(
            profile_id=profile_id,
            model_name=f"{profile.symbol}_{model_type}",
            model_type=model_type,
            model_version=new_version,
            status=ModelStatus.TRAINING,
            parameters=model_params,
            is_deployed=False,
            is_primary=False
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)

        # Create training request
        training_request = TrainingRequest(
            profile_id=profile_id,
            model_id=new_model.id,
            data_id="retraining",  # Not used in new implementation
            epochs=epochs,
            batch_size=batch_size
        )

        # Generate job ID and train model
        job_id = str(uuid.uuid4())
        job_progress[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing",
            "message": "Preparing training...",
            "result": None,
            "error": None
        }

        # Train synchronously (monthly retrainer waits for completion)
        await train_model_task(job_id, training_request, db)

        # Check result
        result = job_progress[job_id]

        if result['status'] == 'completed':
            return {
                'success': True,
                'model_id': new_model.id,
                'model_version': new_version,
                'result': result.get('result', {})
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Training failed'),
                'model_id': new_model.id
            }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in train_model_for_profile:")
        print(error_details)

        return {
            'success': False,
            'error': str(e)
        }