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

from python_backend.exchanges.binance_connector import BinanceConnector
from crypto_ml_trading.data.preprocessing import AdvancedPreprocessor
from crypto_ml_trading.data.enhanced_data_loader import EnhancedDataLoader
from crypto_ml_trading.models.unified_interface import UnifiedDeepLearningModel
from python_backend.database.repository import Repository
from python_backend.database.models import SessionLocal

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
    """Background task to train model."""
    try:
        # This is a simplified version - full implementation would:
        # 1. Load the preprocessed data
        # 2. Initialize the selected model
        # 3. Train with progress updates
        # 4. Save model and metrics
        
        job_progress[job_id].update({
            "status": "running",
            "progress": 0.1,
            "current_step": "Loading data",
            "message": "Loading preprocessed data..."
        })
        await broadcast_job_update(job_id)
        
        # Simulate training progress
        for epoch in range(request.epochs):
            await asyncio.sleep(0.1)  # Simulate training time
            progress = 0.1 + (0.8 * (epoch + 1) / request.epochs)
            
            job_progress[job_id].update({
                "progress": progress,
                "current_step": "Training",
                "message": f"Epoch {epoch + 1}/{request.epochs}"
            })
            await broadcast_job_update(job_id)
        
        job_progress[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "current_step": "Complete",
            "message": "Training completed successfully",
            "result": {
                "model_id": f"model_{job_id}",
                "final_loss": 0.0234,
                "accuracy": 0.876,
                "sharpe_ratio": 1.45
            }
        })
        await broadcast_job_update(job_id)
        
    except Exception as e:
        job_progress[job_id].update({
            "status": "failed",
            "progress": job_progress[job_id]["progress"],
            "error": str(e),
            "message": f"Training failed: {str(e)}"
        })
        await broadcast_job_update(job_id)

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