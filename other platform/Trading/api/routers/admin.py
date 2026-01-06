"""
Admin Management Endpoints
Handles data collection, model training, and system management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging
import asyncio

from database.models import (
    AdminUser, TradingProfile, DataCollectionJob, ModelTrainingJob,
    DataCollectionStatus, JobStatus, SessionLocal
)
from api.routers.admin_auth import get_current_admin, get_superuser_admin
from exchanges.binance_connector import BinanceConnector
from services.websocket_manager import connection_manager
from api.routers.admin_training import train_model_task
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# ==================== Request/Response Models ====================

class DataCollectionRequest(BaseModel):
    symbol: str
    days_back: int = 30
    exchange: str = "binance"

class DataCollectionResponse(BaseModel):
    job_id: str
    symbol: str
    status: str
    message: str

class DataCollectionStatusResponse(BaseModel):
    job_id: str
    symbol: str
    status: str
    progress: int
    current_stage: Optional[str]
    total_records: int
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]

class ModelTrainingRequest(BaseModel):
    symbol: str
    models: Optional[List[str]] = None  # If None, auto-select based on data

class ModelTrainingResponse(BaseModel):
    job_ids: List[str]
    symbol: str
    models: List[str]
    message: str

class ModelStatusResponse(BaseModel):
    job_id: str
    symbol: str
    model_name: str
    status: str
    progress: int
    accuracy: Optional[float]
    started_at: datetime
    completed_at: Optional[datetime]

class AvailableSymbolResponse(BaseModel):
    symbol: str
    name: str
    has_data: bool
    models_trained: bool
    data_updated_at: Optional[datetime]
    last_training: Optional[datetime]
    total_data_points: int

# ==================== Background Tasks ====================

async def collect_data_task(job_id: str, symbol: str, days_back: int, exchange: str):
    """Background task to collect data from exchange"""
    db = SessionLocal()
    try:
        # Get job
        job = db.query(DataCollectionJob).filter(DataCollectionJob.job_id == job_id).first()
        if not job:
            logger.error(f"Job not found: {job_id}")
            return

        # Broadcast started event
        await connection_manager.broadcast_data_collection_started(job_id, symbol, days_back)

        # Update job status
        job.status = DataCollectionStatus.FETCHING
        job.progress = 10
        job.current_stage = "fetching"
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_data_collection_progress(
            job_id, symbol, 10, "fetching", "fetching"
        )

        # Initialize exchange connector
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        connector = BinanceConnector(api_key, api_secret, testnet=testnet)

        # Fetch historical data (1-minute intervals)
        try:
            df = await connector.get_historical_data(symbol, interval='1m', days_back=days_back)
            job.fetched_records = len(df)
            job.progress = 33
            db.commit()

            # Broadcast progress
            await connection_manager.broadcast_data_collection_progress(
                job_id, symbol, 33, "fetching", "fetching"
            )

            logger.info(f"Fetched {len(df)} records for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            job.status = DataCollectionStatus.FAILED
            job.error_message = str(e)
            db.commit()

            # Broadcast failure
            await connection_manager.broadcast_data_collection_failed(job_id, symbol, str(e))
            return

        # Preprocessing stage
        job.status = DataCollectionStatus.PREPROCESSING
        job.progress = 50
        job.current_stage = "preprocessing"
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_data_collection_progress(
            job_id, symbol, 50, "preprocessing", "preprocessing"
        )

        # Basic preprocessing (remove duplicates, handle missing values)
        try:
            # Remove duplicates
            initial_count = len(df)
            df = df[~df.index.duplicated(keep='first')]
            job.duplicate_records = initial_count - len(df)

            # Check for missing values
            missing_count = df.isnull().sum().sum()
            job.missing_data_points = int(missing_count)

            # Forward fill missing values
            df = df.fillna(method='ffill')

            job.processed_records = len(df)
            job.progress = 66
            db.commit()

            # Broadcast progress
            await connection_manager.broadcast_data_collection_progress(
                job_id, symbol, 66, "preprocessing", "preprocessing"
            )

            logger.info(f"Preprocessed {len(df)} records for {symbol}")

        except Exception as e:
            logger.error(f"Error preprocessing data for {symbol}: {e}")
            job.status = DataCollectionStatus.FAILED
            job.error_message = str(e)
            db.commit()

            # Broadcast failure
            await connection_manager.broadcast_data_collection_failed(job_id, symbol, str(e))
            return

        # Storing stage
        job.status = DataCollectionStatus.STORING
        job.progress = 80
        job.current_stage = "storing"
        db.commit()

        # Broadcast progress
        await connection_manager.broadcast_data_collection_progress(
            job_id, symbol, 80, "storing", "storing"
        )

        # Store data (in production, save to file or database)
        try:
            # Create data directory if it doesn't exist
            data_dir = f"data/{symbol}"
            os.makedirs(data_dir, exist_ok=True)

            # Save to CSV
            filename = f"{data_dir}/{symbol}_1m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename)

            job.stored_records = len(df)
            job.total_records = len(df)

            # Calculate quality score
            quality = 100 - (job.missing_data_points / len(df) * 50) - (job.duplicate_records / initial_count * 50)
            job.quality_score = max(0, min(100, quality))

            logger.info(f"Stored {len(df)} records for {symbol} at {filename}")

        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
            job.status = DataCollectionStatus.FAILED
            job.error_message = str(e)
            db.commit()

            # Broadcast failure
            await connection_manager.broadcast_data_collection_failed(job_id, symbol, str(e))
            return

        # Update profile
        profile = db.query(TradingProfile).filter(TradingProfile.symbol == symbol).first()
        if not profile:
            # Create new profile
            profile = TradingProfile(
                symbol=symbol,
                name=symbol,
                profile_type="crypto" if "USDT" in symbol or "BTC" in symbol else "stock",
                exchange=exchange,
                base_currency=symbol.replace("USDT", "").replace("BTC", ""),
                quote_currency="USDT" if "USDT" in symbol else "BTC",
                data_interval='1m',
                has_data=True,
                data_updated_at=datetime.utcnow(),
                total_data_points=len(df)
            )
            db.add(profile)
        else:
            profile.has_data = True
            profile.data_updated_at = datetime.utcnow()
            profile.total_data_points = len(df)
            profile.data_interval = '1m'

        # Complete job
        job.status = DataCollectionStatus.COMPLETED
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

        db.commit()

        # Broadcast completion
        await connection_manager.broadcast_data_collection_completed(
            job_id, symbol, len(df)
        )

        logger.info(f"Data collection completed for {symbol}")

    except Exception as e:
        logger.error(f"Unexpected error in data collection task: {e}")
        try:
            job = db.query(DataCollectionJob).filter(DataCollectionJob.job_id == job_id).first()
            if job:
                job.status = DataCollectionStatus.FAILED
                job.error_message = str(e)
                db.commit()
        except:
            pass
    finally:
        db.close()

# ==================== API Endpoints ====================

@router.post("/data/collect/{symbol}", response_model=DataCollectionResponse)
async def start_data_collection(
    symbol: str,
    background_tasks: BackgroundTasks,
    days_back: int = 30,
    exchange: str = "binance",
    admin: AdminUser = Depends(get_current_admin)
):
    """Start data collection for a symbol"""
    db = SessionLocal()
    try:
        # Create job
        job_id = str(uuid.uuid4())
        job = DataCollectionJob(
            job_id=job_id,
            symbol=symbol,
            status=DataCollectionStatus.PENDING,
            interval='1m',
            days_back=days_back,
            data_source=exchange,
            config={
                "admin_id": admin.id,
                "admin_username": admin.username
            }
        )

        db.add(job)
        db.commit()

        # Start background task
        background_tasks.add_task(collect_data_task, job_id, symbol, days_back, exchange)

        logger.info(f"Data collection started for {symbol} by {admin.username}")

        return DataCollectionResponse(
            job_id=job_id,
            symbol=symbol,
            status="pending",
            message=f"Data collection started for {symbol}"
        )

    except Exception as e:
        logger.error(f"Error starting data collection: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start data collection: {str(e)}"
        )
    finally:
        db.close()

@router.get("/data/status/{job_id}", response_model=DataCollectionStatusResponse)
async def get_data_collection_status(
    job_id: str,
    admin: AdminUser = Depends(get_current_admin)
):
    """Get status of a data collection job"""
    db = SessionLocal()
    try:
        job = db.query(DataCollectionJob).filter(DataCollectionJob.job_id == job_id).first()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        return DataCollectionStatusResponse(
            job_id=job.job_id,
            symbol=job.symbol,
            status=job.status.value,
            progress=job.progress,
            current_stage=job.current_stage,
            total_records=job.total_records,
            error_message=job.error_message,
            started_at=job.started_at,
            completed_at=job.completed_at
        )

    finally:
        db.close()

@router.get("/data/available", response_model=List[AvailableSymbolResponse])
async def get_available_symbols(admin: AdminUser = Depends(get_current_admin)):
    """Get list of symbols with available data"""
    db = SessionLocal()
    try:
        profiles = db.query(TradingProfile).filter(TradingProfile.has_data == True).all()

        return [
            AvailableSymbolResponse(
                symbol=profile.symbol,
                name=profile.name,
                has_data=profile.has_data,
                models_trained=profile.models_trained,
                data_updated_at=profile.data_updated_at,
                last_training=profile.last_training,
                total_data_points=profile.total_data_points
            )
            for profile in profiles
        ]

    finally:
        db.close()

@router.delete("/data/{symbol}")
async def delete_symbol_data(
    symbol: str,
    admin: AdminUser = Depends(get_superuser_admin)
):
    """Delete data for a symbol (superuser only)"""
    db = SessionLocal()
    try:
        profile = db.query(TradingProfile).filter(TradingProfile.symbol == symbol).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Symbol not found"
            )

        # Reset data flags
        profile.has_data = False
        profile.models_trained = False
        profile.data_updated_at = None
        profile.last_training = None
        profile.total_data_points = 0

        db.commit()

        logger.info(f"Data deleted for {symbol} by {admin.username}")

        return {"message": f"Data deleted for {symbol}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete data: {str(e)}"
        )
    finally:
        db.close()

@router.post("/models/train/{symbol}", response_model=ModelTrainingResponse)
async def start_model_training(
    symbol: str,
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    admin: AdminUser = Depends(get_current_admin)
):
    """Start model training for a symbol"""
    db = SessionLocal()
    try:
        # Check if symbol has data
        profile = db.query(TradingProfile).filter(TradingProfile.symbol == symbol).first()

        if not profile or not profile.has_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No data available for {symbol}. Collect data first."
            )

        # Auto-select models if not specified
        models_to_train = request.models or [
            "ARIMA", "GARCH", "GRU_Attention", "CNN_Pattern",
            "LSTM", "Transformer", "XGBoost", "Random_Forest", "LightGBM", "Prophet"
        ]

        job_ids = []

        # Create training jobs for each model
        for model_name in models_to_train:
            job_id = str(uuid.uuid4())
            job = ModelTrainingJob(
                job_id=job_id,
                symbol=symbol,
                profile_id=profile.id,
                model_name=model_name,
                model_type=model_name,
                status=JobStatus.PENDING,
                parameters={"admin_id": admin.id},
                metadata={"started_by": admin.username}
            )

            db.add(job)
            job_ids.append(job_id)

            # Add training task to background tasks (async function directly)
            background_tasks.add_task(
                train_model_task,
                job_id=job_id,
                symbol=symbol,
                profile_id=profile.id,
                model_name=model_name
            )

        db.commit()

        logger.info(f"Model training started for {symbol}: {models_to_train} by {admin.username}")

        return ModelTrainingResponse(
            job_ids=job_ids,
            symbol=symbol,
            models=models_to_train,
            message=f"Training started for {len(models_to_train)} models"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )
    finally:
        db.close()

@router.get("/models/status", response_model=List[ModelStatusResponse])
async def get_all_model_status(admin: AdminUser = Depends(get_current_admin)):
    """Get status of all model training jobs"""
    db = SessionLocal()
    try:
        jobs = db.query(ModelTrainingJob).order_by(ModelTrainingJob.started_at.desc()).limit(50).all()

        return [
            ModelStatusResponse(
                job_id=job.job_id,
                symbol=job.symbol,
                model_name=job.model_name,
                status=job.status.value,
                progress=job.progress,
                accuracy=job.accuracy,
                started_at=job.started_at,
                completed_at=job.completed_at
            )
            for job in jobs
        ]

    finally:
        db.close()

@router.get("/models/status/{symbol}", response_model=List[ModelStatusResponse])
async def get_symbol_model_status(
    symbol: str,
    admin: AdminUser = Depends(get_current_admin)
):
    """Get status of model training jobs for a specific symbol"""
    db = SessionLocal()
    try:
        jobs = db.query(ModelTrainingJob).filter(
            ModelTrainingJob.symbol == symbol
        ).order_by(ModelTrainingJob.started_at.desc()).all()

        return [
            ModelStatusResponse(
                job_id=job.job_id,
                symbol=job.symbol,
                model_name=job.model_name,
                status=job.status.value,
                progress=job.progress,
                accuracy=job.accuracy,
                started_at=job.started_at,
                completed_at=job.completed_at
            )
            for job in jobs
        ]

    finally:
        db.close()

@router.post("/data/backfill/{symbol}")
async def backfill_symbol_data(symbol: str, admin: AdminUser = Depends(get_current_admin)):
    """
    Backfill missing historical data for a symbol.
    Fetches data in batches until caught up with current time.
    """
    from services.data_updater import data_updater

    try:
        result = await data_updater.backfill_missing_data(symbol)
        return result
    except Exception as e:
        logger.error(f"Backfill error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/profiles/{profile_id}/collect-sentiment")
async def collect_sentiment_data(
    profile_id: int,
    days_back: int = 30,
    admin: AdminUser = Depends(get_current_admin)
):
    """
    Collect and store sentiment data for a profile.
    This data is used by models that benefit from sentiment analysis (LSTM, Transformer, etc.).
    """
    from services.sentiment_collector import sentiment_collector

    db = SessionLocal()
    try:
        # Get profile
        profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile with id {profile_id} not found"
            )

        # Collect sentiment data
        result = await sentiment_collector.collect_sentiment(
            symbol=profile.symbol,
            profile_id=profile_id,
            days_back=days_back
        )

        logger.info(f"Sentiment collection completed for {profile.symbol} by {admin.username}")

        return {
            "success": True,
            "message": f"Collected {result['records_created']} sentiment records",
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error collecting sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect sentiment: {str(e)}"
        )
    finally:
        db.close()


@router.get("/profiles/{profile_id}/sentiment")
async def get_profile_sentiment(
    profile_id: int,
    admin: AdminUser = Depends(get_current_admin)
):
    """Get the latest sentiment data for a profile."""
    from services.sentiment_collector import sentiment_collector

    db = SessionLocal()
    try:
        profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile with id {profile_id} not found"
            )

        sentiment = await sentiment_collector.get_latest_sentiment(profile.symbol)

        if not sentiment:
            return {
                "symbol": profile.symbol,
                "has_sentiment": False,
                "message": "No sentiment data available. Run sentiment collection first."
            }

        return {
            "symbol": profile.symbol,
            "has_sentiment": True,
            **sentiment
        }

    finally:
        db.close()
