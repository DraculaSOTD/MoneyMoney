from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

from python_backend.database import get_session, TradingProfile, ProfileModel, ModelTrainingHistory, ProfilePrediction, ProfileMetrics, ProfileType, ModelStatus
from python_backend.database.models import SessionLocal

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

class ProfileTypeEnum(str, Enum):
    crypto = "crypto"
    stock = "stock"
    forex = "forex"
    commodity = "commodity"

class ProfileCreate(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT, AAPL)")
    name: str = Field(..., description="Display name for the profile")
    profile_type: ProfileTypeEnum
    exchange: str = Field(..., description="Exchange name (e.g., binance, nasdaq)")
    description: Optional[str] = None
    base_currency: str = Field(..., description="Base currency (e.g., BTC, USD)")
    quote_currency: str = Field(..., description="Quote currency (e.g., USDT, USD)")

    # Data Configuration
    data_source: str = Field(default="binance")
    timeframe: str = Field(default="1h")
    lookback_days: int = Field(default=365)

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

    # Data Configuration
    data_source: Optional[str] = None
    timeframe: Optional[str] = None
    lookback_days: Optional[int] = Field(None, gt=0)

class ProfileResponse(BaseModel):
    id: int
    symbol: str
    name: str
    profile_type: str
    exchange: str
    description: Optional[str]
    base_currency: str
    quote_currency: str
    created_at: datetime
    updated_at: datetime
    is_active: bool

    # Data Configuration
    data_source: str
    timeframe: str
    lookback_days: int

    # Data Availability Tracking
    has_data: bool = False
    data_updated_at: Optional[datetime] = None
    has_indicators: bool = False
    indicators_updated_at: Optional[datetime] = None
    models_trained: bool = False
    last_training: Optional[datetime] = None
    data_interval: Optional[str] = None
    total_data_points: int = 0

    # Performance Metrics
    total_trades: int
    win_rate: float
    avg_profit: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    last_trade_date: Optional[datetime]

    # Latest metrics
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[float] = None

    # Active models count
    active_models: int = 0
    deployed_models: int = 0

    class Config:
        orm_mode = True

class ModelCreate(BaseModel):
    profile_id: int
    model_name: str
    model_type: str = Field(..., description="Model type (lstm, gru, transformer, etc.)")
    model_version: str = Field(default="1.0.0")
    parameters: dict = Field(default_factory=dict)
    features: List[str] = Field(default_factory=list)
    preprocessing_config: dict = Field(default_factory=dict)

class ModelResponse(BaseModel):
    id: int
    profile_id: int
    model_name: str
    model_type: str
    model_version: str
    status: str
    parameters: dict
    features: List[str]
    preprocessing_config: dict
    last_trained: Optional[datetime]
    training_duration: Optional[float]
    training_samples: Optional[int]
    validation_accuracy: Optional[float]
    validation_loss: Optional[float]
    test_accuracy: Optional[float]
    test_sharpe: Optional[float]
    is_primary: bool
    is_deployed: bool
    deployed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class TrainingHistoryResponse(BaseModel):
    id: int
    profile_id: int
    model_id: int
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]
    status: str
    parameters: dict
    dataset_config: dict
    hardware_info: dict
    epochs_trained: int
    final_train_loss: Optional[float]
    final_val_loss: Optional[float]
    best_epoch: Optional[int]
    best_val_loss: Optional[float]
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    test_accuracy: Optional[float]
    backtest_sharpe: Optional[float]
    backtest_returns: Optional[float]
    backtest_max_drawdown: Optional[float]
    backtest_win_rate: Optional[float]
    
    class Config:
        orm_mode = True

class PredictionResponse(BaseModel):
    id: int
    profile_id: int
    model_id: int
    timestamp: datetime
    prediction_horizon: str
    price_prediction: Optional[float]
    direction_prediction: Optional[str]
    confidence: Optional[float]
    predicted_high: Optional[float]
    predicted_low: Optional[float]
    predicted_volatility: Optional[float]
    signal: Optional[str]
    signal_strength: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    actual_price: Optional[float]
    prediction_error: Optional[float]
    
    class Config:
        orm_mode = True

def get_db():
    """Database session dependency - uses PostgreSQL from environment"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=List[ProfileResponse])
async def get_profiles(
    profile_type: Optional[ProfileTypeEnum] = None,
    is_active: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all trading profiles with optional filtering"""
    query = db.query(TradingProfile)
    
    if profile_type:
        query = query.filter(TradingProfile.profile_type == ProfileType[profile_type.upper()])
    if is_active is not None:
        query = query.filter(TradingProfile.is_active == is_active)
    
    profiles = query.offset(skip).limit(limit).all()
    
    # Enrich with latest metrics and model counts
    enriched_profiles = []
    for profile in profiles:
        profile_dict = profile.__dict__.copy()

        # Get latest metrics from ProfileMetrics table
        latest_metric = db.query(ProfileMetrics).filter(
            ProfileMetrics.profile_id == profile.id
        ).order_by(ProfileMetrics.timestamp.desc()).first()

        if latest_metric:
            profile_dict['current_price'] = latest_metric.current_price
            profile_dict['price_change_24h'] = latest_metric.price_change_24h
            profile_dict['volume_24h'] = latest_metric.volume_24h
        else:
            # If no ProfileMetrics, calculate from MarketData table
            from python_backend.database.models import MarketData
            from datetime import datetime, timedelta
            from sqlalchemy import desc

            # Get latest candle for current price
            latest_candle = db.query(MarketData).filter(
                MarketData.symbol == profile.symbol
            ).order_by(desc(MarketData.timestamp)).first()

            if latest_candle:
                profile_dict['current_price'] = latest_candle.close_price

                # Get candle from 24 hours ago for price change calculation
                twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
                old_candle = db.query(MarketData).filter(
                    MarketData.symbol == profile.symbol,
                    MarketData.timestamp <= twenty_four_hours_ago
                ).order_by(desc(MarketData.timestamp)).first()

                if old_candle and old_candle.close_price:
                    price_change = ((latest_candle.close_price - old_candle.close_price) / old_candle.close_price) * 100
                    profile_dict['price_change_24h'] = round(price_change, 2)

        # Count active and deployed models
        profile_dict['active_models'] = db.query(ProfileModel).filter(
            ProfileModel.profile_id == profile.id,
            ProfileModel.status == ModelStatus.TRAINED
        ).count()

        profile_dict['deployed_models'] = db.query(ProfileModel).filter(
            ProfileModel.profile_id == profile.id,
            ProfileModel.is_deployed == True
        ).count()

        enriched_profiles.append(ProfileResponse(**profile_dict))
    
    return enriched_profiles

@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(profile_id: int, db: Session = Depends(get_db)):
    """Get a specific trading profile by ID"""
    profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    profile_dict = profile.__dict__.copy()
    
    # Get latest metrics
    latest_metric = db.query(ProfileMetrics).filter(
        ProfileMetrics.profile_id == profile.id
    ).order_by(ProfileMetrics.timestamp.desc()).first()
    
    if latest_metric:
        profile_dict['current_price'] = latest_metric.current_price
        profile_dict['price_change_24h'] = latest_metric.price_change_24h
        profile_dict['volume_24h'] = latest_metric.volume_24h
    
    # Count models
    profile_dict['active_models'] = db.query(ProfileModel).filter(
        ProfileModel.profile_id == profile.id,
        ProfileModel.status == ModelStatus.TRAINED
    ).count()
    
    profile_dict['deployed_models'] = db.query(ProfileModel).filter(
        ProfileModel.profile_id == profile.id,
        ProfileModel.is_deployed == True
    ).count()
    
    return ProfileResponse(**profile_dict)

@router.post("/", response_model=ProfileResponse)
async def create_profile(profile: ProfileCreate, db: Session = Depends(get_db)):
    """Create a new trading profile"""
    # Check if profile with same symbol already exists
    existing = db.query(TradingProfile).filter(TradingProfile.symbol == profile.symbol).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Profile with symbol {profile.symbol} already exists")

    # Extract profile_type before unpacking to avoid duplicate parameter
    profile_data = profile.dict()
    profile_type_str = profile_data.pop('profile_type')

    db_profile = TradingProfile(
        **profile_data,
        profile_type=ProfileType[profile_type_str.upper()]
    )

    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)

    # Enrich with computed fields (same pattern as get_profiles)
    profile_dict = db_profile.__dict__.copy()

    # Get latest metrics (will be None for new profile)
    latest_metric = db.query(ProfileMetrics).filter(
        ProfileMetrics.profile_id == db_profile.id
    ).order_by(ProfileMetrics.timestamp.desc()).first()

    if latest_metric:
        profile_dict['current_price'] = latest_metric.current_price
        profile_dict['price_change_24h'] = latest_metric.price_change_24h
        profile_dict['volume_24h'] = latest_metric.volume_24h
    else:
        # Set defaults for new profile
        profile_dict['current_price'] = None
        profile_dict['price_change_24h'] = None
        profile_dict['volume_24h'] = None

    # Count models (will be 0 for new profile)
    profile_dict['active_models'] = db.query(ProfileModel).filter(
        ProfileModel.profile_id == db_profile.id,
        ProfileModel.status == ModelStatus.TRAINED
    ).count()

    profile_dict['deployed_models'] = db.query(ProfileModel).filter(
        ProfileModel.profile_id == db_profile.id,
        ProfileModel.is_deployed == True
    ).count()

    return ProfileResponse(**profile_dict)

@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(profile_id: int, profile_update: ProfileUpdate, db: Session = Depends(get_db)):
    """Update a trading profile"""
    profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    update_data = profile_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(profile, field, value)

    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)

    # Enrich with computed fields (same pattern as get_profiles)
    profile_dict = profile.__dict__.copy()

    # Get latest metrics
    latest_metric = db.query(ProfileMetrics).filter(
        ProfileMetrics.profile_id == profile.id
    ).order_by(ProfileMetrics.timestamp.desc()).first()

    if latest_metric:
        profile_dict['current_price'] = latest_metric.current_price
        profile_dict['price_change_24h'] = latest_metric.price_change_24h
        profile_dict['volume_24h'] = latest_metric.volume_24h
    else:
        profile_dict['current_price'] = None
        profile_dict['price_change_24h'] = None
        profile_dict['volume_24h'] = None

    # Count models
    profile_dict['active_models'] = db.query(ProfileModel).filter(
        ProfileModel.profile_id == profile.id,
        ProfileModel.status == ModelStatus.TRAINED
    ).count()

    profile_dict['deployed_models'] = db.query(ProfileModel).filter(
        ProfileModel.profile_id == profile.id,
        ProfileModel.is_deployed == True
    ).count()

    return ProfileResponse(**profile_dict)

@router.delete("/{profile_id}")
async def delete_profile(profile_id: int, db: Session = Depends(get_db)):
    """Delete a trading profile and all associated data"""
    profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    db.delete(profile)
    db.commit()
    
    return {"message": f"Profile {profile.symbol} deleted successfully"}

@router.get("/{profile_id}/models", response_model=List[ModelResponse])
async def get_profile_models(
    profile_id: int,
    status: Optional[str] = None,
    is_deployed: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get all models for a specific profile"""
    query = db.query(ProfileModel).filter(ProfileModel.profile_id == profile_id)
    
    if status:
        query = query.filter(ProfileModel.status == ModelStatus[status.upper()])
    if is_deployed is not None:
        query = query.filter(ProfileModel.is_deployed == is_deployed)
    
    models = query.all()
    return [ModelResponse(**model.__dict__) for model in models]

@router.post("/{profile_id}/models", response_model=ModelResponse)
async def create_profile_model(profile_id: int, model: ModelCreate, db: Session = Depends(get_db)):
    """Create a new model for a profile"""
    # Verify profile exists
    profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    db_model = ProfileModel(
        profile_id=profile_id,
        **model.dict(exclude={'profile_id'}),
        status=ModelStatus.UNTRAINED
    )
    
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return ModelResponse(**db_model.__dict__)

@router.get("/{profile_id}/training-history", response_model=List[TrainingHistoryResponse])
async def get_training_history(
    profile_id: int,
    model_id: Optional[int] = None,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get training history for a profile"""
    query = db.query(ModelTrainingHistory).filter(
        ModelTrainingHistory.profile_id == profile_id
    )
    
    if model_id:
        query = query.filter(ModelTrainingHistory.model_id == model_id)
    
    history = query.order_by(ModelTrainingHistory.started_at.desc()).limit(limit).all()
    return [TrainingHistoryResponse(**h.__dict__) for h in history]

@router.get("/{profile_id}/predictions", response_model=List[PredictionResponse])
async def get_profile_predictions(
    profile_id: int,
    hours: int = Query(24, ge=1, le=168),
    model_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get recent predictions for a profile"""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    query = db.query(ProfilePrediction).filter(
        ProfilePrediction.profile_id == profile_id,
        ProfilePrediction.timestamp >= since
    )
    
    if model_id:
        query = query.filter(ProfilePrediction.model_id == model_id)
    
    predictions = query.order_by(ProfilePrediction.timestamp.desc()).all()
    return [PredictionResponse(**p.__dict__) for p in predictions]

@router.get("/{profile_id}/metrics/latest")
async def get_latest_metrics(profile_id: int, db: Session = Depends(get_db)):
    """Get the latest metrics for a profile"""
    metrics = db.query(ProfileMetrics).filter(
        ProfileMetrics.profile_id == profile_id
    ).order_by(ProfileMetrics.timestamp.desc()).first()

    if metrics:
        return metrics

    # If no ProfileMetrics, calculate from MarketData
    from python_backend.database.models import MarketData
    from sqlalchemy import desc

    # Get the profile to find its symbol
    profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Get latest candle
    latest_candle = db.query(MarketData).filter(
        MarketData.symbol == profile.symbol
    ).order_by(desc(MarketData.timestamp)).first()

    if not latest_candle:
        raise HTTPException(status_code=404, detail="No market data available for this profile")

    # Calculate price changes from historical candles
    from datetime import timedelta

    now = datetime.now()
    candle_24h_ago = db.query(MarketData).filter(
        MarketData.symbol == profile.symbol,
        MarketData.timestamp <= now - timedelta(hours=24)
    ).order_by(desc(MarketData.timestamp)).first()

    candle_7d_ago = db.query(MarketData).filter(
        MarketData.symbol == profile.symbol,
        MarketData.timestamp <= now - timedelta(days=7)
    ).order_by(desc(MarketData.timestamp)).first()

    candle_30d_ago = db.query(MarketData).filter(
        MarketData.symbol == profile.symbol,
        MarketData.timestamp <= now - timedelta(days=30)
    ).order_by(desc(MarketData.timestamp)).first()

    # Build metrics response
    calculated_metrics = {
        "id": None,
        "profile_id": profile_id,
        "timestamp": latest_candle.timestamp,
        "current_price": latest_candle.close_price,
        "price_change_24h": ((latest_candle.close_price - candle_24h_ago.close_price) / candle_24h_ago.close_price * 100) if candle_24h_ago else None,
        "price_change_7d": ((latest_candle.close_price - candle_7d_ago.close_price) / candle_7d_ago.close_price * 100) if candle_7d_ago else None,
        "price_change_30d": ((latest_candle.close_price - candle_30d_ago.close_price) / candle_30d_ago.close_price * 100) if candle_30d_ago else None,
        "volume_24h": latest_candle.volume,
    }

    return calculated_metrics

@router.get("/{profile_id}/metrics/history")
async def get_metrics_history(
    profile_id: int,
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db)
):
    """Get metrics history for a profile"""
    since = datetime.utcnow() - timedelta(hours=hours)

    metrics = db.query(ProfileMetrics).filter(
        ProfileMetrics.profile_id == profile_id,
        ProfileMetrics.timestamp >= since
    ).order_by(ProfileMetrics.timestamp).all()

    if metrics:
        return metrics

    # If no ProfileMetrics, generate from MarketData
    from python_backend.database.models import MarketData
    from sqlalchemy import desc

    # Get the profile to find its symbol
    profile = db.query(TradingProfile).filter(TradingProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Get candles for the requested time period, sample every 15 minutes
    candles = db.query(MarketData).filter(
        MarketData.symbol == profile.symbol,
        MarketData.timestamp >= since
    ).order_by(MarketData.timestamp).all()

    if not candles:
        return []

    # Sample candles to avoid too many data points (every 15 minutes for chart)
    sample_interval = max(1, len(candles) // 100)  # Max 100 points for chart
    sampled_candles = candles[::sample_interval]

    # Convert to metrics format
    metrics_history = []
    for candle in sampled_candles:
        metrics_history.append({
            "id": None,
            "profile_id": profile_id,
            "timestamp": candle.timestamp,
            "current_price": candle.close_price,
            "volume_24h": candle.volume,
        })

    return metrics_history

@router.put("/{profile_id}/models/{model_id}/deploy")
async def deploy_model(profile_id: int, model_id: int, db: Session = Depends(get_db)):
    """Deploy a model for trading"""
    model = db.query(ProfileModel).filter(
        ProfileModel.id == model_id,
        ProfileModel.profile_id == profile_id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.status != ModelStatus.TRAINED:
        raise HTTPException(status_code=400, detail="Model must be trained before deployment")
    
    # Set all other models for this profile to not primary
    db.query(ProfileModel).filter(
        ProfileModel.profile_id == profile_id,
        ProfileModel.id != model_id
    ).update({"is_primary": False})
    
    # Deploy the model
    model.is_deployed = True
    model.is_primary = True
    model.deployed_at = datetime.utcnow()
    model.status = ModelStatus.DEPLOYED
    
    db.commit()
    
    return {"message": f"Model {model.model_name} deployed successfully"}

@router.put("/{profile_id}/models/{model_id}/undeploy")
async def undeploy_model(profile_id: int, model_id: int, db: Session = Depends(get_db)):
    """Undeploy a model"""
    model = db.query(ProfileModel).filter(
        ProfileModel.id == model_id,
        ProfileModel.profile_id == profile_id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.is_deployed = False
    model.is_primary = False
    model.status = ModelStatus.TRAINED
    
    db.commit()
    
    return {"message": f"Model {model.model_name} undeployed successfully"}