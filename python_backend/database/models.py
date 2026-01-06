from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, Enum as SQLEnum, JSON, Index, ForeignKey, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from decimal import Decimal
import enum

Base = declarative_base()

class OrderSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(enum.Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

class OrderStatus(enum.Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class PositionStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"

class ProfileType(enum.Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"

class ModelStatus(enum.Enum):
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    DEPLOYED = "deployed"

class DataCollectionStatus(enum.Enum):
    PENDING = "pending"
    FETCHING = "fetching"
    PREPROCESSING = "preprocessing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String, unique=True, index=True)
    symbol = Column(String, index=True)
    exchange = Column(String, index=True)
    order_id = Column(String, index=True)
    side = Column(SQLEnum(OrderSide))
    price = Column(Float)
    quantity = Column(Float)
    commission = Column(Float)
    commission_asset = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    strategy = Column(String, index=True)
    
    __table_args__ = (
        Index('ix_trades_symbol_timestamp', 'symbol', 'timestamp'),
        Index('ix_trades_strategy_timestamp', 'strategy', 'timestamp'),
    )

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String, unique=True, index=True)
    client_order_id = Column(String, unique=True, index=True)
    symbol = Column(String, index=True)
    exchange = Column(String, index=True)
    side = Column(SQLEnum(OrderSide))
    type = Column(SQLEnum(OrderType))
    status = Column(SQLEnum(OrderStatus), index=True)
    price = Column(Float)
    quantity = Column(Float)
    executed_qty = Column(Float, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    update_time = Column(DateTime, default=datetime.utcnow)
    strategy = Column(String, index=True)
    
    __table_args__ = (
        Index('ix_orders_symbol_timestamp', 'symbol', 'timestamp'),
        Index('ix_orders_status_timestamp', 'status', 'timestamp'),
    )

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    exchange = Column(String, index=True)
    side = Column(SQLEnum(OrderSide))
    entry_price = Column(Float)
    quantity = Column(Float)
    current_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    status = Column(SQLEnum(PositionStatus), index=True)
    realized_pnl = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)
    fees = Column(Float, default=0)
    strategy = Column(String, index=True)
    
    __table_args__ = (
        Index('ix_positions_status_symbol', 'status', 'symbol'),
        Index('ix_positions_strategy_status', 'strategy', 'status'),
    )

class ModelPrediction(Base):
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    prediction = Column(Float)
    confidence = Column(Float)
    features = Column(JSON)  # Store feature values used
    actual_price = Column(Float, nullable=True)
    error = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('ix_predictions_model_symbol_timestamp', 'model_name', 'symbol', 'timestamp'),
    )

class SystemMetrics(Base):
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metric_type = Column(String, index=True)
    metric_name = Column(String, index=True)
    value = Column(Float)
    metadata_ = Column('metadata', JSON)
    
    __table_args__ = (
        Index('ix_metrics_type_name_timestamp', 'metric_type', 'metric_name', 'timestamp'),
    )

class RiskMetrics(Base):
    __tablename__ = 'risk_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    portfolio_value = Column(Float)
    daily_pnl = Column(Float)
    total_pnl = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    current_drawdown = Column(Float)
    var_95 = Column(Float)  # Value at Risk 95%
    cvar_95 = Column(Float)  # Conditional VaR 95%
    exposure = Column(Float)
    open_positions = Column(Integer)
    
class Alert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String, index=True)
    severity = Column(String, index=True)
    message = Column(String)
    metadata_ = Column('metadata', JSON)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    
class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, index=True)
    strategy_name = Column(String, index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    final_capital = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    parameters = Column(JSON)
    detailed_results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataQuality(Base):
    __tablename__ = 'data_quality'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String, index=True)
    data_type = Column(String, index=True)
    missing_points = Column(Integer)
    outliers = Column(Integer)
    quality_score = Column(Float)
    issues = Column(JSON)

class TradingProfile(Base):
    __tablename__ = 'trading_profiles'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String)
    profile_type = Column(SQLEnum(ProfileType))
    exchange = Column(String)
    description = Column(Text, nullable=True)
    base_currency = Column(String)
    quote_currency = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    auto_update_enabled = Column(Boolean, default=True)  # Enable automatic minute-by-minute updates

    # Data Availability Tracking
    has_data = Column(Boolean, default=False, index=True)
    data_updated_at = Column(DateTime, nullable=True)
    has_indicators = Column(Boolean, default=False, index=True)
    indicators_updated_at = Column(DateTime, nullable=True)
    models_trained = Column(Boolean, default=False, index=True)
    last_training = Column(DateTime, nullable=True)
    data_interval = Column(String, default='1m')  # Enforce 1-minute intervals
    total_data_points = Column(Integer, default=0)
    
    # Data Configuration
    data_source = Column(String, default='binance')
    timeframe = Column(String, default='1h')
    lookback_days = Column(Integer, default=365)

    # Performance Metrics
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    last_trade_date = Column(DateTime, nullable=True)
    
    # Relationships
    models = relationship("ProfileModel", back_populates="profile", cascade="all, delete-orphan")
    training_history = relationship("ModelTrainingHistory", back_populates="profile", cascade="all, delete-orphan")
    predictions = relationship("ProfilePrediction", back_populates="profile", cascade="all, delete-orphan")
    metrics = relationship("ProfileMetrics", back_populates="profile", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_profiles_type_active', 'profile_type', 'is_active'),
    )


class MarketData(Base):
    """OHLCV market data for trading profiles"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True, nullable=False)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id'), index=True)

    # Timestamp
    timestamp = Column(DateTime, index=True, nullable=False)

    # OHLCV Data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Additional fields
    number_of_trades = Column(Integer, nullable=True)
    quote_asset_volume = Column(Float, nullable=True)
    taker_buy_base_volume = Column(Float, nullable=True)
    taker_buy_quote_volume = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('ix_market_data_profile_timestamp', 'profile_id', 'timestamp'),
        UniqueConstraint('symbol', 'timestamp', name='uq_market_data_symbol_timestamp'),
    )


class IndicatorData(Base):
    """Technical indicator data calculated from market data"""
    __tablename__ = 'indicator_data'

    id = Column(Integer, primary_key=True)
    market_data_id = Column(Integer, ForeignKey('market_data.id', ondelete='CASCADE'), nullable=False, index=True)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id', ondelete='CASCADE'), nullable=False, index=True)

    # Moving Averages
    sma_10 = Column(Float, nullable=True)
    sma_20 = Column(Float, nullable=True)
    sma_50 = Column(Float, nullable=True)
    sma_200 = Column(Float, nullable=True)
    ema_12 = Column(Float, nullable=True)
    ema_26 = Column(Float, nullable=True)

    # MACD
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_histogram = Column(Float, nullable=True)

    # RSI
    rsi_14 = Column(Float, nullable=True)

    # Bollinger Bands
    bb_upper = Column(Float, nullable=True)
    bb_middle = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    bb_width = Column(Float, nullable=True)

    # ATR
    atr_14 = Column(Float, nullable=True)

    # Stochastic
    stoch_k = Column(Float, nullable=True)
    stoch_d = Column(Float, nullable=True)

    # ADX
    adx = Column(Float, nullable=True)
    plus_di = Column(Float, nullable=True)
    minus_di = Column(Float, nullable=True)

    # Volume indicators
    obv = Column(Float, nullable=True)

    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)
    config_version = Column(String(50), nullable=True)

    __table_args__ = (
        Index('ix_indicator_data_market_data', 'market_data_id'),
        Index('ix_indicator_data_profile', 'profile_id'),
        UniqueConstraint('market_data_id', name='uq_indicator_market_data'),
    )


class ProfileModel(Base):
    __tablename__ = 'profile_models'
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id'), index=True)
    model_name = Column(String)
    model_type = Column(String)  # 'lstm', 'gru', 'transformer', etc.
    model_version = Column(String)
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.UNTRAINED)
    
    # Model Configuration
    parameters = Column(JSON)  # Model hyperparameters
    features = Column(JSON)  # List of features used
    preprocessing_config = Column(JSON)  # Preprocessing settings
    
    # Training Info
    last_trained = Column(DateTime, nullable=True)
    training_duration = Column(Float, nullable=True)  # in seconds
    training_samples = Column(Integer, nullable=True)
    
    # Performance Metrics
    validation_accuracy = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    test_sharpe = Column(Float, nullable=True)
    
    # Deployment
    is_primary = Column(Boolean, default=False)
    is_deployed = Column(Boolean, default=False)
    deployed_at = Column(DateTime, nullable=True)
    
    # Model Storage
    model_path = Column(String, nullable=True)  # Path to saved model
    scaler_path = Column(String, nullable=True)  # Path to saved scaler
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profile = relationship("TradingProfile", back_populates="models")
    training_history = relationship("ModelTrainingHistory", back_populates="model", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_profile_models_profile_status', 'profile_id', 'status'),
        Index('ix_profile_models_deployed', 'is_deployed', 'profile_id'),
    )

class ModelTrainingHistory(Base):
    __tablename__ = 'model_training_history'
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id'), index=True)
    model_id = Column(Integer, ForeignKey('profile_models.id'), index=True)
    
    # Training Run Info
    run_id = Column(String, unique=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)  # in seconds
    status = Column(String)  # 'running', 'completed', 'failed', 'cancelled'
    
    # Training Configuration
    parameters = Column(JSON)  # Hyperparameters used
    dataset_config = Column(JSON)  # Dataset split, augmentation, etc.
    hardware_info = Column(JSON)  # GPU, memory usage, etc.
    
    # Training Metrics
    epochs_trained = Column(Integer, default=0)
    final_train_loss = Column(Float, nullable=True)
    final_val_loss = Column(Float, nullable=True)
    best_epoch = Column(Integer, nullable=True)
    best_val_loss = Column(Float, nullable=True)
    
    # Performance Metrics
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    
    # Trading Performance (if backtested)
    backtest_sharpe = Column(Float, nullable=True)
    backtest_returns = Column(Float, nullable=True)
    backtest_max_drawdown = Column(Float, nullable=True)
    backtest_win_rate = Column(Float, nullable=True)
    
    # Logs and Artifacts
    training_logs = Column(JSON)  # Epoch-wise metrics
    error_logs = Column(Text, nullable=True)
    artifacts_path = Column(String, nullable=True)  # Path to training artifacts
    
    # Relationships
    profile = relationship("TradingProfile", back_populates="training_history")
    model = relationship("ProfileModel", back_populates="training_history")
    
    __table_args__ = (
        Index('ix_training_history_profile_time', 'profile_id', 'started_at'),
        Index('ix_training_history_model_time', 'model_id', 'started_at'),
    )

class ProfilePrediction(Base):
    __tablename__ = 'profile_predictions'
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id'), index=True)
    model_id = Column(Integer, ForeignKey('profile_models.id'), index=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    prediction_horizon = Column(String)  # '1h', '4h', '1d', etc.
    
    # Predictions
    price_prediction = Column(Float, nullable=True)
    direction_prediction = Column(String, nullable=True)  # 'up', 'down', 'neutral'
    confidence = Column(Float, nullable=True)
    
    # Additional Predictions
    predicted_high = Column(Float, nullable=True)
    predicted_low = Column(Float, nullable=True)
    predicted_volatility = Column(Float, nullable=True)
    
    # Trading Signals
    signal = Column(String, nullable=True)  # 'buy', 'sell', 'hold'
    signal_strength = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # Features Used
    features = Column(JSON)  # Feature values at prediction time
    
    # Actual Values (for tracking)
    actual_price = Column(Float, nullable=True)
    actual_high = Column(Float, nullable=True)
    actual_low = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)
    
    # Relationships
    profile = relationship("TradingProfile", back_populates="predictions")
    
    __table_args__ = (
        Index('ix_predictions_profile_time', 'profile_id', 'timestamp'),
        Index('ix_predictions_profile_model_time', 'profile_id', 'model_id', 'timestamp'),
    )

class ProfileMetrics(Base):
    __tablename__ = 'profile_metrics'
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id'), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Price Metrics
    current_price = Column(Float)
    price_change_24h = Column(Float)
    price_change_7d = Column(Float)
    price_change_30d = Column(Float)
    
    # Volume Metrics
    volume_24h = Column(Float)
    volume_change_24h = Column(Float)
    avg_volume_7d = Column(Float)
    
    # Technical Indicators
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_lower = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Market Metrics
    market_cap = Column(Float, nullable=True)
    circulating_supply = Column(Float, nullable=True)
    
    # Sentiment Metrics
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    social_volume = Column(Integer, nullable=True)
    news_mentions = Column(Integer, nullable=True)
    
    # Custom Metrics
    custom_metrics = Column(JSON)  # For additional metrics
    
    # Relationships
    profile = relationship("TradingProfile", back_populates="metrics")
    
    __table_args__ = (
        Index('ix_metrics_profile_time', 'profile_id', 'timestamp'),
    )

class DataCollectionJob(Base):
    __tablename__ = 'data_collection_jobs'

    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True, index=True)
    symbol = Column(String, index=True)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id'), nullable=True, index=True)

    # Job Status
    status = Column(SQLEnum(DataCollectionStatus), default=DataCollectionStatus.PENDING, index=True)
    progress = Column(Integer, default=0)  # 0-100
    current_stage = Column(String, nullable=True)  # 'fetching', 'preprocessing', 'storing'

    # Collection Configuration
    interval = Column(String, default='1m')
    days_back = Column(Integer, default=30)
    data_source = Column(String, default='binance')

    # Results
    total_records = Column(Integer, default=0)
    fetched_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)
    stored_records = Column(Integer, default=0)

    # Data Quality
    missing_data_points = Column(Integer, default=0)
    duplicate_records = Column(Integer, default=0)
    quality_score = Column(Float, nullable=True)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Error Handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    retry_count = Column(Integer, default=0)

    # Metadata
    config = Column(JSON)  # Store full configuration
    logs = Column(JSON)  # Store stage-by-stage logs

    __table_args__ = (
        Index('ix_data_jobs_symbol_status', 'symbol', 'status'),
        Index('ix_data_jobs_started_at', 'started_at'),
    )

class ModelTrainingJob(Base):
    __tablename__ = 'model_training_jobs'

    id = Column(Integer, primary_key=True)
    job_id = Column(String, unique=True, index=True)
    symbol = Column(String, index=True)
    profile_id = Column(Integer, ForeignKey('trading_profiles.id'), nullable=True, index=True)
    model_id = Column(Integer, ForeignKey('profile_models.id'), nullable=True, index=True)

    # Model Information
    model_name = Column(String, index=True)
    model_type = Column(String)  # 'ARIMA', 'GARCH', 'GRU', 'CNN', etc.
    model_category = Column(String)  # 'statistical', 'deep_learning', 'reinforcement', etc.

    # Job Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    progress = Column(Integer, default=0)  # 0-100
    current_epoch = Column(Integer, nullable=True)
    total_epochs = Column(Integer, nullable=True)

    # Training Configuration
    parameters = Column(JSON)  # Hyperparameters
    features_used = Column(JSON)  # List of features
    dataset_split = Column(JSON)  # Train/val/test split info

    # Training Results
    training_samples = Column(Integer, nullable=True)
    validation_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)

    # Performance Metrics
    final_train_loss = Column(Float, nullable=True)
    final_val_loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)

    # Trading Performance
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Error Handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)

    # Model Artifacts
    model_path = Column(String, nullable=True)
    checkpoint_path = Column(String, nullable=True)
    metrics_path = Column(String, nullable=True)

    # Metadata
    training_logs = Column(JSON)  # Epoch-wise metrics
    job_metadata = Column(JSON)  # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)

    __table_args__ = (
        Index('ix_training_jobs_symbol_status', 'symbol', 'status'),
        Index('ix_training_jobs_profile_status', 'profile_id', 'status'),
        Index('ix_training_jobs_started_at', 'started_at'),
    )

class AdminUser(Base):
    __tablename__ = 'admin_users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)

    # Admin Info
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

    # Security
    last_login = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)

    # Session Management
    session_token = Column(String, nullable=True)
    session_expires = Column(DateTime, nullable=True)

    # Audit
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('admin_users.id'), nullable=True)

    __table_args__ = (
        Index('ix_admin_users_active', 'is_active'),
    )

class SentimentData(Base):
    __tablename__ = 'sentiment_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Sentiment Scores
    overall_sentiment = Column(Float)  # -1 to 1
    confidence = Column(Float)  # 0 to 1

    # Source-specific Sentiment
    twitter_sentiment = Column(Float, nullable=True)
    twitter_volume = Column(Integer, nullable=True)
    reddit_sentiment = Column(Float, nullable=True)
    reddit_volume = Column(Integer, nullable=True)
    news_sentiment = Column(Float, nullable=True)
    news_volume = Column(Integer, nullable=True)

    # Trend Analysis
    sentiment_trend = Column(Float, nullable=True)  # Rate of change
    volatility = Column(Float, nullable=True)  # Sentiment volatility

    # Aggregation Window
    window_size = Column(String)  # '1D', '1W', '1M'
    window_start = Column(DateTime, nullable=True)
    window_end = Column(DateTime, nullable=True)

    # Metadata
    data_sources = Column(JSON)  # List of sources used
    keywords_tracked = Column(JSON)  # Keywords monitored
    raw_data = Column(JSON, nullable=True)  # Store raw sentiment data

    __table_args__ = (
        Index('ix_sentiment_symbol_timestamp', 'symbol', 'timestamp'),
        Index('ix_sentiment_window', 'symbol', 'window_size', 'timestamp'),
    )

def create_tables(engine):
    Base.metadata.create_all(bind=engine)

def get_session(database_url):
    # Add check_same_thread for SQLite
    if database_url.startswith('sqlite'):
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(database_url)
    create_tables(engine)
    Session = sessionmaker(bind=engine)
    return Session()

# Create SessionLocal factory for dependency injection
import os
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres@localhost:5432/trading_platform')

if DATABASE_URL.startswith('sqlite'):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

# Create all tables
create_tables(engine)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)