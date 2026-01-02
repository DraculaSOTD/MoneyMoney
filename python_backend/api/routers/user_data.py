"""
User Data API Router
====================

User-facing endpoints for MoneyMoney integration.
Only returns instruments that have been data-collected AND model-trained.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import pandas as pd

from python_backend.database.models import SessionLocal, TradingProfile, MarketData
from python_backend.services.data_aggregator import DataAggregator
from python_backend.services.sentiment_aggregator import SentimentAggregator
from python_backend.services.data_loader import DataLoader
from python_backend.api.routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["user"])


class InstrumentResponse(BaseModel):
    """User-facing instrument data"""
    symbol: str
    name: str
    category: str
    has_data: bool
    models_trained: bool
    model_accuracy: Optional[float]
    confidence: Optional[int]
    data_updated_at: Optional[str]
    last_training: Optional[str]
    total_data_points: int
    data_interval: str
    current_price: Optional[float]
    change_percent: Optional[float]
    signal: Optional[str]  # 'buy', 'sell', 'hold'


class PredictionResponse(BaseModel):
    """ML model prediction"""
    symbol: str
    model_name: str
    prediction_type: str  # 'price', 'direction', 'volatility'
    predicted_value: float
    confidence: float
    timeframe: str
    timestamp: str


class SignalResponse(BaseModel):
    """Trading signal"""
    symbol: str
    signal: str  # 'buy', 'sell', 'hold'
    confidence: int  # 0-100
    entry_point: Optional[float]
    take_profit: Optional[float]
    stop_loss: Optional[float]
    reasoning: str
    generated_at: str


class OHLCVData(BaseModel):
    """OHLCV candle data"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TimeframeDataResponse(BaseModel):
    """Timeframe data for charts"""
    symbol: str
    timeframe: str
    candles: List[OHLCVData]
    total_candles: int


class SentimentSourceBreakdown(BaseModel):
    """Sentiment breakdown by source"""
    avg_sentiment: float
    total_volume: int
    data_points: int


class SentimentResponse(BaseModel):
    """Sentiment analysis result"""
    symbol: str
    window: str
    overall_sentiment: float  # -1 to 1
    sentiment_label: str  # "Very Positive", "Positive", "Neutral", "Negative", "Very Negative"
    avg_confidence: Optional[float]
    sentiment_trend: float  # Positive = improving, Negative = declining
    sentiment_volatility: float
    recommendation: str
    data_points: int
    latest_update: Optional[str]
    window_start: str
    window_end: str
    source_breakdown: Optional[Dict[str, SentimentSourceBreakdown]]


class SentimentHistoryPoint(BaseModel):
    """Single sentiment data point for charting"""
    timestamp: str
    overall_sentiment: Optional[float]
    confidence: Optional[float]
    twitter_sentiment: Optional[float]
    reddit_sentiment: Optional[float]
    news_sentiment: Optional[float]


class IndicatorData(BaseModel):
    """Single technical indicator"""
    name: str
    value: Optional[float]
    category: str  # 'moving_average', 'oscillator', 'volatility', 'trend', 'volume', 'support_resistance', 'divergence', 'pattern'


class IndicatorsResponse(BaseModel):
    """Collection of technical indicators for an instrument"""
    symbol: str
    timestamp: str
    total_indicators: int
    categories: Dict[str, int]  # Count per category
    indicators: List[IndicatorData]


def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/instruments", response_model=List[InstrumentResponse])
async def get_trained_instruments(
    category: Optional[str] = Query(None, description="Filter by category (crypto, forex, stocks)"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get list of instruments that have data AND trained models.

    **CRITICAL:** Only returns instruments where:
    - `has_data = True`
    - `models_trained = True`

    This ensures users only see instruments ready for analysis.

    **Example:**
    ```
    GET /api/user/instruments
    GET /api/user/instruments?category=crypto
    ```
    """
    try:
        # Query for instruments with data AND trained models
        query = db.query(TradingProfile).filter(
            TradingProfile.has_data == True,
            TradingProfile.models_trained == True
        )

        # Apply category filter if provided
        if category:
            query = query.filter(TradingProfile.profile_type == category)

        profiles = query.all()

        if not profiles:
            return []

        # Convert to response format
        instruments = []
        for profile in profiles:
            # Get latest market data for current price
            latest_data = db.query(MarketData).filter(
                MarketData.symbol == profile.symbol,
                MarketData.profile_id == profile.id
            ).order_by(MarketData.timestamp.desc()).first()

            current_price = None
            change_percent = None

            if latest_data:
                current_price = float(latest_data.close_price)

                # Calculate 24h change if we have enough data
                day_ago_data = db.query(MarketData).filter(
                    MarketData.symbol == profile.symbol,
                    MarketData.profile_id == profile.id,
                    MarketData.timestamp <= datetime.utcnow() - timedelta(days=1)
                ).order_by(MarketData.timestamp.desc()).first()

                if day_ago_data:
                    old_price = float(day_ago_data.close_price)
                    change_percent = ((current_price - old_price) / old_price) * 100

            instruments.append(InstrumentResponse(
                symbol=profile.symbol,
                name=profile.name or profile.symbol,
                category=profile.profile_type,
                has_data=profile.has_data,
                models_trained=profile.models_trained,
                model_accuracy=None,  # TODO: Get from model results
                confidence=None,  # TODO: Get from latest prediction
                data_updated_at=profile.data_updated_at.isoformat() if profile.data_updated_at else None,
                last_training=profile.last_training.isoformat() if profile.last_training else None,
                total_data_points=profile.total_data_points,
                data_interval=profile.data_interval,
                current_price=current_price,
                change_percent=change_percent,
                signal=None  # TODO: Get from latest signal
            ))

        logger.info(f"Returned {len(instruments)} trained instruments for user {current_user.id}")
        return instruments

    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch instruments"
        )


@router.get("/instruments/{symbol}/data/{timeframe}", response_model=TimeframeDataResponse)
async def get_instrument_data(
    symbol: str,
    timeframe: str = Path(..., regex="^(1m|5m|1h|1D|1M)$", description="Timeframe: 1m, 5m, 1h, 1D, 1M"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Number of candles to return"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get OHLCV data for an instrument at a specific timeframe.

    **Timeframes:**
    - `1m`: 1-minute candles
    - `5m`: 5-minute candles
    - `1h`: 1-hour candles
    - `1D`: 1-day candles
    - `1M`: 1-month candles

    **Example:**
    ```
    GET /api/user/instruments/BTCUSDT/data/1h?limit=100
    ```
    Returns the last 100 1-hour candles for BTC/USDT.
    """
    try:
        # Verify instrument has data and models
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.has_data == True,
            TradingProfile.models_trained == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found or not ready for analysis"
            )

        # Get aggregated data
        aggregator = DataAggregator(db)
        df = aggregator.get_aggregated_data(
            symbol=symbol,
            profile_id=profile.id,
            timeframe=timeframe,
            limit=limit,
            use_cache=True
        )

        if df.empty:
            return TimeframeDataResponse(
                symbol=symbol,
                timeframe=timeframe,
                candles=[],
                total_candles=0
            )

        # Convert to response format
        candles = [
            OHLCVData(
                timestamp=row['timestamp'].isoformat(),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            for _, row in df.iterrows()
        ]

        return TimeframeDataResponse(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            total_candles=len(candles)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch instrument data"
        )


@router.get("/instruments/{symbol}/predictions", response_model=List[PredictionResponse])
async def get_instrument_predictions(
    symbol: str,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get ML model predictions for an instrument.

    Returns predictions from all trained models (ARIMA, GARCH, GRU, CNN).

    **Example:**
    ```
    GET /api/user/instruments/BTCUSDT/predictions
    ```

    **TODO:** This endpoint needs to be connected to actual model prediction logic.
    For now, it returns a placeholder structure.
    """
    try:
        # Verify instrument has trained models
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.models_trained == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trained models found for {symbol}"
            )

        # TODO: Implement actual model prediction retrieval
        # For now, return placeholder data
        predictions = [
            PredictionResponse(
                symbol=symbol,
                model_name="ARIMA",
                prediction_type="price",
                predicted_value=0.0,  # TODO: Get from model
                confidence=0.85,
                timeframe="1h",
                timestamp=datetime.utcnow().isoformat()
            ),
            # Add more models as they become available
        ]

        logger.warning(f"Predictions endpoint called for {symbol} - returning placeholder data")
        return predictions

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch predictions"
        )


@router.get("/instruments/{symbol}/signals", response_model=SignalResponse)
async def get_trading_signal(
    symbol: str,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get current trading signal for an instrument.

    Combines ML predictions with technical analysis to generate a trading signal.

    **Signals:**
    - `buy`: Strong buy signal
    - `sell`: Strong sell signal
    - `hold`: No clear signal

    **Example:**
    ```
    GET /api/user/instruments/BTCUSDT/signals
    ```

    **TODO:** This endpoint needs to be connected to actual signal generation logic.
    For now, it returns a placeholder structure.
    """
    try:
        # Verify instrument has trained models
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.models_trained == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trained models found for {symbol}"
            )

        # TODO: Implement actual signal generation logic
        # This should combine:
        # - ML model predictions
        # - Technical indicators
        # - Risk management rules

        # Placeholder signal
        signal = SignalResponse(
            symbol=symbol,
            signal="hold",
            confidence=0,
            entry_point=None,
            take_profit=None,
            stop_loss=None,
            reasoning="Signal generation not yet implemented. Models are trained and ready.",
            generated_at=datetime.utcnow().isoformat()
        )

        logger.warning(f"Signals endpoint called for {symbol} - returning placeholder data")
        return signal

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching signal for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch trading signal"
        )


@router.get("/instruments/{symbol}/stats")
async def get_instrument_stats(
    symbol: str,
    timeframe: str = Query("1D", regex="^(1m|5m|1h|1D|1M)$"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get statistical summary for an instrument.

    Returns price statistics, volume, volatility, etc.

    **Example:**
    ```
    GET /api/user/instruments/BTCUSDT/stats?timeframe=1D
    ```
    """
    try:
        # Verify instrument exists and has data
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.has_data == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found or has no data"
            )

        # Use data aggregator to get stats
        aggregator = DataAggregator(db)
        stats = aggregator.get_stats_for_timeframe(
            symbol=symbol,
            profile_id=profile.id,
            timeframe=timeframe
        )

        if 'error' in stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=stats['error']
            )

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stats for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch statistics"
        )


@router.get("/instruments/{symbol}/sentiment", response_model=SentimentResponse)
async def get_instrument_sentiment(
    symbol: str,
    window: str = Query("1D", regex="^(1D|1W|1M|3M|6M|1Y)$", description="Time window: 1D, 1W, 1M, 3M, 6M, 1Y"),
    include_breakdown: bool = Query(True, description="Include source-specific breakdown"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get sentiment analysis for an instrument within a time window.

    **Time Windows:**
    - `1D`: Last 24 hours
    - `1W`: Last 7 days
    - `1M`: Last 30 days
    - `3M`: Last 90 days
    - `6M`: Last 180 days
    - `1Y`: Last 365 days

    **Response includes:**
    - Overall sentiment score (-1 to 1)
    - Sentiment label (Very Positive to Very Negative)
    - Trend (improving vs declining)
    - Volatility (sentiment stability)
    - Trading recommendation based on sentiment
    - Optional breakdown by source (Twitter, Reddit, News)

    **Example:**
    ```
    GET /api/user/instruments/BTCUSDT/sentiment?window=1W
    ```

    **Note:** Sentiment is adjusted to match user's selected timeframe for consistency
    with price chart views.
    """
    try:
        # Verify instrument exists
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.has_data == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found"
            )

        # Get sentiment data
        aggregator = SentimentAggregator(db)
        sentiment = aggregator.get_sentiment_for_window(
            symbol=symbol,
            window=window,
            include_breakdown=include_breakdown
        )

        if not sentiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No sentiment data available for {symbol} in window {window}"
            )

        logger.info(f"Returned sentiment for {symbol} ({window}): {sentiment['overall_sentiment']:.2f}")
        return sentiment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch sentiment data"
        )


@router.get("/instruments/{symbol}/sentiment/history", response_model=List[SentimentHistoryPoint])
async def get_instrument_sentiment_history(
    symbol: str,
    days_back: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of data points"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get historical sentiment data for charting.

    Returns time-series sentiment data that can be overlaid on price charts.

    **Example:**
    ```
    GET /api/user/instruments/BTCUSDT/sentiment/history?days_back=7&limit=100
    ```
    """
    try:
        # Verify instrument exists
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found"
            )

        # Get historical sentiment
        aggregator = SentimentAggregator(db)
        history = aggregator.get_sentiment_history(
            symbol=symbol,
            days_back=days_back,
            limit=limit
        )

        logger.info(f"Returned {len(history)} sentiment history points for {symbol}")
        return history

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment history for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch sentiment history"
        )


def categorize_indicator(indicator_name: str) -> str:
    """
    Categorize an indicator based on its name.

    Args:
        indicator_name: Name of the indicator

    Returns:
        Category string
    """
    name_upper = indicator_name.upper()

    # Moving Averages
    if any(x in name_upper for x in ['SMA', 'EMA', 'WMA', 'VWMA', 'MA_']):
        return 'moving_average'

    # MACD related
    if any(x in name_upper for x in ['MACD', 'SIGNAL_LINE', 'HISTOGRAM']):
        return 'oscillator'

    # Oscillators
    if any(x in name_upper for x in ['RSI', '%K', '%D', 'STOCH', 'MOMENTUM']):
        return 'oscillator'

    # Volatility
    if any(x in name_upper for x in ['BB', 'BOLLINGER', 'ATR', 'BANDWIDTH', 'WIDTH']):
        return 'volatility'

    # Trend indicators
    if any(x in name_upper for x in ['SAR', 'PARABOLIC', 'ADX', '+DI', '-DI', 'ICHIMOKU', 'TENKAN', 'KIJUN', 'SENKOU', 'CHIKOU']):
        return 'trend'

    # Volume
    if any(x in name_upper for x in ['CMF', 'VOLUME', 'VWAP']):
        return 'volume'

    # Support/Resistance
    if any(x in name_upper for x in ['PP', 'PIVOT', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3', 'SUPPORT', 'RESISTANCE']):
        return 'support_resistance'

    # Divergences
    if 'DIVERGENCE' in name_upper:
        return 'divergence'

    # Patterns
    if any(x in name_upper for x in ['ELLIOTT', 'WAVE', 'PATTERN']):
        return 'pattern'

    # Default
    return 'other'


@router.get("/instruments/{symbol}/indicators", response_model=IndicatorsResponse)
async def get_instrument_indicators(
    symbol: str,
    limit: Optional[int] = Query(200, ge=50, le=500, description="Number of candles to use for indicator calculation"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get all technical indicators for an instrument.

    Returns 70+ technical indicators including:
    - Moving Averages (SMA, EMA, VWMA)
    - MACD components
    - Oscillators (RSI, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR)
    - Trend indicators (Parabolic SAR, ADX, Ichimoku)
    - Volume indicators (CMF)
    - Support/Resistance levels
    - Divergences
    - Elliott Wave patterns

    **Example:**
    ```
    GET /api/user/instruments/BTCUSDT/indicators?limit=200
    ```

    Returns the latest indicator values computed from the last 200 candles.
    """
    try:
        # Verify instrument has data
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.has_data == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found or has no data"
            )

        # Load data with indicators using DataLoader
        loader = DataLoader(db)
        df = loader.get_latest_candles(
            symbol=symbol,
            profile_id=profile.id,
            limit=limit,
            with_indicators=True
        )

        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data available for {symbol}"
            )

        # Get the latest row (most recent indicators)
        latest_row = df.iloc[-1]

        # Get timestamp
        if 'timestamp' in df.columns:
            timestamp = latest_row['timestamp']
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
        else:
            timestamp_str = datetime.utcnow().isoformat()

        # Extract indicators (skip OHLCV columns)
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        indicator_names = [col for col in df.columns if col not in base_cols]

        # Build indicator list with categorization
        indicators_list = []
        category_counts = {}

        for ind_name in indicator_names:
            value = latest_row[ind_name]

            # Convert to float, handle NaN
            if pd.isna(value):
                float_value = None
            else:
                float_value = float(value)

            # Categorize
            category = categorize_indicator(ind_name)

            indicators_list.append(IndicatorData(
                name=ind_name,
                value=float_value,
                category=category
            ))

            # Count categories
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

        logger.info(f"Returned {len(indicators_list)} indicators for {symbol}")

        return IndicatorsResponse(
            symbol=symbol,
            timestamp=timestamp_str,
            total_indicators=len(indicators_list),
            categories=category_counts,
            indicators=indicators_list
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching indicators for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch indicators: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for MoneyMoney integration"""
    return {
        "status": "healthy",
        "service": "Trading Platform User API",
        "timestamp": datetime.utcnow().isoformat()
    }
