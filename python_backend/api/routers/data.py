"""
Data API Router
===============

Provides endpoints for accessing market data with timeframe aggregation.
All data is stored at 1-minute intervals and aggregated on-demand.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

from python_backend.database.models import SessionLocal, TradingProfile, MarketData, IndicatorData
from python_backend.services.data_aggregator import DataAggregator
from python_backend.api.routers.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


class CandleData(BaseModel):
    """Single candle/bar data"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TimeframeDataResponse(BaseModel):
    """Response for timeframe data request"""
    symbol: str
    timeframe: str
    candle_count: int
    start_time: Optional[str]
    end_time: Optional[str]
    candles: List[CandleData]


class MultiTimeframeResponse(BaseModel):
    """Response for multiple timeframe request"""
    symbol: str
    timeframes: Dict[str, List[CandleData]]


class TimeframeStatsResponse(BaseModel):
    """Statistical summary for a timeframe"""
    symbol: str
    timeframe: str
    candle_count: int
    start_time: Optional[str]
    end_time: Optional[str]
    price_high: float
    price_low: float
    price_open: float
    price_close: float
    price_change: float
    price_change_pct: float
    total_volume: float
    avg_volume: float
    volatility: float


class LatestCandleResponse(BaseModel):
    """Latest candle data"""
    symbol: str
    timeframe: str
    candle: Optional[CandleData]


def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_profile_id(symbol: str, user_id: int, db) -> int:
    """Get profile ID for a symbol and user"""
    profile = db.query(TradingProfile).filter(
        TradingProfile.symbol == symbol,
        TradingProfile.user_id == user_id
    ).first()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trading profile found for {symbol}"
        )

    if not profile.has_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data available for {symbol}. Please collect data first."
        )

    return profile.id


@router.get("/{symbol}/{timeframe}", response_model=TimeframeDataResponse)
async def get_data_for_timeframe(
    symbol: str,
    timeframe: str,
    start_time: Optional[datetime] = Query(None, description="Start timestamp (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End timestamp (ISO format)"),
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Maximum number of candles"),
    use_cache: bool = Query(True, description="Use cached data if available"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get market data for a specific timeframe.

    **Supported timeframes:**
    - `1m`: 1-minute candles (raw data)
    - `5m`: 5-minute candles
    - `1h`: 1-hour candles
    - `1D`: 1-day candles
    - `1M`: 1-month candles

    **Example:**
    ```
    GET /data/BTCUSDT/1h?limit=100
    ```
    Returns the last 100 1-hour candles for BTC/USDT.
    """
    try:
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '1h', '1D', '1M']
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )

        # Get profile ID
        profile_id = get_profile_id(symbol, current_user.id, db)

        # Create aggregator
        aggregator = DataAggregator(db)

        # Get aggregated data
        df = aggregator.get_aggregated_data(
            symbol=symbol,
            profile_id=profile_id,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            use_cache=use_cache
        )

        if df.empty:
            return TimeframeDataResponse(
                symbol=symbol,
                timeframe=timeframe,
                candle_count=0,
                start_time=None,
                end_time=None,
                candles=[]
            )

        # Convert to response format
        candles = [
            CandleData(
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
            candle_count=len(candles),
            start_time=df['timestamp'].min().isoformat() if not df.empty else None,
            end_time=df['timestamp'].max().isoformat() if not df.empty else None,
            candles=candles
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch data"
        )


@router.get("/{symbol}/multiple", response_model=MultiTimeframeResponse)
async def get_multiple_timeframes(
    symbol: str,
    timeframes: str = Query(..., description="Comma-separated timeframes (e.g., '1m,5m,1h')"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum candles per timeframe"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get data for multiple timeframes in a single request.

    **Example:**
    ```
    GET /data/BTCUSDT/multiple?timeframes=1m,5m,1h&limit=100
    ```
    Returns the last 100 candles for each timeframe.

    This is more efficient than making separate requests for each timeframe.
    """
    try:
        # Parse timeframes
        timeframe_list = [tf.strip() for tf in timeframes.split(',')]

        # Validate timeframes
        valid_timeframes = ['1m', '5m', '1h', '1D', '1M']
        for tf in timeframe_list:
            if tf not in valid_timeframes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid timeframe '{tf}'. Must be one of: {', '.join(valid_timeframes)}"
                )

        # Get profile ID
        profile_id = get_profile_id(symbol, current_user.id, db)

        # Create aggregator
        aggregator = DataAggregator(db)

        # Get data for all timeframes
        data_dict = aggregator.get_multiple_timeframes(
            symbol=symbol,
            profile_id=profile_id,
            timeframes=timeframe_list,
            limit=limit
        )

        # Convert to response format
        result = {}
        for tf, df in data_dict.items():
            if df.empty:
                result[tf] = []
            else:
                result[tf] = [
                    CandleData(
                        timestamp=row['timestamp'].isoformat(),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume'])
                    )
                    for _, row in df.iterrows()
                ]

        return MultiTimeframeResponse(
            symbol=symbol,
            timeframes=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching multiple timeframes for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch data"
        )


@router.get("/{symbol}/{timeframe}/latest", response_model=LatestCandleResponse)
async def get_latest_candle(
    symbol: str,
    timeframe: str = Path(..., description="Timeframe (1m, 5m, 1h, 1D, 1M)"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get the most recent candle for a symbol and timeframe.

    **Example:**
    ```
    GET /data/BTCUSDT/1h/latest
    ```
    Returns the latest 1-hour candle for BTC/USDT.

    This endpoint always returns fresh data (no caching).
    """
    try:
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '1h', '1D', '1M']
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )

        # Get profile ID
        profile_id = get_profile_id(symbol, current_user.id, db)

        # Create aggregator
        aggregator = DataAggregator(db)

        # Get latest candle
        candle_data = aggregator.get_latest_candle(
            symbol=symbol,
            profile_id=profile_id,
            timeframe=timeframe
        )

        if candle_data is None:
            return LatestCandleResponse(
                symbol=symbol,
                timeframe=timeframe,
                candle=None
            )

        candle = CandleData(**candle_data)

        return LatestCandleResponse(
            symbol=symbol,
            timeframe=timeframe,
            candle=candle
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest candle for {symbol} {timeframe}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch latest candle"
        )


@router.get("/{symbol}/{timeframe}/stats", response_model=TimeframeStatsResponse)
async def get_timeframe_stats(
    symbol: str,
    timeframe: str,
    start_time: Optional[datetime] = Query(None, description="Start timestamp (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End timestamp (ISO format)"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get statistical summary for a timeframe.

    **Returns:**
    - Candle count
    - Price range (high, low, open, close)
    - Price change (absolute and percentage)
    - Volume statistics (total, average)
    - Volatility (standard deviation of returns)

    **Example:**
    ```
    GET /data/BTCUSDT/1h/stats
    ```
    Returns statistics for BTC/USDT 1-hour candles.
    """
    try:
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '1h', '1D', '1M']
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )

        # Get profile ID
        profile_id = get_profile_id(symbol, current_user.id, db)

        # Create aggregator
        aggregator = DataAggregator(db)

        # Get stats
        stats = aggregator.get_stats_for_timeframe(
            symbol=symbol,
            profile_id=profile_id,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        if 'error' in stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=stats['error']
            )

        return TimeframeStatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stats for {symbol} {timeframe}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch statistics"
        )


@router.post("/{symbol}/cache/clear")
async def clear_cache(
    symbol: str,
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Clear aggregation cache for a symbol.

    Useful when new data has been collected and you want to force refresh.

    **Example:**
    ```
    POST /data/BTCUSDT/cache/clear
    ```
    """
    try:
        # Verify symbol exists
        _ = get_profile_id(symbol, current_user.id, db)

        # Clear cache
        aggregator = DataAggregator(db)
        aggregator.clear_cache(symbol)

        logger.info(f"Cache cleared for {symbol} by user {current_user.id}")

        return {
            "message": f"Cache cleared for {symbol}",
            "symbol": symbol
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.get("/available")
async def get_available_symbols(
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get list of symbols with available data for the current user.

    **Returns:**
    - Symbol
    - Data availability status
    - Total data points
    - Last update timestamp
    - Data interval (should always be '1m')
    - Model training status

    **Example:**
    ```
    GET /data/available
    ```
    """
    try:
        # Query profiles with data
        profiles = db.query(TradingProfile).filter(
            TradingProfile.user_id == current_user.id,
            TradingProfile.has_data == True
        ).all()

        result = [
            {
                'symbol': p.symbol,
                'has_data': p.has_data,
                'total_data_points': p.total_data_points,
                'data_updated_at': p.data_updated_at.isoformat() if p.data_updated_at else None,
                'data_interval': p.data_interval,
                'models_trained': p.models_trained,
                'last_training': p.last_training.isoformat() if p.last_training else None
            }
            for p in profiles
        ]

        return {
            'count': len(result),
            'symbols': result
        }

    except Exception as e:
        logger.error(f"Error fetching available symbols: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch available symbols"
        )


class IndicatorPoint(BaseModel):
    """Single indicator data point"""
    timestamp: str
    # Moving Averages
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    # RSI
    rsi_14: Optional[float] = None
    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    # ATR
    atr_14: Optional[float] = None
    # Stochastic
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    # ADX
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    # Volume
    obv: Optional[float] = None


class IndicatorDataResponse(BaseModel):
    """Response for indicator data request"""
    symbol: str
    has_indicators: bool
    indicators_updated_at: Optional[str]
    count: int
    start_time: Optional[str]
    end_time: Optional[str]
    indicators: List[IndicatorPoint]


@router.get("/{symbol}/indicators", response_model=IndicatorDataResponse)
async def get_indicators(
    symbol: str,
    start_time: Optional[datetime] = Query(None, description="Start timestamp (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End timestamp (ISO format)"),
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Maximum number of data points"),
    current_user = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get technical indicator data for a symbol.

    **Returns:**
    - Technical indicators calculated from market data
    - Includes moving averages, MACD, RSI, Bollinger Bands, ATR, Stochastic, ADX, OBV

    **Example:**
    ```
    GET /data/BTCUSDT/indicators?limit=1000
    ```
    Returns the last 1000 indicator data points for BTC/USDT.

    **Note:** Indicators must be preprocessed first by an admin user.
    Use the admin endpoint `/admin/data/preprocess/{symbol}` to calculate indicators.
    """
    try:
        # Get profile
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.user_id == current_user.id
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trading profile found for {symbol}"
            )

        if not profile.has_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data available for {symbol}. Please collect data first."
            )

        # Check if indicators exist
        if not profile.has_indicators:
            return IndicatorDataResponse(
                symbol=symbol,
                has_indicators=False,
                indicators_updated_at=None,
                count=0,
                start_time=None,
                end_time=None,
                indicators=[]
            )

        # Query indicator data
        query = db.query(IndicatorData, MarketData.timestamp).join(
            MarketData, IndicatorData.market_data_id == MarketData.id
        ).filter(
            IndicatorData.profile_id == profile.id
        )

        # Apply time filters
        if start_time:
            query = query.filter(MarketData.timestamp >= start_time)
        if end_time:
            query = query.filter(MarketData.timestamp <= end_time)

        # Order by timestamp
        query = query.order_by(MarketData.timestamp.desc())

        # Apply limit
        if limit:
            query = query.limit(limit)

        # Execute query
        results = query.all()

        if not results:
            return IndicatorDataResponse(
                symbol=symbol,
                has_indicators=True,
                indicators_updated_at=profile.indicators_updated_at.isoformat() if profile.indicators_updated_at else None,
                count=0,
                start_time=None,
                end_time=None,
                indicators=[]
            )

        # Convert to response format
        indicators = []
        timestamps = []

        for indicator_data, timestamp in results:
            indicators.append(IndicatorPoint(
                timestamp=timestamp.isoformat(),
                sma_10=indicator_data.sma_10,
                sma_20=indicator_data.sma_20,
                sma_50=indicator_data.sma_50,
                sma_200=indicator_data.sma_200,
                ema_12=indicator_data.ema_12,
                ema_26=indicator_data.ema_26,
                macd=indicator_data.macd,
                macd_signal=indicator_data.macd_signal,
                macd_histogram=indicator_data.macd_histogram,
                rsi_14=indicator_data.rsi_14,
                bb_upper=indicator_data.bb_upper,
                bb_middle=indicator_data.bb_middle,
                bb_lower=indicator_data.bb_lower,
                bb_width=indicator_data.bb_width,
                atr_14=indicator_data.atr_14,
                stoch_k=indicator_data.stoch_k,
                stoch_d=indicator_data.stoch_d,
                adx=indicator_data.adx,
                plus_di=indicator_data.plus_di,
                minus_di=indicator_data.minus_di,
                obv=indicator_data.obv
            ))
            timestamps.append(timestamp)

        # Reverse to get chronological order (oldest first)
        indicators.reverse()
        timestamps.reverse()

        return IndicatorDataResponse(
            symbol=symbol,
            has_indicators=True,
            indicators_updated_at=profile.indicators_updated_at.isoformat() if profile.indicators_updated_at else None,
            count=len(indicators),
            start_time=timestamps[0].isoformat() if timestamps else None,
            end_time=timestamps[-1].isoformat() if timestamps else None,
            indicators=indicators
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching indicators for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch indicators"
        )
