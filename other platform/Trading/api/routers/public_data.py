"""
Public Data API Router
======================

Public endpoints for MoneyMoney integration (no authentication required).
These endpoints are meant to be called from the Node.js backend which handles its own auth.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import pandas as pd

from database.models import SessionLocal, TradingProfile, MarketData, ProfilePrediction
from services.data_aggregator import DataAggregator
from services.data_loader import DataLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/public", tags=["public"])


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


class IndicatorData(BaseModel):
    """Single indicator value"""
    name: str
    value: Optional[float]
    category: str


class IndicatorsResponse(BaseModel):
    """Technical indicators response"""
    symbol: str
    timestamp: str
    total_indicators: int
    indicators: List[IndicatorData]


class StatsResponse(BaseModel):
    """Statistical summary"""
    symbol: str
    timeframe: str
    high: float
    low: float
    open: float
    close: float
    volume: float
    change_percent: float


def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# NOTE: /indicators route must come BEFORE /{timeframe} to avoid "indicators" being matched as a timeframe
@router.get("/data/{symbol}/indicators", response_model=IndicatorsResponse)
async def get_public_indicators(
    symbol: str,
    limit: Optional[int] = Query(200, ge=1, le=500, description="Number of data points for calculation"),
    db = Depends(get_db)
):
    """
    Get technical indicators for an instrument (public endpoint).
    """
    try:
        # Find the profile
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.is_active == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found"
            )

        # Get data for indicator calculation
        aggregator = DataAggregator(db)
        df = aggregator.get_aggregated_data(
            symbol=symbol,
            profile_id=profile.id,
            timeframe='1h',
            limit=limit,
            use_cache=True
        )

        if df.empty or len(df) < 20:
            return IndicatorsResponse(
                symbol=symbol,
                timestamp=datetime.utcnow().isoformat(),
                total_indicators=0,
                indicators=[]
            )

        # Calculate indicators
        indicators = []

        # Moving Averages
        if len(df) >= 7:
            sma_7 = df['close'].rolling(window=7).mean().iloc[-1]
            indicators.append(IndicatorData(name="SMA 7", value=round(float(sma_7), 4), category="moving_average"))

        if len(df) >= 20:
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            indicators.append(IndicatorData(name="SMA 20", value=round(float(sma_20), 4), category="moving_average"))

        if len(df) >= 50:
            sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            indicators.append(IndicatorData(name="SMA 50", value=round(float(sma_50), 4), category="moving_average"))

        # EMA
        if len(df) >= 12:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
            indicators.append(IndicatorData(name="EMA 12", value=round(float(ema_12), 4), category="moving_average"))

        if len(df) >= 26:
            ema_26 = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
            indicators.append(IndicatorData(name="EMA 26", value=round(float(ema_26), 4), category="moving_average"))

        # RSI (14 period)
        if len(df) >= 14:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1]
            if pd.notna(rsi_value):
                indicators.append(IndicatorData(name="RSI 14", value=round(float(rsi_value), 2), category="oscillator"))

        # MACD
        if len(df) >= 26:
            ema_12_calc = df['close'].ewm(span=12, adjust=False).mean()
            ema_26_calc = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_12_calc - ema_26_calc
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line

            indicators.append(IndicatorData(name="MACD Line", value=round(float(macd_line.iloc[-1]), 4), category="oscillator"))
            indicators.append(IndicatorData(name="MACD Signal", value=round(float(signal_line.iloc[-1]), 4), category="oscillator"))
            indicators.append(IndicatorData(name="MACD Histogram", value=round(float(macd_histogram.iloc[-1]), 4), category="oscillator"))

        # Bollinger Bands
        if len(df) >= 20:
            sma_20_bb = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            upper_band = sma_20_bb + (std_20 * 2)
            lower_band = sma_20_bb - (std_20 * 2)

            indicators.append(IndicatorData(name="Bollinger Upper", value=round(float(upper_band.iloc[-1]), 4), category="volatility"))
            indicators.append(IndicatorData(name="Bollinger Middle", value=round(float(sma_20_bb.iloc[-1]), 4), category="volatility"))
            indicators.append(IndicatorData(name="Bollinger Lower", value=round(float(lower_band.iloc[-1]), 4), category="volatility"))

        # ATR (14 period)
        if len(df) >= 14:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            if pd.notna(atr):
                indicators.append(IndicatorData(name="ATR 14", value=round(float(atr), 4), category="volatility"))

        # Volume indicators
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        indicators.append(IndicatorData(name="Volume", value=round(float(current_volume), 2), category="volume"))
        indicators.append(IndicatorData(name="Avg Volume", value=round(float(avg_volume), 2), category="volume"))
        indicators.append(IndicatorData(name="Volume Ratio", value=round(float(volume_ratio), 2), category="volume"))

        # Stochastic Oscillator (14, 3, 3)
        if len(df) >= 14:
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            stoch_k = 100 * (df['close'] - low_14) / (high_14 - low_14)
            stoch_d = stoch_k.rolling(window=3).mean()
            stoch_k_value = stoch_k.iloc[-1]
            stoch_d_value = stoch_d.iloc[-1]
            if pd.notna(stoch_k_value):
                indicators.append(IndicatorData(name="Stochastic %K", value=round(float(stoch_k_value), 2), category="oscillator"))
            if pd.notna(stoch_d_value):
                indicators.append(IndicatorData(name="Stochastic %D", value=round(float(stoch_d_value), 2), category="oscillator"))

        # Williams %R (14 period)
        if len(df) >= 14:
            high_14 = df['high'].rolling(window=14).max()
            low_14 = df['low'].rolling(window=14).min()
            williams_r = -100 * (high_14 - df['close']) / (high_14 - low_14)
            williams_value = williams_r.iloc[-1]
            if pd.notna(williams_value):
                indicators.append(IndicatorData(name="Williams %R", value=round(float(williams_value), 2), category="oscillator"))

        # CCI (Commodity Channel Index, 20 period)
        if len(df) >= 20:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
            cci = (tp - sma_tp) / (0.015 * mad)
            cci_value = cci.iloc[-1]
            if pd.notna(cci_value):
                indicators.append(IndicatorData(name="CCI 20", value=round(float(cci_value), 2), category="oscillator"))

        # ADX (Average Directional Index, 14 period)
        if len(df) >= 28:  # Need extra data for smoothing
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)

            atr_14 = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()

            adx_value = adx.iloc[-1]
            plus_di_value = plus_di.iloc[-1]
            minus_di_value = minus_di.iloc[-1]

            if pd.notna(adx_value):
                indicators.append(IndicatorData(name="ADX 14", value=round(float(adx_value), 2), category="trend"))
            if pd.notna(plus_di_value):
                indicators.append(IndicatorData(name="+DI 14", value=round(float(plus_di_value), 2), category="trend"))
            if pd.notna(minus_di_value):
                indicators.append(IndicatorData(name="-DI 14", value=round(float(minus_di_value), 2), category="trend"))

        # OBV (On-Balance Volume)
        if len(df) >= 2:
            obv = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
            obv_value = obv.iloc[-1]
            obv_change = ((obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) * 100) if len(df) >= 5 and obv.iloc[-5] != 0 else 0
            indicators.append(IndicatorData(name="OBV", value=round(float(obv_value), 2), category="volume"))
            indicators.append(IndicatorData(name="OBV Change %", value=round(float(obv_change), 2), category="volume"))

        # Money Flow Index (MFI, 14 period)
        if len(df) >= 14:
            tp = (df['high'] + df['low'] + df['close']) / 3
            mf = tp * df['volume']
            pos_mf = mf.where(tp > tp.shift(), 0).rolling(window=14).sum()
            neg_mf = mf.where(tp < tp.shift(), 0).rolling(window=14).sum()
            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
            mfi_value = mfi.iloc[-1]
            if pd.notna(mfi_value):
                indicators.append(IndicatorData(name="MFI 14", value=round(float(mfi_value), 2), category="volume"))

        # Momentum (10 period)
        if len(df) >= 10:
            momentum = df['close'] - df['close'].shift(10)
            roc = (df['close'] / df['close'].shift(10) - 1) * 100
            momentum_value = momentum.iloc[-1]
            roc_value = roc.iloc[-1]
            if pd.notna(momentum_value):
                indicators.append(IndicatorData(name="Momentum 10", value=round(float(momentum_value), 4), category="momentum"))
            if pd.notna(roc_value):
                indicators.append(IndicatorData(name="ROC 10", value=round(float(roc_value), 2), category="momentum"))

        # Price position relative to MAs (trend signals)
        current_price = df['close'].iloc[-1]
        if len(df) >= 20:
            sma_20_val = df['close'].rolling(window=20).mean().iloc[-1]
            price_vs_sma20 = ((current_price - sma_20_val) / sma_20_val) * 100
            indicators.append(IndicatorData(name="Price vs SMA20 %", value=round(float(price_vs_sma20), 2), category="trend"))

        if len(df) >= 50:
            sma_50_val = df['close'].rolling(window=50).mean().iloc[-1]
            price_vs_sma50 = ((current_price - sma_50_val) / sma_50_val) * 100
            indicators.append(IndicatorData(name="Price vs SMA50 %", value=round(float(price_vs_sma50), 2), category="trend"))

        # Bollinger Band %B (position within bands)
        if len(df) >= 20:
            bb_pct = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            indicators.append(IndicatorData(name="Bollinger %B", value=round(float(bb_pct), 2), category="volatility"))

        return IndicatorsResponse(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            total_indicators=len(indicators),
            indicators=indicators
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching indicators for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch indicators: {str(e)}"
        )


@router.get("/data/{symbol}/{timeframe}", response_model=TimeframeDataResponse)
async def get_public_data(
    symbol: str,
    timeframe: str = Path(..., description="Timeframe: 1m, 5m, 1h, 1D, 1M"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Number of candles to return"),
    db = Depends(get_db)
):
    """
    Get OHLCV data for an instrument at a specific timeframe (public endpoint).
    No authentication required - meant for internal service calls.
    """
    try:
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '1h', '1D', '1M']
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )

        # Find the profile
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.is_active == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found"
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
                timestamp=row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else str(row['timestamp']),
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
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch data: {str(e)}"
        )


@router.get("/data/{symbol}/{timeframe}/stats", response_model=StatsResponse)
async def get_public_stats(
    symbol: str,
    timeframe: str = Path(..., description="Timeframe: 1m, 5m, 1h, 1D, 1M"),
    db = Depends(get_db)
):
    """
    Get statistical summary for an instrument (public endpoint).
    """
    try:
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '1h', '1D', '1M']
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )

        # Find the profile
        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == symbol,
            TradingProfile.is_active == True
        ).first()

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instrument {symbol} not found"
            )

        # Get aggregated data for stats
        aggregator = DataAggregator(db)
        df = aggregator.get_aggregated_data(
            symbol=symbol,
            profile_id=profile.id,
            timeframe=timeframe,
            limit=100,
            use_cache=True
        )

        if df.empty:
            return StatsResponse(
                symbol=symbol,
                timeframe=timeframe,
                high=0,
                low=0,
                open=0,
                close=0,
                volume=0,
                change_percent=0
            )

        # Calculate stats
        high = float(df['high'].max())
        low = float(df['low'].min())
        open_price = float(df.iloc[0]['open'])
        close_price = float(df.iloc[-1]['close'])
        total_volume = float(df['volume'].sum())
        change_percent = ((close_price - open_price) / open_price * 100) if open_price > 0 else 0

        return StatsResponse(
            symbol=symbol,
            timeframe=timeframe,
            high=high,
            low=low,
            open=open_price,
            close=close_price,
            volume=total_volume,
            change_percent=round(change_percent, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stats for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stats: {str(e)}"
        )


@router.get("/profiles/{profile_id}/predictions")
async def get_public_predictions(
    profile_id: int,
    limit: Optional[int] = Query(10, ge=1, le=100),
    db = Depends(get_db)
):
    """
    Get predictions for a profile (public endpoint).
    """
    try:
        # Get recent predictions
        predictions = db.query(ProfilePrediction).filter(
            ProfilePrediction.profile_id == profile_id
        ).order_by(ProfilePrediction.timestamp.desc()).limit(limit).all()

        return [
            {
                "id": p.id,
                "profile_id": p.profile_id,
                "model_name": p.model_name,
                "prediction_type": p.prediction_type,
                "predicted_value": float(p.predicted_value) if p.predicted_value else None,
                "actual_value": float(p.actual_value) if p.actual_value else None,
                "confidence": float(p.confidence) if p.confidence else None,
                "timestamp": p.timestamp.isoformat() if p.timestamp else None
            }
            for p in predictions
        ]

    except Exception as e:
        logger.error(f"Error fetching predictions for profile {profile_id}: {e}")
        return []


@router.get("/health")
async def public_health():
    """Health check endpoint (public)."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
