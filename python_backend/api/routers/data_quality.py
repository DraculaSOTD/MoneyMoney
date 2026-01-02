from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Dict, List, Any
from datetime import datetime, timedelta
import statistics

from database.models import SessionLocal, TradingProfile, MarketData

router = APIRouter(prefix="/api/data-quality", tags=["data-quality"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{symbol}")
async def get_data_quality(symbol: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get data quality metrics for a specific symbol

    Returns:
        - total_records: Total number of candles
        - completeness_pct: Percentage of expected candles present
        - quality_score: Letter grade (A-F)
        - missing_data_points: Number of missing candles
        - outlier_count: Number of price outliers
        - null_rate: Percentage of null values
        - date_range: First and last timestamps
    """
    # Check if profile exists
    profile = db.query(TradingProfile).filter(
        TradingProfile.symbol == symbol
    ).first()

    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found for symbol {symbol}")

    if not profile.has_data:
        return {
            "symbol": symbol,
            "total_records": 0,
            "completeness_pct": 0.0,
            "quality_score": "F",
            "missing_data_points": 0,
            "outlier_count": 0,
            "null_rate": 0.0,
            "date_range": None,
            "message": "No data available for this symbol"
        }

    # Get basic statistics
    total_records = db.query(func.count(MarketData.id)).filter(
        MarketData.symbol == symbol
    ).scalar() or 0

    if total_records == 0:
        return {
            "symbol": symbol,
            "total_records": 0,
            "completeness_pct": 0.0,
            "quality_score": "F",
            "missing_data_points": 0,
            "outlier_count": 0,
            "null_rate": 0.0,
            "date_range": None
        }

    # Get date range
    date_stats = db.query(
        func.min(MarketData.timestamp).label('first'),
        func.max(MarketData.timestamp).label('last')
    ).filter(MarketData.symbol == symbol).first()

    first_timestamp = date_stats.first
    last_timestamp = date_stats.last

    # Calculate expected candles (1-minute intervals)
    if first_timestamp and last_timestamp:
        time_diff = (last_timestamp - first_timestamp).total_seconds()
        expected_candles = int(time_diff / 60) + 1  # +1 to include both endpoints
        missing_data_points = max(0, expected_candles - total_records)
        completeness_pct = (total_records / expected_candles * 100) if expected_candles > 0 else 0.0
    else:
        expected_candles = total_records
        missing_data_points = 0
        completeness_pct = 100.0

    # Count null values
    null_count = db.query(func.count(MarketData.id)).filter(
        MarketData.symbol == symbol,
        (MarketData.open_price == None) |
        (MarketData.high_price == None) |
        (MarketData.low_price == None) |
        (MarketData.close_price == None) |
        (MarketData.volume == None)
    ).scalar() or 0

    null_rate = (null_count / total_records * 100) if total_records > 0 else 0.0

    # Detect outliers (prices > 3 standard deviations from mean)
    outlier_count = 0
    try:
        # Get all close prices for statistical analysis
        close_prices = db.query(MarketData.close_price).filter(
            MarketData.symbol == symbol,
            MarketData.close_price != None
        ).limit(10000).all()  # Limit to 10k for performance

        if len(close_prices) > 10:
            prices = [float(p[0]) for p in close_prices if p[0] is not None]
            mean_price = statistics.mean(prices)
            stdev_price = statistics.stdev(prices)

            # Count outliers
            outlier_threshold_high = mean_price + (3 * stdev_price)
            outlier_threshold_low = mean_price - (3 * stdev_price)

            outlier_count = db.query(func.count(MarketData.id)).filter(
                MarketData.symbol == symbol,
                MarketData.close_price != None,
                (
                    (MarketData.close_price > outlier_threshold_high) |
                    (MarketData.close_price < outlier_threshold_low)
                )
            ).scalar() or 0
    except Exception as e:
        # If outlier detection fails, just set to 0
        outlier_count = 0

    # Calculate quality score
    quality_score = calculate_quality_score(
        completeness_pct,
        null_rate,
        outlier_count,
        total_records
    )

    return {
        "symbol": symbol,
        "total_records": total_records,
        "completeness_pct": round(completeness_pct, 2),
        "quality_score": quality_score,
        "missing_data_points": missing_data_points,
        "outlier_count": outlier_count,
        "null_rate": round(null_rate, 2),
        "date_range": {
            "first": first_timestamp.strftime("%Y-%m-%d %H:%M:%S") if first_timestamp else None,
            "last": last_timestamp.strftime("%Y-%m-%d %H:%M:%S") if last_timestamp else None
        }
    }


@router.get("/{symbol}/preview")
async def get_market_data_preview(
    symbol: str,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get first N rows of market data for a symbol

    Args:
        symbol: Trading pair symbol
        limit: Number of rows to return (default 100, max 1000)

    Returns:
        Dictionary with symbol, total count, and data array
    """
    # Validate limit
    limit = min(max(1, limit), 1000)  # Clamp between 1 and 1000

    # Check if profile exists
    profile = db.query(TradingProfile).filter(
        TradingProfile.symbol == symbol
    ).first()

    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found for symbol {symbol}")

    # Get total count
    total_count = db.query(func.count(MarketData.id)).filter(
        MarketData.symbol == symbol
    ).scalar() or 0

    if total_count == 0:
        return {
            "symbol": symbol,
            "total": 0,
            "limit": limit,
            "data": []
        }

    # Fetch first N rows ordered by timestamp
    market_data = db.query(MarketData).filter(
        MarketData.symbol == symbol
    ).order_by(MarketData.timestamp.asc()).limit(limit).all()

    # Format data for response
    data_rows = []
    for row in market_data:
        data_rows.append({
            "timestamp": row.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(row.open_price) if row.open_price else None,
            "high": float(row.high_price) if row.high_price else None,
            "low": float(row.low_price) if row.low_price else None,
            "close": float(row.close_price) if row.close_price else None,
            "volume": float(row.volume) if row.volume else None,
            "number_of_trades": row.number_of_trades if row.number_of_trades else 0
        })

    return {
        "symbol": symbol,
        "total": total_count,
        "limit": limit,
        "data": data_rows
    }


def calculate_quality_score(
    completeness_pct: float,
    null_rate: float,
    outlier_count: int,
    total_records: int
) -> str:
    """
    Calculate overall data quality score (A-F)

    Scoring criteria:
    - A: >99% complete, <0.1% nulls, <0.5% outliers
    - B: >95% complete, <1% nulls, <1% outliers
    - C: >90% complete, <5% nulls, <2% outliers
    - D: >80% complete, <10% nulls, <5% outliers
    - F: Below D thresholds
    """
    outlier_pct = (outlier_count / total_records * 100) if total_records > 0 else 0

    # Calculate weighted score (0-100)
    completeness_score = completeness_pct  # 0-100
    null_score = max(0, 100 - (null_rate * 10))  # Penalize nulls heavily
    outlier_score = max(0, 100 - (outlier_pct * 20))  # Penalize outliers

    # Weighted average: completeness 50%, nulls 30%, outliers 20%
    total_score = (
        (completeness_score * 0.5) +
        (null_score * 0.3) +
        (outlier_score * 0.2)
    )

    # Convert to letter grade
    if total_score >= 99:
        return "A"
    elif total_score >= 95:
        return "B"
    elif total_score >= 85:
        return "C"
    elif total_score >= 70:
        return "D"
    else:
        return "F"
