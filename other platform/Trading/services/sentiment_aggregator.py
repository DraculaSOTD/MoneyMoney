"""
Sentiment Aggregation Service
==============================

Aggregates sentiment data from multiple sources with time-windowed calculations
based on user's selected timeframe. Integrates with existing sentiment analyzer
infrastructure.

Supported windows:
- 1D: Last 24 hours
- 1W: Last 7 days
- 1M: Last 30 days
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
import numpy as np

from database.models import SentimentData, TradingProfile

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Aggregates and provides sentiment data with time-window support.
    """

    # Window definitions in days
    WINDOW_DEFINITIONS = {
        '1D': 1,
        '1W': 7,
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365
    }

    def __init__(self, db: Session):
        """
        Initialize sentiment aggregator.

        Args:
            db: Database session
        """
        self.db = db

    def get_sentiment_for_window(
        self,
        symbol: str,
        window: str = '1D',
        include_breakdown: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get aggregated sentiment for a symbol within a time window.

        Args:
            symbol: Trading symbol
            window: Time window ('1D', '1W', '1M', '3M', '6M', '1Y')
            include_breakdown: Include source-specific breakdown

        Returns:
            Dictionary with sentiment metrics or None
        """
        if window not in self.WINDOW_DEFINITIONS:
            logger.warning(f"Invalid window: {window}, defaulting to 1D")
            window = '1D'

        try:
            # Calculate time range
            days_back = self.WINDOW_DEFINITIONS[window]
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)

            # Query sentiment data for window
            sentiment_records = self.db.query(SentimentData).filter(
                and_(
                    SentimentData.symbol == symbol,
                    SentimentData.timestamp >= start_time,
                    SentimentData.timestamp <= end_time
                )
            ).order_by(SentimentData.timestamp.desc()).all()

            if not sentiment_records:
                logger.info(f"No sentiment data found for {symbol} in window {window}")
                return None

            # Aggregate sentiment scores
            overall_scores = [r.overall_sentiment for r in sentiment_records if r.overall_sentiment is not None]
            confidences = [r.confidence for r in sentiment_records if r.confidence is not None]

            if not overall_scores:
                return None

            # Calculate weighted average (more recent = higher weight)
            weights = self._calculate_time_weights(sentiment_records)
            weighted_sentiment = np.average(overall_scores, weights=weights)

            # Calculate trend (comparing first half vs second half)
            mid_point = len(overall_scores) // 2
            if mid_point > 0:
                first_half_avg = np.mean(overall_scores[:mid_point])
                second_half_avg = np.mean(overall_scores[mid_point:])
                sentiment_trend = second_half_avg - first_half_avg
            else:
                sentiment_trend = 0.0

            # Calculate volatility (standard deviation)
            sentiment_volatility = float(np.std(overall_scores)) if len(overall_scores) > 1 else 0.0

            result = {
                'symbol': symbol,
                'window': window,
                'overall_sentiment': round(float(weighted_sentiment), 3),  # -1 to 1
                'avg_confidence': round(float(np.mean(confidences)), 3) if confidences else None,
                'sentiment_trend': round(float(sentiment_trend), 3),  # Positive = improving, Negative = declining
                'sentiment_volatility': round(sentiment_volatility, 3),
                'data_points': len(sentiment_records),
                'latest_update': sentiment_records[0].timestamp.isoformat() if sentiment_records else None,
                'window_start': start_time.isoformat(),
                'window_end': end_time.isoformat(),
            }

            # Add source-specific breakdown if requested
            if include_breakdown:
                breakdown = self._get_source_breakdown(sentiment_records)
                result['source_breakdown'] = breakdown

            # Add sentiment label
            result['sentiment_label'] = self._get_sentiment_label(weighted_sentiment)

            # Add recommendation
            result['recommendation'] = self._get_sentiment_recommendation(
                weighted_sentiment,
                sentiment_trend,
                sentiment_volatility
            )

            return result

        except Exception as e:
            logger.error(f"Error aggregating sentiment for {symbol}: {e}")
            return None

    def _calculate_time_weights(self, records: List[SentimentData]) -> List[float]:
        """
        Calculate time-based weights (exponential decay - more recent = higher weight).

        Args:
            records: List of sentiment records (newest first)

        Returns:
            List of weights
        """
        if not records:
            return []

        # Exponential decay: weight = e^(-k * age_in_hours)
        decay_rate = 0.05  # Adjust for slower/faster decay

        current_time = datetime.utcnow()
        weights = []

        for record in records:
            age_hours = (current_time - record.timestamp).total_seconds() / 3600
            weight = np.exp(-decay_rate * age_hours)
            weights.append(weight)

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        return weights

    def _get_source_breakdown(self, records: List[SentimentData]) -> Dict[str, Any]:
        """
        Get sentiment breakdown by source (Twitter, Reddit, News).

        Args:
            records: List of sentiment records

        Returns:
            Dictionary with source-specific metrics
        """
        breakdown = {}

        # Twitter sentiment
        twitter_scores = [r.twitter_sentiment for r in records if r.twitter_sentiment is not None]
        twitter_volumes = [r.twitter_volume for r in records if r.twitter_volume is not None]

        if twitter_scores:
            breakdown['twitter'] = {
                'avg_sentiment': round(float(np.mean(twitter_scores)), 3),
                'total_volume': int(sum(twitter_volumes)) if twitter_volumes else 0,
                'data_points': len(twitter_scores)
            }

        # Reddit sentiment
        reddit_scores = [r.reddit_sentiment for r in records if r.reddit_sentiment is not None]
        reddit_volumes = [r.reddit_volume for r in records if r.reddit_volume is not None]

        if reddit_scores:
            breakdown['reddit'] = {
                'avg_sentiment': round(float(np.mean(reddit_scores)), 3),
                'total_volume': int(sum(reddit_volumes)) if reddit_volumes else 0,
                'data_points': len(reddit_scores)
            }

        # News sentiment
        news_scores = [r.news_sentiment for r in records if r.news_sentiment is not None]
        news_volumes = [r.news_volume for r in records if r.news_volume is not None]

        if news_scores:
            breakdown['news'] = {
                'avg_sentiment': round(float(np.mean(news_scores)), 3),
                'total_volume': int(sum(news_volumes)) if news_volumes else 0,
                'data_points': len(news_scores)
            }

        return breakdown

    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """
        Convert sentiment score to human-readable label.

        Args:
            sentiment_score: Score from -1 to 1

        Returns:
            Label string
        """
        if sentiment_score >= 0.5:
            return 'Very Positive'
        elif sentiment_score >= 0.2:
            return 'Positive'
        elif sentiment_score >= -0.2:
            return 'Neutral'
        elif sentiment_score >= -0.5:
            return 'Negative'
        else:
            return 'Very Negative'

    def _get_sentiment_recommendation(
        self,
        sentiment: float,
        trend: float,
        volatility: float
    ) -> str:
        """
        Generate trading recommendation based on sentiment metrics.

        Args:
            sentiment: Overall sentiment score
            trend: Sentiment trend
            volatility: Sentiment volatility

        Returns:
            Recommendation string
        """
        # High volatility = caution
        if volatility > 0.3:
            return 'Caution: High sentiment volatility detected'

        # Strong positive sentiment + positive trend
        if sentiment > 0.3 and trend > 0.1:
            return 'Bullish: Strong positive sentiment with improving trend'

        # Strong negative sentiment + negative trend
        if sentiment < -0.3 and trend < -0.1:
            return 'Bearish: Strong negative sentiment with declining trend'

        # Positive sentiment but negative trend
        if sentiment > 0.2 and trend < -0.1:
            return 'Mixed: Positive sentiment but weakening trend'

        # Negative sentiment but positive trend
        if sentiment < -0.2 and trend > 0.1:
            return 'Mixed: Negative sentiment but improving trend'

        # Neutral
        return 'Neutral: No strong directional sentiment'

    def get_latest_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent sentiment data point for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with latest sentiment or None
        """
        try:
            latest = self.db.query(SentimentData).filter(
                SentimentData.symbol == symbol
            ).order_by(SentimentData.timestamp.desc()).first()

            if not latest:
                return None

            return {
                'symbol': symbol,
                'overall_sentiment': float(latest.overall_sentiment) if latest.overall_sentiment else None,
                'confidence': float(latest.confidence) if latest.confidence else None,
                'sentiment_label': self._get_sentiment_label(latest.overall_sentiment or 0),
                'timestamp': latest.timestamp.isoformat(),
                'twitter_sentiment': float(latest.twitter_sentiment) if latest.twitter_sentiment else None,
                'reddit_sentiment': float(latest.reddit_sentiment) if latest.reddit_sentiment else None,
                'news_sentiment': float(latest.news_sentiment) if latest.news_sentiment else None,
            }

        except Exception as e:
            logger.error(f"Error fetching latest sentiment for {symbol}: {e}")
            return None

    def get_sentiment_history(
        self,
        symbol: str,
        days_back: int = 30,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical sentiment data for charting.

        Args:
            symbol: Trading symbol
            days_back: Number of days to look back
            limit: Maximum number of records to return

        Returns:
            List of sentiment data points
        """
        try:
            start_time = datetime.utcnow() - timedelta(days=days_back)

            query = self.db.query(SentimentData).filter(
                and_(
                    SentimentData.symbol == symbol,
                    SentimentData.timestamp >= start_time
                )
            ).order_by(SentimentData.timestamp.desc())

            if limit:
                query = query.limit(limit)

            records = query.all()

            return [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'overall_sentiment': float(r.overall_sentiment) if r.overall_sentiment else None,
                    'confidence': float(r.confidence) if r.confidence else None,
                    'twitter_sentiment': float(r.twitter_sentiment) if r.twitter_sentiment else None,
                    'reddit_sentiment': float(r.reddit_sentiment) if r.reddit_sentiment else None,
                    'news_sentiment': float(r.news_sentiment) if r.news_sentiment else None,
                }
                for r in records
            ]

        except Exception as e:
            logger.error(f"Error fetching sentiment history for {symbol}: {e}")
            return []

    def compare_sentiment_across_symbols(
        self,
        symbols: List[str],
        window: str = '1D'
    ) -> Dict[str, Any]:
        """
        Compare sentiment across multiple symbols.

        Args:
            symbols: List of trading symbols
            window: Time window

        Returns:
            Dictionary with comparison data
        """
        results = {}

        for symbol in symbols:
            sentiment = self.get_sentiment_for_window(symbol, window, include_breakdown=False)
            if sentiment:
                results[symbol] = {
                    'overall_sentiment': sentiment['overall_sentiment'],
                    'sentiment_label': sentiment['sentiment_label'],
                    'sentiment_trend': sentiment['sentiment_trend'],
                    'data_points': sentiment['data_points']
                }

        # Rank by sentiment
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]['overall_sentiment'],
            reverse=True
        )

        return {
            'window': window,
            'symbols_analyzed': len(results),
            'rankings': [
                {
                    'rank': idx + 1,
                    'symbol': symbol,
                    **data
                }
                for idx, (symbol, data) in enumerate(ranked)
            ],
            'most_positive': ranked[0][0] if ranked else None,
            'most_negative': ranked[-1][0] if ranked else None,
        }


# Utility function for quick access
def get_sentiment(db: Session, symbol: str, window: str = '1D') -> Optional[Dict[str, Any]]:
    """
    Quick helper to get sentiment for a symbol.

    Args:
        db: Database session
        symbol: Trading symbol
        window: Time window

    Returns:
        Sentiment data dictionary or None
    """
    aggregator = SentimentAggregator(db)
    return aggregator.get_sentiment_for_window(symbol, window)
