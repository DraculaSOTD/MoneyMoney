# ==================== Sentiment Data Collector Service ====================
"""
Collects and stores sentiment data for ML model training.
Integrates with NewsSentimentAnalyzer and stores in SentimentData table.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import random

from database.models import SessionLocal, SentimentData, TradingProfile

logger = logging.getLogger(__name__)


class SentimentCollector:
    """
    Collects sentiment data from multiple sources and stores in database.

    For models that benefit from sentiment analysis (like Transformer, LSTM),
    this service provides sentiment features that can be incorporated into training.
    """

    # Crypto keywords for sentiment tracking
    CRYPTO_KEYWORDS = {
        'BTCUSDT': ['bitcoin', 'btc', 'satoshi', 'lightning network'],
        'ETHUSDT': ['ethereum', 'eth', 'vitalik', 'smart contracts', 'defi'],
        'BNBUSDT': ['binance', 'bnb', 'binance coin', 'cz'],
        'SOLUSDT': ['solana', 'sol', 'solana nft'],
        'XRPUSDT': ['xrp', 'ripple', 'ripple labs'],
        'ADAUSDT': ['cardano', 'ada', 'charles hoskinson'],
        'DOGEUSDT': ['dogecoin', 'doge', 'elon musk'],
        'DOTUSDT': ['polkadot', 'dot', 'gavin wood'],
        'LINKUSDT': ['chainlink', 'link', 'oracle'],
        'MATICUSDT': ['polygon', 'matic', 'layer 2'],
    }

    # Default keywords for unknown symbols
    DEFAULT_KEYWORDS = ['crypto', 'cryptocurrency', 'blockchain', 'trading']

    def __init__(self):
        self.db = None

    def _get_keywords(self, symbol: str) -> List[str]:
        """Get keywords to track for a symbol."""
        return self.CRYPTO_KEYWORDS.get(symbol, self.DEFAULT_KEYWORDS + [symbol.lower().replace('usdt', '')])

    async def collect_sentiment(
        self,
        symbol: str,
        profile_id: int,
        days_back: int = 30,
        window_size: str = '1D'
    ) -> Dict:
        """
        Collect and store sentiment data for a trading profile.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            profile_id: Profile ID in database
            days_back: Number of days of historical sentiment to collect
            window_size: Aggregation window ('1D', '1W', '1M')

        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"Collecting sentiment for {symbol} (profile_id={profile_id})")

        self.db = SessionLocal()

        try:
            keywords = self._get_keywords(symbol)
            records_created = 0

            # Generate sentiment data for each day
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)

            current_date = start_date
            while current_date <= end_date:
                # Check if we already have data for this date
                existing = self.db.query(SentimentData).filter(
                    SentimentData.symbol == symbol,
                    SentimentData.timestamp >= current_date,
                    SentimentData.timestamp < current_date + timedelta(days=1)
                ).first()

                if not existing:
                    # Generate sentiment data (simulated for now)
                    # In production, this would call actual sentiment APIs
                    sentiment_record = self._generate_sentiment_record(
                        symbol, current_date, keywords, window_size
                    )

                    self.db.add(sentiment_record)
                    records_created += 1

                current_date += timedelta(days=1)

            self.db.commit()

            logger.info(f"Created {records_created} sentiment records for {symbol}")

            return {
                'symbol': symbol,
                'profile_id': profile_id,
                'records_created': records_created,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'keywords_tracked': keywords
            }

        except Exception as e:
            logger.error(f"Error collecting sentiment: {e}")
            self.db.rollback()
            raise
        finally:
            self.db.close()

    def _generate_sentiment_record(
        self,
        symbol: str,
        timestamp: datetime,
        keywords: List[str],
        window_size: str
    ) -> SentimentData:
        """
        Generate a sentiment data record.

        Note: In production, this would integrate with actual sentiment APIs
        like Twitter API, Reddit API, news aggregators, etc.
        For now, generates simulated but realistic sentiment data.
        """
        # Base sentiment with market-correlated randomness
        # Add some temporal patterns (weekends typically have different sentiment)
        day_of_week = timestamp.weekday()
        base_sentiment = 0.05 if day_of_week < 5 else -0.02  # Slightly bullish on weekdays

        # Add random variation
        overall_sentiment = np.clip(
            base_sentiment + np.random.normal(0, 0.3),
            -1.0, 1.0
        )

        # Confidence based on volume
        confidence = np.clip(np.random.beta(5, 2), 0.5, 1.0)

        # Source-specific sentiment (correlated but with variation)
        twitter_sentiment = np.clip(
            overall_sentiment + np.random.normal(0, 0.15),
            -1.0, 1.0
        )
        reddit_sentiment = np.clip(
            overall_sentiment + np.random.normal(0, 0.2),
            -1.0, 1.0
        )
        news_sentiment = np.clip(
            overall_sentiment + np.random.normal(0, 0.1),
            -1.0, 1.0
        )

        # Volume (higher on weekdays)
        base_volume = 1000 if day_of_week < 5 else 500
        twitter_volume = int(base_volume * np.random.uniform(0.8, 1.5))
        reddit_volume = int(base_volume * 0.3 * np.random.uniform(0.8, 1.5))
        news_volume = int(base_volume * 0.05 * np.random.uniform(0.8, 1.5))

        # Trend and volatility
        sentiment_trend = np.random.normal(0, 0.1)
        volatility = np.abs(np.random.normal(0.2, 0.1))

        # Window times
        if window_size == '1D':
            window_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            window_end = window_start + timedelta(days=1)
        elif window_size == '1W':
            window_start = timestamp - timedelta(days=timestamp.weekday())
            window_start = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
            window_end = window_start + timedelta(weeks=1)
        else:  # 1M
            window_start = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = window_start.replace(day=28) + timedelta(days=4)
            window_end = next_month.replace(day=1)

        return SentimentData(
            symbol=symbol,
            timestamp=timestamp,
            overall_sentiment=float(overall_sentiment),
            confidence=float(confidence),
            twitter_sentiment=float(twitter_sentiment),
            twitter_volume=twitter_volume,
            reddit_sentiment=float(reddit_sentiment),
            reddit_volume=reddit_volume,
            news_sentiment=float(news_sentiment),
            news_volume=news_volume,
            sentiment_trend=float(sentiment_trend),
            volatility=float(volatility),
            window_size=window_size,
            window_start=window_start,
            window_end=window_end,
            data_sources=['twitter_simulated', 'reddit_simulated', 'news_simulated'],
            keywords_tracked=keywords
        )

    def get_sentiment_features(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve sentiment features for model training.

        Args:
            symbol: Trading symbol
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DataFrame with sentiment features indexed by timestamp
        """
        self.db = SessionLocal()

        try:
            records = self.db.query(SentimentData).filter(
                SentimentData.symbol == symbol,
                SentimentData.timestamp >= start_time,
                SentimentData.timestamp <= end_time
            ).order_by(SentimentData.timestamp).all()

            if not records:
                logger.warning(f"No sentiment data found for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for r in records:
                data.append({
                    'timestamp': r.timestamp,
                    'sentiment_overall': r.overall_sentiment,
                    'sentiment_confidence': r.confidence,
                    'sentiment_twitter': r.twitter_sentiment,
                    'sentiment_reddit': r.reddit_sentiment,
                    'sentiment_news': r.news_sentiment,
                    'sentiment_volume': (r.twitter_volume or 0) + (r.reddit_volume or 0) + (r.news_volume or 0),
                    'sentiment_trend': r.sentiment_trend,
                    'sentiment_volatility': r.volatility
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)

            return df

        finally:
            self.db.close()

    async def get_latest_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get the most recent sentiment data for a symbol."""
        self.db = SessionLocal()

        try:
            record = self.db.query(SentimentData).filter(
                SentimentData.symbol == symbol
            ).order_by(SentimentData.timestamp.desc()).first()

            if not record:
                return None

            return {
                'symbol': record.symbol,
                'timestamp': record.timestamp.isoformat(),
                'overall_sentiment': record.overall_sentiment,
                'confidence': record.confidence,
                'twitter_sentiment': record.twitter_sentiment,
                'reddit_sentiment': record.reddit_sentiment,
                'news_sentiment': record.news_sentiment,
                'trend': record.sentiment_trend
            }

        finally:
            self.db.close()


# Singleton instance
sentiment_collector = SentimentCollector()
