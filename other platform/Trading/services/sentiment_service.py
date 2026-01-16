"""
Sentiment Analysis Service
==========================

Provides sentiment analysis for cryptocurrency trading using:
1. Fear & Greed Index from Alternative.me API
2. Reddit sentiment analysis using FinBERT
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed. FinBERT sentiment will be unavailable.")


class SentimentService:
    """
    Sentiment analysis service combining Fear & Greed Index and Reddit sentiment.
    """

    FEAR_GREED_API = "https://api.alternative.me/fng/"

    # Crypto-related subreddits
    CRYPTO_SUBREDDITS = ['cryptocurrency', 'bitcoin', 'CryptoMarkets', 'ethtrader']

    # Symbol to search term mapping
    SYMBOL_KEYWORDS = {
        'BTCUSDT': ['bitcoin', 'btc', 'BTC'],
        'ETHUSDT': ['ethereum', 'eth', 'ETH'],
        'BNBUSDT': ['binance', 'bnb', 'BNB'],
        'SOLUSDT': ['solana', 'sol', 'SOL'],
        'XRPUSDT': ['ripple', 'xrp', 'XRP'],
        'ADAUSDT': ['cardano', 'ada', 'ADA'],
        'DOGEUSDT': ['dogecoin', 'doge', 'DOGE'],
    }

    def __init__(self):
        """Initialize sentiment service."""
        self.finbert_pipeline = None
        self._init_finbert()
        logger.info("Sentiment service initialized (using public Reddit API - no credentials needed)")

    def _init_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("FinBERT unavailable - transformers not installed")
            return

        try:
            # Use FinBERT - pre-trained on financial text
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1  # GPU if available
            )
            logger.info(f"FinBERT initialized on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT: {e}")
            self.finbert_pipeline = None

    async def get_fear_greed_index(self, limit: int = 30) -> Dict[str, Any]:
        """
        Fetch Fear & Greed Index data from Alternative.me API.

        Args:
            limit: Number of days of historical data

        Returns:
            Dict with current value, historical data, and classification
        """
        try:
            url = f"{self.FEAR_GREED_API}?limit={limit}&format=json"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"Fear & Greed API returned {response.status}")
                        return self._default_fear_greed()

                    data = await response.json()

            if 'data' not in data:
                return self._default_fear_greed()

            # Parse response
            fng_data = data['data']
            current = fng_data[0] if fng_data else {}

            # Build historical series
            historical = []
            for item in fng_data:
                historical.append({
                    'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                    'value': int(item['value']),
                    'classification': item['value_classification']
                })

            return {
                'current_value': int(current.get('value', 50)),
                'classification': current.get('value_classification', 'Neutral'),
                'historical': historical,
                'source': 'alternative.me',
                'fetched_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return self._default_fear_greed()

    def _default_fear_greed(self) -> Dict[str, Any]:
        """Return default Fear & Greed data when API fails."""
        return {
            'current_value': 50,
            'classification': 'Neutral',
            'historical': [],
            'source': 'default',
            'fetched_at': datetime.utcnow().isoformat()
        }

    async def get_reddit_sentiment(self, symbol: str = 'BTCUSDT',
                                   limit_per_sub: int = 25) -> Dict[str, Any]:
        """
        Fetch and analyze Reddit posts for sentiment using public JSON API.
        No API key required - uses public Reddit endpoints.

        Args:
            symbol: Trading symbol to search for
            limit_per_sub: Number of posts to fetch per subreddit

        Returns:
            Dict with sentiment scores and post analysis
        """
        try:
            # Get search keywords for symbol
            keywords = self.SYMBOL_KEYWORDS.get(symbol, [symbol.replace('USDT', '').lower()])

            posts = []
            headers = {
                'User-Agent': 'TradingDashboard/1.0 (Sentiment Analysis Bot)'
            }

            async with aiohttp.ClientSession() as session:
                # Fetch hot posts from each subreddit
                for subreddit_name in self.CRYPTO_SUBREDDITS[:3]:  # Limit to 3 subreddits
                    try:
                        url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit={limit_per_sub}"

                        async with session.get(url, headers=headers, timeout=10) as response:
                            if response.status != 200:
                                logger.warning(f"Reddit API returned {response.status} for r/{subreddit_name}")
                                continue

                            data = await response.json()

                            # Parse posts
                            if 'data' in data and 'children' in data['data']:
                                for child in data['data']['children']:
                                    post_data = child.get('data', {})
                                    title = post_data.get('title', '')

                                    # Filter by keywords if symbol-specific
                                    title_lower = title.lower()
                                    if any(kw.lower() in title_lower for kw in keywords):
                                        posts.append({
                                            'title': title,
                                            'score': post_data.get('score', 0),
                                            'num_comments': post_data.get('num_comments', 0),
                                            'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                                            'subreddit': subreddit_name
                                        })

                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.5)

                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout fetching from r/{subreddit_name}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                        continue

            if not posts:
                logger.info(f"No Reddit posts found for {symbol}, returning neutral sentiment")
                return self._default_reddit_sentiment()

            logger.info(f"Found {len(posts)} Reddit posts for {symbol}")

            # Analyze sentiment of post titles
            sentiment_results = await self._analyze_texts([p['title'] for p in posts])

            # Combine results
            for i, post in enumerate(posts):
                if i < len(sentiment_results):
                    post['sentiment'] = sentiment_results[i]

            # Calculate aggregate scores
            positive_count = sum(1 for p in posts if p.get('sentiment', {}).get('label') == 'positive')
            negative_count = sum(1 for p in posts if p.get('sentiment', {}).get('label') == 'negative')
            neutral_count = len(posts) - positive_count - negative_count

            total = len(posts)

            return {
                'total_posts': total,
                'positive_ratio': positive_count / total if total > 0 else 0.33,
                'negative_ratio': negative_count / total if total > 0 else 0.33,
                'neutral_ratio': neutral_count / total if total > 0 else 0.34,
                'sentiment_score': (positive_count - negative_count) / total if total > 0 else 0,
                'posts_analyzed': posts[:10],  # Return top 10 for reference
                'source': 'reddit_public_api',
                'fetched_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return self._default_reddit_sentiment()

    def _default_reddit_sentiment(self) -> Dict[str, Any]:
        """Return default Reddit sentiment when unavailable."""
        return {
            'total_posts': 0,
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34,
            'sentiment_score': 0,
            'posts_analyzed': [],
            'source': 'default',
            'fetched_at': datetime.utcnow().isoformat()
        }

    async def _analyze_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze texts using FinBERT.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment results
        """
        if not self.finbert_pipeline or not texts:
            return [{'label': 'neutral', 'score': 0.5}] * len(texts)

        try:
            # Process in batches to avoid memory issues
            batch_size = 16
            results = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Truncate texts to max 512 tokens
                batch = [t[:512] for t in batch]

                batch_results = await asyncio.to_thread(
                    self.finbert_pipeline, batch
                )
                results.extend(batch_results)

            return results

        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return [{'label': 'neutral', 'score': 0.5}] * len(texts)

    async def get_combined_sentiment(self, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """
        Get combined sentiment from all sources.

        Args:
            symbol: Trading symbol

        Returns:
            Combined sentiment analysis with all features
        """
        # Fetch both data sources in parallel
        fear_greed_task = self.get_fear_greed_index()
        reddit_task = self.get_reddit_sentiment(symbol)

        fear_greed, reddit = await asyncio.gather(fear_greed_task, reddit_task)

        # Calculate combined sentiment score (-1 to 1)
        # Fear & Greed: 0-100, convert to -1 to 1
        fng_normalized = (fear_greed['current_value'] - 50) / 50

        # Reddit: already -1 to 1
        reddit_score = reddit['sentiment_score']

        # Weighted combination (Fear & Greed is more reliable)
        combined_score = 0.6 * fng_normalized + 0.4 * reddit_score

        # Determine signal
        if combined_score > 0.3:
            signal = 'bullish'
            signal_strength = min(combined_score, 1.0)
        elif combined_score < -0.3:
            signal = 'bearish'
            signal_strength = min(abs(combined_score), 1.0)
        else:
            signal = 'neutral'
            signal_strength = 0.5

        return {
            'symbol': symbol,
            'combined_score': combined_score,
            'signal': signal,
            'signal_strength': signal_strength,
            'fear_greed': fear_greed,
            'reddit': reddit,
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_sentiment_features(self, sentiment_data: Dict) -> np.ndarray:
        """
        Extract numerical features from sentiment data for model training.

        Args:
            sentiment_data: Combined sentiment data from get_combined_sentiment()

        Returns:
            Numpy array of features
        """
        features = [
            sentiment_data['combined_score'],
            sentiment_data['fear_greed']['current_value'] / 100,  # Normalize to 0-1
            sentiment_data['reddit']['positive_ratio'],
            sentiment_data['reddit']['negative_ratio'],
            sentiment_data['reddit']['sentiment_score'],
            1 if sentiment_data['signal'] == 'bullish' else (-1 if sentiment_data['signal'] == 'bearish' else 0),
            sentiment_data['signal_strength']
        ]

        return np.array(features, dtype=np.float32)


# Singleton instance
_sentiment_service = None

def get_sentiment_service() -> SentimentService:
    """Get or create sentiment service singleton."""
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
    return _sentiment_service
