"""
Multi-Source Sentiment Data Aggregator.

Collects and aggregates sentiment data from multiple sources including
social media, news, and messaging platforms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import queue
import time
import json
import re
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SentimentDataPoint:
    """Single sentiment data point from any source."""
    source: str
    platform: str
    text: str
    author: str
    timestamp: datetime
    engagement: Dict[str, int]  # likes, retweets, comments, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'platform': self.platform,
            'text': self.text,
            'author': self.author,
            'timestamp': self.timestamp.isoformat(),
            'engagement': self.engagement,
            'metadata': self.metadata
        }


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def fetch_data(self, query: str, limit: int = 100) -> List[SentimentDataPoint]:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def stream_data(self, query: str, callback: Callable) -> None:
        """Stream real-time data from the source."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this data source."""
        pass


class MockTwitterSource(DataSource):
    """Mock Twitter/X data source for demonstration."""
    
    def __init__(self):
        self.source_name = "twitter"
        self.mock_users = [
            "crypto_whale", "bitcoin_maxi", "eth_trader", "defi_degen",
            "nft_collector", "blockchain_dev", "crypto_analyst", "moon_boy"
        ]
        self.mock_templates = [
            "$COIN to the moon! 🚀🚀🚀 #crypto #bullish",
            "Just bought more $COIN on this dip 💎🙌 #HODL",
            "$COIN looking bearish, might see $PRICE soon 📉 #crypto",
            "Breaking: $COIN partnership announced! This is huge! 🔥",
            "Technical analysis: $COIN forming bullish pattern 📈",
            "$COIN is a scam, stay away! 🚫 #cryptoscam",
            "Accumulating $COIN here, great entry point 💰",
            "Sold all my $COIN, taking profits 💸 #trading"
        ]
    
    def fetch_data(self, query: str, limit: int = 100) -> List[SentimentDataPoint]:
        """Fetch mock Twitter data."""
        data_points = []
        
        for i in range(min(limit, 50)):
            # Generate mock tweet
            template = np.random.choice(self.mock_templates)
            coin = query.upper() if query else "BTC"
            text = template.replace("$COIN", coin).replace("$PRICE", str(np.random.randint(10000, 50000)))
            
            author = np.random.choice(self.mock_users)
            
            data_point = SentimentDataPoint(
                source=self.source_name,
                platform="twitter",
                text=text,
                author=author,
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 60)),
                engagement={
                    'likes': np.random.randint(0, 1000),
                    'retweets': np.random.randint(0, 500),
                    'replies': np.random.randint(0, 100)
                },
                metadata={
                    'verified': np.random.choice([True, False]),
                    'followers': np.random.randint(100, 100000)
                }
            )
            
            data_points.append(data_point)
        
        return data_points
    
    def stream_data(self, query: str, callback: Callable) -> None:
        """Stream mock Twitter data."""
        while True:
            # Generate a new tweet every 5-15 seconds
            time.sleep(np.random.randint(5, 15))
            
            data_points = self.fetch_data(query, limit=1)
            if data_points:
                callback(data_points[0])
    
    def get_source_name(self) -> str:
        return self.source_name


class MockRedditSource(DataSource):
    """Mock Reddit data source for demonstration."""
    
    def __init__(self):
        self.source_name = "reddit"
        self.subreddits = ["r/cryptocurrency", "r/bitcoin", "r/ethtrader", "r/cryptomoonshots"]
        self.mock_posts = [
            "DD: Why $COIN is undervalued at current prices",
            "$COIN technical analysis - Cup and handle forming",
            "Lost everything on $COIN, learn from my mistakes",
            "$COIN is the future of finance, here's why",
            "Whale alert: Large $COIN transfer detected",
            "My $COIN price prediction for next month",
            "$COIN vs competitors - detailed comparison"
        ]
    
    def fetch_data(self, query: str, limit: int = 100) -> List[SentimentDataPoint]:
        """Fetch mock Reddit data."""
        data_points = []
        
        for i in range(min(limit, 30)):
            template = np.random.choice(self.mock_posts)
            coin = query.upper() if query else "BTC"
            text = template.replace("$COIN", coin)
            
            subreddit = np.random.choice(self.subreddits)
            
            data_point = SentimentDataPoint(
                source=self.source_name,
                platform="reddit",
                text=text,
                author=f"user_{np.random.randint(1000, 9999)}",
                timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 24)),
                engagement={
                    'upvotes': np.random.randint(0, 5000),
                    'downvotes': np.random.randint(0, 500),
                    'comments': np.random.randint(0, 200)
                },
                metadata={
                    'subreddit': subreddit,
                    'post_type': np.random.choice(['discussion', 'analysis', 'news'])
                }
            )
            
            data_points.append(data_point)
        
        return data_points
    
    def stream_data(self, query: str, callback: Callable) -> None:
        """Stream mock Reddit data."""
        while True:
            time.sleep(np.random.randint(30, 90))
            
            data_points = self.fetch_data(query, limit=1)
            if data_points:
                callback(data_points[0])
    
    def get_source_name(self) -> str:
        return self.source_name


class MockNewsSource(DataSource):
    """Mock news source for demonstration."""
    
    def __init__(self):
        self.source_name = "news"
        self.news_outlets = ["CoinDesk", "CoinTelegraph", "CryptoNews", "Bloomberg Crypto"]
        self.headlines = [
            "$COIN Surges X% Following Major Partnership Announcement",
            "Breaking: $COIN Network Experiences Significant Outage",
            "$COIN Adoption Grows as Major Retailer Accepts Payments",
            "Regulatory Concerns Weigh on $COIN Price",
            "$COIN Technical Analysis: Key Support Levels to Watch",
            "Institutional Investment in $COIN Reaches New High",
            "$COIN Community Votes on Major Protocol Upgrade"
        ]
    
    def fetch_data(self, query: str, limit: int = 100) -> List[SentimentDataPoint]:
        """Fetch mock news data."""
        data_points = []
        
        for i in range(min(limit, 20)):
            template = np.random.choice(self.headlines)
            coin = query.upper() if query else "BTC"
            text = template.replace("$COIN", coin).replace("X", str(np.random.randint(5, 30)))
            
            outlet = np.random.choice(self.news_outlets)
            
            data_point = SentimentDataPoint(
                source=self.source_name,
                platform="news",
                text=text,
                author=outlet,
                timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 48)),
                engagement={
                    'views': np.random.randint(1000, 50000),
                    'shares': np.random.randint(10, 1000)
                },
                metadata={
                    'outlet': outlet,
                    'category': np.random.choice(['breaking', 'analysis', 'opinion'])
                }
            )
            
            data_points.append(data_point)
        
        return data_points
    
    def stream_data(self, query: str, callback: Callable) -> None:
        """Stream mock news data."""
        while True:
            time.sleep(np.random.randint(300, 600))  # News less frequent
            
            data_points = self.fetch_data(query, limit=1)
            if data_points:
                callback(data_points[0])
    
    def get_source_name(self) -> str:
        return self.source_name


class MultiSourceAggregator:
    """
    Aggregates sentiment data from multiple sources.
    
    Features:
    - Multi-source data collection
    - Real-time streaming support
    - Source quality scoring
    - Deduplication
    - Rate limiting
    - Data normalization
    """
    
    def __init__(self, 
                 sources: Optional[List[DataSource]] = None,
                 enable_streaming: bool = True,
                 cache_size: int = 10000):
        """
        Initialize multi-source aggregator.
        
        Args:
            sources: List of data sources
            enable_streaming: Enable real-time streaming
            cache_size: Size of data cache
        """
        self.sources = sources or self._get_default_sources()
        self.enable_streaming = enable_streaming
        
        # Data storage
        self.data_cache = deque(maxlen=cache_size)
        self.source_queues = {source.get_source_name(): queue.Queue() for source in self.sources}
        
        # Deduplication
        self.seen_hashes = deque(maxlen=cache_size * 2)
        
        # Source quality tracking
        self.source_quality = defaultdict(lambda: {
            'total_items': 0,
            'quality_score': 1.0,
            'error_count': 0,
            'last_fetch': None
        })
        
        # Streaming
        self.streaming_active = False
        self.stream_threads = []
        
        # Rate limiting
        self.rate_limits = {
            'twitter': {'requests_per_minute': 15, 'last_request': None},
            'reddit': {'requests_per_minute': 10, 'last_request': None},
            'news': {'requests_per_minute': 5, 'last_request': None}
        }
        
        logger.info(f"Multi-source aggregator initialized with {len(self.sources)} sources")
    
    def _get_default_sources(self) -> List[DataSource]:
        """Get default data sources."""
        return [
            MockTwitterSource(),
            MockRedditSource(),
            MockNewsSource()
        ]
    
    def fetch_all_sources(self, query: str, limit_per_source: int = 50) -> List[SentimentDataPoint]:
        """
        Fetch data from all sources.
        
        Args:
            query: Search query (e.g., coin symbol)
            limit_per_source: Maximum items per source
            
        Returns:
            Aggregated list of sentiment data points
        """
        all_data = []
        
        for source in self.sources:
            source_name = source.get_source_name()
            
            # Check rate limit
            if not self._check_rate_limit(source_name):
                logger.warning(f"Rate limit exceeded for {source_name}")
                continue
            
            try:
                # Fetch data
                data_points = source.fetch_data(query, limit_per_source)
                
                # Filter duplicates
                filtered_points = []
                for point in data_points:
                    if not self._is_duplicate(point):
                        filtered_points.append(point)
                        self._mark_as_seen(point)
                
                all_data.extend(filtered_points)
                
                # Update source quality
                self.source_quality[source_name]['total_items'] += len(filtered_points)
                self.source_quality[source_name]['last_fetch'] = datetime.now()
                
                # Update rate limit
                self.rate_limits[source_name]['last_request'] = datetime.now()
                
                logger.info(f"Fetched {len(filtered_points)} items from {source_name}")
                
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
                self.source_quality[source_name]['error_count'] += 1
                self._update_source_quality_score(source_name)
        
        # Add to cache
        self.data_cache.extend(all_data)
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x.timestamp, reverse=True)
        
        return all_data
    
    def _check_rate_limit(self, source_name: str) -> bool:
        """Check if source is within rate limit."""
        if source_name not in self.rate_limits:
            return True
        
        limit_info = self.rate_limits[source_name]
        if limit_info['last_request'] is None:
            return True
        
        time_since_last = (datetime.now() - limit_info['last_request']).total_seconds()
        min_interval = 60 / limit_info['requests_per_minute']
        
        return time_since_last >= min_interval
    
    def _is_duplicate(self, data_point: SentimentDataPoint) -> bool:
        """Check if data point is a duplicate."""
        # Create hash of key fields
        hash_str = f"{data_point.source}:{data_point.author}:{data_point.text[:50]}"
        data_hash = hash(hash_str)
        
        return data_hash in self.seen_hashes
    
    def _mark_as_seen(self, data_point: SentimentDataPoint) -> None:
        """Mark data point as seen for deduplication."""
        hash_str = f"{data_point.source}:{data_point.author}:{data_point.text[:50]}"
        data_hash = hash(hash_str)
        self.seen_hashes.append(data_hash)
    
    def _update_source_quality_score(self, source_name: str) -> None:
        """Update quality score for a source."""
        quality_info = self.source_quality[source_name]
        
        # Calculate quality score based on error rate
        error_rate = quality_info['error_count'] / max(quality_info['total_items'], 1)
        quality_info['quality_score'] = max(0.1, 1.0 - error_rate)
    
    def start_streaming(self, queries: List[str]) -> None:
        """
        Start streaming data from all sources.
        
        Args:
            queries: List of queries to stream
        """
        if self.streaming_active:
            logger.warning("Streaming already active")
            return
        
        self.streaming_active = True
        
        for source in self.sources:
            for query in queries:
                thread = threading.Thread(
                    target=self._stream_worker,
                    args=(source, query),
                    daemon=True
                )
                thread.start()
                self.stream_threads.append(thread)
        
        logger.info(f"Started streaming with {len(self.stream_threads)} threads")
    
    def _stream_worker(self, source: DataSource, query: str) -> None:
        """Worker thread for streaming data."""
        source_name = source.get_source_name()
        
        def callback(data_point: SentimentDataPoint):
            if not self._is_duplicate(data_point):
                self.source_queues[source_name].put(data_point)
                self._mark_as_seen(data_point)
                self.data_cache.append(data_point)
        
        try:
            source.stream_data(query, callback)
        except Exception as e:
            logger.error(f"Streaming error for {source_name}: {e}")
            self.source_quality[source_name]['error_count'] += 1
    
    def stop_streaming(self) -> None:
        """Stop streaming from all sources."""
        self.streaming_active = False
        logger.info("Streaming stopped")
    
    def get_stream_data(self, timeout: float = 0.1) -> List[SentimentDataPoint]:
        """
        Get data from streaming queues.
        
        Args:
            timeout: Queue timeout in seconds
            
        Returns:
            List of new data points
        """
        new_data = []
        
        for source_name, source_queue in self.source_queues.items():
            while True:
                try:
                    data_point = source_queue.get(timeout=timeout)
                    new_data.append(data_point)
                except queue.Empty:
                    break
        
        return new_data
    
    def get_source_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all sources."""
        stats = {}
        
        for source_name, quality_info in self.source_quality.items():
            # Count items by source in cache
            source_items = [d for d in self.data_cache if d.source == source_name]
            
            stats[source_name] = {
                'total_items': quality_info['total_items'],
                'items_in_cache': len(source_items),
                'quality_score': quality_info['quality_score'],
                'error_count': quality_info['error_count'],
                'last_fetch': quality_info['last_fetch'],
                'queue_size': self.source_queues[source_name].qsize() if source_name in self.source_queues else 0
            }
            
            # Calculate engagement statistics
            if source_items:
                engagement_totals = defaultdict(int)
                for item in source_items:
                    for metric, value in item.engagement.items():
                        engagement_totals[metric] += value
                
                stats[source_name]['avg_engagement'] = {
                    metric: total / len(source_items)
                    for metric, total in engagement_totals.items()
                }
        
        return stats
    
    def get_aggregated_data(self, 
                          hours: int = 1,
                          min_quality_score: float = 0.5) -> pd.DataFrame:
        """
        Get aggregated data from all sources.
        
        Args:
            hours: Hours of data to retrieve
            min_quality_score: Minimum source quality score
            
        Returns:
            DataFrame with aggregated data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter data
        filtered_data = []
        for data_point in self.data_cache:
            if data_point.timestamp > cutoff_time:
                source_quality = self.source_quality[data_point.source]['quality_score']
                if source_quality >= min_quality_score:
                    filtered_data.append(data_point)
        
        if not filtered_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = []
        for point in filtered_data:
            row = {
                'timestamp': point.timestamp,
                'source': point.source,
                'platform': point.platform,
                'text': point.text,
                'author': point.author,
                **{f'engagement_{k}': v for k, v in point.engagement.items()},
                **{f'meta_{k}': v for k, v in point.metadata.items()}
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def calculate_source_weights(self) -> Dict[str, float]:
        """
        Calculate optimal weights for each source based on quality and volume.
        
        Returns:
            Dictionary of source weights
        """
        weights = {}
        total_score = 0
        
        for source_name, quality_info in self.source_quality.items():
            # Weight based on quality score and data volume
            volume_factor = min(1.0, quality_info['total_items'] / 1000)
            combined_score = quality_info['quality_score'] * (0.7 + 0.3 * volume_factor)
            
            weights[source_name] = combined_score
            total_score += combined_score
        
        # Normalize weights
        if total_score > 0:
            for source_name in weights:
                weights[source_name] /= total_score
        
        return weights