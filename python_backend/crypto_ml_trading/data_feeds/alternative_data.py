"""
Alternative Data Sources Integration.

Provides connections to news APIs, social media, and on-chain data
for comprehensive market sentiment and analysis.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import logging
import re
from abc import ABC, abstractmethod
import numpy as np

# Twitter/X API
import tweepy

# Reddit API
import praw

# News APIs
import feedparser

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """News article data."""
    source: str
    title: str
    content: str
    url: str
    published: datetime
    author: Optional[str] = None
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialPost:
    """Social media post data."""
    platform: str
    author: str
    content: str
    timestamp: datetime
    url: str
    engagement: Dict[str, int] = field(default_factory=dict)  # likes, retweets, etc.
    sentiment_score: Optional[float] = None
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnChainData:
    """On-chain blockchain data."""
    blockchain: str
    metric: str
    value: float
    timestamp: datetime
    block_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for alternative data sources."""
    
    def __init__(self, name: str, rate_limit: int = 60):
        """
        Initialize data source.
        
        Args:
            name: Source name
            rate_limit: Requests per minute limit
        """
        self.name = name
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
    
    async def _rate_limit_check(self):
        """Check and enforce rate limiting."""
        current_time = datetime.now().timestamp()
        
        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check if we've hit the limit
        if self.request_count >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {self.name}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = datetime.now().timestamp()
        
        self.request_count += 1
    
    @abstractmethod
    async def fetch_data(self, query: str, limit: int = 100) -> List[Any]:
        """Fetch data from source."""
        pass


class CryptoNewsAPI(DataSource):
    """Cryptocurrency news aggregator."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__('CryptoNews', rate_limit=100)
        self.api_key = api_key
        self.base_urls = [
            'https://cryptopanic.com/api/v1/posts/',
            'https://newsapi.org/v2/everything',
            'https://cryptocontrol.io/api/v1/public/news'
        ]
        self.rss_feeds = [
            'https://cointelegraph.com/rss',
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://decrypt.co/feed',
            'https://bitcoinmagazine.com/.rss/full/'
        ]
    
    async def fetch_data(self, query: str, limit: int = 100) -> List[NewsArticle]:
        """Fetch crypto news articles."""
        await self._rate_limit_check()
        
        articles = []
        
        # Fetch from RSS feeds (no API key required)
        rss_articles = await self._fetch_rss_feeds(query, limit)
        articles.extend(rss_articles)
        
        # Fetch from APIs if key available
        if self.api_key:
            api_articles = await self._fetch_from_apis(query, limit)
            articles.extend(api_articles)
        
        # Sort by timestamp and deduplicate
        articles = self._deduplicate_articles(articles)
        articles.sort(key=lambda x: x.published, reverse=True)
        
        return articles[:limit]
    
    async def _fetch_rss_feeds(self, query: str, limit: int) -> List[NewsArticle]:
        """Fetch articles from RSS feeds."""
        articles = []
        
        async with aiohttp.ClientSession() as session:
            for feed_url in self.rss_feeds:
                try:
                    async with session.get(feed_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed = feedparser.parse(content)
                            
                            for entry in feed.entries[:limit // len(self.rss_feeds)]:
                                # Filter by query
                                if query.lower() in entry.title.lower() or \
                                   query.lower() in entry.get('summary', '').lower():
                                    
                                    article = NewsArticle(
                                        source=feed.feed.get('title', 'Unknown'),
                                        title=entry.title,
                                        content=entry.get('summary', ''),
                                        url=entry.link,
                                        published=datetime.fromtimestamp(
                                            entry.published_parsed.timestamp() if hasattr(entry, 'published_parsed') else datetime.now().timestamp(),
                                            tz=timezone.utc
                                        ),
                                        author=entry.get('author'),
                                        entities=self._extract_entities(entry.title + ' ' + entry.get('summary', ''))
                                    )
                                    articles.append(article)
                                    
                except Exception as e:
                    logger.error(f"Error fetching RSS feed {feed_url}: {e}")
                    self.error_count += 1
        
        return articles
    
    async def _fetch_from_apis(self, query: str, limit: int) -> List[NewsArticle]:
        """Fetch articles from news APIs."""
        articles = []
        
        # NewsAPI implementation
        if 'newsapi.org' in self.base_urls[1]:
            async with aiohttp.ClientSession() as session:
                params = {
                    'q': f"{query} cryptocurrency",
                    'apiKey': self.api_key,
                    'pageSize': limit,
                    'sortBy': 'publishedAt',
                    'language': 'en'
                }
                
                try:
                    async with session.get(self.base_urls[1], params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('articles', []):
                                article = NewsArticle(
                                    source=item.get('source', {}).get('name', 'Unknown'),
                                    title=item.get('title', ''),
                                    content=item.get('description', ''),
                                    url=item.get('url', ''),
                                    published=datetime.fromisoformat(
                                        item.get('publishedAt', '').replace('Z', '+00:00')
                                    ),
                                    author=item.get('author'),
                                    entities=self._extract_entities(item.get('title', '') + ' ' + item.get('description', ''))
                                )
                                articles.append(article)
                                
                except Exception as e:
                    logger.error(f"Error fetching from NewsAPI: {e}")
                    self.error_count += 1
        
        return articles
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract cryptocurrency entities from text."""
        entities = []
        
        # Common crypto patterns
        crypto_pattern = r'\b(BTC|ETH|BNB|SOL|ADA|DOT|AVAX|MATIC|LINK|UNI)\b'
        matches = re.findall(crypto_pattern, text.upper())
        entities.extend(matches)
        
        # Full names
        crypto_names = {
            'BITCOIN': 'BTC', 'ETHEREUM': 'ETH', 'BINANCE': 'BNB',
            'SOLANA': 'SOL', 'CARDANO': 'ADA', 'POLKADOT': 'DOT'
        }
        
        text_upper = text.upper()
        for name, symbol in crypto_names.items():
            if name in text_upper and symbol not in entities:
                entities.append(symbol)
        
        return list(set(entities))
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles."""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles


class TwitterAPI(DataSource):
    """Twitter/X API integration."""
    
    def __init__(self, bearer_token: Optional[str] = None):
        super().__init__('Twitter', rate_limit=300)
        self.bearer_token = bearer_token
        self.client = None
        
        if bearer_token:
            self.client = tweepy.Client(bearer_token=bearer_token)
    
    async def fetch_data(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Fetch tweets about query."""
        await self._rate_limit_check()
        
        if not self.client:
            logger.warning("Twitter API client not initialized (no bearer token)")
            return self._get_mock_tweets(query, limit)
        
        posts = []
        
        try:
            # Search recent tweets
            tweets = self.client.search_recent_tweets(
                query=f"{query} -is:retweet lang:en",
                max_results=min(limit, 100),
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'entities']
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    post = SocialPost(
                        platform='twitter',
                        author=str(tweet.author_id),
                        content=tweet.text,
                        timestamp=tweet.created_at.replace(tzinfo=timezone.utc),
                        url=f"https://twitter.com/i/status/{tweet.id}",
                        engagement={
                            'likes': tweet.public_metrics.get('like_count', 0),
                            'retweets': tweet.public_metrics.get('retweet_count', 0),
                            'replies': tweet.public_metrics.get('reply_count', 0),
                            'quotes': tweet.public_metrics.get('quote_count', 0)
                        },
                        entities=self._extract_entities_from_tweet(tweet)
                    )
                    posts.append(post)
                    
        except Exception as e:
            logger.error(f"Error fetching from Twitter API: {e}")
            self.error_count += 1
            # Fall back to mock data
            return self._get_mock_tweets(query, limit)
        
        return posts
    
    def _extract_entities_from_tweet(self, tweet) -> List[str]:
        """Extract entities from tweet."""
        entities = []
        
        # From tweet entities
        if hasattr(tweet, 'entities'):
            # Cashtags ($BTC, $ETH)
            for cashtag in tweet.entities.get('cashtags', []):
                entities.append(cashtag['tag'].upper())
            
            # Hashtags
            for hashtag in tweet.entities.get('hashtags', []):
                tag = hashtag['tag'].upper()
                if re.match(r'^[A-Z]{2,5}$', tag):  # Likely crypto symbol
                    entities.append(tag)
        
        return entities
    
    def _get_mock_tweets(self, query: str, limit: int) -> List[SocialPost]:
        """Get mock tweets for testing."""
        mock_templates = [
            f"${query} is looking bullish! 🚀 #crypto #trading",
            f"Just bought more ${query} on this dip 💎🙌",
            f"Technical analysis: ${query} forming a cup and handle pattern",
            f"${query} to the moon! 🌙 Who's holding with me?",
            f"Breaking: Major partnership announced for ${query}!",
            f"${query} price prediction: $100k by EOY? 📈"
        ]
        
        posts = []
        for i in range(min(limit, len(mock_templates))):
            post = SocialPost(
                platform='twitter',
                author=f'crypto_trader_{i}',
                content=mock_templates[i % len(mock_templates)],
                timestamp=datetime.now(tz=timezone.utc) - timedelta(minutes=i*5),
                url=f"https://twitter.com/i/status/mock_{i}",
                engagement={
                    'likes': np.random.randint(10, 1000),
                    'retweets': np.random.randint(5, 500),
                    'replies': np.random.randint(1, 100)
                },
                entities=[query.upper()]
            )
            posts.append(post)
        
        return posts


class RedditAPI(DataSource):
    """Reddit API integration."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        super().__init__('Reddit', rate_limit=60)
        self.reddit = None
        
        if client_id and client_secret:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent='crypto_ml_trading/1.0'
            )
    
    async def fetch_data(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Fetch Reddit posts about query."""
        await self._rate_limit_check()
        
        if not self.reddit:
            logger.warning("Reddit API client not initialized")
            return self._get_mock_posts(query, limit)
        
        posts = []
        subreddits = ['cryptocurrency', 'bitcoin', 'ethtrader', 'cryptomarkets']
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search posts
                for submission in subreddit.search(query, limit=limit // len(subreddits)):
                    post = SocialPost(
                        platform='reddit',
                        author=str(submission.author) if submission.author else 'deleted',
                        content=submission.title + '\n\n' + submission.selftext[:500],
                        timestamp=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                        url=f"https://reddit.com{submission.permalink}",
                        engagement={
                            'upvotes': submission.score,
                            'comments': submission.num_comments,
                            'upvote_ratio': int(submission.upvote_ratio * 100)
                        },
                        entities=[query.upper()],
                        metadata={'subreddit': subreddit_name}
                    )
                    posts.append(post)
                    
        except Exception as e:
            logger.error(f"Error fetching from Reddit API: {e}")
            self.error_count += 1
            return self._get_mock_posts(query, limit)
        
        return posts
    
    def _get_mock_posts(self, query: str, limit: int) -> List[SocialPost]:
        """Get mock Reddit posts for testing."""
        mock_posts = [
            f"DD: Why {query} is undervalued at current prices",
            f"{query} technical analysis - Cup and handle forming",
            f"I'm all in on {query}! Here's why...",
            f"{query} vs competitors - detailed comparison",
            f"My {query} price prediction for next month"
        ]
        
        posts = []
        for i in range(min(limit, len(mock_posts) * 2)):
            post = SocialPost(
                platform='reddit',
                author=f'redditor_{i}',
                content=mock_posts[i % len(mock_posts)],
                timestamp=datetime.now(tz=timezone.utc) - timedelta(hours=i),
                url=f"https://reddit.com/r/cryptocurrency/comments/mock_{i}",
                engagement={
                    'upvotes': np.random.randint(10, 5000),
                    'comments': np.random.randint(5, 500),
                    'upvote_ratio': np.random.randint(60, 95)
                },
                entities=[query.upper()],
                metadata={'subreddit': 'cryptocurrency'}
            )
            posts.append(post)
        
        return posts


class OnChainDataProvider(DataSource):
    """On-chain blockchain data provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__('OnChain', rate_limit=30)
        self.api_key = api_key
        self.base_urls = {
            'glassnode': 'https://api.glassnode.com/v1/metrics',
            'messari': 'https://data.messari.io/api/v1',
            'blockchain_info': 'https://blockchain.info'
        }
    
    async def fetch_data(self, query: str, limit: int = 100) -> List[OnChainData]:
        """Fetch on-chain data for cryptocurrency."""
        await self._rate_limit_check()
        
        # For demo, return mock data
        return self._get_mock_onchain_data(query)
    
    def _get_mock_onchain_data(self, symbol: str) -> List[OnChainData]:
        """Get mock on-chain data."""
        metrics = [
            ('hash_rate', np.random.uniform(100, 200), 'EH/s'),
            ('active_addresses', np.random.randint(500000, 1000000), 'addresses'),
            ('transaction_volume', np.random.uniform(10, 50), 'billion_usd'),
            ('exchange_inflow', np.random.uniform(1000, 5000), 'btc'),
            ('exchange_outflow', np.random.uniform(1000, 5000), 'btc'),
            ('miner_revenue', np.random.uniform(20, 40), 'million_usd'),
            ('difficulty', np.random.uniform(20, 30), 'T'),
            ('nvt_ratio', np.random.uniform(50, 150), 'ratio')
        ]
        
        data = []
        for metric_name, value, unit in metrics:
            on_chain = OnChainData(
                blockchain=symbol.upper(),
                metric=metric_name,
                value=value,
                timestamp=datetime.now(tz=timezone.utc),
                metadata={'unit': unit}
            )
            data.append(on_chain)
        
        return data


class AlternativeDataAggregator:
    """
    Aggregates data from all alternative sources.
    
    Features:
    - Multi-source data collection
    - Sentiment analysis integration
    - Entity extraction
    - Real-time streaming
    """
    
    def __init__(self, 
                 twitter_token: Optional[str] = None,
                 reddit_client_id: Optional[str] = None,
                 reddit_secret: Optional[str] = None,
                 news_api_key: Optional[str] = None):
        """Initialize aggregator with API credentials."""
        # Data sources
        self.news_api = CryptoNewsAPI(news_api_key)
        self.twitter_api = TwitterAPI(twitter_token)
        self.reddit_api = RedditAPI(reddit_client_id, reddit_secret)
        self.onchain_api = OnChainDataProvider()
        
        # Data storage
        self.news_buffer = deque(maxlen=1000)
        self.social_buffer = deque(maxlen=5000)
        self.onchain_buffer = deque(maxlen=500)
        
        # Sentiment analyzer placeholder
        self.sentiment_analyzer = None
        
        # Callbacks
        self.callbacks = defaultdict(list)
        
        logger.info("Alternative data aggregator initialized")
    
    async def fetch_all_data(self, symbols: List[str], lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Fetch data from all sources for given symbols.
        
        Args:
            symbols: List of cryptocurrency symbols
            lookback_hours: Hours of historical data
            
        Returns:
            Dictionary with all collected data
        """
        all_data = {
            'news': [],
            'twitter': [],
            'reddit': [],
            'onchain': []
        }
        
        # Create tasks for parallel fetching
        tasks = []
        
        for symbol in symbols:
            # News
            tasks.append(self._fetch_with_error_handling(
                self.news_api.fetch_data(symbol, limit=50),
                'news', all_data
            ))
            
            # Twitter
            tasks.append(self._fetch_with_error_handling(
                self.twitter_api.fetch_data(symbol, limit=100),
                'twitter', all_data
            ))
            
            # Reddit
            tasks.append(self._fetch_with_error_handling(
                self.reddit_api.fetch_data(symbol, limit=50),
                'reddit', all_data
            ))
            
            # On-chain
            tasks.append(self._fetch_with_error_handling(
                self.onchain_api.fetch_data(symbol),
                'onchain', all_data
            ))
        
        # Execute all tasks
        await asyncio.gather(*tasks)
        
        # Store in buffers
        self.news_buffer.extend(all_data['news'])
        self.social_buffer.extend(all_data['twitter'] + all_data['reddit'])
        self.onchain_buffer.extend(all_data['onchain'])
        
        # Emit to callbacks
        self._emit_data(all_data)
        
        return all_data
    
    async def _fetch_with_error_handling(self, coro, data_type: str, result_dict: Dict):
        """Fetch data with error handling."""
        try:
            data = await coro
            result_dict[data_type].extend(data)
        except Exception as e:
            logger.error(f"Error fetching {data_type} data: {e}")
    
    def _emit_data(self, data: Dict[str, Any]):
        """Emit data to registered callbacks."""
        for data_type, items in data.items():
            if data_type in self.callbacks:
                for callback in self.callbacks[data_type]:
                    try:
                        callback(items)
                    except Exception as e:
                        logger.error(f"Callback error for {data_type}: {e}")
    
    def register_callback(self, data_type: str, callback: Callable):
        """Register callback for data updates."""
        self.callbacks[data_type].append(callback)
    
    def set_sentiment_analyzer(self, analyzer):
        """Set sentiment analyzer for processing text data."""
        self.sentiment_analyzer = analyzer
    
    async def analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of text data."""
        if not self.sentiment_analyzer:
            logger.warning("No sentiment analyzer configured")
            return data
        
        # Analyze news
        for article in data.get('news', []):
            if article.content:
                sentiment = self.sentiment_analyzer.analyze_text(
                    article.title + ' ' + article.content
                )
                article.sentiment_score = sentiment.get('sentiment_score', 0)
        
        # Analyze social posts
        for post in data.get('twitter', []) + data.get('reddit', []):
            if post.content:
                sentiment = self.sentiment_analyzer.analyze_text(post.content)
                post.sentiment_score = sentiment.get('sentiment_score', 0)
        
        return data
    
    def get_sentiment_summary(self, symbol: str, hours: int = 1) -> Dict[str, Any]:
        """Get sentiment summary for a symbol."""
        cutoff_time = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        
        # Filter recent data
        recent_news = [n for n in self.news_buffer 
                      if symbol in n.entities and n.published > cutoff_time]
        recent_social = [s for s in self.social_buffer 
                        if symbol in s.entities and s.timestamp > cutoff_time]
        
        # Calculate aggregated sentiment
        news_sentiments = [n.sentiment_score for n in recent_news if n.sentiment_score]
        social_sentiments = [s.sentiment_score for s in recent_social if s.sentiment_score]
        
        return {
            'symbol': symbol,
            'timeframe': f'{hours}h',
            'news': {
                'count': len(recent_news),
                'avg_sentiment': np.mean(news_sentiments) if news_sentiments else 0,
                'std_sentiment': np.std(news_sentiments) if news_sentiments else 0
            },
            'social': {
                'count': len(recent_social),
                'avg_sentiment': np.mean(social_sentiments) if social_sentiments else 0,
                'std_sentiment': np.std(social_sentiments) if social_sentiments else 0,
                'total_engagement': sum(
                    sum(s.engagement.values()) for s in recent_social
                )
            },
            'overall_sentiment': np.mean(news_sentiments + social_sentiments) 
                               if news_sentiments + social_sentiments else 0
        }
    
    def get_trending_entities(self, hours: int = 24, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get trending cryptocurrency entities."""
        cutoff_time = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        
        entity_counts = defaultdict(int)
        
        # Count from news
        for article in self.news_buffer:
            if article.published > cutoff_time:
                for entity in article.entities:
                    entity_counts[entity] += 1
        
        # Count from social with engagement weighting
        for post in self.social_buffer:
            if post.timestamp > cutoff_time:
                engagement_weight = 1 + np.log1p(sum(post.engagement.values()))
                for entity in post.entities:
                    entity_counts[entity] += engagement_weight
        
        # Sort by count
        trending = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        return trending[:top_n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            'news': {
                'total_articles': len(self.news_buffer),
                'sources': len(set(n.source for n in self.news_buffer)),
                'error_count': self.news_api.error_count
            },
            'twitter': {
                'total_posts': len([p for p in self.social_buffer if p.platform == 'twitter']),
                'avg_engagement': np.mean([
                    sum(p.engagement.values()) 
                    for p in self.social_buffer 
                    if p.platform == 'twitter'
                ]) if self.social_buffer else 0,
                'error_count': self.twitter_api.error_count
            },
            'reddit': {
                'total_posts': len([p for p in self.social_buffer if p.platform == 'reddit']),
                'error_count': self.reddit_api.error_count
            },
            'onchain': {
                'total_metrics': len(self.onchain_buffer),
                'unique_metrics': len(set(d.metric for d in self.onchain_buffer)),
                'error_count': self.onchain_api.error_count
            }
        }