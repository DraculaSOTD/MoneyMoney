"""
Social Media Analysis for Cryptocurrency Trading.

Implements comprehensive social media sentiment analysis and monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class SocialMediaPost:
    """Individual social media post data."""
    post_id: str
    platform: str  # 'twitter', 'reddit', 'telegram', 'discord'
    timestamp: datetime
    content: str
    author: str
    metrics: Dict[str, int]  # likes, retweets, comments, etc.
    sentiment_score: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    influence_score: Optional[float] = None


@dataclass
class SentimentMetrics:
    """Aggregated sentiment metrics."""
    timestamp: datetime
    platform: str
    asset_symbol: str
    sentiment_score: float  # -1 to 1
    sentiment_distribution: Dict[str, float]  # positive, negative, neutral percentages
    volume_metrics: Dict[str, int]
    influence_weighted_sentiment: float
    trend_indicators: Dict[str, float]
    confidence_score: float


@dataclass
class SocialInfluencer:
    """Social media influencer tracking."""
    user_id: str
    username: str
    platform: str
    follower_count: int
    influence_score: float
    expertise_areas: List[str]
    recent_posts: deque = field(default_factory=lambda: deque(maxlen=100))
    sentiment_accuracy: Optional[float] = None


@dataclass
class ViralEvent:
    """Viral social media event detection."""
    event_id: str
    timestamp: datetime
    platforms: List[str]
    trigger_post: Optional[SocialMediaPost]
    peak_volume: int
    duration: float  # hours
    sentiment_shift: float
    market_impact: Optional[float] = None
    event_type: str = "unknown"  # 'announcement', 'rumor', 'controversy', etc.


class SocialMediaAnalyzer:
    """
    Advanced social media analysis for cryptocurrency trading.
    
    Features:
    - Multi-platform sentiment analysis
    - Influencer tracking and impact measurement
    - Viral event detection and analysis
    - Topic modeling and trend identification
    - Real-time sentiment monitoring
    - Sentiment-price correlation analysis
    """
    
    def __init__(self,
                 platforms: List[str] = None,
                 sentiment_window: int = 100,
                 viral_threshold: float = 3.0,
                 min_influence_score: float = 0.1):
        """
        Initialize social media analyzer.
        
        Args:
            platforms: List of platforms to monitor
            sentiment_window: Window size for sentiment aggregation
            viral_threshold: Threshold for viral event detection (std devs)
            min_influence_score: Minimum influence score for tracking
        """
        self.platforms = platforms or ['twitter', 'reddit', 'telegram', 'discord']
        self.sentiment_window = sentiment_window
        self.viral_threshold = viral_threshold
        self.min_influence_score = min_influence_score
        
        # Data storage
        self.posts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))  # platform -> posts
        self.sentiment_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # asset -> metrics
        self.influencers: Dict[str, SocialInfluencer] = {}  # user_id -> influencer
        
        # Analysis results
        self.viral_events: deque = deque(maxlen=200)
        self.trending_topics: Dict[str, Dict] = defaultdict(dict)  # timestamp -> topics
        
        # Sentiment lexicons and models
        self.sentiment_lexicon = self._initialize_sentiment_lexicon()
        self.crypto_keywords = self._initialize_crypto_keywords()
        
        # Real-time monitoring
        self.platform_volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=288))  # 24h of 5min buckets
        
    def _initialize_sentiment_lexicon(self) -> Dict[str, float]:
        """Initialize cryptocurrency-specific sentiment lexicon."""
        # Positive sentiment words
        positive_words = {
            'moon', 'mooning', 'bullish', 'bull', 'pump', 'pumping', 'rocket', 'gain', 'gains',
            'profit', 'profits', 'green', 'lambo', 'hodl', 'diamond', 'hands', 'buy', 'buying',
            'accumulate', 'accumulating', 'undervalued', 'cheap', 'opportunity', 'breakout',
            'rally', 'surge', 'boom', 'golden', 'cross', 'support', 'strong', 'solid',
            'fundamental', 'fundamentals', 'adoption', 'partnership', 'upgrade', 'update'
        }
        
        # Negative sentiment words
        negative_words = {
            'bearish', 'bear', 'dump', 'dumping', 'crash', 'crashing', 'red', 'loss', 'losses',
            'sell', 'selling', 'panic', 'fear', 'fud', 'scam', 'rug', 'pull', 'dead', 'rip',
            'overvalued', 'expensive', 'bubble', 'correction', 'resistance', 'weak', 'fragile',
            'hack', 'hacked', 'exploit', 'exploited', 'ban', 'banned', 'regulation', 'decline'
        }
        
        # Create lexicon with scores
        lexicon = {}
        for word in positive_words:
            lexicon[word] = 1.0
        for word in negative_words:
            lexicon[word] = -1.0
            
        return lexicon
    
    def _initialize_crypto_keywords(self) -> Dict[str, List[str]]:
        """Initialize cryptocurrency-specific keywords for each asset."""
        return {
            'bitcoin': ['bitcoin', 'btc', '$btc', 'satoshi', 'sats'],
            'ethereum': ['ethereum', 'eth', '$eth', 'ether', 'vitalik'],
            'binance': ['binance', 'bnb', '$bnb', 'cz', 'binancecoin'],
            'cardano': ['cardano', 'ada', '$ada', 'charles'],
            'solana': ['solana', 'sol', '$sol', 'phantom'],
            'polkadot': ['polkadot', 'dot', '$dot', 'kusama'],
            'chainlink': ['chainlink', 'link', '$link', 'oracle'],
            'polygon': ['polygon', 'matic', '$matic', 'layer2'],
            'avalanche': ['avalanche', 'avax', '$avax', 'subnet'],
            'generic': ['crypto', 'cryptocurrency', 'altcoin', 'defi', 'nft', 'web3', 'blockchain']
        }
    
    def add_social_media_post(self, post: SocialMediaPost) -> None:
        """
        Add a social media post for analysis.
        
        Args:
            post: Social media post data
        """
        # Analyze sentiment
        post.sentiment_score = self._analyze_post_sentiment(post.content)
        
        # Extract topics
        post.topics = self._extract_topics(post.content)
        
        # Calculate influence score if author is tracked
        if post.author in self.influencers:
            influencer = self.influencers[post.author]
            post.influence_score = self._calculate_post_influence(post, influencer)
            influencer.recent_posts.append(post)
        
        # Store post
        self.posts[post.platform].append(post)
        
        # Update platform volume tracking
        self._update_platform_volume(post.platform, post.timestamp)
        
        # Check for viral events
        self._check_viral_event(post)
        
        # Update sentiment metrics for relevant assets
        self._update_sentiment_metrics(post)
    
    def _analyze_post_sentiment(self, content: str) -> float:
        """Analyze sentiment of a social media post."""
        # Clean and tokenize content
        tokens = self._tokenize_content(content)
        
        # Calculate sentiment score
        sentiment_scores = []
        
        for token in tokens:
            if token in self.sentiment_lexicon:
                sentiment_scores.append(self.sentiment_lexicon[token])
        
        if not sentiment_scores:
            return 0.0  # Neutral
        
        # Average sentiment with adjustments
        base_sentiment = np.mean(sentiment_scores)
        
        # Adjust for intensity words
        intensity_multiplier = self._calculate_intensity_multiplier(content)
        
        # Adjust for emojis
        emoji_sentiment = self._analyze_emoji_sentiment(content)
        
        # Combine sentiments
        final_sentiment = base_sentiment * intensity_multiplier + 0.1 * emoji_sentiment
        
        # Clamp to [-1, 1]
        return np.clip(final_sentiment, -1.0, 1.0)
    
    def _tokenize_content(self, content: str) -> List[str]:
        """Tokenize social media content."""
        # Convert to lowercase
        content = content.lower()
        
        # Remove special characters but keep crypto symbols
        content = re.sub(r'[^\w\s$#@]', ' ', content)
        
        # Split into tokens
        tokens = content.split()
        
        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
        
        return tokens
    
    def _calculate_intensity_multiplier(self, content: str) -> float:
        """Calculate intensity multiplier based on content features."""
        multiplier = 1.0
        
        # ALL CAPS increases intensity
        caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        if caps_ratio > 0.5:
            multiplier *= 1.5
        
        # Exclamation marks
        exclamation_count = content.count('!')
        multiplier *= (1.0 + 0.1 * min(exclamation_count, 5))
        
        # Question marks (reduce certainty)
        question_count = content.count('?')
        multiplier *= (1.0 - 0.05 * min(question_count, 3))
        
        # Repeated characters (e.g., "sooooo")
        if re.search(r'(.)\1{2,}', content):
            multiplier *= 1.2
        
        return np.clip(multiplier, 0.5, 3.0)
    
    def _analyze_emoji_sentiment(self, content: str) -> float:
        """Analyze emoji sentiment in content."""
        # Simple emoji sentiment mapping
        positive_emojis = ['😀', '😃', '😄', '😁', '😊', '🙂', '😉', '😍', '🤑', '💰', '📈', '🚀', '🌙', '💎', '🔥']
        negative_emojis = ['😢', '😭', '😡', '😠', '💀', '📉', '💩', '😰', '😱', '🤡']
        
        positive_count = sum(content.count(emoji) for emoji in positive_emojis)
        negative_count = sum(content.count(emoji) for emoji in negative_emojis)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract relevant topics from post content."""
        topics = []
        content_lower = content.lower()
        
        # Check for cryptocurrency mentions
        for asset, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    topics.append(asset)
                    break
        
        # Check for common crypto topics
        topic_keywords = {
            'defi': ['defi', 'decentralized', 'yield', 'farming', 'liquidity'],
            'nft': ['nft', 'opensea', 'collectible', 'art', 'metadata'],
            'regulation': ['sec', 'regulation', 'compliance', 'legal', 'government'],
            'trading': ['trading', 'buy', 'sell', 'hodl', 'swing', 'scalp'],
            'technical_analysis': ['support', 'resistance', 'chart', 'pattern', 'indicator'],
            'news': ['news', 'announcement', 'partnership', 'update', 'launch'],
            'mining': ['mining', 'hash', 'difficulty', 'asic', 'proof'],
            'staking': ['staking', 'validator', 'rewards', 'delegation', 'pos']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return list(set(topics))  # Remove duplicates
    
    def _calculate_post_influence(self, post: SocialMediaPost, influencer: SocialInfluencer) -> float:
        """Calculate influence score for a post."""
        base_influence = influencer.influence_score
        
        # Adjust based on post metrics
        engagement_score = 0.0
        metrics = post.metrics
        
        if 'likes' in metrics:
            engagement_score += metrics['likes'] * 0.1
        if 'retweets' in metrics:
            engagement_score += metrics['retweets'] * 0.3
        if 'comments' in metrics:
            engagement_score += metrics['comments'] * 0.2
        if 'shares' in metrics:
            engagement_score += metrics['shares'] * 0.4
        
        # Normalize engagement score
        normalized_engagement = min(1.0, engagement_score / 1000.0)
        
        # Combine base influence with engagement
        final_influence = base_influence * (0.7 + 0.3 * normalized_engagement)
        
        return min(1.0, final_influence)
    
    def _update_platform_volume(self, platform: str, timestamp: datetime) -> None:
        """Update platform volume tracking."""
        # Round timestamp to 5-minute bucket
        bucket_time = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
        
        # Add to volume tracking
        volume_queue = self.platform_volumes[platform]
        
        # Check if we need to add a new bucket or increment existing
        if volume_queue and volume_queue[-1][0] == bucket_time:
            # Increment existing bucket
            volume_queue[-1] = (bucket_time, volume_queue[-1][1] + 1)
        else:
            # Add new bucket
            volume_queue.append((bucket_time, 1))
    
    def _check_viral_event(self, post: SocialMediaPost) -> None:
        """Check if post indicates start of viral event."""
        platform = post.platform
        timestamp = post.timestamp
        
        # Get recent volume for this platform
        recent_volumes = [vol for time, vol in self.platform_volumes[platform] 
                         if (timestamp - time).total_seconds() <= 3600]  # Last hour
        
        if len(recent_volumes) < 10:
            return  # Need more data
        
        # Calculate z-score of recent volume
        current_volume = recent_volumes[-1] if recent_volumes else 0
        mean_volume = np.mean(recent_volumes[:-1])
        std_volume = np.std(recent_volumes[:-1])
        
        if std_volume > 0:
            z_score = (current_volume - mean_volume) / std_volume
            
            if z_score > self.viral_threshold:
                # Potential viral event
                viral_event = ViralEvent(
                    event_id=f"{platform}_{timestamp.isoformat()}",
                    timestamp=timestamp,
                    platforms=[platform],
                    trigger_post=post,
                    peak_volume=current_volume,
                    duration=0.0,  # Will be updated as event evolves
                    sentiment_shift=0.0,  # Will be calculated
                    event_type=self._classify_viral_event(post)
                )
                
                self.viral_events.append(viral_event)
    
    def _classify_viral_event(self, post: SocialMediaPost) -> str:
        """Classify type of viral event based on post content."""
        content_lower = post.content.lower()
        
        # Keywords for different event types
        announcement_keywords = ['announce', 'launch', 'release', 'update', 'partnership']
        rumor_keywords = ['rumor', 'speculation', 'allegedly', 'sources', 'leaked']
        controversy_keywords = ['scam', 'hack', 'exploit', 'controversy', 'scandal']
        price_keywords = ['pump', 'dump', 'moon', 'crash', 'surge']
        
        if any(word in content_lower for word in announcement_keywords):
            return 'announcement'
        elif any(word in content_lower for word in rumor_keywords):
            return 'rumor'
        elif any(word in content_lower for word in controversy_keywords):
            return 'controversy'
        elif any(word in content_lower for word in price_keywords):
            return 'price_movement'
        else:
            return 'general'
    
    def _update_sentiment_metrics(self, post: SocialMediaPost) -> None:
        """Update sentiment metrics for relevant assets."""
        for topic in post.topics:
            if topic in self.crypto_keywords:
                # This post is about a specific cryptocurrency
                self._add_sentiment_data_point(topic, post)
    
    def _add_sentiment_data_point(self, asset: str, post: SocialMediaPost) -> None:
        """Add sentiment data point for an asset."""
        # Get recent posts for this asset
        asset_posts = []
        cutoff_time = post.timestamp - timedelta(minutes=30)  # 30-minute window
        
        for platform_posts in self.posts.values():
            for p in platform_posts:
                if (asset in p.topics and 
                    p.timestamp >= cutoff_time and 
                    p.timestamp <= post.timestamp):
                    asset_posts.append(p)
        
        if len(asset_posts) < 5:
            return  # Need minimum posts for reliable metrics
        
        # Calculate aggregated metrics
        sentiments = [p.sentiment_score for p in asset_posts if p.sentiment_score is not None]
        if not sentiments:
            return
        
        # Sentiment distribution
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        total_posts = len(sentiments)
        sentiment_distribution = {
            'positive': positive_count / total_posts,
            'negative': negative_count / total_posts,
            'neutral': neutral_count / total_posts
        }
        
        # Volume metrics
        volume_metrics = {
            'total_posts': total_posts,
            'unique_authors': len(set(p.author for p in asset_posts)),
            'total_engagement': sum(sum(p.metrics.values()) for p in asset_posts),
            'platforms': len(set(p.platform for p in asset_posts))
        }
        
        # Influence-weighted sentiment
        influence_weighted_sentiments = []
        for p in asset_posts:
            if p.sentiment_score is not None:
                weight = p.influence_score if p.influence_score else 1.0
                influence_weighted_sentiments.append(p.sentiment_score * weight)
        
        influence_weighted_sentiment = (
            np.mean(influence_weighted_sentiments) if influence_weighted_sentiments 
            else np.mean(sentiments)
        )
        
        # Trend indicators
        trend_indicators = self._calculate_trend_indicators(asset, sentiments, post.timestamp)
        
        # Confidence score
        confidence_score = self._calculate_sentiment_confidence(asset_posts)
        
        # Create sentiment metrics
        metrics = SentimentMetrics(
            timestamp=post.timestamp,
            platform='aggregated',
            asset_symbol=asset,
            sentiment_score=np.mean(sentiments),
            sentiment_distribution=sentiment_distribution,
            volume_metrics=volume_metrics,
            influence_weighted_sentiment=influence_weighted_sentiment,
            trend_indicators=trend_indicators,
            confidence_score=confidence_score
        )
        
        self.sentiment_history[asset].append(metrics)
    
    def _calculate_trend_indicators(self, 
                                  asset: str, 
                                  current_sentiments: List[float],
                                  timestamp: datetime) -> Dict[str, float]:
        """Calculate sentiment trend indicators."""
        indicators = {}
        
        # Get historical sentiment for comparison
        recent_metrics = [m for m in self.sentiment_history[asset] 
                         if (timestamp - m.timestamp).total_seconds() <= 3600]  # Last hour
        
        if len(recent_metrics) >= 3:
            historical_sentiment = np.mean([m.sentiment_score for m in recent_metrics])
            current_sentiment = np.mean(current_sentiments)
            
            indicators['sentiment_change'] = current_sentiment - historical_sentiment
            indicators['sentiment_momentum'] = self._calculate_sentiment_momentum(recent_metrics + [
                SentimentMetrics(timestamp, 'temp', asset, current_sentiment, {}, {}, 0, {}, 0)
            ])
            indicators['volatility'] = np.std([m.sentiment_score for m in recent_metrics])
        else:
            indicators['sentiment_change'] = 0.0
            indicators['sentiment_momentum'] = 0.0
            indicators['volatility'] = np.std(current_sentiments) if len(current_sentiments) > 1 else 0.0
        
        return indicators
    
    def _calculate_sentiment_momentum(self, metrics: List[SentimentMetrics]) -> float:
        """Calculate sentiment momentum using linear regression."""
        if len(metrics) < 3:
            return 0.0
        
        timestamps = [(m.timestamp - metrics[0].timestamp).total_seconds() for m in metrics]
        sentiments = [m.sentiment_score for m in metrics]
        
        # Linear regression slope
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(sentiments)
        sum_xy = sum(x * y for x, y in zip(timestamps, sentiments))
        sum_x2 = sum(x * x for x in timestamps)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Normalize slope to reasonable range
        return np.clip(slope * 3600, -1.0, 1.0)  # Scale by hour
    
    def _calculate_sentiment_confidence(self, posts: List[SocialMediaPost]) -> float:
        """Calculate confidence in sentiment measurement."""
        factors = []
        
        # Sample size factor
        sample_size_factor = min(1.0, len(posts) / 50.0)
        factors.append(sample_size_factor)
        
        # Diversity factor (multiple platforms and authors)
        unique_platforms = len(set(p.platform for p in posts))
        unique_authors = len(set(p.author for p in posts))
        diversity_factor = min(1.0, (unique_platforms / 4.0 + unique_authors / 20.0) / 2.0)
        factors.append(diversity_factor)
        
        # Sentiment consistency factor
        sentiments = [p.sentiment_score for p in posts if p.sentiment_score is not None]
        if len(sentiments) > 1:
            sentiment_std = np.std(sentiments)
            consistency_factor = max(0.0, 1.0 - sentiment_std)
            factors.append(consistency_factor)
        
        # Influence factor
        influenced_posts = [p for p in posts if p.influence_score and p.influence_score > 0.5]
        influence_factor = len(influenced_posts) / len(posts) if posts else 0.0
        factors.append(influence_factor)
        
        return np.mean(factors) if factors else 0.0
    
    def add_influencer(self, influencer: SocialInfluencer) -> None:
        """Add social media influencer for tracking."""
        if influencer.influence_score >= self.min_influence_score:
            self.influencers[influencer.user_id] = influencer
    
    def update_influencer_metrics(self, 
                                user_id: str, 
                                new_metrics: Dict[str, Any]) -> None:
        """Update influencer metrics."""
        if user_id in self.influencers:
            influencer = self.influencers[user_id]
            
            if 'follower_count' in new_metrics:
                influencer.follower_count = new_metrics['follower_count']
            
            if 'influence_score' in new_metrics:
                influencer.influence_score = new_metrics['influence_score']
                
                # Remove influencer if influence drops below threshold
                if influencer.influence_score < self.min_influence_score:
                    del self.influencers[user_id]
    
    def get_asset_sentiment(self, 
                          asset: str, 
                          window_minutes: int = 60) -> Optional[SentimentMetrics]:
        """Get current sentiment metrics for an asset."""
        if asset not in self.sentiment_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.sentiment_history[asset] if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Return most recent metrics
        return recent_metrics[-1]
    
    def get_trending_topics(self, window_minutes: int = 30) -> Dict[str, Dict[str, Any]]:
        """Get currently trending topics across all platforms."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # Collect all topics from recent posts
        topic_counts = defaultdict(int)
        topic_sentiments = defaultdict(list)
        topic_platforms = defaultdict(set)
        
        for platform_posts in self.posts.values():
            for post in platform_posts:
                if post.timestamp >= cutoff_time:
                    for topic in post.topics:
                        topic_counts[topic] += 1
                        if post.sentiment_score is not None:
                            topic_sentiments[topic].append(post.sentiment_score)
                        topic_platforms[topic].add(post.platform)
        
        # Create trending topics summary
        trending = {}
        for topic, count in topic_counts.items():
            if count >= 5:  # Minimum threshold for trending
                sentiments = topic_sentiments[topic]
                
                trending[topic] = {
                    'post_count': count,
                    'avg_sentiment': np.mean(sentiments) if sentiments else 0.0,
                    'sentiment_std': np.std(sentiments) if len(sentiments) > 1 else 0.0,
                    'platforms': list(topic_platforms[topic]),
                    'trending_score': count * (1 + abs(np.mean(sentiments)) if sentiments else 1)
                }
        
        # Sort by trending score
        return dict(sorted(trending.items(), key=lambda x: x[1]['trending_score'], reverse=True))
    
    def get_viral_events(self, window_hours: int = 24) -> List[ViralEvent]:
        """Get viral events from specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        return [event for event in self.viral_events if event.timestamp >= cutoff_time]
    
    def get_influencer_sentiment(self, window_hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Get sentiment analysis from tracked influencers."""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        influencer_analysis = {}
        
        for user_id, influencer in self.influencers.items():
            recent_posts = [p for p in influencer.recent_posts if p.timestamp >= cutoff_time]
            
            if recent_posts:
                sentiments = [p.sentiment_score for p in recent_posts if p.sentiment_score is not None]
                topics = set()
                for post in recent_posts:
                    topics.update(post.topics)
                
                influencer_analysis[user_id] = {
                    'username': influencer.username,
                    'platform': influencer.platform,
                    'influence_score': influencer.influence_score,
                    'post_count': len(recent_posts),
                    'avg_sentiment': np.mean(sentiments) if sentiments else 0.0,
                    'topics_discussed': list(topics),
                    'engagement_total': sum(sum(p.metrics.values()) for p in recent_posts)
                }
        
        return influencer_analysis
    
    def get_platform_analytics(self) -> Dict[str, Dict[str, Any]]:
        """Get analytics for each platform."""
        analytics = {}
        
        for platform in self.platforms:
            recent_posts = list(self.posts[platform])[-100:]  # Last 100 posts
            
            if recent_posts:
                sentiments = [p.sentiment_score for p in recent_posts if p.sentiment_score is not None]
                
                analytics[platform] = {
                    'total_posts': len(self.posts[platform]),
                    'recent_posts': len(recent_posts),
                    'avg_sentiment': np.mean(sentiments) if sentiments else 0.0,
                    'sentiment_std': np.std(sentiments) if len(sentiments) > 1 else 0.0,
                    'unique_authors': len(set(p.author for p in recent_posts)),
                    'top_topics': self._get_platform_top_topics(platform, recent_posts),
                    'volume_trend': self._get_platform_volume_trend(platform)
                }
        
        return analytics
    
    def _get_platform_top_topics(self, platform: str, posts: List[SocialMediaPost]) -> List[Tuple[str, int]]:
        """Get top topics for a platform."""
        topic_counts = defaultdict(int)
        
        for post in posts:
            for topic in post.topics:
                topic_counts[topic] += 1
        
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_platform_volume_trend(self, platform: str) -> str:
        """Get volume trend for a platform."""
        volumes = self.platform_volumes[platform]
        
        if len(volumes) < 6:  # Need at least 30 minutes of data
            return 'insufficient_data'
        
        recent_volume = sum(vol for _, vol in list(volumes)[-6:])  # Last 30 minutes
        previous_volume = sum(vol for _, vol in list(volumes)[-12:-6])  # Previous 30 minutes
        
        if previous_volume == 0:
            return 'unknown'
        
        change_ratio = recent_volume / previous_volume
        
        if change_ratio > 1.2:
            return 'increasing'
        elif change_ratio < 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def correlate_sentiment_with_price(self, 
                                     asset: str,
                                     price_data: List[Tuple[datetime, float]],
                                     window_hours: int = 24) -> Dict[str, float]:
        """Correlate sentiment with price movements."""
        if asset not in self.sentiment_history:
            return {'correlation': 0.0, 'lag_correlation': 0.0, 'significance': 0.0}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        # Get sentiment data
        sentiment_data = [(m.timestamp, m.sentiment_score) for m in self.sentiment_history[asset] 
                         if m.timestamp >= cutoff_time]
        
        # Filter price data
        price_data = [(timestamp, price) for timestamp, price in price_data 
                     if timestamp >= cutoff_time]
        
        if len(sentiment_data) < 10 or len(price_data) < 10:
            return {'correlation': 0.0, 'lag_correlation': 0.0, 'significance': 0.0}
        
        # Align data by timestamp (simplified - would need proper interpolation)
        aligned_data = []
        
        for sent_time, sentiment in sentiment_data:
            # Find closest price point
            closest_price = min(price_data, key=lambda x: abs((x[0] - sent_time).total_seconds()))
            
            if abs((closest_price[0] - sent_time).total_seconds()) <= 1800:  # Within 30 minutes
                aligned_data.append((sentiment, closest_price[1]))
        
        if len(aligned_data) < 5:
            return {'correlation': 0.0, 'lag_correlation': 0.0, 'significance': 0.0}
        
        sentiments, prices = zip(*aligned_data)
        
        # Calculate price returns
        price_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                price_returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(price_returns) < 5:
            return {'correlation': 0.0, 'lag_correlation': 0.0, 'significance': 0.0}
        
        # Correlation analysis
        sentiment_changes = [sentiments[i] - sentiments[i-1] for i in range(1, len(sentiments))]
        
        # Contemporaneous correlation
        if len(sentiment_changes) == len(price_returns):
            correlation = np.corrcoef(sentiment_changes, price_returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Lagged correlation (sentiment leading price)
        if len(sentiment_changes) > len(price_returns):
            lag_sent = sentiment_changes[:len(price_returns)]
            lag_correlation = np.corrcoef(lag_sent, price_returns)[0, 1]
            if np.isnan(lag_correlation):
                lag_correlation = 0.0
        else:
            lag_correlation = 0.0
        
        # Statistical significance (simplified)
        n = len(price_returns)
        significance = abs(correlation) * np.sqrt(n - 2) / np.sqrt(1 - correlation**2) if abs(correlation) < 1 else 0
        
        return {
            'correlation': correlation,
            'lag_correlation': lag_correlation,
            'significance': significance,
            'sample_size': n
        }