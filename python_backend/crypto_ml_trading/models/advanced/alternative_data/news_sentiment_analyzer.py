"""
News Sentiment Analysis for Cryptocurrency Trading.

Implements comprehensive news sentiment analysis and event detection.
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
class NewsArticle:
    """Individual news article data."""
    article_id: str
    timestamp: datetime
    headline: str
    content: str
    source: str
    author: Optional[str] = None
    url: Optional[str] = None
    language: str = "en"
    category: str = "general"  # regulatory, technical, market, adoption, etc.
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    impact_score: Optional[float] = None


@dataclass
class NewsEvent:
    """Detected news event."""
    event_id: str
    timestamp: datetime
    event_type: str  # regulatory, hack, partnership, launch, etc.
    headline: str
    description: str
    affected_assets: List[str]
    sentiment_impact: float
    confidence: float
    source_articles: List[str] = field(default_factory=list)
    market_reaction_prediction: Optional[str] = None


@dataclass
class SentimentTrend:
    """Sentiment trend analysis."""
    timestamp: datetime
    asset_symbol: str
    sentiment_score: float
    trend_direction: str  # bullish, bearish, neutral
    trend_strength: float
    momentum: float
    volatility: float
    confidence: float
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class SourceCredibility:
    """News source credibility metrics."""
    source: str
    credibility_score: float
    bias_score: float  # -1 (bearish bias) to 1 (bullish bias)
    accuracy_history: float
    article_count: int
    avg_impact: float
    specialization_areas: List[str] = field(default_factory=list)


class NewsSentimentAnalyzer:
    """
    Advanced news sentiment analysis for cryptocurrency trading.
    
    Features:
    - Multi-source news aggregation and analysis
    - Event detection and classification
    - Sentiment scoring with confidence intervals
    - Source credibility assessment
    - Market impact prediction
    - Real-time sentiment monitoring
    - Cross-asset sentiment correlation
    - Regulatory and policy analysis
    """
    
    def __init__(self,
                 sentiment_window: int = 60,
                 event_detection_threshold: float = 0.7,
                 sources_to_track: List[str] = None):
        """
        Initialize news sentiment analyzer.
        
        Args:
            sentiment_window: Window for sentiment aggregation (minutes)
            event_detection_threshold: Threshold for event detection
            sources_to_track: List of news sources to track
        """
        self.sentiment_window = sentiment_window
        self.event_detection_threshold = event_detection_threshold
        self.sources_to_track = sources_to_track or [
            'coindesk', 'cointelegraph', 'bloomberg', 'reuters', 'wsj',
            'forbes', 'cnbc', 'decrypt', 'theblock', 'coinbase_blog'
        ]
        
        # Data storage
        self.news_articles: deque = deque(maxlen=5000)
        self.detected_events: deque = deque(maxlen=500)
        self.sentiment_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        # Source tracking
        self.source_credibility: Dict[str, SourceCredibility] = {}
        self.source_performance: Dict[str, Dict] = defaultdict(dict)
        
        # Sentiment models
        self.sentiment_lexicon = self._initialize_sentiment_lexicon()
        self.asset_keywords = self._initialize_asset_keywords()
        self.event_keywords = self._initialize_event_keywords()
        
        # Analysis caches
        self.current_sentiment_scores: Dict[str, float] = defaultdict(float)
        self.sentiment_history: Dict[str, List] = defaultdict(list)
        
        # Initialize source credibility
        self._initialize_source_credibility()
    
    def _initialize_sentiment_lexicon(self) -> Dict[str, float]:
        """Initialize cryptocurrency news sentiment lexicon."""
        # Positive sentiment terms
        positive_terms = {
            # Price/market positive
            'bullish': 2.0, 'bull': 1.5, 'rally': 1.8, 'surge': 2.0, 'pump': 1.5,
            'moon': 1.8, 'gains': 1.5, 'profit': 1.2, 'up': 1.0, 'rise': 1.2,
            'breakout': 1.8, 'support': 1.0, 'strength': 1.2, 'strong': 1.2,
            
            # Adoption/development positive
            'adoption': 1.8, 'partnership': 1.5, 'integration': 1.2, 'launch': 1.3,
            'upgrade': 1.4, 'development': 1.0, 'innovation': 1.3, 'breakthrough': 2.0,
            'milestone': 1.5, 'progress': 1.2, 'success': 1.5, 'achievement': 1.4,
            
            # Regulatory positive
            'approval': 1.8, 'legal': 1.0, 'compliant': 1.2, 'regulated': 1.0,
            'authorized': 1.5, 'licensed': 1.3, 'framework': 1.0,
            
            # Institutional positive
            'institutional': 1.2, 'enterprise': 1.0, 'corporate': 1.0, 'bank': 1.0,
            'investment': 1.2, 'fund': 1.0, 'backing': 1.3, 'support': 1.0
        }
        
        # Negative sentiment terms
        negative_terms = {
            # Price/market negative
            'bearish': -2.0, 'bear': -1.5, 'crash': -2.0, 'dump': -1.8, 'decline': -1.5,
            'fall': -1.2, 'drop': -1.2, 'down': -1.0, 'loss': -1.3, 'losses': -1.3,
            'correction': -1.0, 'resistance': -0.8, 'weak': -1.2, 'weakness': -1.2,
            
            # Security/risk negative
            'hack': -2.5, 'hacked': -2.5, 'breach': -2.0, 'stolen': -2.0, 'scam': -2.5,
            'fraud': -2.5, 'rug': -2.0, 'exploit': -2.0, 'vulnerability': -1.5,
            'risk': -1.0, 'risky': -1.2, 'dangerous': -1.8, 'unsafe': -1.5,
            
            # Regulatory negative
            'ban': -2.5, 'banned': -2.5, 'illegal': -2.0, 'crackdown': -2.0,
            'investigation': -1.5, 'probe': -1.3, 'lawsuit': -1.8, 'fine': -1.5,
            'penalty': -1.5, 'restriction': -1.3, 'regulate': -0.8,
            
            # Market structure negative
            'bubble': -1.8, 'overvalued': -1.5, 'speculation': -1.0, 'volatile': -0.8,
            'uncertainty': -1.2, 'concern': -1.0, 'worry': -1.2, 'fear': -1.5
        }
        
        # Combine lexicons
        lexicon = {}
        lexicon.update(positive_terms)
        lexicon.update(negative_terms)
        
        return lexicon
    
    def _initialize_asset_keywords(self) -> Dict[str, List[str]]:
        """Initialize asset-specific keywords for relevance detection."""
        return {
            'bitcoin': [
                'bitcoin', 'btc', '$btc', 'satoshi', 'nakamoto', 'block', 'blockchain',
                'mining', 'miner', 'hash', 'difficulty', 'halving'
            ],
            'ethereum': [
                'ethereum', 'eth', '$eth', 'ether', 'vitalik', 'buterin', 'gas',
                'smart contract', 'dapp', 'defi', 'eip', 'merge', 'proof of stake'
            ],
            'binance': [
                'binance', 'bnb', '$bnb', 'cz', 'changpeng', 'zhao', 'bsc',
                'binance smart chain', 'binance coin'
            ],
            'cardano': [
                'cardano', 'ada', '$ada', 'charles', 'hoskinson', 'plutus',
                'hydra', 'ouroboros'
            ],
            'solana': [
                'solana', 'sol', '$sol', 'phantom', 'proof of history', 'validator'
            ],
            'polkadot': [
                'polkadot', 'dot', '$dot', 'kusama', 'parachain', 'substrate',
                'gavin', 'wood'
            ],
            'crypto_general': [
                'cryptocurrency', 'crypto', 'digital asset', 'altcoin', 'token',
                'coin', 'market cap', 'trading', 'exchange', 'wallet'
            ]
        }
    
    def _initialize_event_keywords(self) -> Dict[str, List[str]]:
        """Initialize event detection keywords."""
        return {
            'regulatory': [
                'regulation', 'sec', 'cftc', 'treasury', 'irs', 'government',
                'policy', 'law', 'legal', 'compliance', 'jurisdiction'
            ],
            'security': [
                'hack', 'breach', 'exploit', 'vulnerability', 'attack', 'stolen',
                'security', 'audit', 'bug', 'protocol'
            ],
            'partnership': [
                'partnership', 'collaboration', 'agreement', 'alliance', 'deal',
                'integration', 'adoption', 'cooperation'
            ],
            'technical': [
                'upgrade', 'update', 'fork', 'protocol', 'development', 'release',
                'version', 'improvement', 'feature', 'launch'
            ],
            'market': [
                'listing', 'exchange', 'trading', 'price', 'market', 'volume',
                'etf', 'fund', 'investment', 'institutional'
            ],
            'adoption': [
                'adoption', 'mainstream', 'acceptance', 'payment', 'merchant',
                'retail', 'consumer', 'user', 'growth'
            ]
        }
    
    def _initialize_source_credibility(self) -> None:
        """Initialize credibility scores for news sources."""
        # Base credibility scores (would be learned over time)
        source_ratings = {
            'bloomberg': {'credibility': 0.9, 'bias': 0.0, 'specialization': ['market', 'regulatory']},
            'reuters': {'credibility': 0.95, 'bias': 0.0, 'specialization': ['general', 'regulatory']},
            'wsj': {'credibility': 0.9, 'bias': 0.1, 'specialization': ['market', 'institutional']},
            'coindesk': {'credibility': 0.8, 'bias': 0.2, 'specialization': ['technical', 'market']},
            'cointelegraph': {'credibility': 0.7, 'bias': 0.3, 'specialization': ['technical', 'adoption']},
            'theblock': {'credibility': 0.85, 'bias': 0.1, 'specialization': ['technical', 'defi']},
            'decrypt': {'credibility': 0.75, 'bias': 0.2, 'specialization': ['technical', 'culture']},
            'forbes': {'credibility': 0.8, 'bias': 0.1, 'specialization': ['market', 'institutional']},
            'cnbc': {'credibility': 0.8, 'bias': 0.0, 'specialization': ['market', 'regulatory']},
            'coinbase_blog': {'credibility': 0.7, 'bias': 0.4, 'specialization': ['technical', 'adoption']}
        }
        
        for source, ratings in source_ratings.items():
            self.source_credibility[source] = SourceCredibility(
                source=source,
                credibility_score=ratings['credibility'],
                bias_score=ratings['bias'],
                accuracy_history=0.8,  # Default
                article_count=0,
                avg_impact=0.0,
                specialization_areas=ratings['specialization']
            )
    
    def add_news_article(self, article: NewsArticle) -> None:
        """
        Add news article for sentiment analysis.
        
        Args:
            article: News article data
        """
        # Analyze sentiment
        article.sentiment_score = self._analyze_article_sentiment(article)
        
        # Calculate relevance scores for different assets
        article.relevance_score = self._calculate_article_relevance(article)
        
        # Calculate potential impact score
        article.impact_score = self._calculate_impact_score(article)
        
        # Store article
        self.news_articles.append(article)
        
        # Update source metrics
        self._update_source_metrics(article)
        
        # Check for events
        self._detect_news_events(article)
        
        # Update sentiment trends
        self._update_sentiment_trends(article)
    
    def _analyze_article_sentiment(self, article: NewsArticle) -> float:
        """Analyze sentiment of news article."""
        # Combine headline and content for analysis
        text = f"{article.headline} {article.content}"
        
        # Tokenize and clean text
        tokens = self._tokenize_text(text)
        
        # Calculate sentiment scores
        sentiment_scores = []
        
        for token in tokens:
            if token in self.sentiment_lexicon:
                sentiment_scores.append(self.sentiment_lexicon[token])
        
        if not sentiment_scores:
            return 0.0  # Neutral
        
        # Weight by source credibility
        source_credibility = self.source_credibility.get(article.source)
        credibility_weight = source_credibility.credibility_score if source_credibility else 0.5
        
        # Calculate base sentiment
        base_sentiment = np.mean(sentiment_scores)
        
        # Adjust for headline vs content (headline weighted more)
        headline_sentiment = self._analyze_text_sentiment(article.headline)
        content_sentiment = self._analyze_text_sentiment(article.content)
        
        weighted_sentiment = 0.6 * headline_sentiment + 0.4 * content_sentiment
        
        # Apply credibility weighting
        final_sentiment = weighted_sentiment * credibility_weight
        
        # Apply source bias correction
        if source_credibility:
            final_sentiment -= source_credibility.bias_score * 0.1
        
        return np.clip(final_sentiment, -1.0, 1.0)
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of specific text."""
        tokens = self._tokenize_text(text)
        
        sentiment_scores = [self.sentiment_lexicon[token] for token in tokens 
                           if token in self.sentiment_lexicon]
        
        if not sentiment_scores:
            return 0.0
        
        # Apply intensity modifiers
        intensifiers = ['very', 'extremely', 'highly', 'significantly', 'major', 'massive']
        diminishers = ['slightly', 'somewhat', 'minor', 'small', 'limited']
        
        text_lower = text.lower()
        
        intensity_multiplier = 1.0
        if any(word in text_lower for word in intensifiers):
            intensity_multiplier = 1.3
        elif any(word in text_lower for word in diminishers):
            intensity_multiplier = 0.7
        
        return np.mean(sentiment_scores) * intensity_multiplier
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize and clean text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important symbols
        text = re.sub(r'[^\w\s$%]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove stop words and short tokens
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might'
        }
        
        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
        
        return tokens
    
    def _calculate_article_relevance(self, article: NewsArticle) -> Dict[str, float]:
        """Calculate relevance scores for different assets."""
        text = f"{article.headline} {article.content}".lower()
        relevance_scores = {}
        
        for asset, keywords in self.asset_keywords.items():
            relevance_score = 0.0
            
            for keyword in keywords:
                # Count occurrences with different weights
                headline_mentions = article.headline.lower().count(keyword.lower())
                content_mentions = article.content.lower().count(keyword.lower())
                
                # Headlines weighted more heavily
                relevance_score += headline_mentions * 2.0 + content_mentions * 1.0
            
            # Normalize by article length
            total_words = len(text.split())
            normalized_score = relevance_score / max(total_words, 1) * 100
            
            relevance_scores[asset] = min(1.0, normalized_score)
        
        return relevance_scores
    
    def _calculate_impact_score(self, article: NewsArticle) -> float:
        """Calculate potential market impact score."""
        impact_factors = []
        
        # Source credibility factor
        source_credibility = self.source_credibility.get(article.source)
        if source_credibility:
            impact_factors.append(source_credibility.credibility_score * 0.4)
        
        # Sentiment magnitude factor
        sentiment_magnitude = abs(article.sentiment_score or 0.0)
        impact_factors.append(sentiment_magnitude * 0.3)
        
        # Event type factor
        event_type = self._classify_article_event_type(article)
        event_impact_weights = {
            'regulatory': 0.9,
            'security': 0.8,
            'partnership': 0.6,
            'technical': 0.4,
            'market': 0.7,
            'adoption': 0.5,
            'general': 0.2
        }
        impact_factors.append(event_impact_weights.get(event_type, 0.2) * 0.3)
        
        return min(1.0, sum(impact_factors))
    
    def _classify_article_event_type(self, article: NewsArticle) -> str:
        """Classify the type of event in the article."""
        text = f"{article.headline} {article.content}".lower()
        
        # Score each event type
        event_scores = {}
        
        for event_type, keywords in self.event_keywords.items():
            score = sum(text.count(keyword) for keyword in keywords)
            event_scores[event_type] = score
        
        # Return the highest scoring event type
        if event_scores:
            return max(event_scores, key=event_scores.get)
        
        return 'general'
    
    def _update_source_metrics(self, article: NewsArticle) -> None:
        """Update metrics for news source."""
        source = article.source
        
        if source not in self.source_credibility:
            self.source_credibility[source] = SourceCredibility(
                source=source,
                credibility_score=0.5,  # Default
                bias_score=0.0,
                accuracy_history=0.5,
                article_count=0,
                avg_impact=0.0
            )
        
        source_metrics = self.source_credibility[source]
        source_metrics.article_count += 1
        
        # Update average impact
        if article.impact_score:
            source_metrics.avg_impact = (
                (source_metrics.avg_impact * (source_metrics.article_count - 1) + article.impact_score) /
                source_metrics.article_count
            )
    
    def _detect_news_events(self, article: NewsArticle) -> None:
        """Detect significant news events."""
        # Check if article represents a significant event
        if (article.impact_score and 
            article.impact_score > self.event_detection_threshold):
            
            event_type = self._classify_article_event_type(article)
            affected_assets = self._identify_affected_assets(article)
            
            # Generate event
            event = NewsEvent(
                event_id=f"{article.source}_{article.timestamp.isoformat()}_{event_type}",
                timestamp=article.timestamp,
                event_type=event_type,
                headline=article.headline,
                description=article.content[:200] + "..." if len(article.content) > 200 else article.content,
                affected_assets=affected_assets,
                sentiment_impact=article.sentiment_score or 0.0,
                confidence=article.impact_score,
                source_articles=[article.article_id],
                market_reaction_prediction=self._predict_market_reaction(article, event_type)
            )
            
            self.detected_events.append(event)
    
    def _identify_affected_assets(self, article: NewsArticle) -> List[str]:
        """Identify assets affected by the news article."""
        affected_assets = []
        
        if hasattr(article, 'relevance_score') and article.relevance_score:
            # Get assets with high relevance scores
            for asset, score in article.relevance_score.items():
                if score > 0.3:  # Threshold for significant relevance
                    affected_assets.append(asset)
        
        # If no specific assets identified, check for general crypto relevance
        if not affected_assets:
            text = f"{article.headline} {article.content}".lower()
            crypto_general_keywords = self.asset_keywords.get('crypto_general', [])
            
            if any(keyword in text for keyword in crypto_general_keywords):
                affected_assets.append('crypto_general')
        
        return affected_assets
    
    def _predict_market_reaction(self, article: NewsArticle, event_type: str) -> str:
        """Predict market reaction to news event."""
        sentiment = article.sentiment_score or 0.0
        impact = article.impact_score or 0.0
        
        # Reaction prediction based on sentiment, impact, and event type
        if sentiment > 0.3 and impact > 0.6:
            if event_type in ['partnership', 'adoption', 'technical']:
                return 'bullish_strong'
            else:
                return 'bullish_moderate'
        elif sentiment > 0.1 and impact > 0.4:
            return 'bullish_weak'
        elif sentiment < -0.3 and impact > 0.6:
            if event_type in ['regulatory', 'security']:
                return 'bearish_strong'
            else:
                return 'bearish_moderate'
        elif sentiment < -0.1 and impact > 0.4:
            return 'bearish_weak'
        else:
            return 'neutral'
    
    def _update_sentiment_trends(self, article: NewsArticle) -> None:
        """Update sentiment trends for affected assets."""
        if not hasattr(article, 'relevance_score') or not article.relevance_score:
            return
        
        for asset, relevance in article.relevance_score.items():
            if relevance > 0.2:  # Minimum relevance threshold
                self._update_asset_sentiment_trend(asset, article)
    
    def _update_asset_sentiment_trend(self, asset: str, article: NewsArticle) -> None:
        """Update sentiment trend for a specific asset."""
        # Get recent sentiment data
        cutoff_time = article.timestamp - timedelta(minutes=self.sentiment_window)
        
        recent_articles = [
            a for a in self.news_articles
            if (a.timestamp >= cutoff_time and 
                hasattr(a, 'relevance_score') and
                a.relevance_score and
                a.relevance_score.get(asset, 0) > 0.2)
        ]
        
        if len(recent_articles) < 3:
            return  # Need minimum articles for trend analysis
        
        # Calculate trend metrics
        sentiments = [a.sentiment_score for a in recent_articles if a.sentiment_score is not None]
        timestamps = [a.timestamp for a in recent_articles]
        
        if not sentiments:
            return
        
        # Calculate trend direction and strength
        current_sentiment = np.mean(sentiments[-3:])  # Recent sentiment
        historical_sentiment = np.mean(sentiments[:-3]) if len(sentiments) > 3 else 0.0
        
        trend_direction = self._classify_trend_direction(current_sentiment, historical_sentiment)
        trend_strength = abs(current_sentiment - historical_sentiment)
        
        # Calculate momentum (rate of change)
        momentum = self._calculate_sentiment_momentum(sentiments, timestamps)
        
        # Calculate volatility
        volatility = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Calculate confidence
        confidence = self._calculate_trend_confidence(recent_articles, asset)
        
        # Contributing factors
        contributing_factors = self._identify_contributing_factors(recent_articles)
        
        trend = SentimentTrend(
            timestamp=article.timestamp,
            asset_symbol=asset,
            sentiment_score=current_sentiment,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            momentum=momentum,
            volatility=volatility,
            confidence=confidence,
            contributing_factors=contributing_factors
        )
        
        self.sentiment_trends[asset].append(trend)
        self.current_sentiment_scores[asset] = current_sentiment
    
    def _classify_trend_direction(self, current: float, historical: float) -> str:
        """Classify sentiment trend direction."""
        change = current - historical
        
        if change > 0.1:
            return 'bullish'
        elif change < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_sentiment_momentum(self, 
                                    sentiments: List[float], 
                                    timestamps: List[datetime]) -> float:
        """Calculate sentiment momentum using linear regression."""
        if len(sentiments) < 3:
            return 0.0
        
        # Convert timestamps to numeric values (hours since first timestamp)
        base_time = timestamps[0]
        x_values = [(ts - base_time).total_seconds() / 3600 for ts in timestamps]
        
        # Linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(sentiments)
        sum_xy = sum(x * y for x, y in zip(x_values, sentiments))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        return np.clip(slope, -1.0, 1.0)
    
    def _calculate_trend_confidence(self, articles: List[NewsArticle], asset: str) -> float:
        """Calculate confidence in sentiment trend."""
        factors = []
        
        # Sample size factor
        sample_size_factor = min(1.0, len(articles) / 20.0)
        factors.append(sample_size_factor)
        
        # Source diversity factor
        sources = set(a.source for a in articles)
        source_diversity = min(1.0, len(sources) / 5.0)
        factors.append(source_diversity)
        
        # Average source credibility
        credibility_scores = []
        for article in articles:
            if article.source in self.source_credibility:
                credibility_scores.append(self.source_credibility[article.source].credibility_score)
        
        avg_credibility = np.mean(credibility_scores) if credibility_scores else 0.5
        factors.append(avg_credibility)
        
        # Relevance factor
        relevance_scores = [a.relevance_score.get(asset, 0) for a in articles 
                           if hasattr(a, 'relevance_score') and a.relevance_score]
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        factors.append(avg_relevance)
        
        return np.mean(factors)
    
    def _identify_contributing_factors(self, articles: List[NewsArticle]) -> List[str]:
        """Identify factors contributing to sentiment trend."""
        factors = []
        
        # Count event types
        event_counts = defaultdict(int)
        for article in articles:
            event_type = self._classify_article_event_type(article)
            event_counts[event_type] += 1
        
        # Add significant contributing factors
        total_articles = len(articles)
        for event_type, count in event_counts.items():
            if count / total_articles > 0.3:  # 30% threshold
                factors.append(event_type)
        
        return factors
    
    def get_current_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get current sentiment analysis for an asset."""
        if asset not in self.sentiment_trends or not self.sentiment_trends[asset]:
            return None
        
        latest_trend = self.sentiment_trends[asset][-1]
        
        return {
            'asset': asset,
            'sentiment_score': latest_trend.sentiment_score,
            'trend_direction': latest_trend.trend_direction,
            'trend_strength': latest_trend.trend_strength,
            'momentum': latest_trend.momentum,
            'volatility': latest_trend.volatility,
            'confidence': latest_trend.confidence,
            'contributing_factors': latest_trend.contributing_factors,
            'timestamp': latest_trend.timestamp.isoformat()
        }
    
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent significant news events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.detected_events
            if event.timestamp >= cutoff_time
        ]
        
        return [
            {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'headline': event.headline,
                'affected_assets': event.affected_assets,
                'sentiment_impact': event.sentiment_impact,
                'confidence': event.confidence,
                'market_reaction_prediction': event.market_reaction_prediction
            }
            for event in recent_events
        ]
    
    def get_source_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for news sources."""
        performance = {}
        
        for source, credibility in self.source_credibility.items():
            performance[source] = {
                'credibility_score': credibility.credibility_score,
                'bias_score': credibility.bias_score,
                'accuracy_history': credibility.accuracy_history,
                'article_count': credibility.article_count,
                'avg_impact': credibility.avg_impact,
                'specialization_areas': credibility.specialization_areas
            }
        
        return performance
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis summary."""
        summary = {
            'total_articles_analyzed': len(self.news_articles),
            'total_events_detected': len(self.detected_events),
            'sources_tracked': len(self.source_credibility),
            'assets_monitored': list(self.current_sentiment_scores.keys())
        }
        
        # Current sentiment overview
        current_sentiments = {}
        for asset, score in self.current_sentiment_scores.items():
            if asset in self.sentiment_trends and self.sentiment_trends[asset]:
                latest_trend = self.sentiment_trends[asset][-1]
                current_sentiments[asset] = {
                    'sentiment_score': score,
                    'trend_direction': latest_trend.trend_direction,
                    'confidence': latest_trend.confidence
                }
        
        summary['current_sentiments'] = current_sentiments
        
        # Recent event summary
        recent_events = self.get_recent_events(24)
        summary['recent_events_count'] = len(recent_events)
        
        if recent_events:
            summary['latest_event'] = recent_events[-1]
        
        # Top performing sources
        top_sources = sorted(
            self.source_credibility.items(),
            key=lambda x: x[1].credibility_score * x[1].article_count,
            reverse=True
        )[:5]
        
        summary['top_sources'] = [
            {
                'source': source,
                'credibility': metrics.credibility_score,
                'articles': metrics.article_count
            }
            for source, metrics in top_sources
        ]
        
        return summary
    
    def analyze_cross_asset_sentiment_correlation(self) -> Dict[str, float]:
        """Analyze sentiment correlation between different assets."""
        correlations = {}
        
        assets = list(self.current_sentiment_scores.keys())
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                if (asset1 in self.sentiment_trends and 
                    asset2 in self.sentiment_trends and
                    len(self.sentiment_trends[asset1]) > 10 and
                    len(self.sentiment_trends[asset2]) > 10):
                    
                    # Get aligned sentiment scores
                    scores1 = [t.sentiment_score for t in list(self.sentiment_trends[asset1])[-20:]]
                    scores2 = [t.sentiment_score for t in list(self.sentiment_trends[asset2])[-20:]]
                    
                    # Calculate correlation
                    if len(scores1) == len(scores2):
                        correlation = np.corrcoef(scores1, scores2)[0, 1]
                        if not np.isnan(correlation):
                            correlations[f"{asset1}_{asset2}"] = correlation
        
        return correlations