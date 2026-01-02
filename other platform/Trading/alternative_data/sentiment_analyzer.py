import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import re
import aiohttp
import tweepy
import praw
from newsapi import NewsApiClient
import os

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    COMBINED = "combined"

@dataclass
class SentimentData:
    source: SentimentSource
    symbol: str
    sentiment_score: float  # -1 to 1
    volume: int  # Number of mentions/articles
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any]

class CryptoSentimentAnalyzer:
    def __init__(self, config: Dict[str, Any], key_manager):
        self.config = config
        self.key_manager = key_manager
        
        # Initialize API clients
        self.twitter_client = None
        self.reddit_client = None
        self.news_client = None
        
        # Sentiment history
        self.sentiment_history: Dict[str, deque] = {}
        self.max_history = 1000
        
        # Crypto symbols and tickers
        self.crypto_symbols = {
            'BTC': ['bitcoin', 'btc', '₿'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'BNB': ['binance', 'bnb'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'ADA': ['cardano', 'ada'],
            'DOGE': ['dogecoin', 'doge'],
            'MATIC': ['polygon', 'matic'],
            'DOT': ['polkadot', 'dot'],
            'AVAX': ['avalanche', 'avax']
        }
        
        # Sentiment keywords
        self.positive_keywords = [
            'bullish', 'moon', 'pump', 'rally', 'breakout', 'surge',
            'soar', 'boom', 'rocket', 'lambo', 'gains', 'profit',
            'buy', 'long', 'accumulate', 'hodl', 'diamond hands',
            'to the moon', 'all time high', 'ath', 'green', 'up'
        ]
        
        self.negative_keywords = [
            'bearish', 'dump', 'crash', 'plunge', 'collapse', 'tank',
            'sell', 'short', 'red', 'down', 'loss', 'rekt', 'scam',
            'bubble', 'correction', 'capitulation', 'fear', 'panic',
            'dead', 'worthless', 'rug pull', 'exit scam'
        ]
        
        self._initialize_clients()
        
    def _initialize_clients(self):
        try:
            # Twitter
            twitter_key = self.key_manager.get_key('twitter')
            if twitter_key and self.config.get('alternative_data', {}).get('twitter', {}).get('enabled', False):
                auth = tweepy.OAuthHandler(twitter_key.key, twitter_key.secret)
                if twitter_key.metadata and twitter_key.metadata.get('bearer_token'):
                    self.twitter_client = tweepy.Client(bearer_token=twitter_key.metadata['bearer_token'])
                    logger.info("Twitter client initialized")
                else:
                    logger.warning("Twitter bearer token not found")
                    
            # Reddit
            reddit_config = self.config.get('alternative_data', {}).get('reddit', {})
            if reddit_config.get('enabled', False):
                self.reddit_client = praw.Reddit(
                    client_id=os.getenv('REDDIT_CLIENT_ID'),
                    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                    user_agent='CryptoTradingBot/1.0'
                )
                logger.info("Reddit client initialized")
                
            # News API
            news_key = self.key_manager.get_key('newsapi')
            if news_key and self.config.get('alternative_data', {}).get('news', {}).get('enabled', False):
                self.news_client = NewsApiClient(api_key=news_key.key)
                logger.info("News API client initialized")
                
        except Exception as e:
            logger.error(f"Error initializing sentiment clients: {e}")
            
    async def analyze_twitter_sentiment(self, symbol: str, 
                                      hours_back: int = 1) -> Optional[SentimentData]:
        if not self.twitter_client:
            return None
            
        try:
            # Build query
            keywords = self.crypto_symbols.get(symbol, [symbol.lower()])
            query = ' OR '.join([f'#{kw}' for kw in keywords] + keywords)
            query += ' -is:retweet lang:en'
            
            # Search tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return None
                
            # Analyze sentiment
            positive_count = 0
            negative_count = 0
            total_engagement = 0
            
            for tweet in tweets.data:
                text = tweet.text.lower()
                metrics = tweet.public_metrics
                
                # Weight by engagement
                weight = 1 + np.log1p(
                    metrics['retweet_count'] + 
                    metrics['like_count'] + 
                    metrics['reply_count']
                )
                
                # Count sentiment
                pos_score = sum(1 for word in self.positive_keywords if word in text)
                neg_score = sum(1 for word in self.negative_keywords if word in text)
                
                if pos_score > neg_score:
                    positive_count += weight
                elif neg_score > pos_score:
                    negative_count += weight
                    
                total_engagement += weight
                
            # Calculate sentiment score
            if total_engagement > 0:
                sentiment_score = (positive_count - negative_count) / total_engagement
                sentiment_score = np.tanh(sentiment_score)  # Normalize to -1 to 1
            else:
                sentiment_score = 0
                
            # Calculate confidence based on volume
            confidence = min(1.0, np.log1p(len(tweets.data)) / np.log(100))
            
            return SentimentData(
                source=SentimentSource.TWITTER,
                symbol=symbol,
                sentiment_score=sentiment_score,
                volume=len(tweets.data),
                timestamp=datetime.utcnow(),
                confidence=confidence,
                metadata={
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'total_engagement': total_engagement
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {e}")
            return None
            
    async def analyze_reddit_sentiment(self, symbol: str,
                                     hours_back: int = 4) -> Optional[SentimentData]:
        if not self.reddit_client:
            return None
            
        try:
            # Subreddits to monitor
            subreddits = [
                'cryptocurrency', 'CryptoMarkets', 'Bitcoin',
                'ethereum', 'CryptoCurrency', 'binance',
                'CryptoMoonShots', 'SatoshiStreetBets'
            ]
            
            keywords = self.crypto_symbols.get(symbol, [symbol.lower()])
            
            positive_count = 0
            negative_count = 0
            total_posts = 0
            total_score = 0
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Search recent posts
                    for submission in subreddit.search(
                        ' OR '.join(keywords),
                        time_filter='day',
                        limit=25
                    ):
                        # Check if within time window
                        post_time = datetime.fromtimestamp(submission.created_utc)
                        if datetime.utcnow() - post_time > timedelta(hours=hours_back):
                            continue
                            
                        text = (submission.title + ' ' + submission.selftext).lower()
                        
                        # Weight by score
                        weight = 1 + np.log1p(submission.score)
                        
                        # Analyze sentiment
                        pos_score = sum(1 for word in self.positive_keywords if word in text)
                        neg_score = sum(1 for word in self.negative_keywords if word in text)
                        
                        if pos_score > neg_score:
                            positive_count += weight
                        elif neg_score > pos_score:
                            negative_count += weight
                            
                        total_posts += 1
                        total_score += submission.score
                        
                        # Also check top comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list()[:10]:
                            if hasattr(comment, 'body'):
                                comment_text = comment.body.lower()
                                comment_weight = 1 + np.log1p(comment.score)
                                
                                pos = sum(1 for word in self.positive_keywords if word in comment_text)
                                neg = sum(1 for word in self.negative_keywords if word in comment_text)
                                
                                if pos > neg:
                                    positive_count += comment_weight * 0.5
                                elif neg > pos:
                                    negative_count += comment_weight * 0.5
                                    
                except Exception as e:
                    logger.debug(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
                    
            # Calculate sentiment
            if total_posts > 0:
                total_weight = positive_count + negative_count
                if total_weight > 0:
                    sentiment_score = (positive_count - negative_count) / total_weight
                    sentiment_score = np.tanh(sentiment_score)
                else:
                    sentiment_score = 0
                    
                confidence = min(1.0, np.log1p(total_posts) / np.log(50))
            else:
                return None
                
            return SentimentData(
                source=SentimentSource.REDDIT,
                symbol=symbol,
                sentiment_score=sentiment_score,
                volume=total_posts,
                timestamp=datetime.utcnow(),
                confidence=confidence,
                metadata={
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'total_score': total_score,
                    'subreddits': len(subreddits)
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment: {e}")
            return None
            
    async def analyze_news_sentiment(self, symbol: str,
                                   hours_back: int = 24) -> Optional[SentimentData]:
        if not self.news_client:
            return None
            
        try:
            keywords = self.crypto_symbols.get(symbol, [symbol.lower()])
            query = ' OR '.join(keywords)
            
            # Get news articles
            news = self.news_client.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page_size=50,
                from_param=(datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            )
            
            if not news['articles']:
                return None
                
            positive_count = 0
            negative_count = 0
            total_articles = 0
            
            for article in news['articles']:
                # Combine title and description
                text = (
                    (article.get('title', '') + ' ' + 
                     article.get('description', '')).lower()
                )
                
                # Source credibility weight
                source = article.get('source', {}).get('name', '')
                credibility_weight = 1.0
                if source in ['Reuters', 'Bloomberg', 'CNBC', 'CoinDesk', 'CoinTelegraph']:
                    credibility_weight = 1.5
                    
                # Analyze sentiment
                pos_score = sum(1 for word in self.positive_keywords if word in text)
                neg_score = sum(1 for word in self.negative_keywords if word in text)
                
                if pos_score > neg_score:
                    positive_count += credibility_weight
                elif neg_score > pos_score:
                    negative_count += credibility_weight
                    
                total_articles += 1
                
            # Calculate sentiment
            total_weight = positive_count + negative_count
            if total_weight > 0:
                sentiment_score = (positive_count - negative_count) / total_weight
                sentiment_score = np.tanh(sentiment_score)
            else:
                sentiment_score = 0
                
            confidence = min(1.0, np.log1p(total_articles) / np.log(30))
            
            return SentimentData(
                source=SentimentSource.NEWS,
                symbol=symbol,
                sentiment_score=sentiment_score,
                volume=total_articles,
                timestamp=datetime.utcnow(),
                confidence=confidence,
                metadata={
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'sources': len(set(a.get('source', {}).get('name', '') 
                                      for a in news['articles']))
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return None
            
    async def get_combined_sentiment(self, symbol: str) -> Optional[SentimentData]:
        # Gather sentiment from all sources
        tasks = [
            self.analyze_twitter_sentiment(symbol),
            self.analyze_reddit_sentiment(symbol),
            self.analyze_news_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if isinstance(r, SentimentData)]
        
        if not valid_results:
            return None
            
        # Weight by confidence and recency
        total_weight = 0
        weighted_sentiment = 0
        total_volume = 0
        
        source_sentiments = {}
        
        for result in valid_results:
            # Recency weight (decay over 24 hours)
            age_hours = (datetime.utcnow() - result.timestamp).total_seconds() / 3600
            recency_weight = np.exp(-age_hours / 24)
            
            # Source weight
            source_weights = {
                SentimentSource.TWITTER: 0.4,
                SentimentSource.REDDIT: 0.3,
                SentimentSource.NEWS: 0.3
            }
            source_weight = source_weights.get(result.source, 0.2)
            
            # Combined weight
            weight = result.confidence * recency_weight * source_weight
            
            weighted_sentiment += result.sentiment_score * weight
            total_weight += weight
            total_volume += result.volume
            
            source_sentiments[result.source.value] = {
                'score': result.sentiment_score,
                'confidence': result.confidence,
                'volume': result.volume
            }
            
        if total_weight > 0:
            combined_score = weighted_sentiment / total_weight
            combined_confidence = min(1.0, total_weight)
        else:
            combined_score = 0
            combined_confidence = 0
            
        sentiment = SentimentData(
            source=SentimentSource.COMBINED,
            symbol=symbol,
            sentiment_score=combined_score,
            volume=total_volume,
            timestamp=datetime.utcnow(),
            confidence=combined_confidence,
            metadata={
                'sources': source_sentiments,
                'data_points': len(valid_results)
            }
        )
        
        # Store in history
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = deque(maxlen=self.max_history)
        self.sentiment_history[symbol].append(sentiment)
        
        return sentiment
        
    def get_sentiment_trend(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        if symbol not in self.sentiment_history:
            return {'trend': 0, 'volatility': 0, 'data_points': 0}
            
        history = list(self.sentiment_history[symbol])
        
        # Filter by time
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_data = [s for s in history if s.timestamp > cutoff]
        
        if len(recent_data) < 2:
            return {'trend': 0, 'volatility': 0, 'data_points': len(recent_data)}
            
        # Calculate trend
        scores = [s.sentiment_score for s in recent_data]
        times = [(s.timestamp - recent_data[0].timestamp).total_seconds() / 3600 
                for s in recent_data]
        
        if len(scores) > 1:
            # Linear regression for trend
            z = np.polyfit(times, scores, 1)
            trend = z[0]  # Slope
            
            # Volatility
            volatility = np.std(scores)
        else:
            trend = 0
            volatility = 0
            
        return {
            'trend': trend,
            'volatility': volatility,
            'current_sentiment': scores[-1] if scores else 0,
            'average_sentiment': np.mean(scores) if scores else 0,
            'data_points': len(recent_data)
        }
        
    async def start_monitoring(self, symbols: List[str], interval_minutes: int = 15):
        """Start continuous sentiment monitoring"""
        while True:
            try:
                for symbol in symbols:
                    sentiment = await self.get_combined_sentiment(symbol)
                    if sentiment:
                        logger.info(
                            f"Sentiment for {symbol}: {sentiment.sentiment_score:.3f} "
                            f"(confidence: {sentiment.confidence:.2f})"
                        )
                        
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in sentiment monitoring: {e}")
                await asyncio.sleep(60)