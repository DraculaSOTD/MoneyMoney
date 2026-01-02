"""
Enhanced Sentiment Analysis System for Cryptocurrency Trading.

Extends the base sentiment analyzer with advanced features including
multi-source aggregation, real-time processing, and trading integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re
import json
import logging
import threading
import queue

from sentiment_analyzer import SentimentAnalyzer
from transformer_model import SentimentTransformer
from tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SentimentSignal:
    """Sentiment-based trading signal."""
    timestamp: datetime
    asset: str
    sentiment_score: float
    confidence: float
    action: str  # buy, sell, hold
    sources: List[str]
    volume: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntitySentiment:
    """Sentiment for a specific entity (coin, exchange, person)."""
    entity: str
    entity_type: str  # coin, exchange, person, protocol
    sentiment_score: float
    mention_count: int
    contexts: List[str]
    temporal_trend: float


class EnhancedSentimentAnalyzer(SentimentAnalyzer):
    """
    Enhanced sentiment analyzer with advanced capabilities.
    
    New features:
    - Entity-specific sentiment tracking
    - Multi-timeframe analysis
    - Context-aware sentiment
    - Real-time alert system
    - Trading signal generation
    """
    
    def __init__(self, 
                 model: Optional[SentimentTransformer] = None,
                 tokenizer: Optional[SimpleTokenizer] = None,
                 enable_streaming: bool = True,
                 alert_threshold: float = 0.7):
        """
        Initialize enhanced sentiment analyzer.
        
        Args:
            model: Pre-trained transformer model
            tokenizer: Tokenizer instance
            enable_streaming: Enable real-time streaming
            alert_threshold: Threshold for sentiment alerts
        """
        super().__init__(model, tokenizer)
        
        self.enable_streaming = enable_streaming
        self.alert_threshold = alert_threshold
        
        # Entity tracking
        self.entity_sentiments = defaultdict(lambda: deque(maxlen=1000))
        self.entity_patterns = self._load_entity_patterns()
        
        # Multi-timeframe tracking
        self.timeframe_sentiments = {
            '1m': deque(maxlen=60),
            '5m': deque(maxlen=60),
            '15m': deque(maxlen=96),
            '1h': deque(maxlen=168),
            '4h': deque(maxlen=168),
            '1d': deque(maxlen=30)
        }
        
        # Alert system
        self.alert_queue = queue.Queue()
        self.alert_callbacks = []
        
        # Context tracking
        self.context_buffer = deque(maxlen=100)
        self.sentiment_contexts = defaultdict(list)
        
        # Performance tracking
        self.signal_performance = deque(maxlen=1000)
        
        logger.info("Enhanced sentiment analyzer initialized")
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load entity recognition patterns."""
        return {
            'coins': {
                'BTC': ['bitcoin', 'btc', '₿'],
                'ETH': ['ethereum', 'eth', 'ether'],
                'BNB': ['binance coin', 'bnb'],
                'SOL': ['solana', 'sol'],
                'ADA': ['cardano', 'ada'],
                'DOT': ['polkadot', 'dot'],
                'AVAX': ['avalanche', 'avax'],
                'MATIC': ['polygon', 'matic'],
                'LINK': ['chainlink', 'link'],
                'UNI': ['uniswap', 'uni']
            },
            'exchanges': {
                'binance': ['binance', 'bnb exchange'],
                'coinbase': ['coinbase', 'cb'],
                'kraken': ['kraken'],
                'ftx': ['ftx'],
                'kucoin': ['kucoin'],
                'huobi': ['huobi'],
                'okex': ['okex', 'okx']
            },
            'people': {
                'vitalik': ['vitalik', 'buterin'],
                'cz': ['cz', 'changpeng zhao'],
                'saylor': ['saylor', 'michael saylor'],
                'elon': ['elon', 'musk'],
                'armstrong': ['brian armstrong']
            },
            'protocols': {
                'defi': ['defi', 'decentralized finance'],
                'nft': ['nft', 'non-fungible'],
                'dao': ['dao', 'decentralized autonomous'],
                'web3': ['web3', 'web 3.0'],
                'metaverse': ['metaverse', 'meta']
            }
        }
    
    def analyze_with_entities(self, text: str, source: str = 'unknown',
                            author: str = 'unknown') -> Dict[str, Any]:
        """
        Analyze sentiment with entity recognition.
        
        Args:
            text: Input text
            source: Source platform
            author: Author username
            
        Returns:
            Enhanced sentiment analysis with entities
        """
        # Get base sentiment
        base_analysis = self.analyze_text(text, source, author)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Analyze entity-specific sentiment
        entity_sentiments = {}
        for entity_type, detected_entities in entities.items():
            for entity, mentions in detected_entities.items():
                if mentions:
                    # Get context around entity mentions
                    contexts = self._extract_entity_contexts(text, mentions)
                    
                    # Analyze sentiment in context
                    context_sentiments = []
                    for context in contexts:
                        context_analysis = super().analyze_text(context, source, author)
                        context_sentiments.append(context_analysis['sentiment_score'])
                    
                    # Average context sentiment
                    entity_sentiment = np.mean(context_sentiments) if context_sentiments else base_analysis['sentiment_score']
                    
                    entity_sentiments[entity] = {
                        'type': entity_type,
                        'sentiment': entity_sentiment,
                        'mentions': len(mentions),
                        'contexts': contexts[:3]  # Keep top 3 contexts
                    }
                    
                    # Track entity sentiment over time
                    self.entity_sentiments[entity].append({
                        'timestamp': datetime.now(),
                        'sentiment': entity_sentiment,
                        'source': source,
                        'author': author
                    })
        
        # Enhanced analysis
        enhanced_analysis = {
            **base_analysis,
            'entities': entity_sentiments,
            'dominant_entity': self._get_dominant_entity(entity_sentiments),
            'multi_entity_sentiment': self._calculate_multi_entity_sentiment(entity_sentiments),
            'context_consistency': self._calculate_context_consistency(entity_sentiments)
        }
        
        # Update timeframes
        self._update_timeframe_sentiments(enhanced_analysis)
        
        # Check for alerts
        self._check_sentiment_alerts(enhanced_analysis)
        
        # Add to context buffer
        self.context_buffer.append(enhanced_analysis)
        
        return enhanced_analysis
    
    def _extract_entities(self, text: str) -> Dict[str, Dict[str, List[int]]]:
        """Extract entities from text with their positions."""
        text_lower = text.lower()
        entities = defaultdict(lambda: defaultdict(list))
        
        for entity_type, patterns in self.entity_patterns.items():
            for entity, variations in patterns.items():
                for variation in variations:
                    # Find all occurrences
                    pattern = r'\b' + re.escape(variation) + r'\b'
                    for match in re.finditer(pattern, text_lower):
                        entities[entity_type][entity].append(match.start())
        
        return dict(entities)
    
    def _extract_entity_contexts(self, text: str, positions: List[int], 
                               context_window: int = 50) -> List[str]:
        """Extract context around entity mentions."""
        contexts = []
        
        for pos in positions:
            start = max(0, pos - context_window)
            end = min(len(text), pos + context_window)
            context = text[start:end]
            contexts.append(context)
        
        return contexts
    
    def _get_dominant_entity(self, entity_sentiments: Dict[str, Dict]) -> Optional[str]:
        """Get the most mentioned entity."""
        if not entity_sentiments:
            return None
        
        return max(entity_sentiments.items(), 
                  key=lambda x: x[1]['mentions'])[0]
    
    def _calculate_multi_entity_sentiment(self, entity_sentiments: Dict[str, Dict]) -> float:
        """Calculate weighted sentiment across all entities."""
        if not entity_sentiments:
            return 0.0
        
        total_mentions = sum(e['mentions'] for e in entity_sentiments.values())
        if total_mentions == 0:
            return 0.0
        
        weighted_sentiment = sum(
            e['sentiment'] * e['mentions'] 
            for e in entity_sentiments.values()
        ) / total_mentions
        
        return weighted_sentiment
    
    def _calculate_context_consistency(self, entity_sentiments: Dict[str, Dict]) -> float:
        """Calculate how consistent sentiment is across different contexts."""
        all_sentiments = []
        
        for entity_data in entity_sentiments.values():
            if 'sentiment' in entity_data:
                all_sentiments.append(entity_data['sentiment'])
        
        if len(all_sentiments) < 2:
            return 1.0
        
        # Calculate standard deviation as measure of inconsistency
        std_dev = np.std(all_sentiments)
        # Convert to consistency score (lower std = higher consistency)
        consistency = 1.0 / (1.0 + std_dev)
        
        return consistency
    
    def _update_timeframe_sentiments(self, analysis: Dict[str, Any]) -> None:
        """Update multi-timeframe sentiment tracking."""
        current_time = datetime.now()
        sentiment_data = {
            'timestamp': current_time,
            'sentiment': analysis['sentiment_score'],
            'confidence': analysis['confidence'],
            'volume': 1
        }
        
        # Add to all timeframes
        for timeframe in self.timeframe_sentiments:
            self.timeframe_sentiments[timeframe].append(sentiment_data)
    
    def _check_sentiment_alerts(self, analysis: Dict[str, Any]) -> None:
        """Check for sentiment-based alerts."""
        sentiment_score = abs(analysis['sentiment_score'])
        
        # High sentiment alert
        if sentiment_score > self.alert_threshold:
            alert = {
                'type': 'high_sentiment',
                'timestamp': datetime.now(),
                'sentiment': analysis['sentiment_score'],
                'confidence': analysis['confidence'],
                'entities': analysis.get('entities', {}),
                'source': analysis.get('source', 'unknown'),
                'text': analysis.get('text', '')[:200]
            }
            
            self.alert_queue.put(alert)
            self._trigger_alert_callbacks(alert)
        
        # Entity-specific alerts
        for entity, data in analysis.get('entities', {}).items():
            if abs(data['sentiment']) > self.alert_threshold:
                alert = {
                    'type': 'entity_sentiment',
                    'timestamp': datetime.now(),
                    'entity': entity,
                    'entity_type': data['type'],
                    'sentiment': data['sentiment'],
                    'mentions': data['mentions']
                }
                
                self.alert_queue.put(alert)
                self._trigger_alert_callbacks(alert)
    
    def _trigger_alert_callbacks(self, alert: Dict[str, Any]) -> None:
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback for sentiment alerts."""
        self.alert_callbacks.append(callback)
    
    def get_entity_sentiment_history(self, entity: str, 
                                   hours: int = 24) -> pd.DataFrame:
        """Get sentiment history for a specific entity."""
        if entity not in self.entity_sentiments:
            return pd.DataFrame()
        
        history = list(self.entity_sentiments[entity])
        if not history:
            return pd.DataFrame()
        
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours)
        filtered = [h for h in history if h['timestamp'] > cutoff]
        
        if not filtered:
            return pd.DataFrame()
        
        df = pd.DataFrame(filtered)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_timeframe_sentiment(self, timeframe: str) -> Dict[str, Any]:
        """Get aggregated sentiment for a specific timeframe."""
        if timeframe not in self.timeframe_sentiments:
            return {}
        
        data = list(self.timeframe_sentiments[timeframe])
        if not data:
            return {}
        
        sentiments = [d['sentiment'] for d in data]
        confidences = [d['confidence'] for d in data]
        
        # Calculate trend
        if len(sentiments) > 10:
            recent = np.mean(sentiments[-5:])
            older = np.mean(sentiments[-10:-5])
            trend = recent - older
        else:
            trend = 0.0
        
        return {
            'timeframe': timeframe,
            'current_sentiment': sentiments[-1] if sentiments else 0.0,
            'average_sentiment': np.mean(sentiments),
            'sentiment_std': np.std(sentiments),
            'trend': trend,
            'confidence': np.mean(confidences),
            'data_points': len(sentiments)
        }
    
    def generate_enhanced_trading_signal(self, 
                                       asset: str,
                                       lookback_hours: int = 4) -> SentimentSignal:
        """Generate enhanced trading signal with entity awareness."""
        # Get entity sentiment if available
        entity_sentiment = None
        if asset in self.entity_sentiments:
            entity_history = self.get_entity_sentiment_history(asset, lookback_hours)
            if not entity_history.empty:
                entity_sentiment = entity_history['sentiment'].iloc[-1]
        
        # Get multi-timeframe sentiments
        timeframe_data = {}
        for tf in ['5m', '15m', '1h', '4h']:
            timeframe_data[tf] = self.get_timeframe_sentiment(tf)
        
        # Aggregate sentiment across timeframes
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        # Weight recent timeframes more heavily
        weights = {'5m': 0.4, '15m': 0.3, '1h': 0.2, '4h': 0.1}
        
        for tf, weight in weights.items():
            if tf in timeframe_data and timeframe_data[tf]:
                tf_sentiment = timeframe_data[tf]['current_sentiment']
                weighted_sentiment += tf_sentiment * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_sentiment /= total_weight
        
        # Combine with entity sentiment if available
        if entity_sentiment is not None:
            final_sentiment = 0.7 * weighted_sentiment + 0.3 * entity_sentiment
        else:
            final_sentiment = weighted_sentiment
        
        # Determine trading action
        if final_sentiment > 0.3:
            action = 'buy'
            confidence = min(0.9, abs(final_sentiment))
        elif final_sentiment < -0.3:
            action = 'sell'
            confidence = min(0.9, abs(final_sentiment))
        else:
            action = 'hold'
            confidence = 0.5
        
        # Check for sentiment divergence
        short_term = timeframe_data.get('5m', {}).get('current_sentiment', 0)
        long_term = timeframe_data.get('4h', {}).get('average_sentiment', 0)
        
        divergence = abs(short_term - long_term)
        if divergence > 0.5:
            # High divergence reduces confidence
            confidence *= 0.7
        
        # Create signal
        signal = SentimentSignal(
            timestamp=datetime.now(),
            asset=asset,
            sentiment_score=final_sentiment,
            confidence=confidence,
            action=action,
            sources=[tf for tf in timeframe_data if timeframe_data[tf]],
            volume=sum(tf.get('data_points', 0) for tf in timeframe_data.values()),
            metadata={
                'timeframe_sentiments': timeframe_data,
                'entity_sentiment': entity_sentiment,
                'divergence': divergence,
                'dominant_timeframe': max(weights.items(), key=lambda x: x[1])[0]
            }
        )
        
        return signal
    
    def detect_sentiment_anomalies_enhanced(self, 
                                          entities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Enhanced anomaly detection with entity filtering."""
        anomalies = []
        
        # Check overall anomalies
        overall_anomalies = super().detect_sentiment_anomalies()
        anomalies.extend(overall_anomalies)
        
        # Check entity-specific anomalies
        entities_to_check = entities or list(self.entity_sentiments.keys())
        
        for entity in entities_to_check:
            if entity not in self.entity_sentiments:
                continue
            
            entity_history = list(self.entity_sentiments[entity])
            if len(entity_history) < 20:
                continue
            
            # Get recent sentiments
            recent_sentiments = [h['sentiment'] for h in entity_history[-20:]]
            mean_sentiment = np.mean(recent_sentiments)
            std_sentiment = np.std(recent_sentiments)
            
            # Check for sudden changes
            if len(recent_sentiments) >= 5:
                recent_change = recent_sentiments[-1] - np.mean(recent_sentiments[-5:-1])
                
                if abs(recent_change) > 2 * std_sentiment:
                    anomalies.append({
                        'type': 'entity_sentiment_spike',
                        'entity': entity,
                        'timestamp': entity_history[-1]['timestamp'],
                        'change': recent_change,
                        'severity': 'high' if abs(recent_change) > 3 * std_sentiment else 'medium'
                    })
        
        # Check for cross-entity anomalies
        if len(entities_to_check) > 1:
            entity_correlations = self._calculate_entity_correlations(entities_to_check)
            
            for (entity1, entity2), correlation in entity_correlations.items():
                # Historically correlated entities showing divergence
                if correlation > 0.7:
                    recent1 = self._get_recent_entity_sentiment(entity1)
                    recent2 = self._get_recent_entity_sentiment(entity2)
                    
                    if recent1 is not None and recent2 is not None:
                        divergence = abs(recent1 - recent2)
                        
                        if divergence > 0.5:
                            anomalies.append({
                                'type': 'entity_divergence',
                                'entities': [entity1, entity2],
                                'timestamp': datetime.now(),
                                'divergence': divergence,
                                'historical_correlation': correlation,
                                'severity': 'medium'
                            })
        
        return anomalies
    
    def _calculate_entity_correlations(self, entities: List[str]) -> Dict[Tuple[str, str], float]:
        """Calculate sentiment correlations between entities."""
        correlations = {}
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Get aligned sentiment histories
                history1 = list(self.entity_sentiments[entity1])
                history2 = list(self.entity_sentiments[entity2])
                
                if len(history1) < 10 or len(history2) < 10:
                    continue
                
                # Align by timestamp (simplified - in production use proper time alignment)
                sentiments1 = [h['sentiment'] for h in history1[-50:]]
                sentiments2 = [h['sentiment'] for h in history2[-50:]]
                
                min_len = min(len(sentiments1), len(sentiments2))
                if min_len < 10:
                    continue
                
                correlation = np.corrcoef(sentiments1[:min_len], sentiments2[:min_len])[0, 1]
                correlations[(entity1, entity2)] = correlation
        
        return correlations
    
    def _get_recent_entity_sentiment(self, entity: str) -> Optional[float]:
        """Get most recent sentiment for an entity."""
        if entity not in self.entity_sentiments or not self.entity_sentiments[entity]:
            return None
        
        return self.entity_sentiments[entity][-1]['sentiment']
    
    def get_sentiment_report(self, assets: List[str]) -> Dict[str, Any]:
        """Generate comprehensive sentiment report for multiple assets."""
        report = {
            'timestamp': datetime.now(),
            'assets': {},
            'market_sentiment': {},
            'anomalies': [],
            'top_entities': [],
            'sentiment_momentum': {}
        }
        
        # Asset-specific analysis
        for asset in assets:
            signal = self.generate_enhanced_trading_signal(asset)
            
            report['assets'][asset] = {
                'signal': signal.action,
                'sentiment': signal.sentiment_score,
                'confidence': signal.confidence,
                'volume': signal.volume,
                'metadata': signal.metadata
            }
        
        # Overall market sentiment
        all_timeframes = {}
        for tf in self.timeframe_sentiments:
            tf_data = self.get_timeframe_sentiment(tf)
            if tf_data:
                all_timeframes[tf] = tf_data
        
        report['market_sentiment'] = all_timeframes
        
        # Detect anomalies
        report['anomalies'] = self.detect_sentiment_anomalies_enhanced(assets)
        
        # Top mentioned entities
        entity_mentions = defaultdict(int)
        for entity, history in self.entity_sentiments.items():
            entity_mentions[entity] = len(history)
        
        top_entities = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        report['top_entities'] = [
            {
                'entity': entity,
                'mentions': count,
                'recent_sentiment': self._get_recent_entity_sentiment(entity)
            }
            for entity, count in top_entities
        ]
        
        # Sentiment momentum (rate of change)
        for tf in ['15m', '1h', '4h']:
            tf_data = self.get_timeframe_sentiment(tf)
            if tf_data and 'trend' in tf_data:
                report['sentiment_momentum'][tf] = tf_data['trend']
        
        return report