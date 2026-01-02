"""
Advanced Sentiment Feature Extraction.

Provides sophisticated feature extraction for sentiment analysis including
context awareness, temporal patterns, and cross-reference analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentFeatures:
    """Container for extracted sentiment features."""
    # Basic features
    text_length: int
    word_count: int
    avg_word_length: float
    
    # Sentiment indicators
    positive_word_ratio: float
    negative_word_ratio: float
    neutral_word_ratio: float
    
    # Emotion features
    emotion_scores: Dict[str, float]
    dominant_emotion: str
    emotion_intensity: float
    
    # Linguistic features
    exclamation_ratio: float
    question_ratio: float
    caps_ratio: float
    emoji_density: float
    
    # Crypto-specific features
    price_mentions: int
    technical_terms: int
    fomo_fud_score: float
    
    # Temporal features
    time_references: List[str]
    urgency_score: float
    
    # Entity features
    mentioned_coins: List[str]
    mentioned_people: List[str]
    mentioned_exchanges: List[str]
    
    # Context features
    sentiment_consistency: float
    sarcasm_probability: float
    
    # Engagement prediction
    virality_score: float
    credibility_score: float


class SentimentFeatureExtractor:
    """
    Advanced feature extraction for crypto sentiment analysis.
    
    Features:
    - Emotion detection
    - Sarcasm and irony detection
    - Temporal pattern analysis
    - Entity-specific features
    - Context-aware extraction
    """
    
    def __init__(self):
        """Initialize feature extractor with lexicons and patterns."""
        # Load lexicons
        self.positive_words = self._load_positive_lexicon()
        self.negative_words = self._load_negative_lexicon()
        self.emotion_lexicon = self._load_emotion_lexicon()
        
        # Crypto-specific patterns
        self.price_patterns = [
            r'\$[\d,]+\.?\d*[kmb]?',
            r'[\d,]+\.?\d*\s*(usd|usdt|dollars?)',
            r'price\s*(of|at|is)?\s*[\d,]+',
            r'target\s*(:|-|of)?\s*[\d,]+'
        ]
        
        self.technical_terms = {
            'support', 'resistance', 'breakout', 'breakdown', 'pattern',
            'indicator', 'rsi', 'macd', 'ema', 'sma', 'bollinger',
            'fibonacci', 'retracement', 'divergence', 'convergence',
            'oversold', 'overbought', 'consolidation', 'accumulation'
        }
        
        self.urgency_terms = {
            'now', 'immediately', 'urgent', 'asap', 'quickly',
            'last chance', 'don\'t miss', 'hurry', 'limited time',
            'act fast', 'right now', 'today only'
        }
        
        self.sarcasm_indicators = {
            'yeah right', 'sure thing', 'totally', 'obviously',
            'clearly', 'definitely not', 'as if', 'wow'
        }
        
        # Entity patterns
        self.coin_pattern = r'\b[A-Z]{2,5}\b'
        self.mention_pattern = r'@\w+'
        self.exchange_names = {
            'binance', 'coinbase', 'kraken', 'ftx', 'kucoin',
            'huobi', 'okex', 'okx', 'bitfinex', 'bitstamp'
        }
        
        logger.info("Sentiment feature extractor initialized")
    
    def _load_positive_lexicon(self) -> Set[str]:
        """Load positive sentiment words."""
        return {
            'good', 'great', 'excellent', 'amazing', 'wonderful',
            'bullish', 'moon', 'profit', 'gain', 'rise',
            'pump', 'rally', 'breakout', 'strong', 'solid',
            'buy', 'long', 'hodl', 'accumulate', 'undervalued',
            'opportunity', 'potential', 'promising', 'innovative',
            'revolutionary', 'disruptive', 'growth', 'adoption'
        }
    
    def _load_negative_lexicon(self) -> Set[str]:
        """Load negative sentiment words."""
        return {
            'bad', 'terrible', 'awful', 'horrible', 'disaster',
            'bearish', 'crash', 'dump', 'loss', 'fall',
            'scam', 'fraud', 'bubble', 'overvalued', 'risky',
            'sell', 'short', 'avoid', 'warning', 'danger',
            'collapse', 'plunge', 'tank', 'rekt', 'liquidation',
            'failed', 'broken', 'dead', 'worthless', 'shitcoin'
        }
    
    def _load_emotion_lexicon(self) -> Dict[str, Set[str]]:
        """Load emotion-specific lexicons."""
        return {
            'fear': {
                'afraid', 'scared', 'worried', 'anxious', 'panic',
                'concern', 'nervous', 'uncertain', 'doubt', 'risk'
            },
            'anger': {
                'angry', 'furious', 'mad', 'pissed', 'hate',
                'disgusted', 'frustrated', 'annoyed', 'rage'
            },
            'joy': {
                'happy', 'excited', 'thrilled', 'delighted', 'euphoric',
                'celebrate', 'amazing', 'fantastic', 'love'
            },
            'surprise': {
                'shocked', 'surprised', 'unexpected', 'sudden',
                'unbelievable', 'incredible', 'wow', 'omg'
            },
            'anticipation': {
                'expect', 'await', 'hope', 'predict', 'forecast',
                'coming', 'soon', 'future', 'potential'
            },
            'trust': {
                'believe', 'trust', 'confident', 'reliable', 'proven',
                'legitimate', 'authentic', 'genuine', 'solid'
            },
            'disgust': {
                'disgusting', 'revolting', 'gross', 'awful',
                'hate', 'detest', 'abhor', 'loathe'
            },
            'sadness': {
                'sad', 'depressed', 'disappointed', 'unhappy',
                'miserable', 'gloomy', 'down', 'loss'
            }
        }
    
    def extract_features(self, text: str, 
                        context: Optional[List[str]] = None) -> SentimentFeatures:
        """
        Extract comprehensive sentiment features from text.
        
        Args:
            text: Input text
            context: Optional context (previous messages)
            
        Returns:
            Extracted sentiment features
        """
        # Preprocess
        text_lower = text.lower()
        words = text_lower.split()
        
        # Basic features
        text_length = len(text)
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Sentiment word ratios
        positive_words = sum(1 for w in words if w in self.positive_words)
        negative_words = sum(1 for w in words if w in self.negative_words)
        neutral_words = word_count - positive_words - negative_words
        
        positive_ratio = positive_words / max(word_count, 1)
        negative_ratio = negative_words / max(word_count, 1)
        neutral_ratio = neutral_words / max(word_count, 1)
        
        # Emotion detection
        emotion_scores = self._detect_emotions(words)
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else 'neutral'
        emotion_intensity = max(emotion_scores.values()) if emotion_scores else 0.0
        
        # Linguistic features
        exclamation_ratio = text.count('!') / max(text_length, 1)
        question_ratio = text.count('?') / max(text_length, 1)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(text_length, 1)
        emoji_density = self._calculate_emoji_density(text)
        
        # Crypto-specific features
        price_mentions = self._count_price_mentions(text)
        technical_terms = sum(1 for w in words if w in self.technical_terms)
        fomo_fud_score = self._calculate_fomo_fud_score(text_lower)
        
        # Temporal features
        time_references = self._extract_time_references(text_lower)
        urgency_score = self._calculate_urgency_score(text_lower)
        
        # Entity extraction
        mentioned_coins = self._extract_coin_mentions(text)
        mentioned_people = self._extract_people_mentions(text)
        mentioned_exchanges = self._extract_exchange_mentions(text_lower)
        
        # Context features
        sentiment_consistency = self._calculate_sentiment_consistency(text, context)
        sarcasm_probability = self._detect_sarcasm(text, emotion_scores, caps_ratio)
        
        # Engagement prediction
        virality_score = self._predict_virality(
            emotion_intensity, caps_ratio, exclamation_ratio, urgency_score
        )
        credibility_score = self._calculate_credibility(
            technical_terms, price_mentions, caps_ratio, exclamation_ratio
        )
        
        return SentimentFeatures(
            text_length=text_length,
            word_count=word_count,
            avg_word_length=avg_word_length,
            positive_word_ratio=positive_ratio,
            negative_word_ratio=negative_ratio,
            neutral_word_ratio=neutral_ratio,
            emotion_scores=emotion_scores,
            dominant_emotion=dominant_emotion,
            emotion_intensity=emotion_intensity,
            exclamation_ratio=exclamation_ratio,
            question_ratio=question_ratio,
            caps_ratio=caps_ratio,
            emoji_density=emoji_density,
            price_mentions=price_mentions,
            technical_terms=technical_terms,
            fomo_fud_score=fomo_fud_score,
            time_references=time_references,
            urgency_score=urgency_score,
            mentioned_coins=mentioned_coins,
            mentioned_people=mentioned_people,
            mentioned_exchanges=mentioned_exchanges,
            sentiment_consistency=sentiment_consistency,
            sarcasm_probability=sarcasm_probability,
            virality_score=virality_score,
            credibility_score=credibility_score
        )
    
    def _detect_emotions(self, words: List[str]) -> Dict[str, float]:
        """Detect emotions in text."""
        emotion_scores = {}
        
        for emotion, lexicon in self.emotion_lexicon.items():
            score = sum(1 for w in words if w in lexicon) / max(len(words), 1)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
        
        return emotion_scores
    
    def _calculate_emoji_density(self, text: str) -> float:
        """Calculate emoji density in text."""
        # Simple emoji detection (would be more comprehensive in production)
        emoji_pattern = r'[😀-🙏🌀-🗿🚀-🛿🏠-🏿]'
        emoji_count = len(re.findall(emoji_pattern, text))
        return emoji_count / max(len(text), 1)
    
    def _count_price_mentions(self, text: str) -> int:
        """Count price mentions in text."""
        count = 0
        for pattern in self.price_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    def _calculate_fomo_fud_score(self, text: str) -> float:
        """Calculate FOMO (positive) vs FUD (negative) score."""
        fomo_terms = {
            'moon', 'pump', 'rally', 'explosive', 'skyrocket',
            'miss out', 'last chance', 'buy now', 'going up'
        }
        
        fud_terms = {
            'crash', 'dump', 'collapse', 'scam', 'rug pull',
            'bear', 'sell', 'get out', 'dead', 'over'
        }
        
        fomo_count = sum(1 for term in fomo_terms if term in text)
        fud_count = sum(1 for term in fud_terms if term in text)
        
        if fomo_count + fud_count == 0:
            return 0.0
        
        return (fomo_count - fud_count) / (fomo_count + fud_count)
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on time-pressure language."""
        urgency_count = sum(1 for term in self.urgency_terms if term in text)
        return min(1.0, urgency_count / 3)  # Normalize to 0-1
    
    def _extract_time_references(self, text: str) -> List[str]:
        """Extract temporal references from text."""
        time_patterns = [
            r'\b(today|tomorrow|yesterday|now|soon|later)\b',
            r'\b(next|last|this)\s+(week|month|year|day|hour)\b',
            r'\b\d+\s*(hours?|days?|weeks?|months?)\b',
            r'\b(morning|afternoon|evening|night)\b'
        ]
        
        references = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return references
    
    def _extract_coin_mentions(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions."""
        # Find potential coin symbols (2-5 uppercase letters)
        potential_coins = re.findall(self.coin_pattern, text)
        
        # Filter out common non-coin abbreviations
        non_coins = {'USA', 'USD', 'EUR', 'GDP', 'CEO', 'IPO', 'API', 'URL'}
        coins = [coin for coin in potential_coins if coin not in non_coins]
        
        # Also check for full names
        coin_names = {
            'bitcoin': 'BTC', 'ethereum': 'ETH', 'binance': 'BNB',
            'cardano': 'ADA', 'solana': 'SOL', 'polkadot': 'DOT'
        }
        
        text_lower = text.lower()
        for name, symbol in coin_names.items():
            if name in text_lower and symbol not in coins:
                coins.append(symbol)
        
        return list(set(coins))
    
    def _extract_people_mentions(self, text: str) -> List[str]:
        """Extract people mentions (@ mentions)."""
        mentions = re.findall(self.mention_pattern, text)
        return [m[1:] for m in mentions]  # Remove @ symbol
    
    def _extract_exchange_mentions(self, text: str) -> List[str]:
        """Extract exchange mentions."""
        mentioned = []
        for exchange in self.exchange_names:
            if exchange in text:
                mentioned.append(exchange)
        return mentioned
    
    def _calculate_sentiment_consistency(self, text: str, 
                                       context: Optional[List[str]]) -> float:
        """Calculate sentiment consistency with context."""
        if not context:
            return 1.0
        
        # Simple consistency check based on sentiment words
        current_positive = sum(1 for w in text.lower().split() if w in self.positive_words)
        current_negative = sum(1 for w in text.lower().split() if w in self.negative_words)
        current_sentiment = np.sign(current_positive - current_negative)
        
        context_sentiments = []
        for ctx_text in context[-3:]:  # Last 3 messages
            ctx_positive = sum(1 for w in ctx_text.lower().split() if w in self.positive_words)
            ctx_negative = sum(1 for w in ctx_text.lower().split() if w in self.negative_words)
            ctx_sentiment = np.sign(ctx_positive - ctx_negative)
            context_sentiments.append(ctx_sentiment)
        
        if not context_sentiments:
            return 1.0
        
        # Calculate consistency
        agreements = sum(1 for s in context_sentiments if s == current_sentiment)
        return agreements / len(context_sentiments)
    
    def _detect_sarcasm(self, text: str, emotion_scores: Dict[str, float],
                       caps_ratio: float) -> float:
        """Detect sarcasm probability."""
        sarcasm_score = 0.0
        
        # Check for sarcasm indicators
        text_lower = text.lower()
        for indicator in self.sarcasm_indicators:
            if indicator in text_lower:
                sarcasm_score += 0.3
        
        # Mixed emotions suggest sarcasm
        if len(emotion_scores) > 2:
            sarcasm_score += 0.2
        
        # Excessive punctuation
        if text.count('...') > 1 or text.count('!!!') > 0:
            sarcasm_score += 0.1
        
        # Quotation marks around positive words
        if '"' in text and any(word in text for word in self.positive_words):
            sarcasm_score += 0.2
        
        # High caps with positive words (SURE, THIS IS "GREAT")
        if caps_ratio > 0.3 and any(word in text.lower() for word in self.positive_words):
            sarcasm_score += 0.2
        
        return min(1.0, sarcasm_score)
    
    def _predict_virality(self, emotion_intensity: float, caps_ratio: float,
                         exclamation_ratio: float, urgency_score: float) -> float:
        """Predict virality potential of text."""
        # High emotion + urgency + exclamations = viral potential
        virality = (
            emotion_intensity * 0.4 +
            min(caps_ratio * 2, 0.3) +
            min(exclamation_ratio * 3, 0.2) +
            urgency_score * 0.1
        )
        
        return min(1.0, virality)
    
    def _calculate_credibility(self, technical_terms: int, price_mentions: int,
                             caps_ratio: float, exclamation_ratio: float) -> float:
        """Calculate credibility score."""
        # Technical analysis and specific prices increase credibility
        # Excessive caps and exclamations decrease it
        
        positive_factors = (
            min(technical_terms / 5, 0.4) +
            min(price_mentions / 3, 0.3)
        )
        
        negative_factors = (
            min(caps_ratio * 2, 0.3) +
            min(exclamation_ratio * 3, 0.2)
        )
        
        credibility = 0.5 + positive_factors - negative_factors
        
        return max(0.0, min(1.0, credibility))
    
    def extract_temporal_patterns(self, 
                                texts: List[Tuple[str, datetime]]) -> Dict[str, Any]:
        """
        Extract temporal patterns from a sequence of texts.
        
        Args:
            texts: List of (text, timestamp) tuples
            
        Returns:
            Temporal pattern analysis
        """
        if not texts:
            return {}
        
        # Sort by timestamp
        texts.sort(key=lambda x: x[1])
        
        # Extract features for each text
        features_timeline = []
        for text, timestamp in texts:
            features = self.extract_features(text)
            features_timeline.append({
                'timestamp': timestamp,
                'sentiment': features.positive_word_ratio - features.negative_word_ratio,
                'emotion_intensity': features.emotion_intensity,
                'urgency': features.urgency_score,
                'virality': features.virality_score
            })
        
        # Analyze patterns
        df = pd.DataFrame(features_timeline)
        df.set_index('timestamp', inplace=True)
        
        # Calculate trends
        sentiment_trend = np.polyfit(range(len(df)), df['sentiment'], 1)[0]
        emotion_trend = np.polyfit(range(len(df)), df['emotion_intensity'], 1)[0]
        
        # Detect spikes
        sentiment_spikes = []
        if len(df) > 3:
            rolling_mean = df['sentiment'].rolling(3).mean()
            rolling_std = df['sentiment'].rolling(3).std()
            
            for i in range(3, len(df)):
                if abs(df['sentiment'].iloc[i] - rolling_mean.iloc[i]) > 2 * rolling_std.iloc[i]:
                    sentiment_spikes.append({
                        'timestamp': df.index[i],
                        'value': df['sentiment'].iloc[i],
                        'z_score': (df['sentiment'].iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                    })
        
        return {
            'sentiment_trend': sentiment_trend,
            'emotion_trend': emotion_trend,
            'avg_urgency': df['urgency'].mean(),
            'avg_virality': df['virality'].mean(),
            'sentiment_volatility': df['sentiment'].std(),
            'sentiment_spikes': sentiment_spikes,
            'time_span_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600
        }