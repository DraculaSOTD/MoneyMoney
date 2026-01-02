import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from models.nlp.sentiment_transformer.transformer_model import SentimentTransformer
from models.nlp.sentiment_transformer.tokenizer import SimpleTokenizer


class SentimentAnalyzer:
    """
    Cryptocurrency sentiment analysis system.
    
    Features:
    - Social media sentiment analysis
    - News sentiment extraction
    - Sentiment aggregation
    - Trading signal generation
    - Real-time sentiment tracking
    """
    
    def __init__(self, model: Optional[SentimentTransformer] = None,
                 tokenizer: Optional[SimpleTokenizer] = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            model: Pre-trained transformer model
            tokenizer: Tokenizer instance
        """
        self.model = model or SentimentTransformer()
        self.tokenizer = tokenizer or SimpleTokenizer()
        
        # Sentiment history
        self.sentiment_history = []
        
        # Source weights for aggregation
        self.source_weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'news': 0.35,
            'telegram': 0.1
        }
        
        # Influencer weights
        self.influencer_multipliers = self._load_influencer_weights()
        
    def _load_influencer_weights(self) -> Dict[str, float]:
        """Load influencer credibility weights."""
        return {
            # High credibility
            'vitalik': 2.0,
            'cz_binance': 1.8,
            'elonmusk': 1.5,
            'michael_saylor': 1.7,
            
            # Medium credibility
            'crypto_whale': 1.2,
            'crypto_analyst': 1.1,
            
            # Default
            'default': 1.0
        }
    
    def analyze_text(self, text: str, source: str = 'unknown',
                    author: str = 'unknown') -> Dict:
        """
        Analyze sentiment of single text.
        
        Args:
            text: Input text
            source: Source platform
            author: Author username
            
        Returns:
            Sentiment analysis results
        """
        # Tokenize text
        input_ids = self.tokenizer.encode(text)
        input_ids = input_ids.reshape(1, -1)  # Add batch dimension
        attention_mask = self.tokenizer.create_attention_mask(input_ids)
        
        # Get model predictions
        predictions = self.model.predict_sentiment(input_ids, attention_mask)
        
        # Extract features
        features = self.tokenizer.get_sentiment_features(text)
        
        # Calculate weighted sentiment
        base_sentiment = predictions['sentiment_scores'][0]
        
        # Adjust for author credibility
        author_weight = self.influencer_multipliers.get(
            author.lower(), 
            self.influencer_multipliers['default']
        )
        
        weighted_sentiment = base_sentiment * author_weight
        
        # Combine with rule-based features
        combined_sentiment = (
            weighted_sentiment * 0.7 +
            features['emoji_sentiment'] * 0.1 +
            (features['bullish_score'] - features['bearish_score']) * 0.2
        )
        
        return {
            'text': text,
            'source': source,
            'author': author,
            'sentiment_label': predictions['labels'][0],
            'sentiment_score': float(combined_sentiment),
            'confidence': float(predictions['confidence'][0]),
            'features': features,
            'timestamp': datetime.now()
        }
    
    def analyze_batch(self, texts: List[Dict]) -> List[Dict]:
        """
        Analyze batch of texts.
        
        Args:
            texts: List of text dictionaries with 'text', 'source', 'author'
            
        Returns:
            List of sentiment analyses
        """
        results = []
        
        # Prepare batch
        text_contents = [t['text'] for t in texts]
        input_ids = self.tokenizer.batch_encode(text_contents)
        attention_masks = np.array([
            self.tokenizer.create_attention_mask(ids) for ids in input_ids
        ])
        
        # Get predictions
        predictions = self.model.predict_sentiment(input_ids, attention_masks)
        
        # Process each text
        for i, text_data in enumerate(texts):
            features = self.tokenizer.get_sentiment_features(text_data['text'])
            
            # Get author weight
            author_weight = self.influencer_multipliers.get(
                text_data.get('author', 'unknown').lower(),
                self.influencer_multipliers['default']
            )
            
            # Calculate weighted sentiment
            base_sentiment = predictions['sentiment_scores'][i]
            weighted_sentiment = base_sentiment * author_weight
            
            # Combine with features
            combined_sentiment = (
                weighted_sentiment * 0.7 +
                features['emoji_sentiment'] * 0.1 +
                (features['bullish_score'] - features['bearish_score']) * 0.2
            )
            
            result = {
                'text': text_data['text'],
                'source': text_data.get('source', 'unknown'),
                'author': text_data.get('author', 'unknown'),
                'sentiment_label': predictions['labels'][i],
                'sentiment_score': float(combined_sentiment),
                'confidence': float(predictions['confidence'][i]),
                'features': features,
                'timestamp': text_data.get('timestamp', datetime.now())
            }
            
            results.append(result)
            self.sentiment_history.append(result)
            
        return results
    
    def aggregate_sentiment(self, sentiments: List[Dict],
                          time_window: timedelta = timedelta(hours=1)) -> Dict:
        """
        Aggregate sentiments over time window.
        
        Args:
            sentiments: List of sentiment analyses
            time_window: Time window for aggregation
            
        Returns:
            Aggregated sentiment metrics
        """
        if not sentiments:
            return {
                'overall_sentiment': 0.0,
                'sentiment_trend': 0.0,
                'volume': 0,
                'sources': {}
            }
            
        # Filter by time window
        cutoff_time = datetime.now() - time_window
        recent_sentiments = [
            s for s in sentiments 
            if s['timestamp'] > cutoff_time
        ]
        
        if not recent_sentiments:
            return {
                'overall_sentiment': 0.0,
                'sentiment_trend': 0.0,
                'volume': 0,
                'sources': {}
            }
            
        # Aggregate by source
        source_sentiments = {}
        for sentiment in recent_sentiments:
            source = sentiment['source']
            if source not in source_sentiments:
                source_sentiments[source] = []
            source_sentiments[source].append(sentiment['sentiment_score'])
            
        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0
        
        for source, scores in source_sentiments.items():
            source_weight = self.source_weights.get(source, 0.1)
            source_avg = np.mean(scores)
            weighted_sum += source_avg * source_weight * len(scores)
            total_weight += source_weight * len(scores)
            
        overall_sentiment = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate trend
        if len(recent_sentiments) > 10:
            # Split into two halves
            mid_point = len(recent_sentiments) // 2
            first_half = recent_sentiments[:mid_point]
            second_half = recent_sentiments[mid_point:]
            
            first_avg = np.mean([s['sentiment_score'] for s in first_half])
            second_avg = np.mean([s['sentiment_score'] for s in second_half])
            
            sentiment_trend = second_avg - first_avg
        else:
            sentiment_trend = 0.0
            
        # Source breakdown
        source_breakdown = {}
        for source, scores in source_sentiments.items():
            source_breakdown[source] = {
                'average': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores),
                'weight': self.source_weights.get(source, 0.1)
            }
            
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_trend': sentiment_trend,
            'volume': len(recent_sentiments),
            'sources': source_breakdown,
            'bullish_ratio': sum(1 for s in recent_sentiments if s['sentiment_score'] > 0.2) / len(recent_sentiments),
            'bearish_ratio': sum(1 for s in recent_sentiments if s['sentiment_score'] < -0.2) / len(recent_sentiments),
            'extreme_sentiment': any(abs(s['sentiment_score']) > 0.8 for s in recent_sentiments)
        }
    
    def generate_trading_signal(self, aggregated_sentiment: Dict) -> Dict:
        """
        Generate trading signal from sentiment.
        
        Args:
            aggregated_sentiment: Aggregated sentiment metrics
            
        Returns:
            Trading signal
        """
        sentiment_score = aggregated_sentiment['overall_sentiment']
        sentiment_trend = aggregated_sentiment['sentiment_trend']
        volume = aggregated_sentiment['volume']
        
        # Determine signal strength
        signal_strength = abs(sentiment_score)
        
        # Adjust for volume (more data = more confidence)
        volume_factor = min(1.0, volume / 100)
        signal_strength *= volume_factor
        
        # Determine action
        if sentiment_score > 0.3 and sentiment_trend > 0.1:
            action = 'buy'
            confidence = min(0.9, signal_strength)
        elif sentiment_score < -0.3 and sentiment_trend < -0.1:
            action = 'sell'
            confidence = min(0.9, signal_strength)
        elif abs(sentiment_score) < 0.1:
            action = 'hold'
            confidence = 0.5
        else:
            # Mixed signals
            action = 'hold'
            confidence = 0.3
            
        # Check for extreme sentiment (potential reversal)
        if aggregated_sentiment.get('extreme_sentiment', False):
            # Extreme sentiment often precedes reversals
            if action == 'buy':
                confidence *= 0.7  # Reduce confidence
            elif action == 'sell':
                confidence *= 0.7
                
        return {
            'action': action,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'sentiment_trend': sentiment_trend,
            'signal_strength': signal_strength,
            'data_points': volume,
            'reasoning': self._generate_signal_reasoning(
                sentiment_score, sentiment_trend, aggregated_sentiment
            )
        }
    
    def _generate_signal_reasoning(self, sentiment_score: float,
                                 sentiment_trend: float,
                                 aggregated: Dict) -> str:
        """Generate human-readable signal reasoning."""
        reasons = []
        
        if sentiment_score > 0.3:
            reasons.append("Strong positive sentiment")
        elif sentiment_score < -0.3:
            reasons.append("Strong negative sentiment")
        else:
            reasons.append("Neutral sentiment")
            
        if sentiment_trend > 0.1:
            reasons.append("improving sentiment trend")
        elif sentiment_trend < -0.1:
            reasons.append("deteriorating sentiment trend")
            
        if aggregated.get('extreme_sentiment'):
            reasons.append("extreme sentiment detected (potential reversal)")
            
        # Source analysis
        dominant_source = max(
            aggregated['sources'].items(),
            key=lambda x: x[1]['count']
        )[0] if aggregated['sources'] else 'unknown'
        
        reasons.append(f"dominated by {dominant_source} sentiment")
        
        return "; ".join(reasons)
    
    def detect_sentiment_anomalies(self, lookback_hours: int = 24) -> List[Dict]:
        """
        Detect unusual sentiment patterns.
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            List of anomalies
        """
        if len(self.sentiment_history) < 100:
            return []
            
        # Get recent history
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_history = [
            s for s in self.sentiment_history
            if s['timestamp'] > cutoff_time
        ]
        
        if len(recent_history) < 20:
            return []
            
        # Calculate statistics
        sentiment_scores = [s['sentiment_score'] for s in recent_history]
        mean_sentiment = np.mean(sentiment_scores)
        std_sentiment = np.std(sentiment_scores)
        
        anomalies = []
        
        # Detect outliers (3 standard deviations)
        for sentiment in recent_history[-10:]:  # Check last 10
            z_score = (sentiment['sentiment_score'] - mean_sentiment) / (std_sentiment + 1e-6)
            
            if abs(z_score) > 3:
                anomalies.append({
                    'type': 'extreme_sentiment',
                    'sentiment': sentiment,
                    'z_score': z_score,
                    'severity': 'high' if abs(z_score) > 4 else 'medium'
                })
                
        # Detect rapid changes
        if len(recent_history) > 50:
            recent_avg = np.mean([s['sentiment_score'] for s in recent_history[-10:]])
            previous_avg = np.mean([s['sentiment_score'] for s in recent_history[-50:-40]])
            
            change_rate = (recent_avg - previous_avg) / (abs(previous_avg) + 0.1)
            
            if abs(change_rate) > 2:
                anomalies.append({
                    'type': 'rapid_change',
                    'change_rate': change_rate,
                    'recent_avg': recent_avg,
                    'previous_avg': previous_avg,
                    'severity': 'high' if abs(change_rate) > 3 else 'medium'
                })
                
        return anomalies
    
    def get_sentiment_dashboard(self) -> Dict:
        """
        Get comprehensive sentiment dashboard.
        
        Returns:
            Dashboard metrics
        """
        # Aggregate different time windows
        windows = {
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '24h': timedelta(hours=24)
        }
        
        dashboard = {}
        
        for window_name, window_delta in windows.items():
            aggregated = self.aggregate_sentiment(
                self.sentiment_history,
                window_delta
            )
            
            signal = self.generate_trading_signal(aggregated)
            
            dashboard[window_name] = {
                'sentiment': aggregated,
                'signal': signal,
                'data_points': aggregated['volume']
            }
            
        # Detect anomalies
        anomalies = self.detect_sentiment_anomalies()
        
        # Overall metrics
        if self.sentiment_history:
            all_scores = [s['sentiment_score'] for s in self.sentiment_history]
            dashboard['overall'] = {
                'total_analyses': len(self.sentiment_history),
                'average_sentiment': np.mean(all_scores),
                'sentiment_std': np.std(all_scores),
                'positive_ratio': sum(1 for s in all_scores if s > 0) / len(all_scores),
                'anomalies': anomalies
            }
            
        return dashboard