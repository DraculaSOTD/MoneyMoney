"""
Alternative Data Coordinator for Cryptocurrency Trading.

Coordinates all alternative data sources and provides unified insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.alternative_data.social_media_analyzer import SocialMediaAnalyzer, SentimentMetrics
from models.advanced.alternative_data.onchain_analytics import OnChainAnalytics, WhaleMovement
from models.advanced.alternative_data.news_sentiment_analyzer import NewsSentimentAnalyzer, NewsEvent
from models.advanced.alternative_data.economic_indicators import EconomicIndicators, EconomicRegime


@dataclass
class AlternativeDataSignal:
    """Unified alternative data signal."""
    timestamp: datetime
    asset_symbol: str
    signal_type: str
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    sources: List[str]
    components: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketNarrative:
    """Market narrative from alternative data."""
    timestamp: datetime
    narrative_type: str
    title: str
    description: str
    supporting_evidence: List[str]
    confidence: float
    market_implications: List[str]
    affected_assets: List[str]
    duration_estimate: Optional[str] = None


@dataclass
class AnomalyDetection:
    """Anomaly detected across alternative data sources."""
    timestamp: datetime
    anomaly_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_sources: List[str]
    potential_causes: List[str]
    recommended_actions: List[str]
    false_positive_probability: float


class AlternativeDataCoordinator:
    """
    Coordinates all alternative data sources for unified insights.
    
    Features:
    - Multi-source data fusion and correlation
    - Signal aggregation and weighting
    - Cross-validation between data sources
    - Anomaly detection across sources
    - Market narrative generation
    - Predictive signal generation
    - Real-time coordination and monitoring
    - Source reliability assessment
    """
    
    def __init__(self,
                 update_frequency: float = 30.0,
                 signal_confidence_threshold: float = 0.6,
                 anomaly_detection_sensitivity: float = 2.0):
        """
        Initialize alternative data coordinator.
        
        Args:
            update_frequency: Update frequency in seconds
            signal_confidence_threshold: Minimum confidence for signals
            anomaly_detection_sensitivity: Sensitivity for anomaly detection
        """
        self.update_frequency = update_frequency
        self.signal_confidence_threshold = signal_confidence_threshold
        self.anomaly_detection_sensitivity = anomaly_detection_sensitivity
        
        # Initialize data sources
        self.social_media_analyzer = SocialMediaAnalyzer()
        self.onchain_analytics = OnChainAnalytics()
        self.news_sentiment_analyzer = NewsSentimentAnalyzer()
        self.economic_indicators = EconomicIndicators()
        
        # Analysis results
        self.unified_signals: deque = deque(maxlen=1000)
        self.market_narratives: deque = deque(maxlen=200)
        self.detected_anomalies: deque = deque(maxlen=100)
        
        # Source weights and reliability
        self.source_weights = {
            'social_media': 0.2,
            'onchain': 0.3,
            'news': 0.3,
            'economic': 0.2
        }
        self.source_reliability = {
            'social_media': 0.7,
            'onchain': 0.9,
            'news': 0.8,
            'economic': 0.85
        }
        
        # Real-time coordination
        self.is_running = False
        self.coordination_thread = None
        self.data_lock = threading.Lock()
        
        # Cross-validation matrices
        self.source_correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.signal_history: Dict[str, List] = defaultdict(list)
        
        # Market regime tracking
        self.current_market_regime: Optional[str] = None
        self.regime_confidence: float = 0.0
    
    def start_coordination(self) -> None:
        """Start real-time alternative data coordination."""
        if self.is_running:
            return
        
        self.is_running = True
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
        print("Alternative data coordination started")
    
    def stop_coordination(self) -> None:
        """Stop real-time coordination."""
        self.is_running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        print("Alternative data coordination stopped")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.is_running:
            try:
                with self.data_lock:
                    # Generate unified signals
                    self._generate_unified_signals()
                    
                    # Detect anomalies
                    self._detect_cross_source_anomalies()
                    
                    # Generate market narratives
                    self._generate_market_narratives()
                    
                    # Update source correlations
                    self._update_source_correlations()
                    
                    # Assess source reliability
                    self._update_source_reliability()
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                print(f"Error in coordination loop: {e}")
                time.sleep(self.update_frequency)
    
    def update_market_data(self,
                          asset_symbol: str,
                          market_data: Dict[str, Any],
                          timestamp: Optional[datetime] = None) -> AlternativeDataSignal:
        """
        Update all data sources with market data and generate unified signal.
        
        Args:
            asset_symbol: Asset symbol
            market_data: Market data dictionary
            timestamp: Data timestamp
            
        Returns:
            Unified alternative data signal
        """
        timestamp = timestamp or datetime.now()
        
        with self.data_lock:
            # Update each data source
            source_signals = {}
            
            # Social media signals
            if 'social_sentiment' in market_data:
                social_signal = self._extract_social_media_signal(asset_symbol, market_data['social_sentiment'])
                source_signals['social_media'] = social_signal
            
            # On-chain signals
            if 'onchain_data' in market_data:
                onchain_signal = self._extract_onchain_signal(asset_symbol, market_data['onchain_data'])
                source_signals['onchain'] = onchain_signal
            
            # News signals
            if 'news_data' in market_data:
                news_signal = self._extract_news_signal(asset_symbol, market_data['news_data'])
                source_signals['news'] = news_signal
            
            # Economic signals
            if 'economic_data' in market_data:
                economic_signal = self._extract_economic_signal(asset_symbol, market_data['economic_data'])
                source_signals['economic'] = economic_signal
            
            # Generate unified signal
            unified_signal = self._create_unified_signal(asset_symbol, source_signals, timestamp)
            
            if unified_signal.confidence >= self.signal_confidence_threshold:
                self.unified_signals.append(unified_signal)
            
            return unified_signal
    
    def _extract_social_media_signal(self, asset_symbol: str, social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signal from social media data."""
        # Get current sentiment for asset
        sentiment_metrics = self.social_media_analyzer.get_asset_sentiment(asset_symbol)
        
        if not sentiment_metrics:
            return {'signal': 0.0, 'confidence': 0.0, 'components': {}}
        
        # Calculate signal strength
        signal_strength = sentiment_metrics.influence_weighted_sentiment
        
        # Calculate confidence based on volume and consistency
        volume_factor = min(1.0, sentiment_metrics.volume_metrics.get('total_posts', 0) / 100.0)
        consistency_factor = 1.0 - sentiment_metrics.trend_indicators.get('volatility', 0.5)
        confidence = sentiment_metrics.confidence_score * volume_factor * consistency_factor
        
        # Get viral events
        viral_events = self.social_media_analyzer.get_viral_events(6)  # Last 6 hours
        viral_impact = sum(event.sentiment_shift for event in viral_events) / len(viral_events) if viral_events else 0.0
        
        return {
            'signal': np.clip(signal_strength + viral_impact * 0.3, -1.0, 1.0),
            'confidence': confidence,
            'components': {
                'sentiment_score': sentiment_metrics.sentiment_score,
                'influence_weighted': sentiment_metrics.influence_weighted_sentiment,
                'viral_events': len(viral_events),
                'trend_momentum': sentiment_metrics.trend_indicators.get('sentiment_momentum', 0.0)
            }
        }
    
    def _extract_onchain_signal(self, asset_symbol: str, onchain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signal from on-chain data."""
        # Get whale activity
        whale_summary = self.onchain_analytics.get_whale_activity_summary(network=asset_symbol.lower())
        
        if not whale_summary or whale_summary.get('total_movements', 0) == 0:
            return {'signal': 0.0, 'confidence': 0.0, 'components': {}}
        
        # Calculate signal from whale movements
        predicted_impact = whale_summary.get('predicted_market_impact', 0.0)
        
        # Get exchange flows
        flow_analysis = self.onchain_analytics.get_exchange_flow_analysis(asset_symbol.lower())
        
        flow_signal = 0.0
        if flow_analysis:
            net_flow = flow_analysis.get('net_exchange_flow', 0.0)
            # Negative net flow (outflows) is bullish
            flow_signal = -net_flow / max(abs(net_flow), 1000.0)  # Normalize
        
        # Combine signals
        combined_signal = 0.6 * predicted_impact + 0.4 * flow_signal
        
        # Calculate confidence based on data quality
        confidence = min(1.0, whale_summary.get('total_movements', 0) / 10.0) * 0.9  # High confidence for on-chain
        
        return {
            'signal': np.clip(combined_signal, -1.0, 1.0),
            'confidence': confidence,
            'components': {
                'whale_impact': predicted_impact,
                'exchange_flow_signal': flow_signal,
                'whale_movements': whale_summary.get('total_movements', 0),
                'net_flow': flow_analysis.get('net_exchange_flow', 0.0) if flow_analysis else 0.0
            }
        }
    
    def _extract_news_signal(self, asset_symbol: str, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signal from news sentiment data."""
        # Get current sentiment
        current_sentiment = self.news_sentiment_analyzer.get_current_sentiment(asset_symbol)
        
        if not current_sentiment:
            return {'signal': 0.0, 'confidence': 0.0, 'components': {}}
        
        # Base signal from sentiment
        base_signal = current_sentiment['sentiment_score']
        
        # Adjust for momentum
        momentum = current_sentiment.get('momentum', 0.0)
        trend_strength = current_sentiment.get('trend_strength', 0.0)
        
        # Weight by trend direction
        if current_sentiment['trend_direction'] == 'bullish':
            momentum_adjustment = momentum * trend_strength
        elif current_sentiment['trend_direction'] == 'bearish':
            momentum_adjustment = -momentum * trend_strength
        else:
            momentum_adjustment = 0.0
        
        combined_signal = base_signal + momentum_adjustment * 0.3
        
        # Get recent events impact
        recent_events = self.news_sentiment_analyzer.get_recent_events(6)
        event_impact = 0.0
        
        for event in recent_events:
            if asset_symbol in event.get('affected_assets', []):
                event_impact += event.get('sentiment_impact', 0.0) * event.get('confidence', 0.0)
        
        if recent_events:
            event_impact /= len(recent_events)
        
        final_signal = combined_signal + event_impact * 0.2
        
        return {
            'signal': np.clip(final_signal, -1.0, 1.0),
            'confidence': current_sentiment['confidence'],
            'components': {
                'base_sentiment': base_signal,
                'momentum': momentum,
                'trend_direction': current_sentiment['trend_direction'],
                'recent_events_impact': event_impact,
                'events_count': len(recent_events)
            }
        }
    
    def _extract_economic_signal(self, asset_symbol: str, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract signal from economic indicators."""
        # Get current regime
        current_regime = self.economic_indicators.get_current_regime()
        risk_assessment = self.economic_indicators.get_risk_regime_assessment()
        
        # Base signal from economic regime
        regime_signals = {
            'expansion': 0.3,
            'recovery': 0.5,
            'peak': -0.2,
            'recession': -0.7,
            'trough': 0.1
        }
        
        regime_signal = 0.0
        if current_regime:
            regime_type = current_regime['regime_type']
            regime_confidence = current_regime['confidence']
            regime_signal = regime_signals.get(regime_type, 0.0) * regime_confidence
        
        # Risk regime adjustment
        risk_regime = risk_assessment.get('regime', 'neutral')
        risk_adjustment = 0.0
        
        if risk_regime == 'risk_on':
            risk_adjustment = 0.2 * risk_assessment.get('confidence', 0.0)
        elif risk_regime == 'risk_off':
            risk_adjustment = -0.3 * risk_assessment.get('confidence', 0.0)
        
        # Macro correlations (if available)
        macro_correlations = self.economic_indicators.get_macro_correlations(asset_symbol)
        correlation_signal = 0.0
        
        if macro_correlations:
            # Weight correlations by significance
            weighted_correlations = []
            for indicator, corr_data in macro_correlations.items():
                correlation = corr_data['correlation']
                significance = corr_data['significance']
                stability = corr_data['stability']
                
                weighted_corr = correlation * significance * stability
                weighted_correlations.append(weighted_corr)
            
            if weighted_correlations:
                correlation_signal = np.mean(weighted_correlations) * 0.1  # Small adjustment
        
        combined_signal = regime_signal + risk_adjustment + correlation_signal
        
        # Calculate confidence
        regime_conf = current_regime['confidence'] if current_regime else 0.5
        risk_conf = risk_assessment.get('confidence', 0.5)
        overall_confidence = (regime_conf + risk_conf) / 2
        
        return {
            'signal': np.clip(combined_signal, -1.0, 1.0),
            'confidence': overall_confidence,
            'components': {
                'regime_signal': regime_signal,
                'risk_adjustment': risk_adjustment,
                'correlation_signal': correlation_signal,
                'economic_regime': current_regime['regime_type'] if current_regime else 'unknown',
                'risk_regime': risk_regime
            }
        }
    
    def _create_unified_signal(self,
                             asset_symbol: str,
                             source_signals: Dict[str, Dict[str, Any]],
                             timestamp: datetime) -> AlternativeDataSignal:
        """Create unified signal from multiple sources."""
        if not source_signals:
            return AlternativeDataSignal(
                timestamp=timestamp,
                asset_symbol=asset_symbol,
                signal_type='no_data',
                strength=0.0,
                confidence=0.0,
                sources=[],
                components={},
                recommendations=[],
                risk_assessment={}
            )
        
        # Weight signals by source reliability and confidence
        weighted_signals = []
        total_weight = 0.0
        
        for source, signal_data in source_signals.items():
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            reliability = self.source_reliability.get(source, 0.5)
            base_weight = self.source_weights.get(source, 0.25)
            
            # Adjusted weight
            weight = base_weight * reliability * confidence
            
            weighted_signals.append(signal * weight)
            total_weight += weight
        
        # Calculate unified signal
        if total_weight > 0:
            unified_strength = sum(weighted_signals) / total_weight
        else:
            unified_strength = 0.0
        
        # Calculate overall confidence
        confidences = [data['confidence'] for data in source_signals.values()]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Adjust confidence for source agreement
        signals = [data['signal'] for data in source_signals.values()]
        if len(signals) > 1:
            signal_std = np.std(signals)
            agreement_factor = max(0.5, 1.0 - signal_std)  # Lower std = higher agreement
            overall_confidence *= agreement_factor
        
        # Classify signal type
        signal_type = self._classify_signal_type(unified_strength, source_signals)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(unified_strength, signal_type, source_signals)
        
        # Risk assessment
        risk_assessment = self._assess_signal_risk(unified_strength, source_signals, overall_confidence)
        
        return AlternativeDataSignal(
            timestamp=timestamp,
            asset_symbol=asset_symbol,
            signal_type=signal_type,
            strength=unified_strength,
            confidence=overall_confidence,
            sources=list(source_signals.keys()),
            components={source: data['components'] for source, data in source_signals.items()},
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            metadata={
                'source_weights': self.source_weights,
                'source_reliability': {k: self.source_reliability[k] for k in source_signals.keys()}
            }
        )
    
    def _classify_signal_type(self, strength: float, source_signals: Dict[str, Dict[str, Any]]) -> str:
        """Classify the type of unified signal."""
        if abs(strength) < 0.1:
            return 'neutral'
        
        # Check for consensus
        signals = [data['signal'] for data in source_signals.values()]
        positive_signals = sum(1 for s in signals if s > 0.1)
        negative_signals = sum(1 for s in signals if s < -0.1)
        
        if strength > 0.3:
            if positive_signals >= len(signals) * 0.75:
                return 'strong_bullish'
            else:
                return 'bullish'
        elif strength > 0.1:
            return 'weak_bullish'
        elif strength < -0.3:
            if negative_signals >= len(signals) * 0.75:
                return 'strong_bearish'
            else:
                return 'bearish'
        elif strength < -0.1:
            return 'weak_bearish'
        
        return 'mixed'
    
    def _generate_recommendations(self,
                                strength: float,
                                signal_type: str,
                                source_signals: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate trading recommendations based on signal."""
        recommendations = []
        
        # Base recommendations
        if signal_type in ['strong_bullish', 'bullish']:
            recommendations.extend([
                'Consider increasing position size',
                'Monitor for entry opportunities',
                'Set trailing stop losses'
            ])
        elif signal_type in ['strong_bearish', 'bearish']:
            recommendations.extend([
                'Consider reducing position size',
                'Monitor for exit opportunities',
                'Implement hedging strategies'
            ])
        elif signal_type == 'mixed':
            recommendations.extend([
                'Wait for clearer signals',
                'Maintain current positions',
                'Monitor for directional clarity'
            ])
        
        # Source-specific recommendations
        if 'onchain' in source_signals:
            onchain_components = source_signals['onchain']['components']
            if onchain_components.get('whale_movements', 0) > 5:
                recommendations.append('High whale activity - monitor for volatility')
            
            net_flow = onchain_components.get('net_flow', 0)
            if abs(net_flow) > 1000:
                direction = 'inflows' if net_flow > 0 else 'outflows'
                recommendations.append(f'Significant exchange {direction} detected')
        
        if 'news' in source_signals:
            news_components = source_signals['news']['components']
            if news_components.get('events_count', 0) > 2:
                recommendations.append('High news activity - monitor for event-driven moves')
        
        return recommendations
    
    def _assess_signal_risk(self,
                          strength: float,
                          source_signals: Dict[str, Dict[str, Any]],
                          confidence: float) -> Dict[str, float]:
        """Assess risk factors for the signal."""
        risk_factors = {}
        
        # Signal strength risk
        risk_factors['signal_risk'] = min(1.0, abs(strength))
        
        # Confidence risk (inverse)
        risk_factors['confidence_risk'] = 1.0 - confidence
        
        # Source disagreement risk
        signals = [data['signal'] for data in source_signals.values()]
        if len(signals) > 1:
            disagreement = np.std(signals)
            risk_factors['disagreement_risk'] = min(1.0, disagreement)
        else:
            risk_factors['disagreement_risk'] = 0.5
        
        # Data quality risk
        avg_confidence = np.mean([data['confidence'] for data in source_signals.values()])
        risk_factors['data_quality_risk'] = 1.0 - avg_confidence
        
        # Source diversity risk
        source_count = len(source_signals)
        risk_factors['diversity_risk'] = max(0.0, 1.0 - source_count / 4.0)  # Ideal: 4 sources
        
        return risk_factors
    
    def _generate_unified_signals(self) -> None:
        """Generate unified signals for all tracked assets."""
        # This would be called periodically to generate signals
        # Implementation depends on having real-time data feeds
        pass
    
    def _detect_cross_source_anomalies(self) -> None:
        """Detect anomalies across multiple data sources."""
        if len(self.unified_signals) < 10:
            return
        
        recent_signals = list(self.unified_signals)[-20:]
        
        # Check for signal consistency anomalies
        for asset in set(signal.asset_symbol for signal in recent_signals):
            asset_signals = [s for s in recent_signals if s.asset_symbol == asset]
            
            if len(asset_signals) > 5:
                self._detect_signal_anomalies(asset, asset_signals)
    
    def _detect_signal_anomalies(self, asset: str, signals: List[AlternativeDataSignal]) -> None:
        """Detect anomalies in signals for a specific asset."""
        # Check for sudden signal reversals
        strengths = [s.strength for s in signals]
        
        if len(strengths) > 3:
            recent_change = abs(strengths[-1] - strengths[-3])
            historical_volatility = np.std(strengths[:-1])
            
            if recent_change > historical_volatility * self.anomaly_detection_sensitivity:
                anomaly = AnomalyDetection(
                    timestamp=signals[-1].timestamp,
                    anomaly_type='signal_reversal',
                    severity='medium',
                    description=f'Sudden signal reversal detected for {asset}',
                    affected_sources=[],
                    potential_causes=['market_shock', 'data_error', 'regime_change'],
                    recommended_actions=['verify_data_sources', 'check_market_events'],
                    false_positive_probability=0.3
                )
                
                self.detected_anomalies.append(anomaly)
        
        # Check for source disagreement anomalies
        latest_signal = signals[-1]
        source_signals = []
        
        for source, components in latest_signal.components.items():
            if 'signal' in str(components):  # Simplified check
                source_signals.append(source)
        
        if len(source_signals) > 2:
            # This would implement more sophisticated disagreement detection
            pass
    
    def _generate_market_narratives(self) -> None:
        """Generate market narratives from alternative data."""
        if len(self.unified_signals) < 5:
            return
        
        recent_signals = list(self.unified_signals)[-10:]
        
        # Group signals by asset
        asset_signals = defaultdict(list)
        for signal in recent_signals:
            asset_signals[signal.asset_symbol].append(signal)
        
        for asset, signals in asset_signals.items():
            if len(signals) >= 3:
                narrative = self._create_asset_narrative(asset, signals)
                if narrative:
                    self.market_narratives.append(narrative)
    
    def _create_asset_narrative(self, asset: str, signals: List[AlternativeDataSignal]) -> Optional[MarketNarrative]:
        """Create market narrative for an asset."""
        # Analyze signal patterns
        strengths = [s.strength for s in signals]
        avg_strength = np.mean(strengths)
        trend = np.polyfit(range(len(strengths)), strengths, 1)[0]
        
        # Determine narrative type and description
        if avg_strength > 0.3 and trend > 0:
            narrative_type = 'bullish_momentum'
            title = f'{asset} showing strong bullish momentum'
            description = 'Multiple alternative data sources confirm positive sentiment and accumulation patterns'
        elif avg_strength < -0.3 and trend < 0:
            narrative_type = 'bearish_momentum'
            title = f'{asset} under bearish pressure'
            description = 'Alternative data indicates negative sentiment and distribution patterns'
        elif abs(trend) > 0.1:
            narrative_type = 'trend_reversal'
            title = f'{asset} potential trend reversal'
            description = 'Alternative data suggests changing market dynamics'
        else:
            return None
        
        # Collect supporting evidence
        supporting_evidence = []
        latest_signal = signals[-1]
        
        if 'social_media' in latest_signal.sources:
            supporting_evidence.append('Social media sentiment analysis')
        if 'onchain' in latest_signal.sources:
            supporting_evidence.append('On-chain transaction analysis')
        if 'news' in latest_signal.sources:
            supporting_evidence.append('News sentiment analysis')
        if 'economic' in latest_signal.sources:
            supporting_evidence.append('Economic indicators analysis')
        
        # Market implications
        market_implications = []
        if narrative_type == 'bullish_momentum':
            market_implications.extend(['potential_price_appreciation', 'increased_buying_interest'])
        elif narrative_type == 'bearish_momentum':
            market_implications.extend(['potential_price_decline', 'increased_selling_pressure'])
        elif narrative_type == 'trend_reversal':
            market_implications.extend(['increased_volatility', 'directional_uncertainty'])
        
        return MarketNarrative(
            timestamp=signals[-1].timestamp,
            narrative_type=narrative_type,
            title=title,
            description=description,
            supporting_evidence=supporting_evidence,
            confidence=np.mean([s.confidence for s in signals]),
            market_implications=market_implications,
            affected_assets=[asset]
        )
    
    def _update_source_correlations(self) -> None:
        """Update correlations between different data sources."""
        if len(self.unified_signals) < 20:
            return
        
        # This would implement correlation analysis between sources
        # For now, we'll use a simplified placeholder
        
        recent_signals = list(self.unified_signals)[-50:]
        
        # Group by asset and analyze source correlations
        asset_groups = defaultdict(list)
        for signal in recent_signals:
            asset_groups[signal.asset_symbol].append(signal)
        
        for asset, signals in asset_groups.items():
            if len(signals) > 10:
                self._calculate_asset_source_correlations(asset, signals)
    
    def _calculate_asset_source_correlations(self, asset: str, signals: List[AlternativeDataSignal]) -> None:
        """Calculate source correlations for a specific asset."""
        # Extract source signals
        source_data = defaultdict(list)
        
        for signal in signals:
            for source in signal.sources:
                # This would extract the actual signal values from each source
                # Simplified for now
                source_data[source].append(signal.strength)
        
        # Calculate correlations between sources
        sources = list(source_data.keys())
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                if len(source_data[source1]) == len(source_data[source2]) and len(source_data[source1]) > 5:
                    correlation = np.corrcoef(source_data[source1], source_data[source2])[0, 1]
                    if not np.isnan(correlation):
                        self.source_correlations[source1][source2] = correlation
                        self.source_correlations[source2][source1] = correlation
    
    def _update_source_reliability(self) -> None:
        """Update reliability scores for data sources."""
        # This would implement adaptive reliability scoring
        # based on prediction accuracy and consistency
        pass
    
    def get_current_signals(self, asset_symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current alternative data signals."""
        recent_signals = list(self.unified_signals)[-20:] if self.unified_signals else []
        
        if asset_symbol:
            recent_signals = [s for s in recent_signals if s.asset_symbol == asset_symbol]
        
        return [
            {
                'timestamp': signal.timestamp.isoformat(),
                'asset_symbol': signal.asset_symbol,
                'signal_type': signal.signal_type,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'sources': signal.sources,
                'recommendations': signal.recommendations,
                'risk_assessment': signal.risk_assessment
            }
            for signal in recent_signals
        ]
    
    def get_market_narratives(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent market narratives."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_narratives = [
            narrative for narrative in self.market_narratives
            if narrative.timestamp >= cutoff_time
        ]
        
        return [
            {
                'timestamp': narrative.timestamp.isoformat(),
                'narrative_type': narrative.narrative_type,
                'title': narrative.title,
                'description': narrative.description,
                'supporting_evidence': narrative.supporting_evidence,
                'confidence': narrative.confidence,
                'market_implications': narrative.market_implications,
                'affected_assets': narrative.affected_assets
            }
            for narrative in recent_narratives
        ]
    
    def get_anomaly_alerts(self, hours: int = 6) -> List[Dict[str, Any]]:
        """Get recent anomaly alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_anomalies = [
            anomaly for anomaly in self.detected_anomalies
            if anomaly.timestamp >= cutoff_time
        ]
        
        return [
            {
                'timestamp': anomaly.timestamp.isoformat(),
                'anomaly_type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'description': anomaly.description,
                'affected_sources': anomaly.affected_sources,
                'potential_causes': anomaly.potential_causes,
                'recommended_actions': anomaly.recommended_actions,
                'false_positive_probability': anomaly.false_positive_probability
            }
            for anomaly in recent_anomalies
        ]
    
    def get_source_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all data sources."""
        return {
            'source_weights': self.source_weights,
            'source_reliability': self.source_reliability,
            'source_correlations': dict(self.source_correlations),
            'data_sources': {
                'social_media': {
                    'status': 'active',
                    'last_update': datetime.now().isoformat()
                },
                'onchain': {
                    'status': 'active', 
                    'last_update': datetime.now().isoformat()
                },
                'news': {
                    'status': 'active',
                    'last_update': datetime.now().isoformat()
                },
                'economic': {
                    'status': 'active',
                    'last_update': datetime.now().isoformat()
                }
            }
        }
    
    def get_comprehensive_analysis(self, asset_symbol: str) -> Dict[str, Any]:
        """Get comprehensive alternative data analysis for an asset."""
        analysis = {
            'asset_symbol': asset_symbol,
            'timestamp': datetime.now().isoformat(),
            'current_signals': self.get_current_signals(asset_symbol),
            'market_narratives': [n for n in self.get_market_narratives() 
                                if asset_symbol in n['affected_assets']],
            'anomaly_alerts': self.get_anomaly_alerts(),
            'source_performance': self.get_source_performance_summary()
        }
        
        # Add summary assessment
        if analysis['current_signals']:
            latest_signal = analysis['current_signals'][-1]
            analysis['summary'] = {
                'overall_signal': latest_signal['signal_type'],
                'signal_strength': latest_signal['strength'],
                'confidence': latest_signal['confidence'],
                'primary_sources': latest_signal['sources'],
                'key_recommendations': latest_signal['recommendations'][:3]
            }
        else:
            analysis['summary'] = {
                'overall_signal': 'no_data',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'primary_sources': [],
                'key_recommendations': []
            }
        
        return analysis