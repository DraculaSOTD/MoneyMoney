"""
Economic Indicators Analysis for Cryptocurrency Trading.

Implements comprehensive economic data integration and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class EconomicIndicator:
    """Individual economic indicator data."""
    indicator_name: str
    timestamp: datetime
    value: float
    previous_value: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    category: str = "general"  # monetary, fiscal, employment, inflation, etc.
    country: str = "US"
    importance: str = "medium"  # low, medium, high
    frequency: str = "monthly"  # daily, weekly, monthly, quarterly, yearly


@dataclass
class EconomicRegime:
    """Economic regime classification."""
    timestamp: datetime
    regime_type: str  # expansion, recession, recovery, peak, trough
    confidence: float
    key_indicators: List[str]
    duration_estimate: Optional[float] = None  # months
    transition_probability: Dict[str, float] = field(default_factory=dict)


@dataclass
class MacroCorrelation:
    """Correlation between crypto and macro indicators."""
    crypto_asset: str
    indicator_name: str
    correlation: float
    lag_correlation: Dict[int, float]  # lag -> correlation
    significance: float
    relationship_type: str  # positive, negative, complex
    stability: float  # how stable the relationship is over time


@dataclass
class PolicyEvent:
    """Central bank or policy event."""
    event_id: str
    timestamp: datetime
    event_type: str  # rate_decision, qe_announcement, policy_statement
    description: str
    institution: str  # fed, ecb, boe, etc.
    impact_assessment: str  # hawkish, dovish, neutral
    market_reaction_expected: str
    crypto_implications: List[str] = field(default_factory=list)


class EconomicIndicators:
    """
    Economic indicators analysis for cryptocurrency trading.
    
    Features:
    - Comprehensive economic data integration
    - Economic regime detection and classification
    - Macro-crypto correlation analysis
    - Central bank policy impact assessment
    - Leading indicator identification
    - Cross-country economic analysis
    - Risk-on/risk-off regime detection
    - Economic surprise analysis
    """
    
    def __init__(self,
                 correlation_window: int = 252,
                 regime_detection_window: int = 60,
                 countries: List[str] = None):
        """
        Initialize economic indicators analyzer.
        
        Args:
            correlation_window: Window for correlation analysis (days)
            regime_detection_window: Window for regime detection
            countries: List of countries to track
        """
        self.correlation_window = correlation_window
        self.regime_detection_window = regime_detection_window
        self.countries = countries or ['US', 'EU', 'UK', 'JP', 'CN']
        
        # Data storage
        self.indicators: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.crypto_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.policy_events: deque = deque(maxlen=200)
        
        # Analysis results
        self.economic_regimes: deque = deque(maxlen=100)
        self.macro_correlations: Dict[str, MacroCorrelation] = {}
        self.current_regime: Optional[EconomicRegime] = None
        
        # Indicator definitions and importance
        self.indicator_definitions = self._initialize_indicator_definitions()
        self.indicator_importance = self._initialize_indicator_importance()
        
        # Risk-on/risk-off indicators
        self.risk_indicators = self._initialize_risk_indicators()
        
        # Analysis caches
        self.regime_probabilities: Dict[str, float] = defaultdict(float)
        self.economic_surprises: Dict[str, float] = defaultdict(float)
    
    def _initialize_indicator_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize economic indicator definitions."""
        return {
            # Monetary Policy Indicators
            'fed_funds_rate': {
                'category': 'monetary',
                'importance': 'high',
                'frequency': 'meeting',
                'description': 'Federal Reserve policy rate',
                'crypto_impact': 'inverse'
            },
            '10y_treasury': {
                'category': 'monetary',
                'importance': 'high',
                'frequency': 'daily',
                'description': '10-year Treasury yield',
                'crypto_impact': 'complex'
            },
            'yield_curve_slope': {
                'category': 'monetary',
                'importance': 'medium',
                'frequency': 'daily',
                'description': '10Y-2Y yield spread',
                'crypto_impact': 'negative'
            },
            'real_rates': {
                'category': 'monetary',
                'importance': 'high',
                'frequency': 'daily',
                'description': 'Real interest rates (TIPS)',
                'crypto_impact': 'inverse'
            },
            
            # Inflation Indicators
            'cpi_yoy': {
                'category': 'inflation',
                'importance': 'high',
                'frequency': 'monthly',
                'description': 'Consumer Price Index YoY',
                'crypto_impact': 'positive'
            },
            'core_cpi': {
                'category': 'inflation',
                'importance': 'high',
                'frequency': 'monthly',
                'description': 'Core CPI (ex food/energy)',
                'crypto_impact': 'positive'
            },
            'pce_deflator': {
                'category': 'inflation',
                'importance': 'high',
                'frequency': 'monthly',
                'description': 'Fed preferred inflation measure',
                'crypto_impact': 'positive'
            },
            'inflation_expectations': {
                'category': 'inflation',
                'importance': 'medium',
                'frequency': 'daily',
                'description': '5Y5Y inflation expectations',
                'crypto_impact': 'positive'
            },
            
            # Growth Indicators
            'gdp_qoq': {
                'category': 'growth',
                'importance': 'high',
                'frequency': 'quarterly',
                'description': 'GDP quarter-over-quarter',
                'crypto_impact': 'positive'
            },
            'ism_manufacturing': {
                'category': 'growth',
                'importance': 'medium',
                'frequency': 'monthly',
                'description': 'ISM Manufacturing PMI',
                'crypto_impact': 'positive'
            },
            'ism_services': {
                'category': 'growth',
                'importance': 'medium',
                'frequency': 'monthly',
                'description': 'ISM Services PMI',
                'crypto_impact': 'positive'
            },
            'retail_sales': {
                'category': 'growth',
                'importance': 'medium',
                'frequency': 'monthly',
                'description': 'Retail sales MoM',
                'crypto_impact': 'positive'
            },
            
            # Employment Indicators
            'unemployment_rate': {
                'category': 'employment',
                'importance': 'high',
                'frequency': 'monthly',
                'description': 'Unemployment rate',
                'crypto_impact': 'inverse'
            },
            'nonfarm_payrolls': {
                'category': 'employment',
                'importance': 'high',
                'frequency': 'monthly',
                'description': 'Nonfarm payrolls change',
                'crypto_impact': 'positive'
            },
            'job_openings': {
                'category': 'employment',
                'importance': 'medium',
                'frequency': 'monthly',
                'description': 'JOLTS job openings',
                'crypto_impact': 'positive'
            },
            'initial_claims': {
                'category': 'employment',
                'importance': 'medium',
                'frequency': 'weekly',
                'description': 'Initial jobless claims',
                'crypto_impact': 'inverse'
            },
            
            # Market Sentiment Indicators
            'vix': {
                'category': 'sentiment',
                'importance': 'high',
                'frequency': 'daily',
                'description': 'VIX volatility index',
                'crypto_impact': 'complex'
            },
            'dollar_index': {
                'category': 'currency',
                'importance': 'high',
                'frequency': 'daily',
                'description': 'US Dollar Index (DXY)',
                'crypto_impact': 'inverse'
            },
            'gold_price': {
                'category': 'commodity',
                'importance': 'medium',
                'frequency': 'daily',
                'description': 'Gold spot price',
                'crypto_impact': 'positive'
            },
            'high_yield_spreads': {
                'category': 'credit',
                'importance': 'medium',
                'frequency': 'daily',
                'description': 'High yield credit spreads',
                'crypto_impact': 'inverse'
            }
        }
    
    def _initialize_indicator_importance(self) -> Dict[str, float]:
        """Initialize indicator importance weights."""
        importance_weights = {}
        
        for indicator, definition in self.indicator_definitions.items():
            importance = definition['importance']
            if importance == 'high':
                importance_weights[indicator] = 1.0
            elif importance == 'medium':
                importance_weights[indicator] = 0.7
            else:
                importance_weights[indicator] = 0.4
        
        return importance_weights
    
    def _initialize_risk_indicators(self) -> Dict[str, str]:
        """Initialize risk-on/risk-off indicator classifications."""
        return {
            'vix': 'risk_off',
            'high_yield_spreads': 'risk_off',
            'dollar_index': 'risk_off',
            'gold_price': 'risk_off',
            'yield_curve_slope': 'risk_on',
            'ism_manufacturing': 'risk_on',
            'ism_services': 'risk_on',
            'retail_sales': 'risk_on'
        }
    
    def add_economic_data(self,
                         indicator_name: str,
                         data: List[EconomicIndicator]) -> None:
        """
        Add economic indicator data.
        
        Args:
            indicator_name: Name of the economic indicator
            data: List of indicator data points
        """
        for indicator in data:
            # Calculate change if previous value available
            if len(self.indicators[indicator_name]) > 0:
                previous = self.indicators[indicator_name][-1]
                indicator.previous_value = previous.value
                indicator.change = indicator.value - previous.value
                
                if previous.value != 0:
                    indicator.change_percent = (indicator.change / previous.value) * 100
            
            self.indicators[indicator_name].append(indicator)
            
            # Update economic surprise
            self._update_economic_surprise(indicator_name, indicator)
        
        # Update regime analysis
        self._update_regime_analysis()
    
    def add_crypto_price_data(self,
                            symbol: str,
                            prices: List[Tuple[datetime, float]]) -> None:
        """
        Add cryptocurrency price data for correlation analysis.
        
        Args:
            symbol: Cryptocurrency symbol
            prices: List of (timestamp, price) tuples
        """
        for timestamp, price in prices:
            self.crypto_prices[symbol].append({
                'timestamp': timestamp,
                'price': price
            })
        
        # Update correlations
        self._update_macro_correlations(symbol)
    
    def add_policy_event(self, event: PolicyEvent) -> None:
        """
        Add central bank or policy event.
        
        Args:
            event: Policy event data
        """
        # Assess crypto implications
        event.crypto_implications = self._assess_crypto_implications(event)
        
        self.policy_events.append(event)
    
    def _update_economic_surprise(self,
                                indicator_name: str,
                                indicator: EconomicIndicator) -> None:
        """Update economic surprise index for indicator."""
        if indicator_name not in self.indicators or len(self.indicators[indicator_name]) < 5:
            return
        
        # Calculate historical average and standard deviation
        recent_values = [ind.value for ind in list(self.indicators[indicator_name])[-20:]]
        
        if len(recent_values) > 5:
            mean_value = np.mean(recent_values[:-1])  # Exclude current value
            std_value = np.std(recent_values[:-1])
            
            if std_value > 0:
                surprise = (indicator.value - mean_value) / std_value
                self.economic_surprises[indicator_name] = surprise
    
    def _update_regime_analysis(self) -> None:
        """Update economic regime classification."""
        # Get recent indicators for regime analysis
        regime_indicators = self._get_regime_indicators()
        
        if len(regime_indicators) < 3:
            return
        
        # Calculate regime scores
        regime_scores = self._calculate_regime_scores(regime_indicators)
        
        # Classify regime
        regime_type = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[regime_type]
        
        # Identify key indicators driving the regime
        key_indicators = self._identify_key_regime_indicators(regime_indicators, regime_type)
        
        # Create regime object
        regime = EconomicRegime(
            timestamp=datetime.now(),
            regime_type=regime_type,
            confidence=confidence,
            key_indicators=key_indicators,
            transition_probability=self._calculate_transition_probabilities(regime_scores)
        )
        
        self.economic_regimes.append(regime)
        self.current_regime = regime
        
        # Update regime probabilities
        self.regime_probabilities = regime_scores
    
    def _get_regime_indicators(self) -> Dict[str, float]:
        """Get recent values of key regime indicators."""
        regime_indicators = {}
        
        # Key indicators for regime classification
        key_indicators = [
            'fed_funds_rate', '10y_treasury', 'yield_curve_slope', 'cpi_yoy',
            'unemployment_rate', 'ism_manufacturing', 'vix', 'dollar_index'
        ]
        
        for indicator in key_indicators:
            if indicator in self.indicators and self.indicators[indicator]:
                latest = self.indicators[indicator][-1]
                regime_indicators[indicator] = latest.value
        
        return regime_indicators
    
    def _calculate_regime_scores(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Calculate scores for different economic regimes."""
        scores = {
            'expansion': 0.0,
            'recession': 0.0,
            'recovery': 0.0,
            'peak': 0.0,
            'trough': 0.0
        }
        
        # Expansion indicators
        if 'ism_manufacturing' in indicators:
            pmi = indicators['ism_manufacturing']
            if pmi > 55:
                scores['expansion'] += 0.3
            elif pmi > 50:
                scores['expansion'] += 0.1
            else:
                scores['recession'] += 0.2
        
        if 'unemployment_rate' in indicators:
            unemployment = indicators['unemployment_rate']
            if unemployment < 4:
                scores['expansion'] += 0.2
                scores['peak'] += 0.1
            elif unemployment > 6:
                scores['recession'] += 0.3
                scores['trough'] += 0.1
        
        # Yield curve analysis
        if 'yield_curve_slope' in indicators:
            slope = indicators['yield_curve_slope']
            if slope < 0:
                scores['recession'] += 0.4  # Inverted yield curve
            elif slope > 200:  # 2%+
                scores['recovery'] += 0.3
        
        # Inflation analysis
        if 'cpi_yoy' in indicators:
            inflation = indicators['cpi_yoy']
            if inflation > 4:
                scores['peak'] += 0.2
            elif inflation < 1:
                scores['trough'] += 0.2
                scores['recovery'] += 0.1
        
        # Volatility analysis
        if 'vix' in indicators:
            vix = indicators['vix']
            if vix > 30:
                scores['recession'] += 0.2
                scores['trough'] += 0.2
            elif vix < 15:
                scores['expansion'] += 0.1
                scores['peak'] += 0.1
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _identify_key_regime_indicators(self,
                                      indicators: Dict[str, float],
                                      regime_type: str) -> List[str]:
        """Identify key indicators driving the current regime."""
        key_indicators = []
        
        # Regime-specific key indicators
        regime_keys = {
            'expansion': ['ism_manufacturing', 'unemployment_rate', 'retail_sales'],
            'recession': ['yield_curve_slope', 'unemployment_rate', 'ism_manufacturing'],
            'recovery': ['yield_curve_slope', 'initial_claims', 'ism_manufacturing'],
            'peak': ['cpi_yoy', 'fed_funds_rate', 'vix'],
            'trough': ['unemployment_rate', 'vix', 'yield_curve_slope']
        }
        
        relevant_indicators = regime_keys.get(regime_type, [])
        
        for indicator in relevant_indicators:
            if indicator in indicators:
                key_indicators.append(indicator)
        
        return key_indicators[:3]  # Return top 3
    
    def _calculate_transition_probabilities(self,
                                          regime_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate probabilities of transitioning to different regimes."""
        # Simplified transition model
        transitions = {}
        
        current_regime = max(regime_scores, key=regime_scores.get)
        
        # Define transition patterns
        transition_matrix = {
            'expansion': {'peak': 0.3, 'expansion': 0.6, 'recession': 0.1},
            'peak': {'recession': 0.7, 'expansion': 0.2, 'peak': 0.1},
            'recession': {'trough': 0.4, 'recession': 0.5, 'recovery': 0.1},
            'trough': {'recovery': 0.6, 'recession': 0.3, 'trough': 0.1},
            'recovery': {'expansion': 0.5, 'recovery': 0.3, 'recession': 0.2}
        }
        
        if current_regime in transition_matrix:
            transitions = transition_matrix[current_regime]
        
        return transitions
    
    def _update_macro_correlations(self, crypto_symbol: str) -> None:
        """Update macro-crypto correlations."""
        if len(self.crypto_prices[crypto_symbol]) < self.correlation_window:
            return
        
        # Get crypto price data
        crypto_data = list(self.crypto_prices[crypto_symbol])[-self.correlation_window:]
        crypto_returns = self._calculate_returns([p['price'] for p in crypto_data])
        crypto_timestamps = [p['timestamp'] for p in crypto_data]
        
        # Calculate correlations with each indicator
        for indicator_name in self.indicators:
            if len(self.indicators[indicator_name]) < 50:
                continue
            
            correlation = self._calculate_indicator_correlation(
                crypto_returns, crypto_timestamps, indicator_name
            )
            
            if correlation:
                key = f"{crypto_symbol}_{indicator_name}"
                self.macro_correlations[key] = correlation
    
    def _calculate_indicator_correlation(self,
                                       crypto_returns: np.ndarray,
                                       crypto_timestamps: List[datetime],
                                       indicator_name: str) -> Optional[MacroCorrelation]:
        """Calculate correlation between crypto and economic indicator."""
        # Get indicator data
        indicator_data = list(self.indicators[indicator_name])
        
        if len(indicator_data) < 20:
            return None
        
        # Align data by timestamp
        aligned_pairs = []
        
        for i, crypto_timestamp in enumerate(crypto_timestamps):
            # Find closest indicator point
            closest_indicator = self._find_closest_indicator(
                indicator_data, crypto_timestamp
            )
            
            if closest_indicator:
                aligned_pairs.append((crypto_returns[i], closest_indicator.value))
        
        if len(aligned_pairs) < 20:
            return None
        
        crypto_aligned, indicator_aligned = zip(*aligned_pairs)
        
        # Calculate correlation
        correlation = np.corrcoef(crypto_aligned, indicator_aligned)[0, 1]
        
        if np.isnan(correlation):
            return None
        
        # Calculate lagged correlations
        lag_correlations = {}
        for lag in range(1, 6):  # 1-5 period lags
            if len(aligned_pairs) > lag:
                lagged_crypto = crypto_aligned[:-lag]
                lagged_indicator = indicator_aligned[lag:]
                
                if len(lagged_crypto) > 10:
                    lag_corr = np.corrcoef(lagged_crypto, lagged_indicator)[0, 1]
                    if not np.isnan(lag_corr):
                        lag_correlations[lag] = lag_corr
        
        # Classify relationship
        relationship_type = self._classify_relationship(correlation, lag_correlations)
        
        # Calculate significance (simplified)
        n = len(aligned_pairs)
        significance = abs(correlation) * np.sqrt(n - 2) / np.sqrt(1 - correlation**2) if abs(correlation) < 1 else 0
        
        # Calculate stability (how consistent correlation is over time)
        stability = self._calculate_correlation_stability(crypto_aligned, indicator_aligned)
        
        return MacroCorrelation(
            crypto_asset=crypto_timestamps[0].strftime('%Y%m%d'),  # Placeholder
            indicator_name=indicator_name,
            correlation=correlation,
            lag_correlation=lag_correlations,
            significance=significance,
            relationship_type=relationship_type,
            stability=stability
        )
    
    def _find_closest_indicator(self,
                              indicator_data: List[EconomicIndicator],
                              target_timestamp: datetime) -> Optional[EconomicIndicator]:
        """Find closest indicator observation to target timestamp."""
        closest_indicator = None
        min_time_diff = float('inf')
        
        for indicator in indicator_data:
            time_diff = abs((indicator.timestamp - target_timestamp).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_indicator = indicator
        
        # Only use if within reasonable time window (30 days)
        if min_time_diff <= 30 * 24 * 3600:
            return closest_indicator
        
        return None
    
    def _calculate_returns(self, prices: List[float]) -> np.ndarray:
        """Calculate returns from price series."""
        prices = np.array(prices)
        returns = np.diff(np.log(prices))
        return returns
    
    def _classify_relationship(self,
                             correlation: float,
                             lag_correlations: Dict[int, float]) -> str:
        """Classify the type of relationship."""
        if abs(correlation) > 0.3:
            return 'positive' if correlation > 0 else 'negative'
        
        # Check for lagged relationships
        max_lag_corr = max(lag_correlations.values()) if lag_correlations else 0
        if abs(max_lag_corr) > 0.3:
            return 'positive' if max_lag_corr > 0 else 'negative'
        
        return 'complex'
    
    def _calculate_correlation_stability(self,
                                       crypto_data: List[float],
                                       indicator_data: List[float]) -> float:
        """Calculate stability of correlation over time."""
        if len(crypto_data) < 40:
            return 0.5
        
        # Calculate rolling correlations
        window = 20
        rolling_correlations = []
        
        for i in range(window, len(crypto_data)):
            crypto_window = crypto_data[i-window:i]
            indicator_window = indicator_data[i-window:i]
            
            corr = np.corrcoef(crypto_window, indicator_window)[0, 1]
            if not np.isnan(corr):
                rolling_correlations.append(corr)
        
        if len(rolling_correlations) < 5:
            return 0.5
        
        # Stability is inverse of correlation volatility
        correlation_volatility = np.std(rolling_correlations)
        stability = max(0.0, 1.0 - correlation_volatility)
        
        return stability
    
    def _assess_crypto_implications(self, event: PolicyEvent) -> List[str]:
        """Assess cryptocurrency implications of policy events."""
        implications = []
        
        event_type = event.event_type
        impact = event.impact_assessment
        
        # Rate decisions
        if event_type == 'rate_decision':
            if impact == 'hawkish':
                implications.extend([
                    'negative_for_risk_assets',
                    'higher_discount_rates',
                    'potential_capital_outflows'
                ])
            elif impact == 'dovish':
                implications.extend([
                    'positive_for_risk_assets',
                    'lower_discount_rates',
                    'potential_inflation_hedge_demand'
                ])
        
        # QE announcements
        elif event_type == 'qe_announcement':
            implications.extend([
                'increased_money_supply',
                'potential_currency_debasement',
                'positive_for_alternative_stores_of_value'
            ])
        
        # Policy statements
        elif event_type == 'policy_statement':
            if 'inflation' in event.description.lower():
                implications.append('inflation_hedge_narrative')
            if 'digital' in event.description.lower() or 'crypto' in event.description.lower():
                implications.append('direct_crypto_mention')
        
        return implications
    
    def get_current_regime(self) -> Optional[Dict[str, Any]]:
        """Get current economic regime assessment."""
        if not self.current_regime:
            return None
        
        return {
            'regime_type': self.current_regime.regime_type,
            'confidence': self.current_regime.confidence,
            'key_indicators': self.current_regime.key_indicators,
            'transition_probabilities': self.current_regime.transition_probability,
            'timestamp': self.current_regime.timestamp.isoformat()
        }
    
    def get_macro_correlations(self, crypto_symbol: str) -> Dict[str, Dict[str, Any]]:
        """Get macro correlations for a cryptocurrency."""
        correlations = {}
        
        for key, correlation in self.macro_correlations.items():
            if crypto_symbol in key:
                indicator_name = key.replace(f"{crypto_symbol}_", "")
                
                correlations[indicator_name] = {
                    'correlation': correlation.correlation,
                    'lag_correlations': correlation.lag_correlation,
                    'significance': correlation.significance,
                    'relationship_type': correlation.relationship_type,
                    'stability': correlation.stability
                }
        
        return correlations
    
    def get_economic_surprises(self) -> Dict[str, float]:
        """Get current economic surprise index."""
        return dict(self.economic_surprises)
    
    def get_risk_regime_assessment(self) -> Dict[str, Any]:
        """Assess current risk-on/risk-off regime."""
        risk_on_score = 0.0
        risk_off_score = 0.0
        
        for indicator_name, risk_type in self.risk_indicators.items():
            if indicator_name in self.indicators and self.indicators[indicator_name]:
                latest = self.indicators[indicator_name][-1]
                surprise = self.economic_surprises.get(indicator_name, 0.0)
                
                # Weight by importance and surprise
                weight = self.indicator_importance.get(indicator_name, 0.5)
                adjusted_surprise = surprise * weight
                
                if risk_type == 'risk_on':
                    risk_on_score += adjusted_surprise
                else:
                    risk_off_score += adjusted_surprise
        
        # Normalize scores
        total_score = abs(risk_on_score) + abs(risk_off_score)
        if total_score > 0:
            risk_on_prob = abs(risk_on_score) / total_score
            risk_off_prob = abs(risk_off_score) / total_score
        else:
            risk_on_prob = risk_off_prob = 0.5
        
        # Determine regime
        if risk_on_prob > 0.6:
            regime = 'risk_on'
        elif risk_off_prob > 0.6:
            regime = 'risk_off'
        else:
            regime = 'neutral'
        
        return {
            'regime': regime,
            'risk_on_probability': risk_on_prob,
            'risk_off_probability': risk_off_prob,
            'confidence': max(risk_on_prob, risk_off_prob)
        }
    
    def get_recent_policy_events(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent policy events."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_events = [
            event for event in self.policy_events
            if event.timestamp >= cutoff_time
        ]
        
        return [
            {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'description': event.description,
                'institution': event.institution,
                'impact_assessment': event.impact_assessment,
                'crypto_implications': event.crypto_implications
            }
            for event in recent_events
        ]
    
    def get_leading_indicators_analysis(self) -> Dict[str, Any]:
        """Analyze leading economic indicators."""
        leading_indicators = [
            'yield_curve_slope', 'initial_claims', 'ism_manufacturing',
            'high_yield_spreads', 'vix'
        ]
        
        analysis = {}
        
        for indicator in leading_indicators:
            if indicator in self.indicators and len(self.indicators[indicator]) > 5:
                recent_data = list(self.indicators[indicator])[-6:]
                values = [ind.value for ind in recent_data]
                
                # Calculate trend
                trend = np.polyfit(range(len(values)), values, 1)[0]
                
                # Calculate momentum
                if len(values) > 3:
                    recent_avg = np.mean(values[-3:])
                    historical_avg = np.mean(values[:-3])
                    momentum = (recent_avg - historical_avg) / historical_avg if historical_avg != 0 else 0
                else:
                    momentum = 0
                
                analysis[indicator] = {
                    'current_value': values[-1],
                    'trend': 'rising' if trend > 0 else 'falling',
                    'momentum': momentum,
                    'signal': self._interpret_leading_indicator_signal(indicator, trend, momentum)
                }
        
        return analysis
    
    def _interpret_leading_indicator_signal(self,
                                          indicator: str,
                                          trend: float,
                                          momentum: float) -> str:
        """Interpret signal from leading indicator."""
        # Indicator-specific interpretation
        if indicator == 'yield_curve_slope':
            if trend < -0.1 and momentum < -0.05:
                return 'recession_warning'
            elif trend > 0.1:
                return 'growth_positive'
            
        elif indicator == 'initial_claims':
            if trend > 1000 and momentum > 0.05:
                return 'employment_weakness'
            elif trend < -1000:
                return 'employment_strength'
        
        elif indicator == 'vix':
            if trend > 2 and momentum > 0.1:
                return 'risk_aversion_rising'
            elif trend < -2:
                return 'risk_appetite_improving'
        
        return 'neutral'
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive economic analysis."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'current_regime': self.get_current_regime(),
            'risk_assessment': self.get_risk_regime_assessment(),
            'economic_surprises': self.get_economic_surprises(),
            'leading_indicators': self.get_leading_indicators_analysis(),
            'recent_policy_events': self.get_recent_policy_events(7)
        }
        
        # Summary assessment
        regime = analysis['current_regime']['regime_type'] if analysis['current_regime'] else 'unknown'
        risk_regime = analysis['risk_assessment']['regime']
        
        summary_signals = []
        
        if regime == 'recession':
            summary_signals.append('economic_contraction')
        elif regime == 'expansion':
            summary_signals.append('economic_growth')
        
        if risk_regime == 'risk_off':
            summary_signals.append('risk_aversion')
        elif risk_regime == 'risk_on':
            summary_signals.append('risk_appetite')
        
        analysis['summary_signals'] = summary_signals
        
        return analysis