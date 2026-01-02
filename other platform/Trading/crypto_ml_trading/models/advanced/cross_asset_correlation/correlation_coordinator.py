"""
Cross-Asset Correlation Coordinator.

Coordinates all correlation analysis components and provides unified insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.cross_asset_correlation.correlation_analyzer import CorrelationAnalyzer, CorrelationMatrix
from models.advanced.cross_asset_correlation.dynamic_correlation_model import DynamicCorrelationModel, DynamicCorrelationState
from models.advanced.cross_asset_correlation.spillover_analyzer import SpilloverAnalyzer, SpilloverIndex
from models.advanced.cross_asset_correlation.macro_factor_model import MacroFactorModel, FactorExposure


@dataclass
class CorrelationSignal:
    """Unified correlation signal."""
    timestamp: datetime
    signal_type: str
    strength: float
    confidence: float
    components: Dict[str, Any]
    recommendations: List[str]
    risk_implications: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossAssetRegime:
    """Cross-asset market regime."""
    timestamp: datetime
    regime_type: str  # 'risk_on', 'risk_off', 'crisis', 'normal', 'rotation'
    confidence: float
    regime_drivers: List[str]
    affected_assets: List[str]
    expected_duration: Optional[float] = None
    transition_probability: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioImplication:
    """Portfolio implications from correlation analysis."""
    timestamp: datetime
    diversification_benefit: float
    concentration_risk: float
    optimal_weights: Dict[str, float]
    risk_budget_allocation: Dict[str, float]
    hedging_recommendations: List[str]
    rebalancing_urgency: str


class CorrelationCoordinator:
    """
    Coordinates all cross-asset correlation analysis components.
    
    Features:
    - Unified correlation signal generation
    - Cross-asset regime detection
    - Portfolio diversification analysis
    - Risk factor decomposition
    - Hedge ratio optimization
    - Real-time correlation monitoring
    """
    
    def __init__(self,
                 update_frequency: float = 5.0,
                 regime_detection_window: int = 60,
                 portfolio_rebalance_threshold: float = 0.1):
        """
        Initialize correlation coordinator.
        
        Args:
            update_frequency: Update frequency in seconds
            regime_detection_window: Window for regime detection
            portfolio_rebalance_threshold: Threshold for rebalancing signals
        """
        self.update_frequency = update_frequency
        self.regime_detection_window = regime_detection_window
        self.portfolio_rebalance_threshold = portfolio_rebalance_threshold
        
        # Initialize components
        self.correlation_analyzer = CorrelationAnalyzer()
        self.dynamic_model = DynamicCorrelationModel()
        self.spillover_analyzer = SpilloverAnalyzer()
        self.macro_factor_model = MacroFactorModel()
        
        # Analysis results
        self.correlation_signals: deque = deque(maxlen=500)
        self.regime_history: deque = deque(maxlen=200)
        self.portfolio_implications: deque = deque(maxlen=100)
        
        # Current state
        self.current_regime: Optional[CrossAssetRegime] = None
        self.current_signal: Optional[CorrelationSignal] = None
        
        # Component weights for signal aggregation
        self.component_weights = {
            'correlation_analyzer': 0.25,
            'dynamic_model': 0.30,
            'spillover_analyzer': 0.25,
            'macro_factor_model': 0.20
        }
        
        # Asset universe and categories
        self.asset_universe: Dict[str, str] = {}  # symbol -> asset_class
        self.portfolio_weights: Dict[str, float] = {}
        
        # Real-time coordination
        self.is_running = False
        self.coordination_thread = None
        self.data_lock = threading.Lock()
        
    def add_asset_universe(self,
                          asset_mapping: Dict[str, str],
                          initial_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Define the asset universe for analysis.
        
        Args:
            asset_mapping: Dictionary of symbol -> asset_class
            initial_weights: Initial portfolio weights
        """
        self.asset_universe = asset_mapping.copy()
        
        if initial_weights:
            self.portfolio_weights = initial_weights.copy()
        else:
            # Equal weights
            n_assets = len(asset_mapping)
            self.portfolio_weights = {symbol: 1.0/n_assets for symbol in asset_mapping}
    
    def update_market_data(self,
                          market_data: Dict[str, Dict[str, Any]],
                          timestamp: Optional[datetime] = None) -> CorrelationSignal:
        """
        Update all components with new market data.
        
        Args:
            market_data: Dictionary of symbol -> data (prices, returns, etc.)
            timestamp: Data timestamp
            
        Returns:
            Unified correlation signal
        """
        timestamp = timestamp or datetime.now()
        
        with self.data_lock:
            # Update correlation analyzer
            self._update_correlation_analyzer(market_data, timestamp)
            
            # Update dynamic correlation model
            self._update_dynamic_model(market_data, timestamp)
            
            # Update spillover analyzer
            self._update_spillover_analyzer(market_data, timestamp)
            
            # Update macro factor model
            self._update_macro_factor_model(market_data, timestamp)
        
        # Generate unified signal
        signal = self._generate_unified_signal(timestamp)
        
        # Update regime detection
        self._update_regime_detection(timestamp)
        
        # Update portfolio implications
        self._update_portfolio_implications(timestamp)
        
        return signal
    
    def _update_correlation_analyzer(self,
                                   market_data: Dict[str, Dict[str, Any]],
                                   timestamp: datetime) -> None:
        """Update correlation analyzer with new data."""
        for symbol, data in market_data.items():
            if symbol in self.asset_universe:
                asset_class = self.asset_universe[symbol]
                
                if 'prices' in data and 'timestamps' in data:
                    prices = np.array(data['prices'])
                    timestamps = data['timestamps']
                    
                    self.correlation_analyzer.add_asset_data(
                        symbol, asset_class, prices, timestamps
                    )
                elif 'price' in data:
                    # Single price update
                    self.correlation_analyzer.update_asset_data(
                        symbol, data['price'], timestamp
                    )
    
    def _update_dynamic_model(self,
                            market_data: Dict[str, Dict[str, Any]],
                            timestamp: datetime) -> None:
        """Update dynamic correlation model."""
        # Extract returns for dynamic model
        returns_data = {}
        
        for symbol, data in market_data.items():
            if symbol in self.asset_universe and 'returns' in data:
                returns_data[symbol] = data['returns']
            elif symbol in self.asset_universe and 'return' in data:
                returns_data[symbol] = data['return']
        
        if len(returns_data) >= 2:
            if not self.dynamic_model.is_fitted:
                # Initial fitting if we have enough historical data
                symbols = list(returns_data.keys())
                returns_matrix = []
                timestamps_list = []
                
                # Try to get historical data from correlation analyzer
                for symbol in symbols:
                    if symbol in self.correlation_analyzer.asset_data:
                        asset_data = self.correlation_analyzer.asset_data[symbol]
                        if len(asset_data.returns) >= 100:
                            returns_matrix.append(asset_data.returns[-100:])
                            timestamps_list = asset_data.timestamps[-100:]
                
                if len(returns_matrix) >= 2:
                    returns_matrix = np.column_stack(returns_matrix)
                    self.dynamic_model.fit(returns_matrix, symbols, timestamps_list)
            
            if self.dynamic_model.is_fitted:
                # Update with new returns
                symbols = self.dynamic_model.asset_symbols
                new_returns = np.array([returns_data.get(symbol, 0) for symbol in symbols])
                self.dynamic_model.update(new_returns, timestamp)
    
    def _update_spillover_analyzer(self,
                                 market_data: Dict[str, Dict[str, Any]],
                                 timestamp: datetime) -> None:
        """Update spillover analyzer."""
        # Prepare data for spillover analysis
        spillover_data = {}
        
        for symbol, data in market_data.items():
            if symbol in self.asset_universe:
                market_category = self.asset_universe[symbol]
                
                if 'return' in data:
                    spillover_data[symbol] = data['return']
                elif 'returns' in data and len(data['returns']) > 0:
                    spillover_data[symbol] = data['returns'][-1]
        
        if spillover_data:
            self.spillover_analyzer.update_real_time(spillover_data, timestamp)
    
    def _update_macro_factor_model(self,
                                 market_data: Dict[str, Dict[str, Any]],
                                 timestamp: datetime) -> None:
        """Update macro factor model."""
        # Add asset returns
        for symbol, data in market_data.items():
            if symbol in self.asset_universe:
                if 'returns' in data and 'timestamps' in data:
                    returns = np.array(data['returns'])
                    timestamps = data['timestamps']
                    self.macro_factor_model.add_asset_returns(symbol, returns, timestamps)
        
        # Check for macro factor data in market_data
        for symbol, data in market_data.items():
            if symbol.startswith('macro_'):  # Convention for macro factors
                factor_name = symbol.replace('macro_', '')
                
                if 'values' in data and 'timestamps' in data:
                    values = np.array(data['values'])
                    timestamps = data['timestamps']
                    self.macro_factor_model.add_factor_data(factor_name, values, timestamps)
    
    def _generate_unified_signal(self, timestamp: datetime) -> CorrelationSignal:
        """Generate unified correlation signal."""
        components = {}
        confidence_scores = {}
        
        # Correlation analyzer component
        corr_summary = self.correlation_analyzer.get_correlation_summary()
        if corr_summary.get('assets_tracked', 0) > 0:
            corr_signal = self._extract_correlation_signal(corr_summary)
            components['correlation_analysis'] = corr_signal
            confidence_scores['correlation_analysis'] = 0.8
        
        # Dynamic correlation model component
        if self.dynamic_model.is_fitted:
            dynamic_signal = self._extract_dynamic_signal()
            components['dynamic_correlation'] = dynamic_signal
            confidence_scores['dynamic_correlation'] = 0.9
        
        # Spillover analyzer component
        spillover_summary = self.spillover_analyzer.get_spillover_summary()
        if spillover_summary.get('assets_analyzed', 0) > 0:
            spillover_signal = self._extract_spillover_signal(spillover_summary)
            components['spillover_analysis'] = spillover_signal
            confidence_scores['spillover_analysis'] = 0.7
        
        # Macro factor model component
        if self.macro_factor_model.is_fitted:
            factor_signal = self._extract_factor_signal()
            components['macro_factors'] = factor_signal
            confidence_scores['macro_factors'] = 0.6
        
        # Aggregate signals
        if components:
            aggregated_signal = self._aggregate_component_signals(components, confidence_scores)
            overall_confidence = np.mean(list(confidence_scores.values()))
        else:
            aggregated_signal = 0.0
            overall_confidence = 0.1
        
        # Determine signal type and recommendations
        signal_type, recommendations, risk_implications = self._classify_correlation_signal(
            aggregated_signal, components
        )
        
        signal = CorrelationSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            strength=abs(aggregated_signal),
            confidence=overall_confidence,
            components=components,
            recommendations=recommendations,
            risk_implications=risk_implications,
            metadata={
                'component_weights': self.component_weights,
                'asset_universe_size': len(self.asset_universe)
            }
        )
        
        self.correlation_signals.append(signal)
        self.current_signal = signal
        
        return signal
    
    def _extract_correlation_signal(self, corr_summary: Dict[str, Any]) -> float:
        """Extract signal from correlation analysis."""
        if 'latest_analysis' not in corr_summary:
            return 0.0
        
        latest = corr_summary['latest_analysis']
        concentration = latest.get('concentration_metrics', {})
        
        # High concentration = negative signal for diversification
        avg_correlation = concentration.get('avg_correlation', 0.5)
        eigenvalue_concentration = concentration.get('eigenvalue_concentration', 0.5)
        
        # Normalize to -1 to 1 scale
        signal = -2 * (avg_correlation - 0.5) - (eigenvalue_concentration - 0.5)
        
        return np.clip(signal, -1.0, 1.0)
    
    def _extract_dynamic_signal(self) -> float:
        """Extract signal from dynamic correlation model."""
        current_corr = self.dynamic_model.get_current_correlation_matrix()
        
        if current_corr is None:
            return 0.0
        
        # Calculate average off-diagonal correlation
        n = current_corr.shape[0]
        if n < 2:
            return 0.0
        
        off_diagonal = current_corr[np.triu_indices(n, k=1)]
        avg_correlation = np.mean(off_diagonal)
        
        # Compare with historical average if available
        if len(self.dynamic_model.correlation_states) > 10:
            historical_correlations = []
            for state in list(self.dynamic_model.correlation_states)[-10:]:
                hist_matrix = state.correlation_matrix
                if hist_matrix.shape == current_corr.shape:
                    hist_off_diag = hist_matrix[np.triu_indices(n, k=1)]
                    historical_correlations.append(np.mean(hist_off_diag))
            
            if historical_correlations:
                hist_avg = np.mean(historical_correlations)
                signal = -(avg_correlation - hist_avg) * 5  # Scale factor
                return np.clip(signal, -1.0, 1.0)
        
        # Default: negative signal for high correlation
        return np.clip(-2 * (avg_correlation - 0.5), -1.0, 1.0)
    
    def _extract_spillover_signal(self, spillover_summary: Dict[str, Any]) -> float:
        """Extract signal from spillover analysis."""
        if 'latest_spillover' not in spillover_summary:
            return 0.0
        
        latest = spillover_summary['latest_spillover']
        total_spillover = latest.get('total_spillover', 0)
        
        # High spillover = negative signal
        signal = -total_spillover / 50.0  # Normalize assuming max spillover ~50%
        
        return np.clip(signal, -1.0, 1.0)
    
    def _extract_factor_signal(self) -> float:
        """Extract signal from macro factor model."""
        factor_summary = self.macro_factor_model.get_factor_summary()
        
        if 'model_performance' not in factor_summary:
            return 0.0
        
        performance = factor_summary['model_performance']
        avg_systematic_risk = performance.get('avg_systematic_risk', 0.5)
        
        # High systematic risk = negative signal for diversification
        signal = -2 * (avg_systematic_risk - 0.5)
        
        return np.clip(signal, -1.0, 1.0)
    
    def _aggregate_component_signals(self,
                                   components: Dict[str, float],
                                   confidences: Dict[str, float]) -> float:
        """Aggregate component signals."""
        total_weight = 0
        weighted_sum = 0
        
        for component, signal in components.items():
            # Map component names to weights
            component_key = None
            if 'correlation' in component:
                component_key = 'correlation_analyzer'
            elif 'dynamic' in component:
                component_key = 'dynamic_model'
            elif 'spillover' in component:
                component_key = 'spillover_analyzer'
            elif 'macro' in component or 'factor' in component:
                component_key = 'macro_factor_model'
            
            if component_key:
                weight = self.component_weights.get(component_key, 1.0)
                confidence = confidences.get(component, 0.5)
                
                adjusted_weight = weight * confidence
                weighted_sum += signal * adjusted_weight
                total_weight += adjusted_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _classify_correlation_signal(self,
                                   signal: float,
                                   components: Dict[str, float]) -> Tuple[str, List[str], Dict[str, float]]:
        """Classify correlation signal and generate recommendations."""
        recommendations = []
        risk_implications = {}
        
        # Determine signal type
        if signal > 0.3:
            signal_type = 'diversification_opportunity'
            recommendations.extend([
                'Good diversification conditions',
                'Consider rebalancing to take advantage of low correlations',
                'Increase position sizes in uncorrelated assets'
            ])
            risk_implications = {
                'portfolio_risk': 0.3,  # Lower risk due to diversification
                'concentration_risk': 0.2,
                'tail_risk': 0.4
            }
        
        elif signal > 0.1:
            signal_type = 'neutral_correlation'
            recommendations.extend([
                'Normal correlation environment',
                'Maintain current portfolio allocation',
                'Monitor for changes in correlation structure'
            ])
            risk_implications = {
                'portfolio_risk': 0.5,
                'concentration_risk': 0.4,
                'tail_risk': 0.5
            }
        
        elif signal > -0.2:
            signal_type = 'increasing_correlation'
            recommendations.extend([
                'Rising correlations detected',
                'Consider reducing position sizes',
                'Review hedging strategies'
            ])
            risk_implications = {
                'portfolio_risk': 0.7,
                'concentration_risk': 0.6,
                'tail_risk': 0.7
            }
        
        else:
            signal_type = 'high_correlation_regime'
            recommendations.extend([
                'High correlation regime - limited diversification benefits',
                'Reduce risk exposure',
                'Implement defensive strategies',
                'Consider alternative asset classes'
            ])
            risk_implications = {
                'portfolio_risk': 0.9,
                'concentration_risk': 0.8,
                'tail_risk': 0.9
            }
        
        # Add component-specific recommendations
        if 'spillover_analysis' in components and components['spillover_analysis'] < -0.5:
            recommendations.append('High spillover effects detected - monitor contagion risk')
        
        if 'macro_factors' in components and components['macro_factors'] < -0.5:
            recommendations.append('Macro factors showing high systematic risk')
        
        return signal_type, recommendations, risk_implications
    
    def _update_regime_detection(self, timestamp: datetime) -> None:
        """Update cross-asset regime detection."""
        if len(self.correlation_signals) < 10:
            return
        
        # Analyze recent signals for regime classification
        recent_signals = list(self.correlation_signals)[-20:]
        
        # Calculate regime indicators
        avg_signal = np.mean([s.strength for s in recent_signals])
        signal_trend = self._calculate_signal_trend(recent_signals)
        
        # Get component-specific indicators
        spillover_summary = self.spillover_analyzer.get_spillover_summary()
        corr_regime = self.correlation_analyzer.get_correlation_regime()
        
        # Classify regime
        regime_type = self._classify_cross_asset_regime(
            avg_signal, signal_trend, spillover_summary, corr_regime
        )
        
        # Identify regime drivers
        regime_drivers = self._identify_regime_drivers(recent_signals[-1])
        
        # Affected assets
        affected_assets = list(self.asset_universe.keys())
        
        # Calculate regime confidence
        confidence = self._calculate_regime_confidence(recent_signals)
        
        regime = CrossAssetRegime(
            timestamp=timestamp,
            regime_type=regime_type,
            confidence=confidence,
            regime_drivers=regime_drivers,
            affected_assets=affected_assets
        )
        
        self.regime_history.append(regime)
        self.current_regime = regime
    
    def _calculate_signal_trend(self, signals: List[CorrelationSignal]) -> float:
        """Calculate trend in correlation signals."""
        if len(signals) < 5:
            return 0.0
        
        strengths = [s.strength for s in signals]
        x = np.arange(len(strengths))
        
        # Linear trend
        trend = np.polyfit(x, strengths, 1)[0]
        
        return trend
    
    def _classify_cross_asset_regime(self,
                                   avg_signal: float,
                                   signal_trend: float,
                                   spillover_summary: Dict,
                                   corr_regime: Optional[str]) -> str:
        """Classify cross-asset market regime."""
        # Get spillover indicators
        total_spillover = 0
        if 'latest_spillover' in spillover_summary:
            total_spillover = spillover_summary['latest_spillover'].get('total_spillover', 0)
        
        # Regime classification logic
        if total_spillover > 40 or corr_regime == 'crisis':
            return 'crisis'
        elif avg_signal < -0.5 and signal_trend < -0.1:
            return 'risk_off'
        elif avg_signal > 0.3 and signal_trend > 0.1:
            return 'risk_on'
        elif abs(signal_trend) > 0.2:
            return 'rotation'
        else:
            return 'normal'
    
    def _identify_regime_drivers(self, latest_signal: CorrelationSignal) -> List[str]:
        """Identify drivers of current regime."""
        drivers = []
        
        components = latest_signal.components
        
        # Check which components are contributing most
        if 'spillover_analysis' in components and abs(components['spillover_analysis']) > 0.5:
            drivers.append('spillover_effects')
        
        if 'correlation_analysis' in components and abs(components['correlation_analysis']) > 0.5:
            drivers.append('correlation_structure')
        
        if 'macro_factors' in components and abs(components['macro_factors']) > 0.5:
            drivers.append('macro_environment')
        
        if 'dynamic_correlation' in components and abs(components['dynamic_correlation']) > 0.5:
            drivers.append('dynamic_correlations')
        
        return drivers if drivers else ['unknown']
    
    def _calculate_regime_confidence(self, signals: List[CorrelationSignal]) -> float:
        """Calculate confidence in regime classification."""
        if not signals:
            return 0.0
        
        # Confidence based on signal consistency and component agreement
        avg_confidence = np.mean([s.confidence for s in signals])
        
        # Consistency of signal direction
        signal_directions = [1 if s.strength > 0 else -1 for s in signals]
        consistency = abs(np.mean(signal_directions))
        
        return min(1.0, avg_confidence * consistency)
    
    def _update_portfolio_implications(self, timestamp: datetime) -> None:
        """Update portfolio implications from correlation analysis."""
        if not self.current_signal or not self.asset_universe:
            return
        
        # Calculate diversification benefit
        diversification_benefit = self._calculate_diversification_benefit()
        
        # Calculate concentration risk
        concentration_risk = self._calculate_concentration_risk()
        
        # Optimize portfolio weights
        optimal_weights = self._optimize_portfolio_weights()
        
        # Risk budget allocation
        risk_budget = self._calculate_risk_budget_allocation()
        
        # Hedging recommendations
        hedging_recs = self._generate_hedging_recommendations()
        
        # Rebalancing urgency
        rebalancing_urgency = self._assess_rebalancing_urgency()
        
        implication = PortfolioImplication(
            timestamp=timestamp,
            diversification_benefit=diversification_benefit,
            concentration_risk=concentration_risk,
            optimal_weights=optimal_weights,
            risk_budget_allocation=risk_budget,
            hedging_recommendations=hedging_recs,
            rebalancing_urgency=rebalancing_urgency
        )
        
        self.portfolio_implications.append(implication)
    
    def _calculate_diversification_benefit(self) -> float:
        """Calculate current diversification benefit."""
        # Get current correlation matrix
        corr_matrix = self.correlation_analyzer.get_correlation_matrix()
        
        if corr_matrix is None:
            return 0.5  # Default
        
        # Calculate diversification ratio
        diversification_ratio = self.correlation_analyzer.get_diversification_ratio()
        
        if diversification_ratio is None:
            return 0.5
        
        # Normalize to 0-1 scale (higher is better)
        return min(1.0, diversification_ratio / 2.0)
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk."""
        # Based on correlation concentration and portfolio weights
        corr_summary = self.correlation_analyzer.get_correlation_summary()
        
        if 'latest_analysis' not in corr_summary:
            return 0.5
        
        concentration_metrics = corr_summary['latest_analysis'].get('concentration_metrics', {})
        eigenvalue_concentration = concentration_metrics.get('eigenvalue_concentration', 0.5)
        
        # Weight concentration in portfolio
        weight_concentration = self._calculate_weight_concentration()
        
        # Combined concentration risk
        concentration_risk = 0.7 * eigenvalue_concentration + 0.3 * weight_concentration
        
        return min(1.0, concentration_risk)
    
    def _calculate_weight_concentration(self) -> float:
        """Calculate concentration in portfolio weights."""
        if not self.portfolio_weights:
            return 0.0
        
        weights = np.array(list(self.portfolio_weights.values()))
        
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights ** 2)
        
        # Normalize (1/n for equal weights, 1 for single asset)
        n_assets = len(weights)
        normalized_hhi = (hhi - 1/n_assets) / (1 - 1/n_assets) if n_assets > 1 else 1.0
        
        return max(0.0, normalized_hhi)
    
    def _optimize_portfolio_weights(self) -> Dict[str, float]:
        """Optimize portfolio weights based on current correlations."""
        # Simple optimization based on correlation structure
        # In practice, would use more sophisticated optimization
        
        corr_matrix = self.correlation_analyzer.get_correlation_matrix()
        
        if corr_matrix is None or len(corr_matrix.asset_symbols) == 0:
            return self.portfolio_weights.copy()
        
        symbols = corr_matrix.asset_symbols
        matrix = corr_matrix.correlation_matrix
        
        try:
            # Inverse correlation weighting (simplified)
            inv_corr = np.linalg.pinv(matrix)
            weights = np.sum(inv_corr, axis=1)
            weights = weights / np.sum(weights)  # Normalize
            
            # Ensure positive weights
            weights = np.maximum(weights, 0.01)
            weights = weights / np.sum(weights)
            
            return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
        except:
            # Fall back to equal weights
            n_assets = len(symbols)
            return {symbol: 1.0/n_assets for symbol in symbols}
    
    def _calculate_risk_budget_allocation(self) -> Dict[str, float]:
        """Calculate risk budget allocation."""
        # Simplified risk budgeting based on volatility and correlation
        optimal_weights = self._optimize_portfolio_weights()
        
        # Assume equal risk contribution target
        return {asset: 1.0/len(optimal_weights) for asset in optimal_weights}
    
    def _generate_hedging_recommendations(self) -> List[str]:
        """Generate hedging recommendations."""
        recommendations = []
        
        if not self.current_regime:
            return recommendations
        
        regime_type = self.current_regime.regime_type
        
        if regime_type == 'crisis':
            recommendations.extend([
                'Increase hedge ratio significantly',
                'Consider safe haven assets',
                'Implement tail risk hedging'
            ])
        elif regime_type == 'risk_off':
            recommendations.extend([
                'Moderate hedging increase',
                'Focus on quality assets',
                'Reduce beta exposure'
            ])
        elif regime_type == 'risk_on':
            recommendations.extend([
                'Reduce hedge ratio',
                'Increase risk asset exposure',
                'Consider momentum strategies'
            ])
        
        return recommendations
    
    def _assess_rebalancing_urgency(self) -> str:
        """Assess urgency of portfolio rebalancing."""
        if not self.portfolio_implications:
            return 'low'
        
        current_impl = self.portfolio_implications[-1]
        
        # Check concentration risk
        if current_impl.concentration_risk > 0.8:
            return 'high'
        elif current_impl.concentration_risk > 0.6:
            return 'medium'
        
        # Check diversification benefit decline
        if len(self.portfolio_implications) > 1:
            prev_impl = self.portfolio_implications[-2]
            benefit_change = current_impl.diversification_benefit - prev_impl.diversification_benefit
            
            if benefit_change < -0.2:
                return 'high'
            elif benefit_change < -0.1:
                return 'medium'
        
        return 'low'
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'asset_universe_size': len(self.asset_universe),
            'components_status': {
                'correlation_analyzer': {
                    'assets_tracked': len(self.correlation_analyzer.asset_data),
                    'current_regime': self.correlation_analyzer.get_correlation_regime()
                },
                'dynamic_model': {
                    'is_fitted': self.dynamic_model.is_fitted,
                    'n_assets': self.dynamic_model.n_assets
                },
                'spillover_analyzer': {
                    'assets_analyzed': len(self.spillover_analyzer.asset_symbols),
                    'recent_events': len(self.spillover_analyzer.spillover_events)
                },
                'macro_factor_model': {
                    'is_fitted': self.macro_factor_model.is_fitted,
                    'factors_available': len(self.macro_factor_model.macro_factors)
                }
            }
        }
        
        if self.current_signal:
            status['current_signal'] = {
                'type': self.current_signal.signal_type,
                'strength': self.current_signal.strength,
                'confidence': self.current_signal.confidence,
                'recommendations': self.current_signal.recommendations
            }
        
        if self.current_regime:
            status['current_regime'] = {
                'type': self.current_regime.regime_type,
                'confidence': self.current_regime.confidence,
                'drivers': self.current_regime.regime_drivers
            }
        
        if self.portfolio_implications:
            latest_impl = self.portfolio_implications[-1]
            status['portfolio_implications'] = {
                'diversification_benefit': latest_impl.diversification_benefit,
                'concentration_risk': latest_impl.concentration_risk,
                'rebalancing_urgency': latest_impl.rebalancing_urgency
            }
        
        return status
    
    def start_coordination(self) -> None:
        """Start real-time coordination."""
        if self.is_running:
            return
        
        self.is_running = True
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True
        )
        self.coordination_thread.start()
        print("Cross-asset correlation coordination started")
    
    def stop_coordination(self) -> None:
        """Stop real-time coordination."""
        self.is_running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        print("Cross-asset correlation coordination stopped")
    
    def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.is_running:
            try:
                # Periodic maintenance and analysis
                self._periodic_model_updates()
                self._cleanup_old_data()
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                print(f"Error in coordination loop: {e}")
                time.sleep(self.update_frequency)
    
    def _periodic_model_updates(self) -> None:
        """Perform periodic model updates."""
        # Re-estimate models if enough new data
        if not self.macro_factor_model.is_fitted and len(self.asset_universe) > 0:
            try:
                self.macro_factor_model.estimate_factor_model()
            except Exception as e:
                print(f"Failed to estimate factor model: {e}")
        
        # Update factor exposures
        if self.macro_factor_model.is_fitted:
            self.macro_factor_model.update_factor_exposures()
        
        # Detect regime changes
        self.macro_factor_model.detect_regime_changes()
        
        # Detect flight-to-quality events
        self.spillover_analyzer.detect_flight_to_quality()
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data from components."""
        # Components handle their own data cleanup with deque maxlen
        pass