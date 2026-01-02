"""
Risk Coordinator - Central Risk Management System.

Coordinates all risk management components including position sizing, portfolio risk
management, stress testing, and hedging to provide unified risk oversight.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations

# Import other risk management components
from models.advanced.risk_management.position_sizer import PositionSizer, PositionSizeResult
from models.advanced.risk_management.portfolio_risk_manager import PortfolioRiskManager, VaRResult, RiskAlert
from models.advanced.risk_management.stress_tester import StressTester, StressTestResult
from models.advanced.risk_management.hedging_manager import HedgingManager, HedgingSignal


@dataclass
class RiskLimits:
    """Comprehensive risk limits configuration."""
    max_portfolio_var: float = 0.05  # 5% daily VaR
    max_concentration: float = 0.25  # 25% max single asset
    max_leverage: float = 2.0
    max_correlation_exposure: float = 0.6  # 60% max correlated assets
    max_sector_exposure: float = 0.4  # 40% max sector exposure
    max_drawdown: float = 0.15  # 15% max drawdown
    min_liquidity_buffer: float = 0.05  # 5% cash buffer
    stress_test_threshold: float = 0.2  # 20% max stress loss


@dataclass
class RiskStatus:
    """Current risk status assessment."""
    timestamp: datetime
    overall_risk_level: str  # low, medium, high, critical
    risk_score: float  # 0-1 scale
    limit_breaches: List[str]
    active_alerts: int
    recommendations: List[str]
    next_review_time: datetime


@dataclass
class RiskReport:
    """Comprehensive risk report."""
    timestamp: datetime
    portfolio_value: float
    risk_summary: Dict[str, Any]
    position_analysis: Dict[str, Any]
    var_analysis: Dict[str, Any]
    stress_test_results: Dict[str, Any]
    hedging_analysis: Dict[str, Any]
    limit_monitoring: Dict[str, Any]
    recommendations: List[str]
    action_items: List[str]


class RiskCoordinator:
    """
    Central risk management coordination system.
    
    Features:
    - Unified risk monitoring and reporting
    - Coordinated risk limit enforcement
    - Integrated position sizing and hedging
    - Real-time risk alerting
    - Automated risk response actions
    - Comprehensive stress testing coordination
    - Risk reporting and analytics
    - Cross-component risk optimization
    """
    
    def __init__(self,
                 risk_limits: Optional[RiskLimits] = None,
                 monitoring_frequency: int = 5,  # minutes
                 auto_risk_response: bool = True):
        """
        Initialize risk coordinator.
        
        Args:
            risk_limits: Risk limits configuration
            monitoring_frequency: Risk monitoring frequency (minutes)
            auto_risk_response: Enable automatic risk response actions
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.monitoring_frequency = monitoring_frequency
        self.auto_risk_response = auto_risk_response
        
        # Initialize risk management components
        self.position_sizer = PositionSizer()
        self.portfolio_risk_manager = PortfolioRiskManager()
        self.stress_tester = StressTester()
        self.hedging_manager = HedgingManager()
        
        # Risk monitoring state
        self.current_risk_status: Optional[RiskStatus] = None
        self.risk_history: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=500)
        
        # Portfolio state
        self.current_portfolio: Dict[str, float] = {}
        self.portfolio_value: float = 0.0
        self.asset_prices: Dict[str, float] = {}
        self.market_data: Dict[str, Any] = {}
        
        # Risk metrics tracking
        self.risk_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.performance_tracking: deque = deque(maxlen=500)
        
        # Last monitoring time
        self.last_monitoring: Optional[datetime] = None
        
        # Risk response actions
        self.response_actions: Dict[str, Callable] = {
            'reduce_position': self._reduce_risky_positions,
            'increase_hedging': self._increase_hedging,
            'stress_test': self._emergency_stress_test,
            'liquidity_check': self._check_liquidity_adequacy,
            'alert_management': self._escalate_to_management
        }
        
        # Configuration
        self.risk_appetite = 'moderate'  # conservative, moderate, aggressive
        self.emergency_mode = False
    
    def update_market_data(self,
                          portfolio_positions: Dict[str, float],
                          portfolio_value: float,
                          asset_prices: Dict[str, float],
                          market_context: Dict[str, Any]) -> None:
        """
        Update market data and trigger risk monitoring.
        
        Args:
            portfolio_positions: Current portfolio positions
            portfolio_value: Current portfolio value
            asset_prices: Current asset prices
            market_context: Market context information
        """
        # Update internal state
        self.current_portfolio = portfolio_positions.copy()
        self.portfolio_value = portfolio_value
        self.asset_prices = asset_prices.copy()
        self.market_data = market_context.copy()
        
        # Update all risk management components
        self._update_all_components()
        
        # Perform risk monitoring
        self._perform_risk_monitoring()
        
        # Execute automatic responses if enabled
        if self.auto_risk_response:
            self._execute_auto_responses()
    
    def _update_all_components(self) -> None:
        """Update all risk management components with current data."""
        # Calculate volatilities (simplified)
        asset_volatilities = {}
        for asset in self.asset_prices:
            # In practice, would calculate from price history
            asset_volatilities[asset] = self.market_data.get(f'{asset}_volatility', 0.02)
        
        # Update portfolio risk manager
        self.portfolio_risk_manager.update_portfolio_data(
            self.current_portfolio,
            self.asset_prices,
            datetime.now()
        )
        
        # Update hedging manager
        self.hedging_manager.update_portfolio_data(
            self.current_portfolio,
            self.portfolio_value,
            self.asset_prices,
            asset_volatilities
        )
        
        # Update stress tester
        if self.asset_prices:
            for asset, price in self.asset_prices.items():
                self.stress_tester.add_market_data(
                    asset, [price], [datetime.now()]
                )
    
    def _perform_risk_monitoring(self) -> None:
        """Perform comprehensive risk monitoring."""
        timestamp = datetime.now()
        
        # Skip if monitoring too frequent
        if (self.last_monitoring and 
            (timestamp - self.last_monitoring).total_seconds() < self.monitoring_frequency * 60):
            return
        
        # Calculate current risk metrics
        risk_metrics = self._calculate_comprehensive_risk_metrics()
        
        # Check risk limits
        limit_breaches = self._check_risk_limits(risk_metrics)
        
        # Assess overall risk level
        risk_level, risk_score = self._assess_risk_level(risk_metrics, limit_breaches)
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(risk_metrics, limit_breaches)
        
        # Create risk status
        self.current_risk_status = RiskStatus(
            timestamp=timestamp,
            overall_risk_level=risk_level,
            risk_score=risk_score,
            limit_breaches=limit_breaches,
            active_alerts=len(self.portfolio_risk_manager.risk_alerts),
            recommendations=recommendations,
            next_review_time=timestamp + timedelta(minutes=self.monitoring_frequency)
        )
        
        # Store in history
        self.risk_history.append(self.current_risk_status)
        
        # Update last monitoring time
        self.last_monitoring = timestamp
        
        # Store risk metrics
        for metric, value in risk_metrics.items():
            if isinstance(value, (int, float)):
                self.risk_metrics_history[metric].append(value)
    
    def _calculate_comprehensive_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics across all components."""
        metrics = {}
        
        # Portfolio VaR metrics
        try:
            var_result = self.portfolio_risk_manager.calculate_var(
                confidence_level=0.95, time_horizon=1
            )
            metrics['var_95_1d'] = var_result.var_value
            metrics['cvar_95_1d'] = var_result.cvar_value
            metrics['var_ratio'] = var_result.var_value / self.portfolio_value if self.portfolio_value > 0 else 0
        except:
            metrics['var_95_1d'] = 0
            metrics['cvar_95_1d'] = 0
            metrics['var_ratio'] = 0
        
        # Portfolio concentration metrics
        if self.current_portfolio:
            position_values = [abs(pos) for pos in self.current_portfolio.values()]
            metrics['max_concentration'] = max(position_values) if position_values else 0
            metrics['herfindahl_index'] = sum(pos**2 for pos in position_values) if position_values else 0
        
        # Leverage metrics
        gross_exposure = sum(abs(pos) for pos in self.current_portfolio.values())
        net_exposure = sum(self.current_portfolio.values())
        metrics['gross_leverage'] = gross_exposure
        metrics['net_leverage'] = abs(net_exposure)
        
        # Stress test metrics
        if hasattr(self.stress_tester, 'stress_results') and self.stress_tester.stress_results:
            latest_stress = list(self.stress_tester.stress_results)[-1]
            metrics['worst_case_loss'] = latest_stress.portfolio_impact.get('total_loss_percentage', 0)
        else:
            metrics['worst_case_loss'] = 0
        
        # Hedging metrics
        hedge_summary = self.hedging_manager.get_hedge_summary()
        metrics['hedge_coverage'] = hedge_summary.get('hedge_budget_used', 0)
        metrics['active_hedges'] = hedge_summary.get('active_hedges', 0)
        
        # Market regime metrics
        metrics['market_volatility'] = self.market_data.get('market_volatility', 0.02)
        metrics['market_regime'] = self.market_data.get('market_regime', 'normal')
        
        return metrics
    
    def _check_risk_limits(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """Check for risk limit breaches."""
        breaches = []
        
        # VaR limit
        if risk_metrics.get('var_ratio', 0) > self.risk_limits.max_portfolio_var:
            breaches.append(f"Portfolio VaR ({risk_metrics['var_ratio']:.2%}) exceeds limit ({self.risk_limits.max_portfolio_var:.2%})")
        
        # Concentration limit
        if risk_metrics.get('max_concentration', 0) > self.risk_limits.max_concentration:
            breaches.append(f"Position concentration ({risk_metrics['max_concentration']:.2%}) exceeds limit ({self.risk_limits.max_concentration:.2%})")
        
        # Leverage limits
        if risk_metrics.get('gross_leverage', 0) > self.risk_limits.max_leverage:
            breaches.append(f"Gross leverage ({risk_metrics['gross_leverage']:.2f}) exceeds limit ({self.risk_limits.max_leverage:.2f})")
        
        # Stress test limit
        if risk_metrics.get('worst_case_loss', 0) > self.risk_limits.stress_test_threshold:
            breaches.append(f"Stress test loss ({risk_metrics['worst_case_loss']:.2%}) exceeds limit ({self.risk_limits.stress_test_threshold:.2%})")
        
        return breaches
    
    def _assess_risk_level(self, 
                          risk_metrics: Dict[str, Any], 
                          limit_breaches: List[str]) -> Tuple[str, float]:
        """Assess overall risk level and score."""
        risk_factors = []
        
        # VaR factor
        var_ratio = risk_metrics.get('var_ratio', 0)
        var_factor = min(1.0, var_ratio / self.risk_limits.max_portfolio_var)
        risk_factors.append(var_factor)
        
        # Concentration factor
        concentration = risk_metrics.get('max_concentration', 0)
        concentration_factor = min(1.0, concentration / self.risk_limits.max_concentration)
        risk_factors.append(concentration_factor)
        
        # Leverage factor
        leverage = risk_metrics.get('gross_leverage', 0)
        leverage_factor = min(1.0, leverage / self.risk_limits.max_leverage)
        risk_factors.append(leverage_factor)
        
        # Stress test factor
        stress_loss = risk_metrics.get('worst_case_loss', 0)
        stress_factor = min(1.0, stress_loss / self.risk_limits.stress_test_threshold)
        risk_factors.append(stress_factor)
        
        # Market regime factor
        market_vol = risk_metrics.get('market_volatility', 0.02)
        vol_factor = min(1.0, market_vol / 0.05)  # 5% threshold
        risk_factors.append(vol_factor)
        
        # Calculate overall risk score
        risk_score = np.mean(risk_factors)
        
        # Adjust for limit breaches
        if limit_breaches:
            risk_score = min(1.0, risk_score + len(limit_breaches) * 0.2)
        
        # Determine risk level
        if risk_score >= 0.8 or len(limit_breaches) >= 3:
            risk_level = "critical"
        elif risk_score >= 0.6 or len(limit_breaches) >= 2:
            risk_level = "high"
        elif risk_score >= 0.4 or len(limit_breaches) >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return risk_level, risk_score
    
    def _generate_risk_recommendations(self, 
                                     risk_metrics: Dict[str, Any], 
                                     limit_breaches: List[str]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # VaR-based recommendations
        if risk_metrics.get('var_ratio', 0) > self.risk_limits.max_portfolio_var * 0.8:
            recommendations.append("Consider reducing portfolio risk exposure")
            recommendations.append("Review position sizing methodology")
        
        # Concentration recommendations
        if risk_metrics.get('max_concentration', 0) > self.risk_limits.max_concentration * 0.8:
            recommendations.append("Reduce concentration in largest positions")
            recommendations.append("Diversify across more assets")
        
        # Leverage recommendations
        if risk_metrics.get('gross_leverage', 0) > self.risk_limits.max_leverage * 0.8:
            recommendations.append("Reduce leverage through position size reduction")
        
        # Hedging recommendations
        if risk_metrics.get('hedge_coverage', 0) < 0.3:
            recommendations.append("Consider increasing hedging coverage")
        
        # Stress test recommendations
        if risk_metrics.get('worst_case_loss', 0) > self.risk_limits.stress_test_threshold * 0.8:
            recommendations.append("Prepare for potential stress scenarios")
            recommendations.append("Review tail risk hedging strategies")
        
        # Market regime recommendations
        market_regime = risk_metrics.get('market_regime', 'normal')
        if market_regime in ['volatile', 'bear']:
            recommendations.append("Adjust risk appetite for current market regime")
            recommendations.append("Consider defensive positioning")
        
        # Emergency recommendations
        if len(limit_breaches) >= 2:
            recommendations.append("URGENT: Multiple risk limits breached - immediate action required")
            recommendations.append("Consider emergency position reduction")
        
        return recommendations
    
    def _execute_auto_responses(self) -> None:
        """Execute automatic risk response actions."""
        if not self.current_risk_status:
            return
        
        risk_level = self.current_risk_status.overall_risk_level
        limit_breaches = self.current_risk_status.limit_breaches
        
        # Critical risk level responses
        if risk_level == "critical" or len(limit_breaches) >= 3:
            self._emergency_risk_response()
        
        # High risk level responses
        elif risk_level == "high" or len(limit_breaches) >= 2:
            self._high_risk_response()
        
        # Medium risk level responses
        elif risk_level == "medium":
            self._medium_risk_response()
    
    def _emergency_risk_response(self) -> None:
        """Execute emergency risk response actions."""
        self.emergency_mode = True
        
        print("EMERGENCY RISK RESPONSE ACTIVATED")
        
        # Immediate actions
        self._reduce_risky_positions(severity=0.3)  # Reduce by 30%
        self._increase_hedging(urgency="critical")
        self._emergency_stress_test()
        self._escalate_to_management()
        
        # Create emergency alert
        alert = f"EMERGENCY: Critical risk level detected at {datetime.now()}"
        self.alert_history.append({
            'timestamp': datetime.now(),
            'type': 'emergency',
            'message': alert,
            'risk_level': 'critical'
        })
    
    def _high_risk_response(self) -> None:
        """Execute high risk response actions."""
        print("HIGH RISK RESPONSE ACTIVATED")
        
        # Moderate actions
        self._reduce_risky_positions(severity=0.15)  # Reduce by 15%
        self._increase_hedging(urgency="high")
        self._check_liquidity_adequacy()
        
        # Create alert
        alert = f"HIGH RISK: Risk limits breached at {datetime.now()}"
        self.alert_history.append({
            'timestamp': datetime.now(),
            'type': 'high_risk',
            'message': alert,
            'risk_level': 'high'
        })
    
    def _medium_risk_response(self) -> None:
        """Execute medium risk response actions."""
        print("MEDIUM RISK RESPONSE ACTIVATED")
        
        # Conservative actions
        self._increase_hedging(urgency="medium")
        self._check_liquidity_adequacy()
        
        # Create alert
        alert = f"MEDIUM RISK: Elevated risk detected at {datetime.now()}"
        self.alert_history.append({
            'timestamp': datetime.now(),
            'type': 'medium_risk',
            'message': alert,
            'risk_level': 'medium'
        })
    
    def _reduce_risky_positions(self, severity: float = 0.1) -> None:
        """Reduce risky positions based on severity level."""
        # Identify riskiest positions
        risky_positions = []
        
        for asset, position in self.current_portfolio.items():
            if abs(position) > self.risk_limits.max_concentration * 0.8:
                risky_positions.append((asset, position, abs(position)))
        
        # Sort by position size
        risky_positions.sort(key=lambda x: x[2], reverse=True)
        
        # Reduce positions
        for asset, position, size in risky_positions[:3]:  # Top 3 risky positions
            reduction = size * severity
            new_position = position - (reduction if position > 0 else -reduction)
            
            print(f"Risk Response: Reducing {asset} position from {position:.3f} to {new_position:.3f}")
            
            # In practice, would send orders to execution system
            # self.current_portfolio[asset] = new_position
    
    def _increase_hedging(self, urgency: str = "medium") -> None:
        """Increase hedging coverage."""
        hedging_recommendations = self.hedging_manager.get_hedging_recommendations()
        
        immediate_actions = hedging_recommendations.get('immediate_actions', [])
        
        # Execute high-priority hedging actions
        for action in immediate_actions:
            if action.get('urgency') == urgency or urgency == "critical":
                print(f"Risk Response: Executing hedge {action.get('action')} for {action.get('asset')}")
                
                # In practice, would execute hedge through hedging manager
                # self.hedging_manager._execute_hedging_signal(create_signal_from_action(action))
    
    def _emergency_stress_test(self) -> None:
        """Run emergency stress test."""
        try:
            # Run comprehensive stress test
            stress_results = self.stress_tester.run_comprehensive_stress_test(
                self.current_portfolio, self.portfolio_value
            )
            
            # Analyze worst-case scenarios
            worst_scenarios = []
            for scenario_id, result in stress_results.items():
                loss_pct = result.portfolio_impact.get('total_loss_percentage', 0)
                worst_scenarios.append((scenario_id, loss_pct))
            
            worst_scenarios.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Emergency Stress Test: Worst case loss {worst_scenarios[0][1]:.2%} in scenario {worst_scenarios[0][0]}")
            
        except Exception as e:
            print(f"Error running emergency stress test: {e}")
    
    def _check_liquidity_adequacy(self) -> None:
        """Check portfolio liquidity adequacy."""
        # Calculate liquidity score (simplified)
        total_liquidity = 0.0
        
        for asset, position in self.current_portfolio.items():
            # Assume major assets have higher liquidity
            major_assets = ['bitcoin', 'ethereum', 'btc', 'eth', 'usdt', 'usdc']
            liquidity_score = 0.9 if any(major in asset.lower() for major in major_assets) else 0.6
            
            position_value = abs(position) * self.portfolio_value
            total_liquidity += position_value * liquidity_score
        
        liquidity_ratio = total_liquidity / self.portfolio_value if self.portfolio_value > 0 else 0
        
        if liquidity_ratio < 0.7:  # 70% threshold
            print(f"Risk Response: Low liquidity detected ({liquidity_ratio:.2%})")
            # In practice, would recommend increasing liquid assets
    
    def _escalate_to_management(self) -> None:
        """Escalate critical risk issues to management."""
        escalation_message = {
            'timestamp': datetime.now(),
            'risk_level': self.current_risk_status.overall_risk_level if self.current_risk_status else 'unknown',
            'portfolio_value': self.portfolio_value,
            'limit_breaches': self.current_risk_status.limit_breaches if self.current_risk_status else [],
            'immediate_action_required': True
        }
        
        print(f"ESCALATION TO MANAGEMENT: {escalation_message}")
        
        # In practice, would send notifications to management systems
    
    def calculate_optimal_position_size(self,
                                      asset_symbol: str,
                                      prediction: Dict[str, Any],
                                      method: Optional[str] = None) -> PositionSizeResult:
        """Calculate optimal position size considering all risk factors."""
        # Get base position size recommendation
        base_result = self.position_sizer.calculate_position_size(
            asset_symbol, prediction, self.portfolio_value, 
            self.current_portfolio.get(asset_symbol, 0.0), method
        )
        
        # Apply risk coordinator adjustments
        adjusted_size = self._apply_risk_adjustments(asset_symbol, base_result.recommended_size)
        
        # Update result
        base_result.recommended_size = adjusted_size
        base_result.rationale.append("Adjusted by Risk Coordinator for portfolio-level constraints")
        
        return base_result
    
    def _apply_risk_adjustments(self, asset_symbol: str, base_size: float) -> float:
        """Apply risk coordinator adjustments to position size."""
        adjusted_size = base_size
        
        # Current risk level adjustment
        if self.current_risk_status:
            risk_level = self.current_risk_status.overall_risk_level
            
            if risk_level == "critical":
                adjusted_size *= 0.3  # Reduce by 70%
            elif risk_level == "high":
                adjusted_size *= 0.6  # Reduce by 40%
            elif risk_level == "medium":
                adjusted_size *= 0.8  # Reduce by 20%
        
        # Concentration limit adjustment
        current_position = self.current_portfolio.get(asset_symbol, 0.0)
        new_total_position = abs(current_position) + abs(adjusted_size)
        
        if new_total_position > self.risk_limits.max_concentration:
            max_additional = self.risk_limits.max_concentration - abs(current_position)
            adjusted_size = min(adjusted_size, max(0, max_additional))
        
        # Market regime adjustment
        market_regime = self.market_data.get('market_regime', 'normal')
        if market_regime in ['volatile', 'bear']:
            adjusted_size *= 0.7  # Reduce in adverse regimes
        
        return max(0.0, adjusted_size)
    
    def generate_comprehensive_report(self) -> RiskReport:
        """Generate comprehensive risk management report."""
        timestamp = datetime.now()
        
        # Gather data from all components
        portfolio_metrics = self.portfolio_risk_manager.calculate_portfolio_metrics()
        hedge_summary = self.hedging_manager.get_hedge_summary()
        stress_summary = self.stress_tester.get_stress_test_summary()
        
        # Calculate current risk metrics
        current_metrics = self._calculate_comprehensive_risk_metrics()
        
        # Position analysis
        position_analysis = {
            'total_positions': len(self.current_portfolio),
            'largest_position': max(abs(pos) for pos in self.current_portfolio.values()) if self.current_portfolio else 0,
            'concentration_hhi': current_metrics.get('herfindahl_index', 0),
            'gross_exposure': current_metrics.get('gross_leverage', 0),
            'net_exposure': current_metrics.get('net_leverage', 0)
        }
        
        # VaR analysis
        var_analysis = {
            'daily_var_95': current_metrics.get('var_95_1d', 0),
            'daily_cvar_95': current_metrics.get('cvar_95_1d', 0),
            'var_ratio': current_metrics.get('var_ratio', 0),
            'var_trend': 'stable'  # Would calculate from history
        }
        
        # Limit monitoring
        limit_breaches = self._check_risk_limits(current_metrics)
        limit_monitoring = {
            'breaches_count': len(limit_breaches),
            'breach_details': limit_breaches,
            'compliance_score': max(0, 1.0 - len(limit_breaches) * 0.2)
        }
        
        # Recommendations and actions
        recommendations = self._generate_risk_recommendations(current_metrics, limit_breaches)
        action_items = self._generate_action_items(current_metrics, limit_breaches)
        
        return RiskReport(
            timestamp=timestamp,
            portfolio_value=self.portfolio_value,
            risk_summary={
                'risk_level': self.current_risk_status.overall_risk_level if self.current_risk_status else 'unknown',
                'risk_score': self.current_risk_status.risk_score if self.current_risk_status else 0,
                'emergency_mode': self.emergency_mode,
                'last_monitoring': self.last_monitoring.isoformat() if self.last_monitoring else None
            },
            position_analysis=position_analysis,
            var_analysis=var_analysis,
            stress_test_results=stress_summary,
            hedging_analysis=hedge_summary,
            limit_monitoring=limit_monitoring,
            recommendations=recommendations,
            action_items=action_items
        )
    
    def _generate_action_items(self, 
                              risk_metrics: Dict[str, Any], 
                              limit_breaches: List[str]) -> List[str]:
        """Generate specific action items for risk management."""
        actions = []
        
        # Immediate actions for breaches
        if limit_breaches:
            actions.append("IMMEDIATE: Address risk limit breaches")
            actions.append("Review and adjust position sizes")
        
        # Hedging actions
        if risk_metrics.get('hedge_coverage', 0) < 0.2:
            actions.append("Increase hedging coverage to 20%+ of portfolio")
        
        # Stress testing actions
        if risk_metrics.get('worst_case_loss', 0) > 0.15:
            actions.append("Run detailed stress test analysis")
            actions.append("Review tail risk hedging strategies")
        
        # Monitoring actions
        actions.append("Continue regular risk monitoring")
        actions.append("Update risk models with latest market data")
        
        return actions
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get data for risk management dashboard."""
        current_metrics = self._calculate_comprehensive_risk_metrics()
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'risk_status': {
                'level': self.current_risk_status.overall_risk_level if self.current_risk_status else 'unknown',
                'score': self.current_risk_status.risk_score if self.current_risk_status else 0,
                'active_alerts': len(self.alert_history)
            },
            'key_metrics': {
                'var_95': current_metrics.get('var_95_1d', 0),
                'max_concentration': current_metrics.get('max_concentration', 0),
                'leverage': current_metrics.get('gross_leverage', 0),
                'hedge_coverage': current_metrics.get('hedge_coverage', 0)
            },
            'limit_status': {
                'var_limit': current_metrics.get('var_ratio', 0) / self.risk_limits.max_portfolio_var,
                'concentration_limit': current_metrics.get('max_concentration', 0) / self.risk_limits.max_concentration,
                'leverage_limit': current_metrics.get('gross_leverage', 0) / self.risk_limits.max_leverage
            },
            'recent_alerts': [
                {
                    'timestamp': alert['timestamp'].isoformat(),
                    'type': alert['type'],
                    'message': alert['message']
                }
                for alert in list(self.alert_history)[-5:]
            ]
        }
        
        return dashboard
    
    def set_risk_appetite(self, appetite: str) -> None:
        """
        Set risk appetite level.
        
        Args:
            appetite: 'conservative', 'moderate', or 'aggressive'
        """
        if appetite not in ['conservative', 'moderate', 'aggressive']:
            raise ValueError("Risk appetite must be 'conservative', 'moderate', or 'aggressive'")
        
        self.risk_appetite = appetite
        
        # Adjust risk limits based on appetite
        if appetite == 'conservative':
            self.risk_limits.max_portfolio_var *= 0.7
            self.risk_limits.max_concentration *= 0.8
            self.risk_limits.max_leverage *= 0.8
        elif appetite == 'aggressive':
            self.risk_limits.max_portfolio_var *= 1.3
            self.risk_limits.max_concentration *= 1.2
            self.risk_limits.max_leverage *= 1.2
        
        print(f"Risk appetite set to {appetite}")
    
    def emergency_shutdown(self) -> None:
        """Emergency shutdown of all risky positions."""
        print("EMERGENCY SHUTDOWN INITIATED")
        
        self.emergency_mode = True
        
        # Close all positions (in practice, would send market orders)
        for asset in self.current_portfolio:
            print(f"Emergency: Closing position in {asset}")
        
        # Close all hedge positions
        for position_id in list(self.hedging_manager.active_hedges.keys()):
            self.hedging_manager.close_hedge_position(position_id, "emergency_shutdown")
        
        # Create emergency log
        emergency_log = {
            'timestamp': datetime.now(),
            'action': 'emergency_shutdown',
            'portfolio_value_at_shutdown': self.portfolio_value,
            'positions_closed': len(self.current_portfolio),
            'hedges_closed': len(self.hedging_manager.active_hedges)
        }
        
        self.alert_history.append(emergency_log)
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all risk management components."""
        return {
            'position_sizer': {
                'active': True,
                'last_sizing': len(self.position_sizer.position_history),
                'risk_budget_used': self.position_sizer.risk_budget.used_budget
            },
            'portfolio_risk_manager': {
                'active': True,
                'last_var_calculation': len(self.portfolio_risk_manager.var_results),
                'active_alerts': len(self.portfolio_risk_manager.risk_alerts)
            },
            'stress_tester': {
                'active': True,
                'stress_tests_run': len(self.stress_tester.stress_results),
                'scenarios_available': len(self.stress_tester.scenario_library)
            },
            'hedging_manager': {
                'active': True,
                'active_hedges': len(self.hedging_manager.active_hedges),
                'hedge_coverage': self.hedging_manager.get_hedge_summary().get('hedge_budget_used', 0)
            },
            'coordinator': {
                'monitoring_active': self.last_monitoring is not None,
                'emergency_mode': self.emergency_mode,
                'risk_appetite': self.risk_appetite,
                'auto_response_enabled': self.auto_risk_response
            }
        }