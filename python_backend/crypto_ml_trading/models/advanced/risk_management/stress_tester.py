"""
Advanced Stress Testing and Scenario Analysis System.

Implements comprehensive stress testing methodologies including Monte Carlo
simulations, historical scenarios, and custom shock analysis.
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


@dataclass
class StressScenario:
    """Stress testing scenario definition."""
    scenario_id: str
    scenario_name: str
    scenario_type: str  # historical, monte_carlo, custom, regulatory
    description: str
    parameters: Dict[str, Any]
    severity_level: str  # mild, moderate, severe, extreme
    probability: float  # estimated probability of occurrence
    asset_shocks: Dict[str, float] = field(default_factory=dict)
    correlation_shocks: Dict[str, float] = field(default_factory=dict)
    volatility_shocks: Dict[str, float] = field(default_factory=dict)
    liquidity_shocks: Dict[str, float] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Result of stress test execution."""
    timestamp: datetime
    scenario_id: str
    portfolio_impact: Dict[str, float]
    asset_impacts: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    liquidity_impact: Dict[str, float]
    capital_adequacy: Dict[str, float]
    recovery_time_estimate: float
    stress_severity: str
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration."""
    n_simulations: int = 10000
    time_horizon: int = 22  # trading days
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    correlation_decay: float = 0.95
    volatility_clustering: bool = True
    fat_tails: bool = True
    jump_diffusion: bool = True


class StressTester:
    """
    Advanced stress testing and scenario analysis system.
    
    Features:
    - Historical scenario replay
    - Monte Carlo stress simulations
    - Custom shock scenarios
    - Regulatory stress tests
    - Multi-factor stress testing
    - Correlation breakdown scenarios
    - Liquidity stress testing
    - Recovery time analysis
    """
    
    def __init__(self,
                 monte_carlo_config: Optional[MonteCarloConfig] = None,
                 max_scenario_history: int = 500,
                 stress_frequency: int = 24):  # hours
        """
        Initialize stress testing system.
        
        Args:
            monte_carlo_config: Monte Carlo simulation configuration
            max_scenario_history: Maximum scenarios to store
            stress_frequency: Frequency of stress testing (hours)
        """
        self.monte_carlo_config = monte_carlo_config or MonteCarloConfig()
        self.max_scenario_history = max_scenario_history
        self.stress_frequency = stress_frequency
        
        # Scenario library
        self.scenario_library: Dict[str, StressScenario] = {}
        self.custom_scenarios: Dict[str, StressScenario] = {}
        
        # Historical data and results
        self.stress_results: deque = deque(maxlen=max_scenario_history)
        self.scenario_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Market data storage
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Correlation and covariance matrices
        self.correlation_matrix: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        self.asset_symbols: List[str] = []
        
        # Stress testing state
        self.last_stress_test: Optional[datetime] = None
        self.current_stress_level: str = "normal"
        
        # Initialize predefined scenarios
        self._initialize_scenario_library()
    
    def _initialize_scenario_library(self) -> None:
        """Initialize predefined stress scenarios."""
        scenarios = [
            # Historical crisis scenarios
            StressScenario(
                scenario_id="2008_financial_crisis",
                scenario_name="2008 Financial Crisis",
                scenario_type="historical",
                description="Replication of 2008 financial crisis conditions",
                parameters={"start_date": "2008-09-15", "duration_days": 90},
                severity_level="extreme",
                probability=0.01,
                asset_shocks={"equity": -0.4, "credit": -0.6, "commodities": -0.3},
                correlation_shocks={"all_assets": 0.8},
                volatility_shocks={"all_assets": 3.0},
                liquidity_shocks={"credit": -0.7, "small_cap": -0.5}
            ),
            
            # COVID-19 pandemic scenario
            StressScenario(
                scenario_id="covid19_pandemic",
                scenario_name="COVID-19 Pandemic",
                scenario_type="historical",
                description="March 2020 pandemic market crash",
                parameters={"start_date": "2020-03-01", "duration_days": 30},
                severity_level="severe",
                probability=0.05,
                asset_shocks={"equity": -0.35, "oil": -0.6, "travel": -0.7},
                correlation_shocks={"risk_assets": 0.9},
                volatility_shocks={"all_assets": 4.0},
                liquidity_shocks={"corporate_bonds": -0.4}
            ),
            
            # Crypto winter scenario
            StressScenario(
                scenario_id="crypto_winter_2022",
                scenario_name="Crypto Winter 2022",
                scenario_type="historical",
                description="Crypto market collapse 2022",
                parameters={"start_date": "2022-05-01", "duration_days": 180},
                severity_level="severe",
                probability=0.1,
                asset_shocks={"bitcoin": -0.7, "ethereum": -0.8, "altcoins": -0.9},
                correlation_shocks={"crypto": 0.95},
                volatility_shocks={"crypto": 2.5},
                liquidity_shocks={"defi_tokens": -0.8}
            ),
            
            # Interest rate shock
            StressScenario(
                scenario_id="interest_rate_shock",
                scenario_name="Sudden Interest Rate Spike",
                scenario_type="custom",
                description="Central bank emergency rate hike",
                parameters={"rate_increase": 0.05, "duration_days": 1},
                severity_level="moderate",
                probability=0.15,
                asset_shocks={"bonds": -0.15, "growth_stocks": -0.25, "real_estate": -0.2},
                correlation_shocks={"duration_sensitive": 0.7},
                volatility_shocks={"bonds": 2.0, "equities": 1.5}
            ),
            
            # Liquidity crisis
            StressScenario(
                scenario_id="liquidity_crisis",
                scenario_name="Market Liquidity Crisis",
                scenario_type="custom",
                description="Sudden liquidity shortage across markets",
                parameters={"liquidity_reduction": 0.6, "duration_days": 7},
                severity_level="severe",
                probability=0.08,
                asset_shocks={"small_cap": -0.3, "emerging_markets": -0.4},
                correlation_shocks={"illiquid_assets": 0.9},
                volatility_shocks={"all_assets": 2.0},
                liquidity_shocks={"all_assets": -0.6}
            ),
            
            # Regulatory shock
            StressScenario(
                scenario_id="crypto_regulation_shock",
                scenario_name="Crypto Regulatory Crackdown",
                scenario_type="custom",
                description="Major regulatory restrictions on crypto",
                parameters={"severity": "high", "geographic_scope": "global"},
                severity_level="severe",
                probability=0.2,
                asset_shocks={"bitcoin": -0.5, "ethereum": -0.6, "defi": -0.8},
                correlation_shocks={"crypto": 0.9},
                volatility_shocks={"crypto": 3.0},
                liquidity_shocks={"crypto_exchanges": -0.9}
            )
        ]
        
        for scenario in scenarios:
            self.scenario_library[scenario.scenario_id] = scenario
    
    def add_market_data(self,
                       asset_symbol: str,
                       prices: List[float],
                       timestamps: List[datetime]) -> None:
        """
        Add market data for stress testing.
        
        Args:
            asset_symbol: Asset symbol
            prices: Price series
            timestamps: Corresponding timestamps
        """
        # Store price history
        for price, timestamp in zip(prices, timestamps):
            self.price_history[asset_symbol].append({
                'price': price,
                'timestamp': timestamp
            })
        
        # Calculate and store returns
        self._update_returns(asset_symbol)
        
        # Update volatility estimates
        self._update_volatility(asset_symbol)
        
        # Update correlation matrix
        self._update_correlation_matrix()
    
    def _update_returns(self, asset_symbol: str) -> None:
        """Update return series for asset."""
        prices = [p['price'] for p in self.price_history[asset_symbol]]
        
        if len(prices) > 1:
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    return_val = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(return_val)
            
            # Store recent returns
            if returns:
                for ret in returns[-50:]:  # Store last 50 returns
                    self.return_history[asset_symbol].append(ret)
    
    def _update_volatility(self, asset_symbol: str) -> None:
        """Update volatility estimates for asset."""
        returns = list(self.return_history[asset_symbol])
        
        if len(returns) > 20:
            # Calculate rolling volatility
            window_size = min(30, len(returns))
            recent_returns = returns[-window_size:]
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
            
            self.volatility_history[asset_symbol].append(volatility)
    
    def _update_correlation_matrix(self) -> None:
        """Update correlation matrix between assets."""
        # Get assets with sufficient data
        assets_with_data = [
            asset for asset in self.return_history
            if len(self.return_history[asset]) > 50
        ]
        
        if len(assets_with_data) < 2:
            return
        
        # Prepare return matrix
        min_length = min(len(self.return_history[asset]) for asset in assets_with_data)
        return_matrix = np.zeros((min_length, len(assets_with_data)))
        
        for i, asset in enumerate(assets_with_data):
            returns = list(self.return_history[asset])[-min_length:]
            return_matrix[:, i] = returns
        
        # Calculate correlation and covariance matrices
        self.correlation_matrix = np.corrcoef(return_matrix.T)
        self.covariance_matrix = np.cov(return_matrix.T)
        self.asset_symbols = assets_with_data
    
    def run_stress_test(self,
                       scenario_id: str,
                       portfolio_positions: Dict[str, float],
                       portfolio_value: float,
                       custom_parameters: Optional[Dict[str, Any]] = None) -> StressTestResult:
        """
        Run stress test for specific scenario.
        
        Args:
            scenario_id: ID of scenario to test
            portfolio_positions: Current portfolio positions
            portfolio_value: Current portfolio value
            custom_parameters: Custom scenario parameters
            
        Returns:
            Stress test result
        """
        timestamp = datetime.now()
        
        # Get scenario
        if scenario_id in self.scenario_library:
            scenario = self.scenario_library[scenario_id]
        elif scenario_id in self.custom_scenarios:
            scenario = self.custom_scenarios[scenario_id]
        else:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        # Apply custom parameters if provided
        if custom_parameters:
            scenario = self._customize_scenario(scenario, custom_parameters)
        
        # Execute stress test based on scenario type
        if scenario.scenario_type == "historical":
            result = self._run_historical_stress_test(scenario, portfolio_positions, portfolio_value)
        elif scenario.scenario_type == "monte_carlo":
            result = self._run_monte_carlo_stress_test(scenario, portfolio_positions, portfolio_value)
        else:
            result = self._run_custom_stress_test(scenario, portfolio_positions, portfolio_value)
        
        # Store result
        self.stress_results.append(result)
        self.scenario_performance[scenario_id].append(result.portfolio_impact.get('total_loss', 0.0))
        
        return result
    
    def _customize_scenario(self,
                          base_scenario: StressScenario,
                          custom_parameters: Dict[str, Any]) -> StressScenario:
        """Customize scenario with new parameters."""
        # Create a copy of the base scenario
        custom_scenario = StressScenario(
            scenario_id=f"{base_scenario.scenario_id}_custom",
            scenario_name=f"{base_scenario.scenario_name} (Custom)",
            scenario_type=base_scenario.scenario_type,
            description=f"Customized: {base_scenario.description}",
            parameters={**base_scenario.parameters, **custom_parameters},
            severity_level=base_scenario.severity_level,
            probability=base_scenario.probability,
            asset_shocks=base_scenario.asset_shocks.copy(),
            correlation_shocks=base_scenario.correlation_shocks.copy(),
            volatility_shocks=base_scenario.volatility_shocks.copy(),
            liquidity_shocks=base_scenario.liquidity_shocks.copy()
        )
        
        # Apply customizations
        if 'severity_multiplier' in custom_parameters:
            multiplier = custom_parameters['severity_multiplier']
            for asset in custom_scenario.asset_shocks:
                custom_scenario.asset_shocks[asset] *= multiplier
        
        return custom_scenario
    
    def _run_historical_stress_test(self,
                                  scenario: StressScenario,
                                  portfolio_positions: Dict[str, float],
                                  portfolio_value: float) -> StressTestResult:
        """Run historical stress test."""
        # Apply asset shocks
        portfolio_impact = {}
        asset_impacts = {}
        
        total_loss = 0.0
        for asset, position in portfolio_positions.items():
            # Find applicable shock
            asset_shock = 0.0
            for shock_category, shock_value in scenario.asset_shocks.items():
                if self._asset_matches_category(asset, shock_category):
                    asset_shock = max(asset_shock, abs(shock_value))
            
            # Calculate impact
            position_value = position * portfolio_value
            asset_loss = position_value * asset_shock
            total_loss += asset_loss
            
            asset_impacts[asset] = asset_loss
        
        portfolio_impact = {
            'total_loss': total_loss,
            'total_loss_percentage': total_loss / portfolio_value if portfolio_value > 0 else 0,
            'remaining_value': portfolio_value - total_loss
        }
        
        # Calculate risk metrics
        risk_metrics = self._calculate_stress_risk_metrics(
            scenario, portfolio_positions, portfolio_value, total_loss
        )
        
        # Performance metrics
        performance_metrics = {
            'max_drawdown': total_loss / portfolio_value if portfolio_value > 0 else 0,
            'recovery_probability': max(0.0, 1.0 - risk_metrics.get('stress_severity_score', 0.5)),
            'var_breach_magnitude': max(0.0, total_loss - risk_metrics.get('current_var', 0))
        }
        
        # Liquidity impact
        liquidity_impact = self._calculate_liquidity_impact(scenario, portfolio_positions)
        
        # Capital adequacy
        capital_adequacy = self._calculate_capital_adequacy(portfolio_value, total_loss)
        
        # Recovery time estimate
        recovery_time = self._estimate_recovery_time(scenario, total_loss, portfolio_value)
        
        # Generate recommendations
        recommendations = self._generate_stress_recommendations(
            scenario, portfolio_impact, risk_metrics
        )
        
        return StressTestResult(
            timestamp=datetime.now(),
            scenario_id=scenario.scenario_id,
            portfolio_impact=portfolio_impact,
            asset_impacts=asset_impacts,
            risk_metrics=risk_metrics,
            performance_metrics=performance_metrics,
            liquidity_impact=liquidity_impact,
            capital_adequacy=capital_adequacy,
            recovery_time_estimate=recovery_time,
            stress_severity=scenario.severity_level,
            recommendations=recommendations,
            metadata={
                'scenario_type': scenario.scenario_type,
                'scenario_probability': scenario.probability
            }
        )
    
    def _run_monte_carlo_stress_test(self,
                                   scenario: StressScenario,
                                   portfolio_positions: Dict[str, float],
                                   portfolio_value: float) -> StressTestResult:
        """Run Monte Carlo stress test."""
        if self.covariance_matrix is None or len(self.asset_symbols) == 0:
            # Fallback to simple stress test
            return self._run_custom_stress_test(scenario, portfolio_positions, portfolio_value)
        
        # Monte Carlo parameters
        n_sims = self.monte_carlo_config.n_simulations
        time_horizon = self.monte_carlo_config.time_horizon
        
        # Portfolio weights
        weights = np.zeros(len(self.asset_symbols))
        for i, asset in enumerate(self.asset_symbols):
            weights[i] = portfolio_positions.get(asset, 0.0)
        
        # Generate random returns with stress adjustments
        stressed_covariance = self._apply_stress_to_covariance(scenario)
        
        # Generate scenarios
        portfolio_returns = []
        for _ in range(n_sims):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(
                np.zeros(len(self.asset_symbols)), 
                stressed_covariance, 
                time_horizon
            )
            
            # Calculate cumulative portfolio return
            daily_portfolio_returns = random_returns @ weights
            cumulative_return = np.prod(1 + daily_portfolio_returns) - 1
            portfolio_returns.append(cumulative_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate losses (negative returns)
        portfolio_losses = -portfolio_returns * portfolio_value
        portfolio_losses = portfolio_losses[portfolio_losses > 0]  # Only losses
        
        # Portfolio impact statistics
        if len(portfolio_losses) > 0:
            total_loss = np.mean(portfolio_losses)
            max_loss = np.max(portfolio_losses)
            var_95 = np.percentile(portfolio_losses, 95)
            var_99 = np.percentile(portfolio_losses, 99)
        else:
            total_loss = max_loss = var_95 = var_99 = 0.0
        
        portfolio_impact = {
            'expected_loss': total_loss,
            'maximum_loss': max_loss,
            'var_95': var_95,
            'var_99': var_99,
            'loss_probability': len(portfolio_losses) / n_sims,
            'total_loss': total_loss,  # For compatibility
            'total_loss_percentage': total_loss / portfolio_value if portfolio_value > 0 else 0
        }
        
        # Asset-level impacts (simplified)
        asset_impacts = {}
        for asset in portfolio_positions:
            asset_shock = scenario.asset_shocks.get(asset, 0.1)  # Default 10% shock
            asset_impacts[asset] = portfolio_positions[asset] * portfolio_value * asset_shock
        
        # Risk metrics
        risk_metrics = {
            'monte_carlo_var_95': var_95,
            'monte_carlo_var_99': var_99,
            'expected_shortfall': np.mean(portfolio_losses[portfolio_losses > var_95]) if len(portfolio_losses[portfolio_losses > var_95]) > 0 else 0,
            'tail_expectation': np.mean(portfolio_losses[portfolio_losses > var_99]) if len(portfolio_losses[portfolio_losses > var_99]) > 0 else 0,
            'simulation_count': n_sims,
            'time_horizon_days': time_horizon
        }
        
        # Other metrics (reuse from historical method)
        performance_metrics = {
            'max_drawdown': max_loss / portfolio_value if portfolio_value > 0 else 0,
            'recovery_probability': max(0.0, 1.0 - portfolio_impact['loss_probability']),
            'var_breach_magnitude': max(0.0, var_99 - var_95)
        }
        
        liquidity_impact = self._calculate_liquidity_impact(scenario, portfolio_positions)
        capital_adequacy = self._calculate_capital_adequacy(portfolio_value, total_loss)
        recovery_time = self._estimate_recovery_time(scenario, total_loss, portfolio_value)
        recommendations = self._generate_stress_recommendations(scenario, portfolio_impact, risk_metrics)
        
        return StressTestResult(
            timestamp=datetime.now(),
            scenario_id=scenario.scenario_id,
            portfolio_impact=portfolio_impact,
            asset_impacts=asset_impacts,
            risk_metrics=risk_metrics,
            performance_metrics=performance_metrics,
            liquidity_impact=liquidity_impact,
            capital_adequacy=capital_adequacy,
            recovery_time_estimate=recovery_time,
            stress_severity=scenario.severity_level,
            recommendations=recommendations,
            metadata={
                'scenario_type': 'monte_carlo',
                'simulation_parameters': {
                    'n_simulations': n_sims,
                    'time_horizon': time_horizon,
                    'confidence_levels': self.monte_carlo_config.confidence_levels
                }
            }
        )
    
    def _run_custom_stress_test(self,
                              scenario: StressScenario,
                              portfolio_positions: Dict[str, float],
                              portfolio_value: float) -> StressTestResult:
        """Run custom stress test."""
        # Similar to historical but with more flexible shock application
        return self._run_historical_stress_test(scenario, portfolio_positions, portfolio_value)
    
    def _apply_stress_to_covariance(self, scenario: StressScenario) -> np.ndarray:
        """Apply stress scenario to covariance matrix."""
        if self.covariance_matrix is None:
            return np.eye(len(self.asset_symbols)) * 0.02**2  # Default covariance
        
        stressed_cov = self.covariance_matrix.copy()
        
        # Apply volatility shocks
        for shock_category, shock_multiplier in scenario.volatility_shocks.items():
            for i, asset in enumerate(self.asset_symbols):
                if self._asset_matches_category(asset, shock_category):
                    # Scale diagonal elements (variances)
                    stressed_cov[i, i] *= shock_multiplier**2
        
        # Apply correlation shocks
        for shock_category, target_correlation in scenario.correlation_shocks.items():
            # Simplified: increase all correlations toward target
            if shock_category == "all_assets":
                correlation_matrix = np.corrcoef(stressed_cov)
                # Blend with target correlation
                blend_factor = 0.5
                target_corr_matrix = np.full_like(correlation_matrix, target_correlation)
                np.fill_diagonal(target_corr_matrix, 1.0)
                
                new_corr = (1 - blend_factor) * correlation_matrix + blend_factor * target_corr_matrix
                
                # Convert back to covariance
                volatilities = np.sqrt(np.diag(stressed_cov))
                stressed_cov = np.outer(volatilities, volatilities) * new_corr
        
        return stressed_cov
    
    def _asset_matches_category(self, asset: str, category: str) -> bool:
        """Check if asset matches shock category."""
        asset_lower = asset.lower()
        category_lower = category.lower()
        
        category_mappings = {
            'all_assets': True,
            'crypto': any(crypto in asset_lower for crypto in ['btc', 'eth', 'bitcoin', 'ethereum', 'crypto']),
            'equity': any(eq in asset_lower for eq in ['stock', 'equity', 'shares']),
            'bonds': any(bond in asset_lower for bond in ['bond', 'treasury', 'corporate']),
            'commodities': any(comm in asset_lower for comm in ['gold', 'oil', 'copper', 'commodity']),
            'bitcoin': 'btc' in asset_lower or 'bitcoin' in asset_lower,
            'ethereum': 'eth' in asset_lower or 'ethereum' in asset_lower,
            'altcoins': ('crypto' in asset_lower or 'coin' in asset_lower) and not ('btc' in asset_lower or 'eth' in asset_lower)
        }
        
        return category_mappings.get(category_lower, category_lower in asset_lower)
    
    def _calculate_stress_risk_metrics(self,
                                     scenario: StressScenario,
                                     portfolio_positions: Dict[str, float],
                                     portfolio_value: float,
                                     total_loss: float) -> Dict[str, float]:
        """Calculate risk metrics under stress."""
        metrics = {}
        
        # Stress severity score
        severity_scores = {'mild': 0.2, 'moderate': 0.4, 'severe': 0.7, 'extreme': 1.0}
        metrics['stress_severity_score'] = severity_scores.get(scenario.severity_level, 0.5)
        
        # Portfolio concentration under stress
        position_values = [abs(pos) for pos in portfolio_positions.values()]
        if position_values:
            max_position = max(position_values)
            metrics['concentration_risk'] = max_position
            
            # Herfindahl index
            total_abs_positions = sum(position_values)
            if total_abs_positions > 0:
                normalized_positions = [pos / total_abs_positions for pos in position_values]
                metrics['herfindahl_index'] = sum(pos**2 for pos in normalized_positions)
        
        # Liquidity risk under stress
        liquidity_risk = 0.0
        for asset, position in portfolio_positions.items():
            asset_liquidity_shock = 0.0
            for category, shock in scenario.liquidity_shocks.items():
                if self._asset_matches_category(asset, category):
                    asset_liquidity_shock = max(asset_liquidity_shock, abs(shock))
            
            position_weight = abs(position)
            liquidity_risk += position_weight * asset_liquidity_shock
        
        metrics['liquidity_risk'] = liquidity_risk
        
        # Correlation risk
        if scenario.correlation_shocks:
            avg_correlation_shock = np.mean(list(scenario.correlation_shocks.values()))
            metrics['correlation_risk'] = avg_correlation_shock
        
        # Tail risk
        metrics['tail_risk'] = total_loss / portfolio_value if portfolio_value > 0 else 0
        
        return metrics
    
    def _calculate_liquidity_impact(self,
                                  scenario: StressScenario,
                                  portfolio_positions: Dict[str, float]) -> Dict[str, float]:
        """Calculate liquidity impact under stress."""
        impact = {
            'total_liquidity_loss': 0.0,
            'assets_affected': 0,
            'liquidity_score': 1.0
        }
        
        affected_assets = 0
        total_liquidity_impact = 0.0
        
        for asset, position in portfolio_positions.items():
            asset_liquidity_shock = 0.0
            for category, shock in scenario.liquidity_shocks.items():
                if self._asset_matches_category(asset, category):
                    asset_liquidity_shock = max(asset_liquidity_shock, abs(shock))
            
            if asset_liquidity_shock > 0:
                affected_assets += 1
                total_liquidity_impact += abs(position) * asset_liquidity_shock
        
        impact['assets_affected'] = affected_assets
        impact['total_liquidity_loss'] = total_liquidity_impact
        impact['liquidity_score'] = max(0.0, 1.0 - total_liquidity_impact)
        
        return impact
    
    def _calculate_capital_adequacy(self,
                                  portfolio_value: float,
                                  stress_loss: float) -> Dict[str, float]:
        """Calculate capital adequacy under stress."""
        remaining_capital = portfolio_value - stress_loss
        
        adequacy = {
            'remaining_capital': remaining_capital,
            'capital_ratio': remaining_capital / portfolio_value if portfolio_value > 0 else 0,
            'stress_test_passed': remaining_capital > 0,
            'buffer_ratio': remaining_capital / stress_loss if stress_loss > 0 else float('inf')
        }
        
        return adequacy
    
    def _estimate_recovery_time(self,
                              scenario: StressScenario,
                              stress_loss: float,
                              portfolio_value: float) -> float:
        """Estimate recovery time from stress scenario."""
        # Simple heuristic based on severity and historical data
        severity_multipliers = {
            'mild': 1.0,
            'moderate': 2.0,
            'severe': 6.0,
            'extreme': 12.0
        }
        
        base_recovery_months = severity_multipliers.get(scenario.severity_level, 3.0)
        
        # Adjust based on loss magnitude
        loss_ratio = stress_loss / portfolio_value if portfolio_value > 0 else 0
        loss_adjustment = 1.0 + (loss_ratio * 10)  # Scale by loss severity
        
        estimated_months = base_recovery_months * loss_adjustment
        
        return estimated_months
    
    def _generate_stress_recommendations(self,
                                       scenario: StressScenario,
                                       portfolio_impact: Dict[str, float],
                                       risk_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        # Loss-based recommendations
        loss_ratio = portfolio_impact.get('total_loss_percentage', 0)
        if loss_ratio > 0.2:
            recommendations.append("Consider reducing portfolio risk through diversification")
        if loss_ratio > 0.1:
            recommendations.append("Review position sizing to limit maximum loss per scenario")
        
        # Concentration recommendations
        concentration = risk_metrics.get('concentration_risk', 0)
        if concentration > 0.3:
            recommendations.append("High concentration risk - consider reducing large positions")
        
        # Liquidity recommendations
        liquidity_risk = risk_metrics.get('liquidity_risk', 0)
        if liquidity_risk > 0.3:
            recommendations.append("Improve portfolio liquidity to handle stress scenarios")
        
        # Correlation recommendations
        correlation_risk = risk_metrics.get('correlation_risk', 0)
        if correlation_risk > 0.8:
            recommendations.append("Add uncorrelated assets to reduce correlation risk")
        
        # Scenario-specific recommendations
        if scenario.scenario_type == "regulatory":
            recommendations.append("Monitor regulatory developments and prepare compliance strategies")
        elif "liquidity" in scenario.scenario_id:
            recommendations.append("Maintain adequate cash reserves for liquidity crises")
        elif "crypto" in scenario.scenario_id:
            recommendations.append("Consider crypto-specific hedging strategies")
        
        return recommendations
    
    def run_comprehensive_stress_test(self,
                                    portfolio_positions: Dict[str, float],
                                    portfolio_value: float) -> Dict[str, StressTestResult]:
        """Run comprehensive stress test across all scenarios."""
        results = {}
        
        for scenario_id in self.scenario_library:
            try:
                result = self.run_stress_test(scenario_id, portfolio_positions, portfolio_value)
                results[scenario_id] = result
            except Exception as e:
                print(f"Error running stress test for {scenario_id}: {e}")
        
        return results
    
    def create_custom_scenario(self,
                             scenario_id: str,
                             scenario_name: str,
                             asset_shocks: Dict[str, float],
                             description: str = "",
                             severity_level: str = "moderate") -> StressScenario:
        """Create custom stress scenario."""
        scenario = StressScenario(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            scenario_type="custom",
            description=description or f"Custom scenario: {scenario_name}",
            parameters={},
            severity_level=severity_level,
            probability=0.1,  # Default probability
            asset_shocks=asset_shocks
        )
        
        self.custom_scenarios[scenario_id] = scenario
        return scenario
    
    def get_stress_test_summary(self) -> Dict[str, Any]:
        """Get comprehensive stress test summary."""
        if not self.stress_results:
            return {'status': 'no_stress_tests'}
        
        recent_results = list(self.stress_results)[-10:]
        
        summary = {
            'total_stress_tests': len(self.stress_results),
            'scenarios_tested': len(set(r.scenario_id for r in recent_results)),
            'last_test_time': recent_results[-1].timestamp.isoformat() if recent_results else None,
            'current_stress_level': self.current_stress_level
        }
        
        # Aggregate statistics
        loss_ratios = [r.portfolio_impact.get('total_loss_percentage', 0) for r in recent_results]
        if loss_ratios:
            summary['stress_statistics'] = {
                'avg_loss_ratio': np.mean(loss_ratios),
                'max_loss_ratio': np.max(loss_ratios),
                'min_loss_ratio': np.min(loss_ratios),
                'stress_volatility': np.std(loss_ratios)
            }
        
        # Worst-case scenarios
        worst_scenarios = sorted(recent_results, key=lambda x: x.portfolio_impact.get('total_loss_percentage', 0), reverse=True)[:3]
        summary['worst_scenarios'] = [
            {
                'scenario_id': r.scenario_id,
                'loss_percentage': r.portfolio_impact.get('total_loss_percentage', 0),
                'severity': r.stress_severity
            }
            for r in worst_scenarios
        ]
        
        return summary
    
    def get_scenario_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance of different stress scenarios."""
        analysis = {}
        
        for scenario_id, performance_history in self.scenario_performance.items():
            if len(performance_history) > 1:
                analysis[scenario_id] = {
                    'test_count': len(performance_history),
                    'avg_loss': np.mean(performance_history),
                    'max_loss': np.max(performance_history),
                    'loss_volatility': np.std(performance_history),
                    'trend': 'increasing' if performance_history[-1] > performance_history[0] else 'decreasing'
                }
        
        return analysis