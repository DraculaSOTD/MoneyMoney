"""
Portfolio Risk Manager for Advanced Risk Assessment.

Implements comprehensive portfolio-level risk management including VaR, CVaR,
risk decomposition, and dynamic risk monitoring.
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
class VaRResult:
    """Value at Risk calculation result."""
    timestamp: datetime
    confidence_level: float
    time_horizon: int  # days
    var_value: float
    cvar_value: float
    method: str
    portfolio_value: float
    risk_contribution: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecomposition:
    """Risk decomposition analysis."""
    timestamp: datetime
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: Dict[str, float]
    asset_contributions: Dict[str, float]
    concentration_metrics: Dict[str, float]
    diversification_ratio: float


@dataclass
class RiskAlert:
    """Risk alert notification."""
    timestamp: datetime
    alert_type: str
    severity: str  # low, medium, high, critical
    description: str
    current_value: float
    threshold_value: float
    affected_assets: List[str]
    recommended_actions: List[str]
    auto_triggered: bool = False


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio risk metrics."""
    timestamp: datetime
    portfolio_value: float
    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    risk_metrics: Dict[str, float]
    exposure_metrics: Dict[str, float]
    liquidity_metrics: Dict[str, float]


class PortfolioRiskManager:
    """
    Advanced portfolio risk management system.
    
    Features:
    - Multiple VaR methodologies (parametric, historical, Monte Carlo)
    - Conditional VaR (Expected Shortfall) calculation
    - Risk decomposition and attribution
    - Dynamic correlation and covariance estimation
    - Real-time risk monitoring and alerting
    - Stress testing integration
    - Liquidity risk assessment
    - Concentration risk analysis
    """
    
    def __init__(self,
                 confidence_levels: List[float] = None,
                 time_horizons: List[int] = None,
                 monitoring_frequency: int = 5,  # minutes
                 alert_thresholds: Dict[str, float] = None):
        """
        Initialize portfolio risk manager.
        
        Args:
            confidence_levels: VaR confidence levels
            time_horizons: Time horizons for risk calculation (days)
            monitoring_frequency: Risk monitoring frequency (minutes)
            alert_thresholds: Risk alert thresholds
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.999]
        self.time_horizons = time_horizons or [1, 5, 10, 22]  # 1D, 1W, 2W, 1M
        self.monitoring_frequency = monitoring_frequency
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'var_95_daily': 0.05,  # 5% daily VaR threshold
            'max_drawdown': 0.15,  # 15% maximum drawdown
            'concentration': 0.3,  # 30% single asset concentration
            'correlation_spike': 0.9,  # 90% correlation spike
            'liquidity_risk': 0.2  # 20% liquidity risk
        }
        
        # Data storage
        self.portfolio_data: deque = deque(maxlen=1000)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.position_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        # Risk calculations
        self.var_results: deque = deque(maxlen=500)
        self.risk_decompositions: deque = deque(maxlen=200)
        self.risk_alerts: deque = deque(maxlen=100)
        self.portfolio_metrics: deque = deque(maxlen=1000)
        
        # Covariance and correlation tracking
        self.covariance_matrix: Optional[np.ndarray] = None
        self.correlation_matrix: Optional[np.ndarray] = None
        self.asset_symbols: List[str] = []
        
        # Risk monitoring state
        self.last_risk_check: Optional[datetime] = None
        self.current_risk_level: str = "normal"  # normal, elevated, high, critical
        
        # VaR calculation methods
        self.var_methods = {
            'parametric': self._parametric_var,
            'historical': self._historical_var,
            'monte_carlo': self._monte_carlo_var,
            'cornish_fisher': self._cornish_fisher_var
        }
    
    def update_portfolio_data(self,
                            portfolio_positions: Dict[str, float],
                            asset_prices: Dict[str, float],
                            timestamp: Optional[datetime] = None) -> None:
        """
        Update portfolio data for risk calculations.
        
        Args:
            portfolio_positions: Current positions {asset: weight}
            asset_prices: Current asset prices {asset: price}
            timestamp: Data timestamp
        """
        timestamp = timestamp or datetime.now()
        
        # Store portfolio snapshot
        portfolio_value = sum(
            position * asset_prices.get(asset, 0) 
            for asset, position in portfolio_positions.items()
        )
        
        portfolio_snapshot = {
            'timestamp': timestamp,
            'positions': portfolio_positions.copy(),
            'prices': asset_prices.copy(),
            'portfolio_value': portfolio_value
        }
        
        self.portfolio_data.append(portfolio_snapshot)
        
        # Update price history
        for asset, price in asset_prices.items():
            self.price_history[asset].append({
                'timestamp': timestamp,
                'price': price
            })
        
        # Update position history
        for asset, position in portfolio_positions.items():
            self.position_history[asset].append({
                'timestamp': timestamp,
                'position': position
            })
        
        # Update return history
        self._update_return_history()
        
        # Update covariance matrix
        self._update_covariance_matrix()
        
        # Trigger risk monitoring
        self._check_risk_alerts(timestamp)
    
    def _update_return_history(self) -> None:
        """Update return history for all assets."""
        for asset in self.price_history:
            prices = [p['price'] for p in self.price_history[asset]]
            
            if len(prices) > 1:
                # Calculate recent returns
                recent_returns = []
                for i in range(max(1, len(prices)-10), len(prices)):
                    if prices[i-1] != 0:
                        return_val = (prices[i] - prices[i-1]) / prices[i-1]
                        recent_returns.append(return_val)
                
                # Store returns
                for ret in recent_returns:
                    self.return_history[asset].append(ret)
    
    def _update_covariance_matrix(self) -> None:
        """Update covariance and correlation matrices."""
        # Get assets with sufficient return data
        assets_with_data = [
            asset for asset in self.return_history
            if len(self.return_history[asset]) > 20
        ]
        
        if len(assets_with_data) < 2:
            return
        
        # Prepare return matrix
        min_length = min(len(self.return_history[asset]) for asset in assets_with_data)
        return_matrix = np.zeros((min_length, len(assets_with_data)))
        
        for i, asset in enumerate(assets_with_data):
            returns = list(self.return_history[asset])[-min_length:]
            return_matrix[:, i] = returns
        
        # Calculate covariance and correlation
        self.covariance_matrix = np.cov(return_matrix.T)
        self.correlation_matrix = np.corrcoef(return_matrix.T)
        self.asset_symbols = assets_with_data
    
    def calculate_var(self,
                     confidence_level: float = 0.95,
                     time_horizon: int = 1,
                     method: str = 'parametric') -> VaRResult:
        """
        Calculate Value at Risk for portfolio.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: VaR calculation method
            
        Returns:
            VaR calculation result
        """
        if not self.portfolio_data:
            raise ValueError("No portfolio data available for VaR calculation")
        
        timestamp = datetime.now()
        
        # Get VaR calculation method
        var_func = self.var_methods.get(method, self._parametric_var)
        
        # Calculate VaR
        var_result = var_func(confidence_level, time_horizon, timestamp)
        
        # Calculate CVaR (Expected Shortfall)
        cvar_value = self._calculate_cvar(confidence_level, time_horizon, method)
        var_result.cvar_value = cvar_value
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(var_result.var_value)
        var_result.risk_contribution = risk_contributions
        
        # Store result
        self.var_results.append(var_result)
        
        return var_result
    
    def _parametric_var(self,
                       confidence_level: float,
                       time_horizon: int,
                       timestamp: datetime) -> VaRResult:
        """Parametric VaR calculation."""
        if not self.portfolio_data:
            raise ValueError("No portfolio data")
        
        latest_portfolio = self.portfolio_data[-1]
        portfolio_value = latest_portfolio['portfolio_value']
        positions = latest_portfolio['positions']
        
        if self.covariance_matrix is None or len(self.asset_symbols) == 0:
            # Fallback calculation
            portfolio_volatility = 0.02  # Default 2% daily volatility
        else:
            # Calculate portfolio weights
            weights = np.zeros(len(self.asset_symbols))
            for i, asset in enumerate(self.asset_symbols):
                weights[i] = positions.get(asset, 0.0)
            
            # Portfolio variance
            portfolio_variance = weights.T @ self.covariance_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Scale for time horizon
        scaled_volatility = portfolio_volatility * np.sqrt(time_horizon)
        
        # Z-score for confidence level
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)
        
        # VaR calculation (positive value)
        var_value = abs(z_score * scaled_volatility * portfolio_value)
        
        return VaRResult(
            timestamp=timestamp,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            var_value=var_value,
            cvar_value=0.0,  # Will be calculated separately
            method='parametric',
            portfolio_value=portfolio_value,
            metadata={
                'portfolio_volatility': portfolio_volatility,
                'scaled_volatility': scaled_volatility,
                'z_score': z_score
            }
        )
    
    def _historical_var(self,
                       confidence_level: float,
                       time_horizon: int,
                       timestamp: datetime) -> VaRResult:
        """Historical VaR calculation."""
        if len(self.portfolio_data) < 100:
            # Fall back to parametric if insufficient data
            return self._parametric_var(confidence_level, time_horizon, timestamp)
        
        # Calculate historical portfolio returns
        portfolio_returns = []
        
        for i in range(1, len(self.portfolio_data)):
            prev_portfolio = self.portfolio_data[i-1]
            curr_portfolio = self.portfolio_data[i]
            
            if prev_portfolio['portfolio_value'] != 0:
                return_val = ((curr_portfolio['portfolio_value'] - 
                              prev_portfolio['portfolio_value']) / 
                              prev_portfolio['portfolio_value'])
                portfolio_returns.append(return_val)
        
        if not portfolio_returns:
            return self._parametric_var(confidence_level, time_horizon, timestamp)
        
        portfolio_returns = np.array(portfolio_returns)
        current_portfolio_value = self.portfolio_data[-1]['portfolio_value']
        
        # Scale for time horizon (approximate)
        if time_horizon > 1:
            # Simple scaling - in practice would use more sophisticated methods
            portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # Historical VaR (percentile)
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(portfolio_returns, var_percentile)
        var_value = abs(var_return * current_portfolio_value)
        
        return VaRResult(
            timestamp=timestamp,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            var_value=var_value,
            cvar_value=0.0,
            method='historical',
            portfolio_value=current_portfolio_value,
            metadata={
                'historical_returns_count': len(portfolio_returns),
                'var_return': var_return,
                'return_percentile': var_percentile
            }
        )
    
    def _monte_carlo_var(self,
                        confidence_level: float,
                        time_horizon: int,
                        timestamp: datetime) -> VaRResult:
        """Monte Carlo VaR calculation."""
        if self.covariance_matrix is None or len(self.asset_symbols) == 0:
            return self._parametric_var(confidence_level, time_horizon, timestamp)
        
        latest_portfolio = self.portfolio_data[-1]
        portfolio_value = latest_portfolio['portfolio_value']
        positions = latest_portfolio['positions']
        
        # Portfolio weights
        weights = np.zeros(len(self.asset_symbols))
        for i, asset in enumerate(self.asset_symbols):
            weights[i] = positions.get(asset, 0.0)
        
        # Monte Carlo simulation parameters
        n_simulations = 10000
        
        # Generate random returns
        mean_returns = np.zeros(len(self.asset_symbols))  # Assume zero mean for risk calculation
        
        # Generate multivariate normal random returns
        random_returns = np.random.multivariate_normal(
            mean_returns, self.covariance_matrix, n_simulations
        )
        
        # Scale for time horizon
        if time_horizon > 1:
            random_returns *= np.sqrt(time_horizon)
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = random_returns @ weights
        
        # Calculate portfolio values
        portfolio_values = portfolio_value * (1 + portfolio_returns)
        portfolio_losses = portfolio_value - portfolio_values
        
        # VaR as percentile of losses
        var_percentile = confidence_level * 100
        var_value = np.percentile(portfolio_losses, var_percentile)
        
        return VaRResult(
            timestamp=timestamp,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            var_value=max(0, var_value),
            cvar_value=0.0,
            method='monte_carlo',
            portfolio_value=portfolio_value,
            metadata={
                'n_simulations': n_simulations,
                'portfolio_returns_std': np.std(portfolio_returns),
                'portfolio_returns_mean': np.mean(portfolio_returns)
            }
        )
    
    def _cornish_fisher_var(self,
                           confidence_level: float,
                           time_horizon: int,
                           timestamp: datetime) -> VaRResult:
        """Cornish-Fisher expansion VaR (accounts for skewness and kurtosis)."""
        # First calculate parametric VaR
        parametric_result = self._parametric_var(confidence_level, time_horizon, timestamp)
        
        # Calculate higher moments if we have sufficient return data
        if len(self.portfolio_data) < 100:
            return parametric_result
        
        # Calculate portfolio returns
        portfolio_returns = []
        for i in range(1, len(self.portfolio_data)):
            prev_portfolio = self.portfolio_data[i-1]
            curr_portfolio = self.portfolio_data[i]
            
            if prev_portfolio['portfolio_value'] != 0:
                return_val = ((curr_portfolio['portfolio_value'] - 
                              prev_portfolio['portfolio_value']) / 
                              prev_portfolio['portfolio_value'])
                portfolio_returns.append(return_val)
        
        if len(portfolio_returns) < 50:
            return parametric_result
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate moments
        returns_std = np.std(portfolio_returns)
        skewness = self._calculate_skewness(portfolio_returns)
        kurtosis = self._calculate_kurtosis(portfolio_returns)
        
        # Cornish-Fisher adjustment
        from scipy.stats import norm
        z = norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher expansion
        cf_adjustment = (
            z + 
            (z**2 - 1) * skewness / 6 + 
            (z**3 - 3*z) * (kurtosis - 3) / 24 - 
            (2*z**3 - 5*z) * skewness**2 / 36
        )
        
        # Adjusted VaR
        portfolio_value = parametric_result.portfolio_value
        cf_volatility = returns_std * np.sqrt(time_horizon)
        cf_var = abs(cf_adjustment * cf_volatility * portfolio_value)
        
        return VaRResult(
            timestamp=timestamp,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            var_value=cf_var,
            cvar_value=0.0,
            method='cornish_fisher',
            portfolio_value=portfolio_value,
            metadata={
                'skewness': skewness,
                'kurtosis': kurtosis,
                'cf_adjustment': cf_adjustment,
                'parametric_var': parametric_result.var_value
            }
        )
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        n = len(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.sum(((returns - mean_return) / std_return) ** 3) / n
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        n = len(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 3.0  # Normal distribution kurtosis
        
        kurtosis = np.sum(((returns - mean_return) / std_return) ** 4) / n
        return kurtosis
    
    def _calculate_cvar(self,
                       confidence_level: float,
                       time_horizon: int,
                       method: str) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if method == 'historical' and len(self.portfolio_data) >= 100:
            # Historical CVaR
            portfolio_returns = []
            for i in range(1, len(self.portfolio_data)):
                prev_portfolio = self.portfolio_data[i-1]
                curr_portfolio = self.portfolio_data[i]
                
                if prev_portfolio['portfolio_value'] != 0:
                    return_val = ((curr_portfolio['portfolio_value'] - 
                                  prev_portfolio['portfolio_value']) / 
                                  prev_portfolio['portfolio_value'])
                    portfolio_returns.append(return_val)
            
            if portfolio_returns:
                portfolio_returns = np.array(portfolio_returns)
                current_portfolio_value = self.portfolio_data[-1]['portfolio_value']
                
                # Scale for time horizon
                if time_horizon > 1:
                    portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
                
                # CVaR as mean of returns below VaR threshold
                var_percentile = (1 - confidence_level) * 100
                var_threshold = np.percentile(portfolio_returns, var_percentile)
                
                tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
                if len(tail_returns) > 0:
                    cvar_return = np.mean(tail_returns)
                    return abs(cvar_return * current_portfolio_value)
        
        # Parametric CVaR approximation
        if self.var_results:
            latest_var = self.var_results[-1]
            # CVaR is typically 1.2-1.4 times VaR for normal distribution
            return latest_var.var_value * 1.3
        
        return 0.0
    
    def _calculate_risk_contributions(self, total_var: float) -> Dict[str, float]:
        """Calculate individual asset risk contributions to portfolio VaR."""
        if (self.covariance_matrix is None or 
            len(self.asset_symbols) == 0 or
            not self.portfolio_data):
            return {}
        
        latest_portfolio = self.portfolio_data[-1]
        positions = latest_portfolio['positions']
        
        # Portfolio weights
        weights = np.zeros(len(self.asset_symbols))
        for i, asset in enumerate(self.asset_symbols):
            weights[i] = positions.get(asset, 0.0)
        
        # Portfolio variance
        portfolio_variance = weights.T @ self.covariance_matrix @ weights
        
        if portfolio_variance <= 0:
            return {}
        
        # Marginal VaR contributions
        marginal_contributions = (self.covariance_matrix @ weights) / np.sqrt(portfolio_variance)
        
        # Component VaR
        component_vars = weights * marginal_contributions * total_var
        
        # Convert to dictionary
        risk_contributions = {}
        for i, asset in enumerate(self.asset_symbols):
            risk_contributions[asset] = float(component_vars[i])
        
        return risk_contributions
    
    def calculate_risk_decomposition(self) -> RiskDecomposition:
        """Calculate comprehensive risk decomposition."""
        timestamp = datetime.now()
        
        if not self.portfolio_data or self.covariance_matrix is None:
            return RiskDecomposition(
                timestamp=timestamp,
                total_risk=0.0,
                systematic_risk=0.0,
                idiosyncratic_risk=0.0,
                factor_contributions={},
                asset_contributions={},
                concentration_metrics={},
                diversification_ratio=1.0
            )
        
        latest_portfolio = self.portfolio_data[-1]
        positions = latest_portfolio['positions']
        
        # Portfolio weights
        weights = np.zeros(len(self.asset_symbols))
        for i, asset in enumerate(self.asset_symbols):
            weights[i] = positions.get(asset, 0.0)
        
        # Total portfolio risk
        portfolio_variance = weights.T @ self.covariance_matrix @ weights
        total_risk = np.sqrt(portfolio_variance)
        
        # Calculate systematic vs idiosyncratic risk (simplified)
        # In practice, would use factor models
        systematic_ratio = 0.7  # Assume 70% systematic risk
        systematic_risk = total_risk * systematic_ratio
        idiosyncratic_risk = total_risk * (1 - systematic_ratio)
        
        # Asset contributions
        marginal_contributions = (self.covariance_matrix @ weights) / np.sqrt(portfolio_variance) if portfolio_variance > 0 else np.zeros(len(weights))
        asset_contributions = {}
        for i, asset in enumerate(self.asset_symbols):
            asset_contributions[asset] = float(weights[i] * marginal_contributions[i] * total_risk)
        
        # Concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(weights)
        
        # Diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(weights)
        
        decomposition = RiskDecomposition(
            timestamp=timestamp,
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            factor_contributions={},  # Would be calculated with factor models
            asset_contributions=asset_contributions,
            concentration_metrics=concentration_metrics,
            diversification_ratio=diversification_ratio
        )
        
        self.risk_decompositions.append(decomposition)
        
        return decomposition
    
    def _calculate_concentration_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio concentration metrics."""
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights ** 2)
        
        # Effective number of assets
        effective_assets = 1 / hhi if hhi > 0 else 0
        
        # Maximum weight
        max_weight = np.max(weights) if len(weights) > 0 else 0
        
        # Weight entropy
        weights_positive = weights[weights > 0]
        if len(weights_positive) > 0:
            weight_entropy = -np.sum(weights_positive * np.log(weights_positive))
        else:
            weight_entropy = 0
        
        return {
            'herfindahl_index': float(hhi),
            'effective_assets': float(effective_assets),
            'max_weight': float(max_weight),
            'weight_entropy': float(weight_entropy)
        }
    
    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate diversification ratio."""
        if self.covariance_matrix is None or len(weights) == 0:
            return 1.0
        
        # Weighted average volatility
        asset_volatilities = np.sqrt(np.diag(self.covariance_matrix))
        weighted_avg_vol = np.sum(weights * asset_volatilities)
        
        # Portfolio volatility
        portfolio_variance = weights.T @ self.covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Diversification ratio
        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        else:
            return 1.0
    
    def _check_risk_alerts(self, timestamp: datetime) -> None:
        """Check for risk alert conditions."""
        # Skip if too soon since last check
        if (self.last_risk_check and 
            (timestamp - self.last_risk_check).total_seconds() < self.monitoring_frequency * 60):
            return
        
        self.last_risk_check = timestamp
        
        # Calculate current VaR
        try:
            current_var = self.calculate_var(confidence_level=0.95, time_horizon=1)
            
            # Check VaR threshold
            var_ratio = current_var.var_value / current_var.portfolio_value
            if var_ratio > self.alert_thresholds['var_95_daily']:
                alert = RiskAlert(
                    timestamp=timestamp,
                    alert_type="var_breach",
                    severity="high",
                    description=f"Daily VaR exceeds threshold: {var_ratio:.2%}",
                    current_value=var_ratio,
                    threshold_value=self.alert_thresholds['var_95_daily'],
                    affected_assets=list(current_var.risk_contribution.keys()),
                    recommended_actions=["Review position sizes", "Consider hedging"],
                    auto_triggered=True
                )
                self.risk_alerts.append(alert)
                self.current_risk_level = "high"
        
        except Exception as e:
            print(f"Error calculating VaR for alert check: {e}")
        
        # Check concentration risk
        if self.portfolio_data:
            latest_portfolio = self.portfolio_data[-1]
            positions = latest_portfolio['positions']
            
            max_position = max(positions.values()) if positions else 0
            if max_position > self.alert_thresholds['concentration']:
                alert = RiskAlert(
                    timestamp=timestamp,
                    alert_type="concentration_risk",
                    severity="medium",
                    description=f"High concentration in single asset: {max_position:.2%}",
                    current_value=max_position,
                    threshold_value=self.alert_thresholds['concentration'],
                    affected_assets=[asset for asset, pos in positions.items() if pos == max_position],
                    recommended_actions=["Reduce position size", "Diversify holdings"],
                    auto_triggered=True
                )
                self.risk_alerts.append(alert)
        
        # Check correlation spike
        if self.correlation_matrix is not None:
            max_correlation = np.max(self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)])
            if max_correlation > self.alert_thresholds['correlation_spike']:
                alert = RiskAlert(
                    timestamp=timestamp,
                    alert_type="correlation_spike",
                    severity="medium",
                    description=f"High correlation detected: {max_correlation:.2f}",
                    current_value=max_correlation,
                    threshold_value=self.alert_thresholds['correlation_spike'],
                    affected_assets=self.asset_symbols,
                    recommended_actions=["Review diversification", "Consider uncorrelated assets"],
                    auto_triggered=True
                )
                self.risk_alerts.append(alert)
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        timestamp = datetime.now()
        
        if not self.portfolio_data:
            return PortfolioMetrics(
                timestamp=timestamp,
                portfolio_value=0.0,
                total_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                cvar_95=0.0,
                risk_metrics={},
                exposure_metrics={},
                liquidity_metrics={}
            )
        
        latest_portfolio = self.portfolio_data[-1]
        portfolio_value = latest_portfolio['portfolio_value']
        
        # Calculate returns
        portfolio_returns = []
        portfolio_values = [p['portfolio_value'] for p in self.portfolio_data]
        
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] != 0:
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                portfolio_returns.append(ret)
        
        # Basic metrics
        if portfolio_returns:
            total_return = (portfolio_value / portfolio_values[0] - 1) if portfolio_values[0] != 0 else 0
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            mean_return = np.mean(portfolio_returns) * 252  # Annualized
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
        else:
            total_return = volatility = sharpe_ratio = max_drawdown = 0.0
        
        # VaR metrics
        try:
            var_result = self.calculate_var(confidence_level=0.95, time_horizon=1)
            var_95 = var_result.var_value / portfolio_value if portfolio_value > 0 else 0
            cvar_95 = var_result.cvar_value / portfolio_value if portfolio_value > 0 else 0
        except:
            var_95 = cvar_95 = 0.0
        
        # Risk metrics
        risk_metrics = {
            'volatility': volatility,
            'beta': 1.0,  # Would calculate vs benchmark
            'correlation_avg': np.mean(self.correlation_matrix) if self.correlation_matrix is not None else 0.0,
            'tracking_error': 0.0  # Would calculate vs benchmark
        }
        
        # Exposure metrics
        positions = latest_portfolio['positions']
        exposure_metrics = {
            'gross_exposure': sum(abs(pos) for pos in positions.values()),
            'net_exposure': sum(positions.values()),
            'long_exposure': sum(pos for pos in positions.values() if pos > 0),
            'short_exposure': sum(pos for pos in positions.values() if pos < 0),
            'leverage': sum(abs(pos) for pos in positions.values())
        }
        
        # Liquidity metrics (simplified)
        liquidity_metrics = {
            'liquidity_score': 0.8,  # Would calculate based on market data
            'avg_bid_ask_spread': 0.001,  # Would get from market data
            'daily_turnover_ratio': 0.1  # Would calculate from trading data
        }
        
        metrics = PortfolioMetrics(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            total_return=total_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            risk_metrics=risk_metrics,
            exposure_metrics=exposure_metrics,
            liquidity_metrics=liquidity_metrics
        )
        
        self.portfolio_metrics.append(metrics)
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'current_risk_level': self.current_risk_level,
            'monitoring_frequency_minutes': self.monitoring_frequency,
            'alert_thresholds': self.alert_thresholds.copy()
        }
        
        # Recent VaR results
        if self.var_results:
            latest_var = self.var_results[-1]
            summary['latest_var'] = {
                'var_95_1d': latest_var.var_value,
                'cvar_95_1d': latest_var.cvar_value,
                'var_ratio': latest_var.var_value / latest_var.portfolio_value,
                'method': latest_var.method,
                'confidence': latest_var.confidence_level
            }
        
        # Recent risk decomposition
        if self.risk_decompositions:
            latest_decomp = self.risk_decompositions[-1]
            summary['risk_decomposition'] = {
                'total_risk': latest_decomp.total_risk,
                'systematic_risk': latest_decomp.systematic_risk,
                'idiosyncratic_risk': latest_decomp.idiosyncratic_risk,
                'diversification_ratio': latest_decomp.diversification_ratio,
                'concentration_metrics': latest_decomp.concentration_metrics
            }
        
        # Recent alerts
        recent_alerts = [alert for alert in self.risk_alerts if 
                        (datetime.now() - alert.timestamp).total_seconds() < 3600]  # Last hour
        summary['recent_alerts'] = [
            {
                'type': alert.alert_type,
                'severity': alert.severity,
                'description': alert.description,
                'timestamp': alert.timestamp.isoformat()
            }
            for alert in recent_alerts
        ]
        
        # Portfolio metrics
        if self.portfolio_metrics:
            latest_metrics = self.portfolio_metrics[-1]
            summary['portfolio_metrics'] = {
                'portfolio_value': latest_metrics.portfolio_value,
                'total_return': latest_metrics.total_return,
                'volatility': latest_metrics.volatility,
                'sharpe_ratio': latest_metrics.sharpe_ratio,
                'max_drawdown': latest_metrics.max_drawdown
            }
        
        return summary
    
    def get_risk_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get risk alerts from specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.risk_alerts
            if alert.timestamp >= cutoff_time
        ]
        
        return [
            {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'severity': alert.severity,
                'description': alert.description,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'affected_assets': alert.affected_assets,
                'recommended_actions': alert.recommended_actions,
                'auto_triggered': alert.auto_triggered
            }
            for alert in recent_alerts
        ]