"""
GPU-Accelerated Portfolio Optimization.

Provides advanced portfolio optimization strategies including
mean-variance optimization, risk parity, and dynamic rebalancing.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp

try:
    import cupy as cp_gpu
    CUPY_AVAILABLE = True
except ImportError:
    cp_gpu = None
    CUPY_AVAILABLE = False

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Asset:
    """Asset representation for portfolio."""
    symbol: str
    current_price: float
    quantity: float = 0.0
    value: float = 0.0
    weight: float = 0.0
    expected_return: float = 0.0
    volatility: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional Value at Risk
    diversification_ratio: float
    effective_assets: float  # Measure of diversification
    metadata: Dict[str, Any] = field(default_factory=dict)


class PortfolioOptimizer:
    """
    GPU-accelerated portfolio optimization.
    
    Features:
    - Mean-variance optimization
    - Risk parity allocation
    - Maximum Sharpe ratio
    - Minimum variance
    - Black-Litterman model
    - Dynamic rebalancing
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 enable_gpu: bool = True,
                 rebalance_threshold: float = 0.05):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            enable_gpu: Enable GPU acceleration
            rebalance_threshold: Threshold for rebalancing (5% default)
        """
        self.risk_free_rate = risk_free_rate
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.rebalance_threshold = rebalance_threshold
        
        self.gpu_manager = get_gpu_manager() if enable_gpu else None
        self.xp = cp_gpu if self.enable_gpu else np
        
        # Historical data cache
        self.returns_cache = {}
        self.covariance_cache = {}
        
        logger.info(f"Portfolio optimizer initialized {'with GPU' if self.enable_gpu else 'with CPU'}")
    
    def optimize_portfolio(self, 
                          assets: List[Asset],
                          returns_data: pd.DataFrame,
                          optimization_method: str = 'max_sharpe',
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Optimize portfolio allocation.
        
        Args:
            assets: List of assets
            returns_data: Historical returns DataFrame
            optimization_method: 'max_sharpe', 'min_variance', 'risk_parity', 'equal_weight'
            constraints: Optional constraints dict
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        # Calculate expected returns and covariance
        expected_returns = self._calculate_expected_returns(returns_data)
        cov_matrix = self._calculate_covariance_matrix(returns_data)
        
        # Apply optimization method
        if optimization_method == 'max_sharpe':
            weights = self._optimize_max_sharpe(expected_returns, cov_matrix, constraints)
        elif optimization_method == 'min_variance':
            weights = self._optimize_min_variance(cov_matrix, constraints)
        elif optimization_method == 'risk_parity':
            weights = self._optimize_risk_parity(cov_matrix)
        elif optimization_method == 'equal_weight':
            weights = np.ones(len(assets)) / len(assets)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights, expected_returns, cov_matrix)
        
        return {
            'weights': weights,
            'expected_return': metrics['expected_return'],
            'volatility': metrics['volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'metrics': metrics
        }
    
    def _calculate_expected_returns(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate expected returns using GPU."""
        if self.enable_gpu:
            returns_gpu = cp_gpu.asarray(returns_data.values)
            
            # Exponentially weighted mean for more recent data emphasis
            weights = cp_gpu.exp(cp_gpu.linspace(-2, 0, len(returns_gpu)))
            weights /= weights.sum()
            
            expected_returns = cp_gpu.average(returns_gpu, axis=0, weights=weights)
            
            # Annualize returns (assuming daily data)
            expected_returns = expected_returns * 252
            
            return cp_gpu.asnumpy(expected_returns)
        else:
            # CPU fallback
            weights = np.exp(np.linspace(-2, 0, len(returns_data)))
            weights /= weights.sum()
            
            expected_returns = np.average(returns_data.values, axis=0, weights=weights)
            return expected_returns * 252
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix using GPU."""
        if self.enable_gpu:
            returns_gpu = cp_gpu.asarray(returns_data.values)
            
            # Demean returns
            returns_demeaned = returns_gpu - cp_gpu.mean(returns_gpu, axis=0)
            
            # Calculate covariance
            cov_matrix = cp_gpu.dot(returns_demeaned.T, returns_demeaned) / (len(returns_demeaned) - 1)
            
            # Annualize covariance
            cov_matrix = cov_matrix * 252
            
            # Shrinkage for numerical stability
            cov_matrix = self._apply_shrinkage(cov_matrix)
            
            return cp_gpu.asnumpy(cov_matrix)
        else:
            # CPU fallback
            cov_matrix = returns_data.cov().values * 252
            return self._apply_shrinkage(cov_matrix)
    
    def _apply_shrinkage(self, cov_matrix: Union[np.ndarray, cp_gpu.ndarray], 
                        shrinkage_factor: float = 0.1) -> Union[np.ndarray, cp_gpu.ndarray]:
        """Apply Ledoit-Wolf shrinkage to covariance matrix."""
        xp = cp_gpu if isinstance(cov_matrix, cp_gpu.ndarray) else np
        
        # Shrinkage target: diagonal matrix with average variance
        avg_variance = xp.mean(xp.diag(cov_matrix))
        shrinkage_target = xp.eye(len(cov_matrix)) * avg_variance
        
        # Apply shrinkage
        return (1 - shrinkage_factor) * cov_matrix + shrinkage_factor * shrinkage_target
    
    def _optimize_max_sharpe(self, expected_returns: np.ndarray, 
                           cov_matrix: np.ndarray,
                           constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Optimize for maximum Sharpe ratio using GPU."""
        n_assets = len(expected_returns)
        
        if self.enable_gpu and self.gpu_manager:
            # GPU optimization using PyTorch
            returns_tensor = torch.tensor(expected_returns, dtype=torch.float32, device=self.gpu_manager.device)
            cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32, device=self.gpu_manager.device)
            
            # Initialize weights
            weights = torch.ones(n_assets, device=self.gpu_manager.device) / n_assets
            weights.requires_grad = True
            
            # Optimizer
            optimizer = optim.Adam([weights], lr=0.01)
            
            # Optimization loop
            for _ in range(1000):
                optimizer.zero_grad()
                
                # Portfolio return and risk
                portfolio_return = torch.sum(weights * returns_tensor)
                portfolio_variance = torch.sum(weights @ cov_tensor @ weights)
                portfolio_std = torch.sqrt(portfolio_variance)
                
                # Negative Sharpe ratio (to minimize)
                sharpe = -(portfolio_return - self.risk_free_rate) / portfolio_std
                
                # Constraints
                weight_sum_penalty = 100 * (torch.sum(weights) - 1) ** 2
                positive_weight_penalty = 100 * torch.sum(torch.relu(-weights))
                
                # Apply additional constraints
                if constraints:
                    if 'max_weight' in constraints:
                        max_weight = constraints['max_weight']
                        max_weight_penalty = 100 * torch.sum(torch.relu(weights - max_weight))
                        sharpe = sharpe + max_weight_penalty
                
                loss = sharpe + weight_sum_penalty + positive_weight_penalty
                
                loss.backward()
                optimizer.step()
            
            # Normalize weights
            weights_final = weights.detach()
            weights_final = torch.relu(weights_final)  # Ensure positive
            weights_final = weights_final / torch.sum(weights_final)
            
            return weights_final.cpu().numpy()
        else:
            # CPU optimization using scipy
            return self._optimize_max_sharpe_cpu(expected_returns, cov_matrix, constraints)
    
    def _optimize_max_sharpe_cpu(self, expected_returns: np.ndarray, 
                                cov_matrix: np.ndarray,
                                constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """CPU fallback for max Sharpe optimization."""
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
            return -(portfolio_return - self.risk_free_rate) / portfolio_std
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        if constraints and 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            bounds = [(0, max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        return result.x
    
    def _optimize_min_variance(self, cov_matrix: np.ndarray,
                              constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Optimize for minimum variance portfolio."""
        n_assets = len(cov_matrix)
        
        # Use cvxpy for convex optimization
        weights = cp.Variable(n_assets)
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        if constraints and 'max_weight' in constraints:
            constraints_list.append(weights <= constraints['max_weight'])
        
        # Problem
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
        problem.solve()
        
        return weights.value
    
    def _optimize_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize for risk parity (equal risk contribution)."""
        n_assets = len(cov_matrix)
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Minimize squared differences from equal contribution
            target_contrib = 1 / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(0.01, 1) for _ in range(n_assets)]  # Min 1% weight
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_contribution,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        return result.x
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray,
                                   expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        # Basic metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Value at Risk (VaR) and Conditional VaR
        confidence_level = 0.95
        z_score = norm.ppf(1 - confidence_level)
        var_95 = portfolio_return + z_score * portfolio_volatility
        cvar_95 = portfolio_return - portfolio_volatility * norm.pdf(z_score) / (1 - confidence_level)
        
        # Diversification ratio
        weighted_vols = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        diversification_ratio = weighted_vols / portfolio_volatility
        
        # Effective number of assets (inverse HHI)
        hhi = np.sum(weights ** 2)
        effective_assets = 1 / hhi if hhi > 0 else 1
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'diversification_ratio': diversification_ratio,
            'effective_assets': effective_assets,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights[weights > 0.001])  # Exclude near-zero weights
        }
    
    def calculate_efficient_frontier(self, 
                                   expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   n_portfolios: int = 100) -> Dict[str, np.ndarray]:
        """Calculate efficient frontier portfolios."""
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)
        
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        frontier_volatilities = []
        frontier_weights = []
        
        for target_return in target_returns:
            # Optimize for minimum variance given target return
            n_assets = len(expected_returns)
            
            # Use cvxpy
            weights = cp.Variable(n_assets)
            portfolio_variance = cp.quad_form(weights, cov_matrix)
            
            constraints_list = [
                cp.sum(weights) == 1,
                weights >= 0,
                weights @ expected_returns == target_return
            ]
            
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
            
            try:
                problem.solve()
                if weights.value is not None:
                    frontier_volatilities.append(np.sqrt(portfolio_variance.value))
                    frontier_weights.append(weights.value)
            except:
                continue
        
        return {
            'returns': target_returns[:len(frontier_volatilities)],
            'volatilities': np.array(frontier_volatilities),
            'weights': np.array(frontier_weights),
            'sharpe_ratios': (target_returns[:len(frontier_volatilities)] - self.risk_free_rate) / 
                           np.array(frontier_volatilities)
        }
    
    def check_rebalancing_needed(self, 
                                current_weights: np.ndarray,
                                target_weights: np.ndarray) -> bool:
        """Check if portfolio rebalancing is needed."""
        weight_diffs = np.abs(current_weights - target_weights)
        max_diff = np.max(weight_diffs)
        
        return max_diff > self.rebalance_threshold
    
    def calculate_rebalancing_trades(self,
                                   current_portfolio: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   total_value: float) -> Dict[str, float]:
        """Calculate trades needed for rebalancing."""
        trades = {}
        
        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_value = current_portfolio.get(symbol, 0)
            
            trade_value = target_value - current_value
            if abs(trade_value) > total_value * 0.001:  # Min 0.1% trade
                trades[symbol] = trade_value
        
        return trades
    
    def monte_carlo_optimization(self,
                               expected_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               n_simulations: int = 10000,
                               time_horizon: int = 252) -> Dict[str, Any]:
        """Run Monte Carlo simulations for portfolio optimization."""
        n_assets = len(expected_returns)
        
        if self.enable_gpu:
            # GPU-accelerated Monte Carlo
            returns_gpu = cp_gpu.asarray(expected_returns)
            cov_gpu = cp_gpu.asarray(cov_matrix)
            
            # Generate random weights
            random_weights = cp_gpu.random.dirichlet(
                cp_gpu.ones(n_assets), 
                size=n_simulations
            )
            
            # Calculate portfolio metrics for each simulation
            portfolio_returns = random_weights @ returns_gpu
            
            # Portfolio variances (vectorized)
            portfolio_vars = cp_gpu.sum(
                random_weights @ cov_gpu * random_weights, 
                axis=1
            )
            portfolio_vols = cp_gpu.sqrt(portfolio_vars)
            
            # Sharpe ratios
            sharpe_ratios = (portfolio_returns - self.risk_free_rate) / portfolio_vols
            
            # Find best portfolio
            best_idx = cp_gpu.argmax(sharpe_ratios)
            best_weights = random_weights[best_idx]
            
            return {
                'best_weights': cp_gpu.asnumpy(best_weights),
                'best_sharpe': float(sharpe_ratios[best_idx]),
                'all_returns': cp_gpu.asnumpy(portfolio_returns),
                'all_volatilities': cp_gpu.asnumpy(portfolio_vols),
                'all_sharpes': cp_gpu.asnumpy(sharpe_ratios)
            }
        else:
            # CPU fallback
            random_weights = np.random.dirichlet(np.ones(n_assets), size=n_simulations)
            portfolio_returns = random_weights @ expected_returns
            portfolio_vars = np.sum(random_weights @ cov_matrix * random_weights, axis=1)
            portfolio_vols = np.sqrt(portfolio_vars)
            sharpe_ratios = (portfolio_returns - self.risk_free_rate) / portfolio_vols
            
            best_idx = np.argmax(sharpe_ratios)
            
            return {
                'best_weights': random_weights[best_idx],
                'best_sharpe': sharpe_ratios[best_idx],
                'all_returns': portfolio_returns,
                'all_volatilities': portfolio_vols,
                'all_sharpes': sharpe_ratios
            }