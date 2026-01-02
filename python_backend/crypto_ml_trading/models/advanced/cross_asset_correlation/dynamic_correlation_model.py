"""
Dynamic Correlation Modeling.

Implements advanced dynamic correlation models for time-varying correlation analysis.
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
class DynamicCorrelationState:
    """State of dynamic correlation model."""
    timestamp: datetime
    correlation_matrix: np.ndarray
    conditional_variances: np.ndarray
    conditional_correlations: np.ndarray
    model_parameters: Dict[str, float]
    log_likelihood: float
    regime_probabilities: Optional[np.ndarray] = None


@dataclass
class CorrelationForecast:
    """Correlation forecast from dynamic model."""
    timestamp: datetime
    forecast_horizon: int
    forecasted_correlations: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    regime_forecasts: Optional[np.ndarray] = None
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelDiagnostics:
    """Dynamic correlation model diagnostics."""
    model_type: str
    aic: float
    bic: float
    log_likelihood: float
    parameter_significance: Dict[str, float]
    residual_diagnostics: Dict[str, float]
    forecast_accuracy: Dict[str, float]


class DynamicCorrelationModel:
    """
    Dynamic correlation modeling for time-varying correlations.
    
    Features:
    - DCC (Dynamic Conditional Correlation) modeling
    - BEKK (Baba-Engle-Kraft-Kroner) multivariate GARCH
    - Regime-switching correlation models
    - Rolling window correlation estimation
    - Correlation forecasting with uncertainty
    - Model selection and diagnostics
    """
    
    def __init__(self,
                 model_type: str = 'dcc',
                 estimation_window: int = 252,
                 forecast_horizon: int = 5,
                 n_regimes: int = 2):
        """
        Initialize dynamic correlation model.
        
        Args:
            model_type: Type of model ('dcc', 'bekk', 'rolling', 'regime_switching')
            estimation_window: Window size for parameter estimation
            forecast_horizon: Default forecast horizon
            n_regimes: Number of regimes for regime-switching models
        """
        self.model_type = model_type
        self.estimation_window = estimation_window
        self.forecast_horizon = forecast_horizon
        self.n_regimes = n_regimes
        
        # Model state
        self.is_fitted = False
        self.asset_symbols: List[str] = []
        self.n_assets = 0
        
        # Model parameters
        self.parameters: Dict[str, Any] = {}
        self.estimation_history: deque = deque(maxlen=100)
        
        # Data storage
        self.return_data: deque = deque(maxlen=1000)
        self.correlation_states: deque = deque(maxlen=500)
        self.forecasts: deque = deque(maxlen=200)
        
        # Model diagnostics
        self.diagnostics: Optional[ModelDiagnostics] = None
        
    def fit(self,
            returns: np.ndarray,
            asset_symbols: List[str],
            timestamps: Optional[List[datetime]] = None) -> None:
        """
        Fit dynamic correlation model to return data.
        
        Args:
            returns: Return matrix (T x N)
            asset_symbols: Asset symbol names
            timestamps: Optional timestamps for each observation
        """
        self.asset_symbols = asset_symbols
        self.n_assets = len(asset_symbols)
        
        if returns.shape[1] != self.n_assets:
            raise ValueError("Number of assets in returns must match asset_symbols")
        
        # Store data
        for i, return_vec in enumerate(returns):
            timestamp = timestamps[i] if timestamps else datetime.now()
            self.return_data.append({
                'timestamp': timestamp,
                'returns': return_vec,
                'asset_symbols': asset_symbols
            })
        
        # Fit model based on type
        if self.model_type == 'dcc':
            self._fit_dcc_model(returns)
        elif self.model_type == 'bekk':
            self._fit_bekk_model(returns)
        elif self.model_type == 'rolling':
            self._fit_rolling_model(returns)
        elif self.model_type == 'regime_switching':
            self._fit_regime_switching_model(returns)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.is_fitted = True
        
        # Calculate diagnostics
        self._calculate_diagnostics(returns)
    
    def _fit_dcc_model(self, returns: np.ndarray) -> None:
        """Fit Dynamic Conditional Correlation (DCC) model."""
        T, N = returns.shape
        
        # Step 1: Estimate univariate GARCH models for each asset
        garch_params = {}
        conditional_variances = np.zeros((T, N))
        standardized_returns = np.zeros((T, N))
        
        for i in range(N):
            asset_returns = returns[:, i]
            params = self._fit_univariate_garch(asset_returns)
            garch_params[self.asset_symbols[i]] = params
            
            # Calculate conditional variances
            conditional_variances[:, i] = self._calculate_conditional_variance(
                asset_returns, params
            )
            
            # Standardized returns
            standardized_returns[:, i] = asset_returns / np.sqrt(conditional_variances[:, i])
        
        # Step 2: Estimate DCC parameters
        dcc_params = self._estimate_dcc_parameters(standardized_returns)
        
        # Step 3: Calculate dynamic correlations
        dynamic_correlations = self._calculate_dynamic_correlations(
            standardized_returns, dcc_params
        )
        
        # Store parameters
        self.parameters = {
            'garch_params': garch_params,
            'dcc_params': dcc_params,
            'conditional_variances': conditional_variances,
            'dynamic_correlations': dynamic_correlations
        }
    
    def _fit_univariate_garch(self, returns: np.ndarray) -> Dict[str, float]:
        """Fit univariate GARCH(1,1) model."""
        # Simple GARCH(1,1) estimation using method of moments
        # In practice, would use maximum likelihood estimation
        
        returns_squared = returns ** 2
        
        # Initial estimates
        omega = np.var(returns) * 0.1
        alpha = 0.1
        beta = 0.8
        
        # Simple iterative estimation (placeholder for MLE)
        for _ in range(10):
            # Calculate conditional variances
            conditional_var = np.zeros(len(returns))
            conditional_var[0] = np.var(returns)
            
            for t in range(1, len(returns)):
                conditional_var[t] = omega + alpha * returns_squared[t-1] + beta * conditional_var[t-1]
            
            # Update parameters (simplified)
            residuals = returns_squared - conditional_var
            
            # Basic parameter updates
            omega = max(0.001, omega + 0.01 * np.mean(residuals))
            alpha = max(0.01, min(0.3, alpha + 0.001 * np.corrcoef(returns_squared[:-1], residuals[1:])[0,1]))
            beta = max(0.6, min(0.95, beta - 0.001 * np.std(residuals)))
            
            # Ensure stationarity
            if alpha + beta >= 1:
                alpha = 0.1
                beta = 0.8
        
        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta
        }
    
    def _calculate_conditional_variance(self,
                                     returns: np.ndarray,
                                     params: Dict[str, float]) -> np.ndarray:
        """Calculate conditional variance from GARCH parameters."""
        omega = params['omega']
        alpha = params['alpha']
        beta = params['beta']
        
        T = len(returns)
        conditional_var = np.zeros(T)
        conditional_var[0] = np.var(returns)
        
        for t in range(1, T):
            conditional_var[t] = omega + alpha * returns[t-1]**2 + beta * conditional_var[t-1]
        
        return conditional_var
    
    def _estimate_dcc_parameters(self, standardized_returns: np.ndarray) -> Dict[str, float]:
        """Estimate DCC parameters."""
        T, N = standardized_returns.shape
        
        # Calculate sample correlation matrix
        sample_corr = np.corrcoef(standardized_returns.T)
        
        # Initial DCC parameters
        a = 0.01  # DCC alpha
        b = 0.95  # DCC beta
        
        # Simple estimation (in practice, would use MLE)
        # This is a simplified implementation
        
        return {
            'alpha': a,
            'beta': b,
            'unconditional_corr': sample_corr
        }
    
    def _calculate_dynamic_correlations(self,
                                      standardized_returns: np.ndarray,
                                      dcc_params: Dict[str, float]) -> np.ndarray:
        """Calculate dynamic conditional correlations."""
        T, N = standardized_returns.shape
        
        alpha = dcc_params['alpha']
        beta = dcc_params['beta']
        R_bar = dcc_params['unconditional_corr']
        
        # Initialize
        Q = np.zeros((T, N, N))
        R = np.zeros((T, N, N))
        
        # Initial Q
        Q[0] = R_bar
        
        for t in range(1, T):
            # Update Q
            u_t = standardized_returns[t-1:t].T  # Column vector
            Q[t] = (1 - alpha - beta) * R_bar + alpha * (u_t @ u_t.T) + beta * Q[t-1]
            
            # Calculate R from Q
            q_diag = np.sqrt(np.diag(Q[t]))
            R[t] = Q[t] / np.outer(q_diag, q_diag)
            
            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(R[t])
            eigenvals = np.maximum(eigenvals, 1e-8)
            R[t] = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return R
    
    def _fit_bekk_model(self, returns: np.ndarray) -> None:
        """Fit BEKK multivariate GARCH model."""
        # Simplified BEKK implementation
        # Full implementation would require constrained optimization
        
        T, N = returns.shape
        
        # Initialize parameters
        C = np.eye(N) * 0.1  # Constant term
        A = np.eye(N) * 0.1  # ARCH effect
        B = np.eye(N) * 0.8  # GARCH effect
        
        # Simple estimation (placeholder)
        self.parameters = {
            'C': C,
            'A': A,
            'B': B,
            'model_type': 'bekk'
        }
    
    def _fit_rolling_model(self, returns: np.ndarray) -> None:
        """Fit rolling window correlation model."""
        window = min(self.estimation_window, len(returns))
        
        self.parameters = {
            'window_size': window,
            'model_type': 'rolling'
        }
    
    def _fit_regime_switching_model(self, returns: np.ndarray) -> None:
        """Fit regime-switching correlation model."""
        # Simplified regime-switching model
        # Full implementation would use EM algorithm
        
        T, N = returns.shape
        
        # Initialize regime probabilities
        regime_probs = np.random.rand(T, self.n_regimes)
        regime_probs = regime_probs / regime_probs.sum(axis=1, keepdims=True)
        
        # Estimate regime-specific correlation matrices
        regime_correlations = []
        for k in range(self.n_regimes):
            # Weighted correlation matrix for regime k
            weights = regime_probs[:, k]
            weighted_returns = returns * weights.reshape(-1, 1)
            
            # Calculate weighted correlation
            if np.sum(weights) > 0:
                corr_matrix = np.corrcoef(weighted_returns.T)
            else:
                corr_matrix = np.eye(N)
            
            regime_correlations.append(corr_matrix)
        
        # Transition probabilities (simplified)
        transition_matrix = np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
        
        self.parameters = {
            'n_regimes': self.n_regimes,
            'regime_correlations': regime_correlations,
            'transition_matrix': transition_matrix,
            'regime_probabilities': regime_probs,
            'model_type': 'regime_switching'
        }
    
    def update(self,
               new_returns: np.ndarray,
               timestamp: Optional[datetime] = None) -> DynamicCorrelationState:
        """
        Update model with new return data.
        
        Args:
            new_returns: New return vector
            timestamp: Optional timestamp
            
        Returns:
            Updated correlation state
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating")
        
        if len(new_returns) != self.n_assets:
            raise ValueError("New returns must match number of assets")
        
        timestamp = timestamp or datetime.now()
        
        # Add new data
        self.return_data.append({
            'timestamp': timestamp,
            'returns': new_returns,
            'asset_symbols': self.asset_symbols
        })
        
        # Update model based on type
        if self.model_type == 'dcc':
            state = self._update_dcc(new_returns, timestamp)
        elif self.model_type == 'rolling':
            state = self._update_rolling(new_returns, timestamp)
        elif self.model_type == 'regime_switching':
            state = self._update_regime_switching(new_returns, timestamp)
        else:
            # For other models, use simple update
            state = self._simple_update(new_returns, timestamp)
        
        self.correlation_states.append(state)
        return state
    
    def _update_dcc(self,
                   new_returns: np.ndarray,
                   timestamp: datetime) -> DynamicCorrelationState:
        """Update DCC model with new observation."""
        # Get DCC parameters
        dcc_params = self.parameters['dcc_params']
        alpha = dcc_params['alpha']
        beta = dcc_params['beta']
        R_bar = dcc_params['unconditional_corr']
        
        # Calculate new conditional variances (simplified)
        garch_params = self.parameters['garch_params']
        conditional_variances = np.zeros(self.n_assets)
        
        for i, symbol in enumerate(self.asset_symbols):
            params = garch_params[symbol]
            # Use last variance and new return
            if self.correlation_states:
                last_var = self.correlation_states[-1].conditional_variances[i]
            else:
                last_var = np.var([obs['returns'][i] for obs in self.return_data[-50:]])
            
            conditional_variances[i] = (
                params['omega'] +
                params['alpha'] * new_returns[i]**2 +
                params['beta'] * last_var
            )
        
        # Standardize returns
        standardized_returns = new_returns / np.sqrt(conditional_variances)
        
        # Update Q matrix
        if self.correlation_states:
            last_Q = self.correlation_states[-1].conditional_correlations
        else:
            last_Q = R_bar
        
        u_t = standardized_returns.reshape(-1, 1)
        Q_new = (1 - alpha - beta) * R_bar + alpha * (u_t @ u_t.T) + beta * last_Q
        
        # Calculate correlation matrix
        q_diag = np.sqrt(np.diag(Q_new))
        R_new = Q_new / np.outer(q_diag, q_diag)
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(R_new)
        eigenvals = np.maximum(eigenvals, 1e-8)
        R_new = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return DynamicCorrelationState(
            timestamp=timestamp,
            correlation_matrix=R_new,
            conditional_variances=conditional_variances,
            conditional_correlations=Q_new,
            model_parameters=dcc_params,
            log_likelihood=0.0  # Would calculate properly in full implementation
        )
    
    def _update_rolling(self,
                       new_returns: np.ndarray,
                       timestamp: datetime) -> DynamicCorrelationState:
        """Update rolling correlation model."""
        window = self.parameters['window_size']
        
        # Get recent returns
        recent_data = list(self.return_data)[-window:]
        if len(recent_data) < window:
            recent_data = list(self.return_data)
        
        returns_matrix = np.array([obs['returns'] for obs in recent_data])
        
        # Calculate rolling correlation
        correlation_matrix = np.corrcoef(returns_matrix.T)
        
        # Handle case where we have only one observation
        if np.isnan(correlation_matrix).any():
            correlation_matrix = np.eye(self.n_assets)
        
        # Calculate conditional variances (rolling)
        conditional_variances = np.var(returns_matrix, axis=0)
        
        return DynamicCorrelationState(
            timestamp=timestamp,
            correlation_matrix=correlation_matrix,
            conditional_variances=conditional_variances,
            conditional_correlations=correlation_matrix,
            model_parameters=self.parameters,
            log_likelihood=0.0
        )
    
    def _update_regime_switching(self,
                               new_returns: np.ndarray,
                               timestamp: datetime) -> DynamicCorrelationState:
        """Update regime-switching model."""
        # Simplified regime probability update
        # Full implementation would use filtering equations
        
        regime_correlations = self.parameters['regime_correlations']
        transition_matrix = self.parameters['transition_matrix']
        
        # Calculate likelihood for each regime
        regime_likelihoods = np.zeros(self.n_regimes)
        
        for k in range(self.n_regimes):
            # Simplified likelihood calculation
            corr_matrix = regime_correlations[k]
            
            # Multivariate normal likelihood (simplified)
            try:
                inv_corr = np.linalg.inv(corr_matrix)
                likelihood = np.exp(-0.5 * new_returns.T @ inv_corr @ new_returns)
                regime_likelihoods[k] = likelihood
            except:
                regime_likelihoods[k] = 1.0
        
        # Normalize probabilities
        if np.sum(regime_likelihoods) > 0:
            regime_probabilities = regime_likelihoods / np.sum(regime_likelihoods)
        else:
            regime_probabilities = np.ones(self.n_regimes) / self.n_regimes
        
        # Calculate weighted correlation matrix
        correlation_matrix = np.zeros((self.n_assets, self.n_assets))
        for k in range(self.n_regimes):
            correlation_matrix += regime_probabilities[k] * regime_correlations[k]
        
        return DynamicCorrelationState(
            timestamp=timestamp,
            correlation_matrix=correlation_matrix,
            conditional_variances=np.ones(self.n_assets),  # Simplified
            conditional_correlations=correlation_matrix,
            model_parameters=self.parameters,
            log_likelihood=0.0,
            regime_probabilities=regime_probabilities
        )
    
    def _simple_update(self,
                      new_returns: np.ndarray,
                      timestamp: datetime) -> DynamicCorrelationState:
        """Simple update for unsupported models."""
        # Use exponential smoothing
        if self.correlation_states:
            old_corr = self.correlation_states[-1].correlation_matrix
            new_corr = 0.95 * old_corr + 0.05 * np.outer(new_returns, new_returns)
        else:
            new_corr = np.eye(self.n_assets)
        
        return DynamicCorrelationState(
            timestamp=timestamp,
            correlation_matrix=new_corr,
            conditional_variances=np.ones(self.n_assets),
            conditional_correlations=new_corr,
            model_parameters=self.parameters,
            log_likelihood=0.0
        )
    
    def forecast(self,
                horizon: Optional[int] = None,
                confidence_level: float = 0.95) -> CorrelationForecast:
        """
        Forecast correlations for specified horizon.
        
        Args:
            horizon: Forecast horizon (default: self.forecast_horizon)
            confidence_level: Confidence level for intervals
            
        Returns:
            Correlation forecast
        """
        if not self.is_fitted or not self.correlation_states:
            raise ValueError("Model must be fitted and have states for forecasting")
        
        horizon = horizon or self.forecast_horizon
        current_state = self.correlation_states[-1]
        
        if self.model_type == 'dcc':
            forecast = self._forecast_dcc(current_state, horizon, confidence_level)
        elif self.model_type == 'regime_switching':
            forecast = self._forecast_regime_switching(current_state, horizon, confidence_level)
        else:
            forecast = self._forecast_simple(current_state, horizon, confidence_level)
        
        self.forecasts.append(forecast)
        return forecast
    
    def _forecast_dcc(self,
                     current_state: DynamicCorrelationState,
                     horizon: int,
                     confidence_level: float) -> CorrelationForecast:
        """Forecast using DCC model."""
        dcc_params = self.parameters['dcc_params']
        alpha = dcc_params['alpha']
        beta = dcc_params['beta']
        R_bar = dcc_params['unconditional_corr']
        
        # Multi-step forecast
        forecasted_correlations = np.zeros((horizon, self.n_assets, self.n_assets))
        
        Q = current_state.conditional_correlations
        
        for h in range(horizon):
            # Forecast Q (mean-reverting to R_bar)
            persistence = (alpha + beta) ** h
            Q = persistence * Q + (1 - persistence) * R_bar
            
            # Convert to correlation
            q_diag = np.sqrt(np.diag(Q))
            R = Q / np.outer(q_diag, q_diag)
            
            forecasted_correlations[h] = R
        
        # Calculate confidence intervals (simplified)
        forecast_std = np.std([state.correlation_matrix for state in list(self.correlation_states)[-20:]], axis=0)
        z_score = 1.96 if confidence_level == 0.95 else 2.58
        
        confidence_intervals = {
            'lower': forecasted_correlations - z_score * forecast_std,
            'upper': forecasted_correlations + z_score * forecast_std
        }
        
        return CorrelationForecast(
            timestamp=current_state.timestamp,
            forecast_horizon=horizon,
            forecasted_correlations=forecasted_correlations,
            confidence_intervals=confidence_intervals,
            uncertainty_metrics={'forecast_std': np.mean(forecast_std)}
        )
    
    def _forecast_regime_switching(self,
                                 current_state: DynamicCorrelationState,
                                 horizon: int,
                                 confidence_level: float) -> CorrelationForecast:
        """Forecast using regime-switching model."""
        transition_matrix = self.parameters['transition_matrix']
        regime_correlations = self.parameters['regime_correlations']
        current_probs = current_state.regime_probabilities
        
        # Forecast regime probabilities
        regime_forecasts = np.zeros((horizon, self.n_regimes))
        regime_forecasts[0] = current_probs
        
        for h in range(1, horizon):
            regime_forecasts[h] = regime_forecasts[h-1] @ transition_matrix
        
        # Forecast correlations
        forecasted_correlations = np.zeros((horizon, self.n_assets, self.n_assets))
        
        for h in range(horizon):
            corr_forecast = np.zeros((self.n_assets, self.n_assets))
            for k in range(self.n_regimes):
                corr_forecast += regime_forecasts[h, k] * regime_correlations[k]
            forecasted_correlations[h] = corr_forecast
        
        return CorrelationForecast(
            timestamp=current_state.timestamp,
            forecast_horizon=horizon,
            forecasted_correlations=forecasted_correlations,
            confidence_intervals={},
            regime_forecasts=regime_forecasts
        )
    
    def _forecast_simple(self,
                        current_state: DynamicCorrelationState,
                        horizon: int,
                        confidence_level: float) -> CorrelationForecast:
        """Simple forecast (random walk or mean reversion)."""
        current_corr = current_state.correlation_matrix
        
        # Simple mean reversion to long-term average
        if len(self.correlation_states) > 10:
            long_term_corr = np.mean([state.correlation_matrix for state in list(self.correlation_states)[-50:]], axis=0)
            reversion_speed = 0.1
        else:
            long_term_corr = current_corr
            reversion_speed = 0.0
        
        forecasted_correlations = np.zeros((horizon, self.n_assets, self.n_assets))
        
        for h in range(horizon):
            decay_factor = np.exp(-reversion_speed * (h + 1))
            forecasted_correlations[h] = decay_factor * current_corr + (1 - decay_factor) * long_term_corr
        
        return CorrelationForecast(
            timestamp=current_state.timestamp,
            forecast_horizon=horizon,
            forecasted_correlations=forecasted_correlations,
            confidence_intervals={}
        )
    
    def _calculate_diagnostics(self, returns: np.ndarray) -> None:
        """Calculate model diagnostics."""
        # Simplified diagnostics
        log_likelihood = 0.0  # Would calculate properly
        n_params = self._count_parameters()
        n_obs = len(returns)
        
        # Information criteria
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_obs)
        
        self.diagnostics = ModelDiagnostics(
            model_type=self.model_type,
            aic=aic,
            bic=bic,
            log_likelihood=log_likelihood,
            parameter_significance={},
            residual_diagnostics={},
            forecast_accuracy={}
        )
    
    def _count_parameters(self) -> int:
        """Count number of model parameters."""
        if self.model_type == 'dcc':
            return 2 + 3 * self.n_assets  # DCC params + GARCH params per asset
        elif self.model_type == 'bekk':
            return 3 * self.n_assets * self.n_assets  # BEKK parameter matrices
        elif self.model_type == 'regime_switching':
            return self.n_regimes * self.n_assets * (self.n_assets + 1) // 2  # Correlation matrices
        else:
            return 1  # Rolling window
    
    def get_current_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get current correlation matrix."""
        if self.correlation_states:
            return self.correlation_states[-1].correlation_matrix
        return None
    
    def get_correlation_time_series(self,
                                  asset1_idx: int,
                                  asset2_idx: int) -> Tuple[List[datetime], List[float]]:
        """Get time series of correlation between two assets."""
        timestamps = []
        correlations = []
        
        for state in self.correlation_states:
            timestamps.append(state.timestamp)
            correlations.append(state.correlation_matrix[asset1_idx, asset2_idx])
        
        return timestamps, correlations
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'n_assets': self.n_assets,
            'asset_symbols': self.asset_symbols,
            'estimation_window': self.estimation_window,
            'n_observations': len(self.return_data),
            'n_states': len(self.correlation_states)
        }
        
        if self.diagnostics:
            summary['diagnostics'] = {
                'aic': self.diagnostics.aic,
                'bic': self.diagnostics.bic,
                'log_likelihood': self.diagnostics.log_likelihood
            }
        
        if self.correlation_states:
            latest_state = self.correlation_states[-1]
            summary['latest_state'] = {
                'timestamp': latest_state.timestamp.isoformat(),
                'avg_correlation': np.mean(latest_state.correlation_matrix[np.triu_indices(self.n_assets, k=1)]),
                'max_correlation': np.max(latest_state.correlation_matrix[np.triu_indices(self.n_assets, k=1)]),
                'min_correlation': np.min(latest_state.correlation_matrix[np.triu_indices(self.n_assets, k=1)])
            }
            
            if latest_state.regime_probabilities is not None:
                summary['latest_state']['regime_probabilities'] = latest_state.regime_probabilities.tolist()
        
        return summary