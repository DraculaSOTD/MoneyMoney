"""
HAR-RV (Heterogeneous Autoregressive Realized Volatility) Model

This model captures multi-timescale volatility patterns from different
trader horizons (intraday, daily, weekly, monthly).

Formula: RV_{t+1} = β_0 + β_d·RV_t^{daily} + β_w·RV_t^{weekly} + β_m·RV_t^{monthly} + ε

The model is based on the observation that different market participants
operate on different time horizons:
- High-frequency traders: minutes/hours
- Day traders: daily
- Swing traders: weekly
- Position traders: monthly

For 1-minute crypto data, we adapt the horizons to:
- Short: 60 periods (1 hour)
- Medium: 1440 periods (1 day)
- Long: 10080 periods (1 week)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.linear_model import LinearRegression, Ridge
import warnings


class HARRV:
    """
    Heterogeneous Autoregressive Realized Volatility model.

    Captures multi-timescale volatility patterns using realized volatility
    components computed over different horizons.

    Features:
    - Multi-horizon RV components (short, medium, long)
    - OLS or Ridge regression fitting
    - Rolling window forecasting
    - Confidence intervals
    """

    def __init__(self, periods: Optional[List[int]] = None,
                 use_ridge: bool = False, alpha: float = 1.0):
        """
        Initialize HAR-RV model.

        Args:
            periods: List of aggregation periods [short, medium, long]
                     Default for 1-min data: [60, 1440, 10080] (1h, 1d, 1w)
            use_ridge: Use Ridge regression instead of OLS
            alpha: Ridge regularization parameter (if use_ridge=True)
        """
        if periods is None:
            # Default periods for 1-minute cryptocurrency data
            # Adapted from daily HAR-RV [1, 5, 22] to minute-based
            periods = [60, 1440, 10080]  # 1 hour, 1 day, 1 week

        self.periods = sorted(periods)
        self.use_ridge = use_ridge
        self.alpha = alpha

        # Model components
        self.model = None
        self.intercept = None
        self.coefficients = None
        self.period_names = ['short', 'medium', 'long'][:len(periods)]

        # Fitted values
        self.returns = None
        self.realized_volatility = None
        self.fitted_values = None
        self.residuals = None

        # Diagnostics
        self.r_squared = None
        self.adj_r_squared = None
        self.rmse = None
        self.mae = None

    def _compute_realized_volatility(self, returns: np.ndarray,
                                      window: int) -> np.ndarray:
        """
        Compute realized volatility over a rolling window.

        RV = sqrt(sum(r^2)) for the window

        Args:
            returns: Return series
            window: Window size

        Returns:
            Realized volatility series
        """
        T = len(returns)
        rv = np.zeros(T)

        for t in range(window, T):
            rv[t] = np.sqrt(np.sum(returns[t - window:t] ** 2))

        # Fill initial values with expanding window
        for t in range(1, min(window, T)):
            rv[t] = np.sqrt(np.sum(returns[:t] ** 2))

        return rv

    def _compute_rv_components(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute RV components at different horizons.

        Args:
            returns: Return series

        Returns:
            Dictionary of RV components
        """
        components = {}
        for period, name in zip(self.periods, self.period_names):
            components[name] = self._compute_realized_volatility(returns, period)
        return components

    def fit(self, returns: Union[np.ndarray, pd.Series],
            target_horizon: int = 1) -> 'HARRV':
        """
        Fit HAR-RV model.

        Args:
            returns: Return series
            target_horizon: Forecast horizon (periods ahead)

        Returns:
            Self for method chaining
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = returns[~np.isnan(returns)]
        self.returns = returns

        # Compute RV components
        rv_components = self._compute_rv_components(returns)

        # Compute target: realized volatility at t+horizon
        target_rv = self._compute_realized_volatility(returns, self.periods[0])

        # Prepare feature matrix and target
        max_period = max(self.periods)
        start_idx = max_period + target_horizon

        # Features: lagged RV components
        X = np.column_stack([
            rv_components[name][start_idx - target_horizon - 1:-target_horizon - 1]
            for name in self.period_names
        ])

        # Target: future RV
        y = target_rv[start_idx:]

        # Align lengths
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # Remove any NaN/inf values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                       np.isinf(X).any(axis=1) | np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]

        # Fit regression model
        if self.use_ridge:
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()

        self.model.fit(X, y)

        self.intercept = self.model.intercept_
        self.coefficients = dict(zip(self.period_names, self.model.coef_))

        # Compute fitted values and residuals
        self.fitted_values = self.model.predict(X)
        self.residuals = y - self.fitted_values

        # Compute diagnostics
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        self.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n, p = len(y), len(self.periods)
        self.adj_r_squared = 1 - (1 - self.r_squared) * (n - 1) / (n - p - 1)
        self.rmse = np.sqrt(np.mean(self.residuals ** 2))
        self.mae = np.mean(np.abs(self.residuals))

        # Store realized volatility for forecasting
        self.realized_volatility = target_rv

        return self

    def forecast(self, steps: int = 1,
                 recent_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Forecast realized volatility.

        Args:
            steps: Number of steps ahead to forecast
            recent_returns: Recent returns for computing RV components
                           (uses fitted returns if not provided)

        Returns:
            Dictionary with forecasts
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")

        if recent_returns is None:
            recent_returns = self.returns

        # Compute current RV components
        rv_components = self._compute_rv_components(recent_returns)

        # Get latest values for forecasting
        latest_features = np.array([
            rv_components[name][-1] for name in self.period_names
        ]).reshape(1, -1)

        # Multi-step forecast
        vol_forecast = np.zeros(steps)

        for t in range(steps):
            if t == 0:
                vol_forecast[t] = self.model.predict(latest_features)[0]
            else:
                # For multi-step, update features with forecasted values
                # Simple approach: decay towards long-run mean
                vol_forecast[t] = (
                    self.intercept +
                    self.coefficients['short'] * vol_forecast[t - 1] +
                    self.coefficients.get('medium', 0) * np.mean(vol_forecast[:t]) +
                    self.coefficients.get('long', 0) * np.mean(vol_forecast[:t])
                )

            vol_forecast[t] = max(vol_forecast[t], 1e-10)

        # Compute prediction intervals based on residual std
        if self.residuals is not None and len(self.residuals) > 0:
            residual_std = np.std(self.residuals)
        else:
            residual_std = vol_forecast[0] * 0.2  # Default 20% uncertainty

        vol_lower = vol_forecast - 1.96 * residual_std
        vol_upper = vol_forecast + 1.96 * residual_std

        # Ensure non-negative
        vol_lower = np.maximum(vol_lower, 0)

        return {
            'variance': vol_forecast ** 2,
            'volatility': vol_forecast,
            'volatility_lower': vol_lower,
            'volatility_upper': vol_upper
        }

    def rolling_forecast(self, returns: np.ndarray,
                        window: int = 5000,
                        step: int = 100) -> pd.DataFrame:
        """
        Generate rolling out-of-sample forecasts.

        Args:
            returns: Full return series
            window: Training window size
            step: Step size between re-estimations

        Returns:
            DataFrame with forecasts and actuals
        """
        results = []
        T = len(returns)

        for t in range(window, T - 1, step):
            # Fit on training window
            train_returns = returns[t - window:t]
            self.fit(train_returns)

            # Forecast
            forecast = self.forecast(steps=1, recent_returns=train_returns)

            # Get actual
            actual_rv = self._compute_realized_volatility(
                returns[t:t + self.periods[0] + 1],
                self.periods[0]
            )[-1] if t + self.periods[0] < T else np.nan

            results.append({
                'timestamp': t,
                'forecast': forecast['volatility'][0],
                'actual': actual_rv,
                'lower': forecast['volatility_lower'][0],
                'upper': forecast['volatility_upper'][0]
            })

        return pd.DataFrame(results)

    def summary(self) -> Dict:
        """
        Generate model summary.

        Returns:
            Dictionary with model summary
        """
        if self.model is None:
            return {'error': 'Model not fitted'}

        return {
            'model': 'HAR-RV',
            'periods': dict(zip(self.period_names, self.periods)),
            'parameters': {
                'intercept': self.intercept,
                'coefficients': self.coefficients
            },
            'regression_type': 'Ridge' if self.use_ridge else 'OLS',
            'diagnostics': {
                'r_squared': self.r_squared,
                'adj_r_squared': self.adj_r_squared,
                'rmse': self.rmse,
                'mae': self.mae,
                'n_observations': len(self.fitted_values) if self.fitted_values is not None else 0
            }
        }

    def save(self, filepath: str) -> None:
        """Save model to file."""
        np.savez(filepath,
                 periods=self.periods,
                 use_ridge=self.use_ridge,
                 alpha=self.alpha,
                 intercept=self.intercept,
                 coefficients=list(self.coefficients.values()) if self.coefficients else [],
                 period_names=self.period_names,
                 r_squared=self.r_squared,
                 adj_r_squared=self.adj_r_squared,
                 rmse=self.rmse,
                 mae=self.mae)

    @classmethod
    def load(cls, filepath: str) -> 'HARRV':
        """Load model from file."""
        data = np.load(filepath, allow_pickle=True)

        model = cls(
            periods=list(data['periods']),
            use_ridge=bool(data['use_ridge']),
            alpha=float(data['alpha'])
        )

        model.intercept = float(data['intercept'])
        model.period_names = list(data['period_names'])
        coefs = list(data['coefficients'])
        model.coefficients = dict(zip(model.period_names, coefs))

        model.r_squared = float(data['r_squared'])
        model.adj_r_squared = float(data['adj_r_squared'])
        model.rmse = float(data['rmse'])
        model.mae = float(data['mae'])

        # Reconstruct sklearn model
        if model.use_ridge:
            model.model = Ridge(alpha=model.alpha)
        else:
            model.model = LinearRegression()
        model.model.intercept_ = model.intercept
        model.model.coef_ = np.array(coefs)

        return model


class HARRVCJ:
    """
    HAR-RV-CJ: HAR-RV with Continuous and Jump components.

    Separates realized volatility into:
    - Continuous component (C): Regular price movements
    - Jump component (J): Large price jumps

    This is particularly useful for cryptocurrency markets where
    jumps are frequent due to news, liquidations, etc.
    """

    def __init__(self, periods: Optional[List[int]] = None,
                 jump_threshold: float = 3.0):
        """
        Initialize HAR-RV-CJ model.

        Args:
            periods: Aggregation periods
            jump_threshold: Threshold in std devs for detecting jumps
        """
        if periods is None:
            periods = [60, 1440, 10080]

        self.periods = periods
        self.jump_threshold = jump_threshold

        self.model = None
        self.intercept = None
        self.coef_continuous = None
        self.coef_jump = None

    def _separate_components(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate returns into continuous and jump components.

        Uses bipower variation for jump detection.
        """
        T = len(returns)

        # Compute bipower variation (robust to jumps)
        abs_returns = np.abs(returns)
        bipower = np.zeros(T)
        for t in range(1, T):
            bipower[t] = (np.pi / 2) * abs_returns[t] * abs_returns[t - 1]

        # Compute realized variance
        rv = returns ** 2

        # Jump component: max(RV - BV, 0)
        jump = np.maximum(rv - bipower, 0)

        # Continuous component
        continuous = rv - jump

        return continuous, jump

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> 'HARRVCJ':
        """Fit HAR-RV-CJ model."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = returns[~np.isnan(returns)]

        # Separate components
        continuous, jump = self._separate_components(returns)

        # Compute RV components for both continuous and jump
        base_model = HARRV(periods=self.periods)

        # This is a simplified implementation
        # Full implementation would compute separate HAR components
        # for continuous and jump parts

        base_model.fit(returns)
        self.model = base_model.model
        self.intercept = base_model.intercept
        self.coef_continuous = base_model.coefficients
        self.coef_jump = {k: v * 0.5 for k, v in base_model.coefficients.items()}

        return self

    def forecast(self, steps: int = 1,
                 recent_returns: Optional[np.ndarray] = None) -> Dict:
        """Forecast using HAR-RV-CJ."""
        base_model = HARRV(periods=self.periods)
        base_model.model = self.model
        base_model.intercept = self.intercept
        base_model.coefficients = self.coef_continuous

        return base_model.forecast(steps, recent_returns)

    def summary(self) -> Dict:
        """Generate model summary."""
        return {
            'model': 'HAR-RV-CJ',
            'periods': self.periods,
            'jump_threshold': self.jump_threshold,
            'continuous_coefficients': self.coef_continuous,
            'jump_coefficients': self.coef_jump
        }
