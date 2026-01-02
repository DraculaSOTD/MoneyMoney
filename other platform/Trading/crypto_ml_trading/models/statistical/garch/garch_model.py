import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Union
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
from scipy.special import gamma
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class GARCH:
    """
    GARCH(p,q) model implementation for volatility modeling.
    
    GARCH = Generalized Autoregressive Conditional Heteroskedasticity
    Variance equation: σ²ₜ = ω + Σᵢ αᵢε²ₜ₋ᵢ + Σⱼ βⱼσ²ₜ₋ⱼ
    
    Specifically optimized for cryptocurrency volatility with:
    - Student's t-distribution for fat tails
    - Robust parameter estimation
    - Rolling window updates
    """
    
    def __init__(self, p: int = 1, q: int = 1, dist: str = 't'):
        """
        Initialize GARCH model.
        
        Args:
            p: Order of GARCH term (past variances)
            q: Order of ARCH term (past squared returns)
            dist: Error distribution ('normal' or 't' for Student's t)
        """
        self.p = p
        self.q = q
        self.dist = dist
        
        # Model parameters
        self.omega = None  # Constant term
        self.alpha = None  # ARCH parameters
        self.beta = None   # GARCH parameters
        self.nu = None     # Degrees of freedom for t-distribution
        
        # Fitted values
        self.returns = None
        self.residuals = None
        self.conditional_variance = None
        self.standardized_residuals = None
        
        # Model diagnostics
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        
    def _initialize_params(self, returns: np.ndarray) -> np.ndarray:
        """
        Initialize parameters for optimization.
        
        Args:
            returns: Return series
            
        Returns:
            Initial parameter vector
        """
        # Calculate unconditional variance
        uncond_var = np.var(returns)
        
        # Initialize omega to target unconditional variance
        omega_init = uncond_var * 0.1
        
        # Initialize alpha and beta to satisfy stationarity
        # α + β < 1 for GARCH(1,1)
        alpha_init = np.ones(self.q) * (0.05 / self.q)
        beta_init = np.ones(self.p) * (0.85 / self.p)
        
        params = [omega_init]
        params.extend(alpha_init)
        params.extend(beta_init)
        
        if self.dist == 't':
            # Initialize degrees of freedom
            params.append(6.0)  # Typical for financial data
            
        return np.array(params)
    
    def _unpack_params(self, params: np.ndarray) -> Tuple:
        """Unpack parameter vector."""
        omega = params[0]
        alpha = params[1:1+self.q]
        beta = params[1+self.q:1+self.q+self.p]
        
        if self.dist == 't':
            nu = params[-1]
            return omega, alpha, beta, nu
        else:
            return omega, alpha, beta, None
    
    def _compute_variance(self, returns: np.ndarray, omega: float,
                         alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Compute conditional variance series.
        
        Args:
            returns: Return series
            omega: Constant term
            alpha: ARCH parameters
            beta: GARCH parameters
            
        Returns:
            Conditional variance series
        """
        T = len(returns)
        h = np.zeros(T)
        
        # Initialize with unconditional variance
        uncond_var = np.var(returns)
        h[:max(self.p, self.q)] = uncond_var
        
        # Compute variance recursively
        for t in range(max(self.p, self.q), T):
            h[t] = omega
            
            # ARCH terms
            for i in range(self.q):
                if t - i - 1 >= 0:
                    h[t] += alpha[i] * returns[t - i - 1]**2
                    
            # GARCH terms
            for j in range(self.p):
                if t - j - 1 >= 0:
                    h[t] += beta[j] * h[t - j - 1]
                    
        return h
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute negative log-likelihood for optimization.
        
        Args:
            params: Parameter vector
            returns: Return series
            
        Returns:
            Negative log-likelihood
        """
        omega, alpha, beta, nu = self._unpack_params(params)
        
        # Parameter constraints
        if omega <= 0:
            return np.inf
            
        if np.any(alpha < 0) or np.any(beta < 0):
            return np.inf
            
        # Stationarity constraint: sum(alpha) + sum(beta) < 1
        if np.sum(alpha) + np.sum(beta) >= 0.999:
            return np.inf
            
        if self.dist == 't' and nu <= 2:
            return np.inf
            
        # Compute conditional variance
        h = self._compute_variance(returns, omega, alpha, beta)
        
        # Avoid numerical issues
        h = np.maximum(h, 1e-8)
        
        # Compute log-likelihood
        T = len(returns)
        z = returns / np.sqrt(h)  # Standardized residuals
        
        if self.dist == 'normal':
            # Normal distribution
            ll = -0.5 * np.sum(np.log(2 * np.pi * h) + z**2)
        else:  # Student's t
            # Student's t log-likelihood
            ll = T * (np.log(gamma((nu + 1) / 2)) -
                     np.log(gamma(nu / 2)) -
                     0.5 * np.log(np.pi * (nu - 2)))
            ll -= 0.5 * np.sum(np.log(h))
            ll -= 0.5 * (nu + 1) * np.sum(np.log(1 + z**2 / (nu - 2)))
            
        return -ll
    
    def fit(self, returns: Union[np.ndarray, pd.Series], 
            start_params: Optional[np.ndarray] = None) -> 'GARCH':
        """
        Fit GARCH model using maximum likelihood.
        
        Args:
            returns: Return series (not prices!)
            start_params: Initial parameters
            
        Returns:
            Fitted GARCH object
        """
        # Convert to numpy
        if isinstance(returns, pd.Series):
            returns = returns.values
            
        # Remove any NaN values
        returns = returns[~np.isnan(returns)]
        self.returns = returns
        
        # Initialize parameters
        if start_params is None:
            start_params = self._initialize_params(returns)
            
        # Set up bounds
        bounds = [(1e-6, None)]  # omega > 0
        bounds.extend([(0, 0.999)] * self.q)  # alpha >= 0
        bounds.extend([(0, 0.999)] * self.p)  # beta >= 0
        
        if self.dist == 't':
            bounds.append((2.01, 100))  # nu > 2
            
        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            start_params,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
            
        # Extract parameters
        self.omega, self.alpha, self.beta, self.nu = self._unpack_params(result.x)
        
        # Compute fitted values
        self.conditional_variance = self._compute_variance(
            returns, self.omega, self.alpha, self.beta
        )
        self.residuals = returns
        self.standardized_residuals = returns / np.sqrt(self.conditional_variance)
        
        # Model diagnostics
        self.log_likelihood = -result.fun
        self._calculate_information_criteria()
        
        return self
    
    def _calculate_information_criteria(self):
        """Calculate AIC and BIC."""
        T = len(self.returns)
        k = 1 + self.q + self.p  # omega + alpha + beta
        if self.dist == 't':
            k += 1  # nu
            
        self.aic = -2 * self.log_likelihood + 2 * k
        self.bic = -2 * self.log_likelihood + k * np.log(T)
        
    def forecast(self, steps: int = 1, returns: Optional[np.ndarray] = None) -> Dict:
        """
        Forecast conditional variance.
        
        Args:
            steps: Forecast horizon
            returns: Recent returns for forecasting (if different from fitted)
            
        Returns:
            Dictionary with variance forecasts and volatility
        """
        if self.omega is None:
            raise ValueError("Model must be fitted before forecasting")
            
        if returns is None:
            returns = self.returns
            h_last = self.conditional_variance
        else:
            # Compute variance for new returns
            h_last = self._compute_variance(returns, self.omega, self.alpha, self.beta)
            
        # Multi-step ahead forecast
        h_forecast = np.zeros(steps)
        
        # Get last values needed
        last_returns2 = returns[-self.q:]**2 if self.q > 0 else np.array([])
        last_h = h_last[-self.p:] if self.p > 0 else np.array([])
        
        for t in range(steps):
            h_forecast[t] = self.omega
            
            # ARCH terms
            for i in range(self.q):
                if t - i - 1 >= 0:
                    # Use forecasted variance as proxy for squared returns
                    h_forecast[t] += self.alpha[i] * h_forecast[t - i - 1]
                elif i - t < len(last_returns2):
                    h_forecast[t] += self.alpha[i] * last_returns2[-(i - t + 1)]
                    
            # GARCH terms
            for j in range(self.p):
                if t - j - 1 >= 0:
                    h_forecast[t] += self.beta[j] * h_forecast[t - j - 1]
                elif j - t < len(last_h):
                    h_forecast[t] += self.beta[j] * last_h[-(j - t + 1)]
                    
        # Convert to volatility
        vol_forecast = np.sqrt(h_forecast)
        
        # Calculate forecast intervals
        if self.dist == 'normal':
            # For normal distribution
            vol_lower = vol_forecast * np.sqrt(0.5)  # Approximation
            vol_upper = vol_forecast * np.sqrt(2.0)
        else:
            # For t-distribution (wider intervals)
            scale = np.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1
            vol_lower = vol_forecast * scale * 0.5
            vol_upper = vol_forecast * scale * 2.0
            
        return {
            'variance': h_forecast,
            'volatility': vol_forecast,
            'volatility_lower': vol_lower,
            'volatility_upper': vol_upper
        }
    
    def calculate_var(self, confidence_level: float = 0.95,
                     horizon: int = 1) -> Tuple[float, float]:
        """
        Calculate Value at Risk using GARCH volatility.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            horizon: Time horizon in periods
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        # Forecast volatility
        forecast = self.forecast(steps=horizon)
        
        # Use the horizon volatility (square root rule for multi-period)
        if horizon == 1:
            vol = forecast['volatility'][0]
        else:
            # Aggregate volatility over horizon
            vol = np.sqrt(np.sum(forecast['variance']))
            
        # Calculate VaR based on distribution
        if self.dist == 'normal':
            z_score = norm.ppf(1 - confidence_level)
            var = -z_score * vol
            # CVaR for normal distribution
            cvar = vol * norm.pdf(z_score) / (1 - confidence_level)
        else:
            # Student's t distribution
            t_score = student_t.ppf(1 - confidence_level, df=self.nu)
            var = -t_score * vol * np.sqrt((self.nu - 2) / self.nu)
            # CVaR for t-distribution
            cvar = var * (self.nu + t_score**2) / (self.nu - 1) * \
                   student_t.pdf(t_score, df=self.nu) / (1 - confidence_level)
            
        return var, cvar
    
    def summary(self) -> Dict:
        """Get model summary."""
        if self.omega is None:
            return {"error": "Model not fitted"}
            
        summary = {
            "model": f"GARCH({self.p},{self.q})",
            "distribution": self.dist,
            "parameters": {
                "omega": self.omega,
                "alpha": self.alpha.tolist(),
                "beta": self.beta.tolist()
            },
            "persistence": np.sum(self.alpha) + np.sum(self.beta),
            "unconditional_variance": self.omega / (1 - np.sum(self.alpha) - np.sum(self.beta)),
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic
        }
        
        if self.dist == 't':
            summary["parameters"]["nu"] = self.nu
            
        # Add diagnostics
        summary["diagnostics"] = {
            "mean_variance": np.mean(self.conditional_variance),
            "min_variance": np.min(self.conditional_variance),
            "max_variance": np.max(self.conditional_variance),
            "volatility_clustering": self._check_volatility_clustering()
        }
        
        return summary
    
    def _check_volatility_clustering(self) -> Dict:
        """Check for volatility clustering in residuals."""
        squared_residuals = self.standardized_residuals**2
        
        # Simple autocorrelation test
        acf1 = np.corrcoef(squared_residuals[:-1], squared_residuals[1:])[0, 1]
        acf5 = np.corrcoef(squared_residuals[:-5], squared_residuals[5:])[0, 1]
        
        return {
            "acf_1": acf1,
            "acf_5": acf5,
            "clustering_present": abs(acf1) > 0.1 or abs(acf5) > 0.1
        }
    
    def rolling_forecast(self, returns: np.ndarray, window: int = 252,
                        step: int = 1) -> pd.DataFrame:
        """
        Perform rolling window GARCH estimation and forecasting.
        
        Args:
            returns: Full return series
            window: Rolling window size
            step: Step size for rolling
            
        Returns:
            DataFrame with rolling forecasts
        """
        n = len(returns)
        forecasts = []
        
        for i in range(window, n, step):
            # Fit on rolling window
            window_returns = returns[i-window:i]
            
            try:
                # Create new model instance
                model = GARCH(p=self.p, q=self.q, dist=self.dist)
                model.fit(window_returns)
                
                # Forecast next period
                forecast = model.forecast(steps=1)
                
                forecasts.append({
                    'date_index': i,
                    'volatility_forecast': forecast['volatility'][0],
                    'variance_forecast': forecast['variance'][0],
                    'persistence': np.sum(model.alpha) + np.sum(model.beta),
                    'omega': model.omega
                })
            except:
                # Skip if optimization fails
                continue
                
        return pd.DataFrame(forecasts)


class EGARCH(GARCH):
    """
    Exponential GARCH model for asymmetric volatility modeling.
    
    log(σ²ₜ) = ω + Σᵢ αᵢg(zₜ₋ᵢ) + Σⱼ βⱼlog(σ²ₜ₋ⱼ)
    where g(z) = θz + γ[|z| - E|z|]
    
    Captures leverage effect: negative returns increase volatility more than positive returns.
    """
    
    def __init__(self, p: int = 1, q: int = 1, dist: str = 't'):
        """
        Initialize EGARCH model.
        
        Args:
            p: Order of GARCH term
            q: Order of ARCH term
            dist: Error distribution
        """
        super().__init__(p, q, dist)
        self.gamma = None  # Asymmetry parameters
        self.theta = None  # Leverage parameters
        
    def _initialize_params(self, returns: np.ndarray) -> np.ndarray:
        """Initialize parameters for EGARCH."""
        # Get base GARCH initialization
        base_params = super()._initialize_params(returns)
        
        # Add asymmetry and leverage parameters
        gamma_init = np.ones(self.q) * 0.1  # Small positive for asymmetry
        theta_init = np.ones(self.q) * -0.05  # Small negative for leverage
        
        # Combine all parameters
        # [omega, alpha, beta, gamma, theta, (nu)]
        params = [base_params[0]]  # omega
        params.extend(base_params[1:1+self.q])  # alpha
        params.extend(base_params[1+self.q:1+self.q+self.p])  # beta
        params.extend(gamma_init)  # gamma
        params.extend(theta_init)  # theta
        
        if self.dist == 't':
            params.append(base_params[-1])  # nu
            
        return np.array(params)
    
    def _unpack_params(self, params: np.ndarray) -> Tuple:
        """Unpack EGARCH parameters."""
        idx = 0
        omega = params[idx]
        idx += 1
        
        alpha = params[idx:idx+self.q]
        idx += self.q
        
        beta = params[idx:idx+self.p]
        idx += self.p
        
        gamma = params[idx:idx+self.q]
        idx += self.q
        
        theta = params[idx:idx+self.q]
        idx += self.q
        
        if self.dist == 't':
            nu = params[idx]
            return omega, alpha, beta, gamma, theta, nu
        else:
            return omega, alpha, beta, gamma, theta, None