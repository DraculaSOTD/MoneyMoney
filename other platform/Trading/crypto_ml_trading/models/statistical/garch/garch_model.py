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

    def save(self, filepath: str) -> None:
        """
        Save model parameters to file.

        Args:
            filepath: Path to save the model (should end with .npz)
        """
        if self.omega is None:
            raise ValueError("Model must be fitted before saving")

        np.savez(filepath,
            p=self.p, q=self.q, dist=self.dist,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            nu=self.nu if self.nu is not None else 0,
            returns=self.returns,
            residuals=self.residuals,
            conditional_variance=self.conditional_variance,
            standardized_residuals=self.standardized_residuals,
            log_likelihood=self.log_likelihood,
            aic=self.aic,
            bic=self.bic
        )

    @classmethod
    def load(cls, filepath: str) -> 'GARCH':
        """
        Load model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded GARCH model
        """
        data = np.load(filepath, allow_pickle=True)

        model = cls(p=int(data['p']), q=int(data['q']), dist=str(data['dist']))
        model.omega = float(data['omega'])
        model.alpha = data['alpha']
        model.beta = data['beta']
        nu_val = float(data['nu'])
        model.nu = nu_val if nu_val != 0 else None
        model.returns = data['returns']
        model.residuals = data['residuals']
        model.conditional_variance = data['conditional_variance']
        model.standardized_residuals = data['standardized_residuals']
        model.log_likelihood = float(data['log_likelihood'])
        model.aic = float(data['aic'])
        model.bic = float(data['bic'])

        return model


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


class GJRGARCH:
    """
    GJR-GARCH (Glosten-Jagannathan-Runkle) model for asymmetric volatility.

    Variance equation:
    σ²ₜ = ω + Σᵢ (αᵢ + γᵢ·I(εₜ₋ᵢ<0))ε²ₜ₋ᵢ + Σⱼ βⱼσ²ₜ₋ⱼ

    where I(εₜ₋ᵢ<0) is an indicator function that equals 1 when the
    previous return was negative (bad news) and 0 otherwise.

    This captures the leverage effect in financial markets where negative
    returns tend to increase volatility more than positive returns of the
    same magnitude - a key stylized fact in cryptocurrency markets.
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = 't'):
        """
        Initialize GJR-GARCH model.

        Args:
            p: Order of GARCH term (past variances)
            q: Order of ARCH term (past squared returns)
            dist: Error distribution ('normal' or 't' for Student's t)
        """
        self.p = p
        self.q = q
        self.dist = dist

        # Model parameters
        self.omega = None   # Constant term
        self.alpha = None   # ARCH parameters (symmetric)
        self.gamma = None   # Asymmetry parameters (leverage effect)
        self.beta = None    # GARCH parameters
        self.nu = None      # Degrees of freedom for t-distribution

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

        Uses moment-based initialization for better convergence.
        """
        unconditional_var = np.var(returns)

        # Initialize omega (intercept)
        omega_init = 0.1 * unconditional_var

        # Initialize alpha (ARCH, symmetric effect)
        alpha_init = np.ones(self.q) * 0.03 / self.q

        # Initialize gamma (asymmetry/leverage effect)
        # Typically positive: negative returns increase vol more
        gamma_init = np.ones(self.q) * 0.04 / self.q

        # Initialize beta (GARCH persistence)
        beta_init = np.ones(self.p) * 0.85 / self.p

        # Combine parameters
        params = [omega_init]
        params.extend(alpha_init)
        params.extend(gamma_init)
        params.extend(beta_init)

        if self.dist == 't':
            params.append(8.0)  # nu (degrees of freedom)

        return np.array(params)

    def _compute_variance(self, returns: np.ndarray, omega: float,
                         alpha: np.ndarray, gamma: np.ndarray,
                         beta: np.ndarray) -> np.ndarray:
        """
        Compute conditional variance series with asymmetric effect.

        The GJR-GARCH variance is:
        h[t] = omega + sum(alpha[i] * eps[t-i]^2)
                     + sum(gamma[i] * eps[t-i]^2 * I(eps[t-i]<0))
                     + sum(beta[j] * h[t-j])
        """
        T = len(returns)
        h = np.zeros(T)

        # Initial variance (unconditional)
        initial_var = np.var(returns)
        max_lag = max(self.p, self.q)

        for t in range(T):
            h[t] = omega

            # ARCH terms with asymmetry
            for i in range(self.q):
                if t - i - 1 >= 0:
                    eps_sq = returns[t - i - 1] ** 2
                    # Indicator: 1 if return was negative, 0 otherwise
                    indicator = 1.0 if returns[t - i - 1] < 0 else 0.0
                    h[t] += alpha[i] * eps_sq + gamma[i] * indicator * eps_sq
                elif t < max_lag:
                    h[t] += (alpha[i] + 0.5 * gamma[i]) * initial_var

            # GARCH terms
            for j in range(self.p):
                if t - j - 1 >= 0:
                    h[t] += beta[j] * h[t - j - 1]
                elif t < max_lag:
                    h[t] += beta[j] * initial_var

            # Ensure positive variance
            h[t] = max(h[t], 1e-10)

        return h

    def _negative_log_likelihood(self, params: np.ndarray,
                                  returns: np.ndarray) -> float:
        """
        Compute negative log-likelihood for MLE estimation.
        """
        # Unpack parameters
        idx = 0
        omega = params[idx]
        idx += 1

        alpha = params[idx:idx + self.q]
        idx += self.q

        gamma = params[idx:idx + self.q]
        idx += self.q

        beta = params[idx:idx + self.p]
        idx += self.p

        if self.dist == 't':
            nu = params[idx]
        else:
            nu = None

        # Parameter constraints
        if omega <= 0:
            return np.inf
        if np.any(alpha < 0) or np.any(gamma < 0) or np.any(beta < 0):
            return np.inf
        # Stationarity: sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1
        persistence = np.sum(alpha) + 0.5 * np.sum(gamma) + np.sum(beta)
        if persistence >= 0.999:
            return np.inf
        if self.dist == 't' and nu <= 2:
            return np.inf

        # Compute conditional variance
        h = self._compute_variance(returns, omega, alpha, gamma, beta)

        # Compute log-likelihood
        z = returns / np.sqrt(h)  # Standardized residuals

        if self.dist == 'normal':
            ll = -0.5 * np.sum(np.log(2 * np.pi * h) + z ** 2)
        else:
            # Student's t log-likelihood
            ll = len(returns) * (
                np.log(gamma(0.5 * (nu + 1))) -
                np.log(gamma(0.5 * nu)) -
                0.5 * np.log(np.pi * (nu - 2))
            )
            ll -= 0.5 * np.sum(np.log(h))
            ll -= 0.5 * (nu + 1) * np.sum(np.log(1 + z ** 2 / (nu - 2)))

        return -ll  # Return negative for minimization

    def fit(self, returns: Union[np.ndarray, pd.Series],
            start_params: Optional[np.ndarray] = None) -> 'GJRGARCH':
        """
        Fit GJR-GARCH model using maximum likelihood estimation.

        Args:
            returns: Return series
            start_params: Initial parameter values

        Returns:
            Self for method chaining
        """
        # Convert to numpy array
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = returns[~np.isnan(returns)]
        self.returns = returns

        # Initialize parameters
        if start_params is None:
            start_params = self._initialize_params(returns)

        # Set up bounds
        bounds = [(1e-6, None)]  # omega > 0
        bounds.extend([(0, 0.999)] * self.q)  # alpha >= 0
        bounds.extend([(0, 0.999)] * self.q)  # gamma >= 0
        bounds.extend([(0, 0.999)] * self.p)  # beta >= 0

        if self.dist == 't':
            bounds.append((2.01, 100))  # nu > 2

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._negative_log_likelihood,
                start_params,
                args=(returns,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )

        # Extract optimal parameters
        params = result.x
        idx = 0

        self.omega = params[idx]
        idx += 1

        self.alpha = params[idx:idx + self.q]
        idx += self.q

        self.gamma = params[idx:idx + self.q]
        idx += self.q

        self.beta = params[idx:idx + self.p]
        idx += self.p

        if self.dist == 't':
            self.nu = params[idx]

        # Compute fitted values
        self.conditional_variance = self._compute_variance(
            returns, self.omega, self.alpha, self.gamma, self.beta
        )
        self.residuals = returns
        self.standardized_residuals = returns / np.sqrt(self.conditional_variance)

        # Compute log-likelihood and information criteria
        self.log_likelihood = -result.fun

        T = len(returns)
        k = 1 + self.q + self.q + self.p  # omega + alpha + gamma + beta
        if self.dist == 't':
            k += 1  # nu

        self.aic = -2 * self.log_likelihood + 2 * k
        self.bic = -2 * self.log_likelihood + k * np.log(T)

        return self

    def forecast(self, steps: int = 1,
                 returns: Optional[np.ndarray] = None) -> Dict:
        """
        Forecast conditional variance.

        Args:
            steps: Forecast horizon
            returns: Recent returns for forecasting

        Returns:
            Dictionary with variance forecasts and volatility
        """
        if self.omega is None:
            raise ValueError("Model must be fitted before forecasting")

        if returns is None:
            returns = self.returns
            h_last = self.conditional_variance
        else:
            h_last = self._compute_variance(
                returns, self.omega, self.alpha, self.gamma, self.beta
            )

        # Multi-step ahead forecast
        h_forecast = np.zeros(steps)

        # Get last values needed
        last_returns = returns[-self.q:] if self.q > 0 else np.array([])
        last_h = h_last[-self.p:] if self.p > 0 else np.array([])

        # Expected value of indicator: P(r < 0) ≈ 0.5 for symmetric returns
        # This is used for future forecasts where we don't know the sign
        expected_indicator = 0.5

        for t in range(steps):
            h_forecast[t] = self.omega

            # ARCH terms
            for i in range(self.q):
                if t - i - 1 >= 0:
                    # Use forecasted variance as proxy for expected squared return
                    h_forecast[t] += (self.alpha[i] + expected_indicator * self.gamma[i]) * h_forecast[t - i - 1]
                elif i - t < len(last_returns):
                    eps_sq = last_returns[-(i - t + 1)] ** 2
                    indicator = 1.0 if last_returns[-(i - t + 1)] < 0 else 0.0
                    h_forecast[t] += self.alpha[i] * eps_sq + self.gamma[i] * indicator * eps_sq

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
            vol_lower = vol_forecast * np.sqrt(0.5)
            vol_upper = vol_forecast * np.sqrt(2.0)
        else:
            scale = np.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1
            vol_lower = vol_forecast * scale * 0.5
            vol_upper = vol_forecast * scale * 2.0

        return {
            'variance': h_forecast,
            'volatility': vol_forecast,
            'volatility_lower': vol_lower,
            'volatility_upper': vol_upper
        }

    def summary(self) -> Dict:
        """
        Generate model summary with diagnostics.

        Returns:
            Dictionary with model summary
        """
        if self.omega is None:
            return {'error': 'Model not fitted'}

        # Persistence includes asymmetric effect
        persistence = np.sum(self.alpha) + 0.5 * np.sum(self.gamma) + np.sum(self.beta)

        # Half-life of volatility shocks
        if 0 < persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf

        # Unconditional variance
        if persistence < 1:
            uncond_var = self.omega / (1 - persistence)
        else:
            uncond_var = np.inf

        # Leverage ratio: effect of negative vs positive shocks
        # (alpha + gamma) / alpha - measures asymmetry
        avg_alpha = np.mean(self.alpha)
        avg_gamma = np.mean(self.gamma)
        if avg_alpha > 0:
            leverage_ratio = (avg_alpha + avg_gamma) / avg_alpha
        else:
            leverage_ratio = np.inf

        return {
            'model': f'GJR-GARCH({self.p},{self.q})',
            'distribution': self.dist,
            'parameters': {
                'omega': self.omega,
                'alpha': self.alpha.tolist(),
                'gamma': self.gamma.tolist(),
                'beta': self.beta.tolist(),
                'nu': self.nu if self.dist == 't' else None
            },
            'persistence': persistence,
            'half_life': half_life,
            'unconditional_variance': uncond_var,
            'unconditional_volatility': np.sqrt(uncond_var) if uncond_var < np.inf else np.inf,
            'leverage_ratio': leverage_ratio,
            'diagnostics': {
                'log_likelihood': self.log_likelihood,
                'aic': self.aic,
                'bic': self.bic,
                'n_observations': len(self.returns) if self.returns is not None else 0
            }
        }

    def save(self, filepath: str) -> None:
        """Save model parameters to file."""
        np.savez(filepath,
                 p=self.p, q=self.q, dist=self.dist,
                 omega=self.omega,
                 alpha=self.alpha,
                 gamma=self.gamma,
                 beta=self.beta,
                 nu=self.nu if self.nu else 0,
                 returns=self.returns,
                 residuals=self.residuals,
                 conditional_variance=self.conditional_variance,
                 standardized_residuals=self.standardized_residuals,
                 log_likelihood=self.log_likelihood,
                 aic=self.aic,
                 bic=self.bic)

    @classmethod
    def load(cls, filepath: str) -> 'GJRGARCH':
        """Load model from file."""
        data = np.load(filepath, allow_pickle=True)

        model = cls(p=int(data['p']), q=int(data['q']), dist=str(data['dist']))
        model.omega = float(data['omega'])
        model.alpha = data['alpha']
        model.gamma = data['gamma']
        model.beta = data['beta']
        model.nu = float(data['nu']) if data['nu'] != 0 else None
        model.returns = data['returns']
        model.residuals = data['residuals']
        model.conditional_variance = data['conditional_variance']
        model.standardized_residuals = data['standardized_residuals']
        model.log_likelihood = float(data['log_likelihood'])
        model.aic = float(data['aic'])
        model.bic = float(data['bic'])

        return model


class GARCHX:
    """
    GARCH-X model with exogenous variables.

    Variance equation:
    σ²ₜ = ω + Σᵢ αᵢε²ₜ₋ᵢ + Σⱼ βⱼσ²ₜ₋ⱼ + Σₖ δₖXₜ₋₁,ₖ

    where X is a matrix of exogenous variables that can include:
    - Lagged realized volatility
    - Volume/volume ratio
    - Jump intensity
    - Other external factors

    This allows the model to incorporate additional information
    beyond past returns and variances.
    """

    def __init__(self, p: int = 1, q: int = 1, n_exog: int = 1, dist: str = 't'):
        """
        Initialize GARCH-X model.

        Args:
            p: Order of GARCH term
            q: Order of ARCH term
            n_exog: Number of exogenous variables
            dist: Error distribution
        """
        self.p = p
        self.q = q
        self.n_exog = n_exog
        self.dist = dist

        # Model parameters
        self.omega = None
        self.alpha = None
        self.beta = None
        self.delta = None  # Exogenous variable coefficients
        self.nu = None

        # Fitted values
        self.returns = None
        self.exog = None
        self.residuals = None
        self.conditional_variance = None
        self.standardized_residuals = None

        # Diagnostics
        self.log_likelihood = None
        self.aic = None
        self.bic = None

    def _initialize_params(self, returns: np.ndarray, exog: np.ndarray) -> np.ndarray:
        """Initialize parameters for optimization."""
        unconditional_var = np.var(returns)

        omega_init = 0.1 * unconditional_var
        alpha_init = np.ones(self.q) * 0.05 / self.q
        beta_init = np.ones(self.p) * 0.85 / self.p

        # Initialize delta based on correlation with variance proxy
        delta_init = np.ones(self.n_exog) * 0.01

        params = [omega_init]
        params.extend(alpha_init)
        params.extend(beta_init)
        params.extend(delta_init)

        if self.dist == 't':
            params.append(8.0)

        return np.array(params)

    def _compute_variance(self, returns: np.ndarray, exog: np.ndarray,
                         omega: float, alpha: np.ndarray, beta: np.ndarray,
                         delta: np.ndarray) -> np.ndarray:
        """Compute conditional variance with exogenous variables."""
        T = len(returns)
        h = np.zeros(T)

        initial_var = np.var(returns)
        max_lag = max(self.p, self.q)

        for t in range(T):
            h[t] = omega

            # ARCH terms
            for i in range(self.q):
                if t - i - 1 >= 0:
                    h[t] += alpha[i] * returns[t - i - 1] ** 2
                elif t < max_lag:
                    h[t] += alpha[i] * initial_var

            # GARCH terms
            for j in range(self.p):
                if t - j - 1 >= 0:
                    h[t] += beta[j] * h[t - j - 1]
                elif t < max_lag:
                    h[t] += beta[j] * initial_var

            # Exogenous terms (lagged by 1)
            if t > 0:
                for k in range(self.n_exog):
                    h[t] += delta[k] * exog[t - 1, k]

            h[t] = max(h[t], 1e-10)

        return h

    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray,
                                  exog: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        idx = 0
        omega = params[idx]
        idx += 1

        alpha = params[idx:idx + self.q]
        idx += self.q

        beta = params[idx:idx + self.p]
        idx += self.p

        delta = params[idx:idx + self.n_exog]
        idx += self.n_exog

        if self.dist == 't':
            nu = params[idx]
        else:
            nu = None

        # Constraints
        if omega <= 0:
            return np.inf
        if np.any(alpha < 0) or np.any(beta < 0):
            return np.inf
        persistence = np.sum(alpha) + np.sum(beta)
        if persistence >= 0.999:
            return np.inf
        if self.dist == 't' and nu <= 2:
            return np.inf

        h = self._compute_variance(returns, exog, omega, alpha, beta, delta)
        z = returns / np.sqrt(h)

        if self.dist == 'normal':
            ll = -0.5 * np.sum(np.log(2 * np.pi * h) + z ** 2)
        else:
            ll = len(returns) * (
                np.log(gamma(0.5 * (nu + 1))) -
                np.log(gamma(0.5 * nu)) -
                0.5 * np.log(np.pi * (nu - 2))
            )
            ll -= 0.5 * np.sum(np.log(h))
            ll -= 0.5 * (nu + 1) * np.sum(np.log(1 + z ** 2 / (nu - 2)))

        return -ll

    def fit(self, returns: Union[np.ndarray, pd.Series],
            exog: np.ndarray,
            start_params: Optional[np.ndarray] = None) -> 'GARCHX':
        """
        Fit GARCH-X model.

        Args:
            returns: Return series
            exog: Exogenous variables matrix (T x n_exog)
            start_params: Initial parameters

        Returns:
            Self for method chaining
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = returns[~np.isnan(returns)]

        # Ensure exog has correct shape
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)

        # Align lengths
        min_len = min(len(returns), len(exog))
        returns = returns[:min_len]
        exog = exog[:min_len]

        self.returns = returns
        self.exog = exog
        self.n_exog = exog.shape[1]

        if start_params is None:
            start_params = self._initialize_params(returns, exog)

        # Bounds
        bounds = [(1e-6, None)]  # omega
        bounds.extend([(0, 0.999)] * self.q)  # alpha
        bounds.extend([(0, 0.999)] * self.p)  # beta
        bounds.extend([(-1, 1)] * self.n_exog)  # delta (can be negative)

        if self.dist == 't':
            bounds.append((2.01, 100))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._negative_log_likelihood,
                start_params,
                args=(returns, exog),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )

        params = result.x
        idx = 0

        self.omega = params[idx]
        idx += 1

        self.alpha = params[idx:idx + self.q]
        idx += self.q

        self.beta = params[idx:idx + self.p]
        idx += self.p

        self.delta = params[idx:idx + self.n_exog]
        idx += self.n_exog

        if self.dist == 't':
            self.nu = params[idx]

        self.conditional_variance = self._compute_variance(
            returns, exog, self.omega, self.alpha, self.beta, self.delta
        )
        self.residuals = returns
        self.standardized_residuals = returns / np.sqrt(self.conditional_variance)

        self.log_likelihood = -result.fun

        T = len(returns)
        k = 1 + self.q + self.p + self.n_exog
        if self.dist == 't':
            k += 1

        self.aic = -2 * self.log_likelihood + 2 * k
        self.bic = -2 * self.log_likelihood + k * np.log(T)

        return self

    def forecast(self, steps: int = 1,
                 future_exog: Optional[np.ndarray] = None) -> Dict:
        """
        Forecast conditional variance.

        Args:
            steps: Forecast horizon
            future_exog: Future exogenous values (steps x n_exog)

        Returns:
            Dictionary with forecasts
        """
        if self.omega is None:
            raise ValueError("Model must be fitted before forecasting")

        if future_exog is None:
            # Use last observed exogenous values
            future_exog = np.tile(self.exog[-1], (steps, 1))

        if future_exog.ndim == 1:
            future_exog = future_exog.reshape(-1, 1)

        h_forecast = np.zeros(steps)
        last_returns = self.returns[-self.q:] if self.q > 0 else np.array([])
        last_h = self.conditional_variance[-self.p:] if self.p > 0 else np.array([])

        for t in range(steps):
            h_forecast[t] = self.omega

            # ARCH terms
            for i in range(self.q):
                if t - i - 1 >= 0:
                    h_forecast[t] += self.alpha[i] * h_forecast[t - i - 1]
                elif i - t < len(last_returns):
                    h_forecast[t] += self.alpha[i] * last_returns[-(i - t + 1)] ** 2

            # GARCH terms
            for j in range(self.p):
                if t - j - 1 >= 0:
                    h_forecast[t] += self.beta[j] * h_forecast[t - j - 1]
                elif j - t < len(last_h):
                    h_forecast[t] += self.beta[j] * last_h[-(j - t + 1)]

            # Exogenous terms
            for k in range(self.n_exog):
                h_forecast[t] += self.delta[k] * future_exog[t, k]

            h_forecast[t] = max(h_forecast[t], 1e-10)

        vol_forecast = np.sqrt(h_forecast)

        if self.dist == 'normal':
            vol_lower = vol_forecast * np.sqrt(0.5)
            vol_upper = vol_forecast * np.sqrt(2.0)
        else:
            scale = np.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1
            vol_lower = vol_forecast * scale * 0.5
            vol_upper = vol_forecast * scale * 2.0

        return {
            'variance': h_forecast,
            'volatility': vol_forecast,
            'volatility_lower': vol_lower,
            'volatility_upper': vol_upper
        }

    def summary(self) -> Dict:
        """Generate model summary."""
        if self.omega is None:
            return {'error': 'Model not fitted'}

        persistence = np.sum(self.alpha) + np.sum(self.beta)

        return {
            'model': f'GARCH-X({self.p},{self.q}) with {self.n_exog} exogenous',
            'distribution': self.dist,
            'parameters': {
                'omega': self.omega,
                'alpha': self.alpha.tolist(),
                'beta': self.beta.tolist(),
                'delta': self.delta.tolist(),
                'nu': self.nu if self.dist == 't' else None
            },
            'persistence': persistence,
            'exogenous_effects': {f'delta_{i}': self.delta[i] for i in range(self.n_exog)},
            'diagnostics': {
                'log_likelihood': self.log_likelihood,
                'aic': self.aic,
                'bic': self.bic
            }
        }

    def save(self, filepath: str) -> None:
        """Save model to file."""
        np.savez(filepath,
                 p=self.p, q=self.q, n_exog=self.n_exog, dist=self.dist,
                 omega=self.omega, alpha=self.alpha, beta=self.beta,
                 delta=self.delta, nu=self.nu if self.nu else 0,
                 returns=self.returns, exog=self.exog,
                 conditional_variance=self.conditional_variance,
                 log_likelihood=self.log_likelihood,
                 aic=self.aic, bic=self.bic)

    @classmethod
    def load(cls, filepath: str) -> 'GARCHX':
        """Load model from file."""
        data = np.load(filepath, allow_pickle=True)

        model = cls(
            p=int(data['p']), q=int(data['q']),
            n_exog=int(data['n_exog']), dist=str(data['dist'])
        )
        model.omega = float(data['omega'])
        model.alpha = data['alpha']
        model.beta = data['beta']
        model.delta = data['delta']
        model.nu = float(data['nu']) if data['nu'] != 0 else None
        model.returns = data['returns']
        model.exog = data['exog']
        model.conditional_variance = data['conditional_variance']
        model.log_likelihood = float(data['log_likelihood'])
        model.aic = float(data['aic'])
        model.bic = float(data['bic'])

        return model