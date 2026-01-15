import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class ARIMA:
    """
    Custom ARIMA(p,d,q) implementation without statsmodels.
    
    ARIMA = AutoRegressive Integrated Moving Average
    - AR(p): p autoregressive terms
    - I(d): d differencing operations
    - MA(q): q moving average terms
    """
    
    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        """
        Initialize ARIMA model.
        
        Args:
            p: Order of autoregression
            d: Degree of differencing
            q: Order of moving average
        """
        self.p = p
        self.d = d
        self.q = q
        
        # Model parameters
        self.ar_params = None
        self.ma_params = None
        self.intercept = None
        self.sigma2 = None
        
        # Fitted values
        self.residuals = None
        self.fitted_values = None
        self.data = None
        self.differenced_data = None
        
        # Information criteria
        self.aic = None
        self.bic = None
        self.log_likelihood = None
        
    def difference(self, data: np.ndarray, d: int) -> np.ndarray:
        """
        Apply differencing to make series stationary.
        
        Args:
            data: Time series data
            d: Number of differences
            
        Returns:
            Differenced series
        """
        diff_data = data.copy()
        for _ in range(d):
            diff_data = np.diff(diff_data)
        return diff_data
    
    def inverse_difference(self, diff_data: np.ndarray, original_data: np.ndarray,
                          d: int) -> np.ndarray:
        """
        Inverse differencing operation.
        
        Args:
            diff_data: Differenced data
            original_data: Original data for initial values
            d: Number of differences applied
            
        Returns:
            Reconstructed series
        """
        result = diff_data.copy()
        
        for i in range(d):
            # Get the appropriate initial values
            if i == 0:
                init_values = original_data[-(d-i):][:len(result)]
            else:
                init_values = np.cumsum(np.concatenate([[original_data[-(d-i)]], result[:-1]]))[-len(result):]
            
            # Reconstruct by cumulative sum
            result = np.cumsum(np.concatenate([[init_values[0]], result]))
            
        return result
    
    def _negative_log_likelihood(self, params: np.ndarray, endog: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for optimization.
        
        Args:
            params: Parameter vector [intercept, ar_params, ma_params, sigma2]
            endog: Endogenous variable (differenced if d > 0)
            
        Returns:
            Negative log-likelihood
        """
        # Extract parameters
        intercept = params[0]
        ar_params = params[1:1+self.p] if self.p > 0 else np.array([])
        ma_params = params[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
        sigma2 = params[-1]
        
        if sigma2 <= 0:
            return np.inf
        
        n = len(endog)
        residuals = np.zeros(n)
        
        # Initialize
        for t in range(max(self.p, self.q), n):
            # AR component
            ar_term = intercept
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_term += ar_params[i] * endog[t - i - 1]
            
            # MA component
            ma_term = 0
            for j in range(self.q):
                if t - j - 1 >= 0:
                    ma_term += ma_params[j] * residuals[t - j - 1]
            
            # Calculate residual
            residuals[t] = endog[t] - ar_term - ma_term
        
        # Calculate log-likelihood
        valid_residuals = residuals[max(self.p, self.q):]
        n_valid = len(valid_residuals)
        
        log_likelihood = -0.5 * n_valid * np.log(2 * np.pi * sigma2)
        log_likelihood -= 0.5 * np.sum(valid_residuals**2) / sigma2
        
        return -log_likelihood
    
    def fit(self, data: Union[np.ndarray, pd.Series], method: str = 'mle',
            start_params: Optional[np.ndarray] = None) -> 'ARIMA':
        """
        Fit ARIMA model using maximum likelihood estimation.
        
        Args:
            data: Time series data
            method: Estimation method ('mle' for maximum likelihood)
            start_params: Initial parameter values
            
        Returns:
            Fitted ARIMA object
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        
        self.data = data.copy()
        
        # Apply differencing
        if self.d > 0:
            self.differenced_data = self.difference(data, self.d)
        else:
            self.differenced_data = data.copy()
        
        # Initialize parameters
        if start_params is None:
            # Simple initialization
            intercept_init = np.mean(self.differenced_data)
            ar_init = np.random.randn(self.p) * 0.1 if self.p > 0 else np.array([])
            ma_init = np.random.randn(self.q) * 0.1 if self.q > 0 else np.array([])
            sigma2_init = np.var(self.differenced_data)
            
            start_params = np.concatenate([
                [intercept_init],
                ar_init,
                ma_init,
                [sigma2_init]
            ])
        
        # Optimize parameters
        if method == 'mle':
            # Bounds for parameters
            bounds = [(None, None)]  # intercept
            bounds.extend([(-0.999, 0.999)] * self.p)  # AR parameters (stationarity)
            bounds.extend([(-0.999, 0.999)] * self.q)  # MA parameters (invertibility)
            bounds.append((1e-6, None))  # sigma2
            
            # Optimize
            result = minimize(
                self._negative_log_likelihood,
                start_params,
                args=(self.differenced_data,),
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if not result.success:
                warnings.warn(f"Optimization failed: {result.message}")
            
            # Extract fitted parameters
            fitted_params = result.x
            self.intercept = fitted_params[0]
            self.ar_params = fitted_params[1:1+self.p] if self.p > 0 else np.array([])
            self.ma_params = fitted_params[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
            self.sigma2 = fitted_params[-1]
            
            # Calculate fitted values and residuals
            self._calculate_fitted_values()
            
            # Calculate information criteria
            self.log_likelihood = -result.fun
            self._calculate_information_criteria()
        
        return self
    
    def _calculate_fitted_values(self):
        """Calculate fitted values and residuals."""
        n = len(self.differenced_data)
        self.fitted_values = np.zeros(n)
        self.residuals = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            # AR component
            ar_term = self.intercept
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_term += self.ar_params[i] * self.differenced_data[t - i - 1]
            
            # MA component
            ma_term = 0
            for j in range(self.q):
                if t - j - 1 >= 0:
                    ma_term += self.ma_params[j] * self.residuals[t - j - 1]
            
            # Fitted value
            self.fitted_values[t] = ar_term + ma_term
            self.residuals[t] = self.differenced_data[t] - self.fitted_values[t]
    
    def _calculate_information_criteria(self):
        """Calculate AIC and BIC."""
        n = len(self.differenced_data)
        k = 1 + self.p + self.q + 1  # intercept + AR + MA + sigma2
        
        self.aic = -2 * self.log_likelihood + 2 * k
        self.bic = -2 * self.log_likelihood + k * np.log(n)
    
    def predict(self, steps: int = 1, return_conf_int: bool = False,
                alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Predictions or tuple of (predictions, lower_bound, upper_bound)
        """
        if self.ar_params is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get the last values needed for prediction
        last_values = self.differenced_data[-self.p:] if self.p > 0 else np.array([])
        last_residuals = self.residuals[-self.q:] if self.q > 0 else np.array([])
        
        predictions = []
        
        for h in range(steps):
            # AR component
            pred = self.intercept
            
            # Use previous predictions for multi-step ahead
            for i in range(self.p):
                if i < len(last_values):
                    pred += self.ar_params[i] * last_values[-(i+1)]
                elif len(predictions) > i - len(last_values):
                    pred += self.ar_params[i] * predictions[-(i-len(last_values)+1)]
            
            # MA component (assume future shocks are zero)
            for j in range(self.q):
                if j < len(last_residuals):
                    pred += self.ma_params[j] * last_residuals[-(j+1)]
            
            predictions.append(pred)
            
            # Update last_values for multi-step prediction
            if self.p > 0:
                last_values = np.append(last_values, pred)[-self.p:]
        
        predictions = np.array(predictions)
        
        # Inverse difference to get predictions in original scale
        if self.d > 0:
            # Need the last d values from original data
            last_original = self.data[-self.d:]
            predictions = self.inverse_difference(predictions, last_original, self.d)
            # Trim to keep only the forecasted steps (inverse_difference adds d extra elements)
            predictions = predictions[-steps:]

        if return_conf_int:
            # Calculate prediction intervals
            # For multi-step ahead, variance increases
            forecast_variance = np.zeros(steps)
            
            # MA representation for calculating forecast variance
            psi_weights = self._calculate_psi_weights(steps)
            
            for h in range(steps):
                forecast_variance[h] = self.sigma2 * np.sum(psi_weights[:h+1]**2)
            
            z_score = norm.ppf(1 - alpha/2)
            margin = z_score * np.sqrt(forecast_variance)
            
            lower = predictions - margin
            upper = predictions + margin
            
            return predictions, lower, upper
        
        return predictions
    
    def _calculate_psi_weights(self, steps: int) -> np.ndarray:
        """
        Calculate psi weights for MA representation of ARIMA model.
        
        Args:
            steps: Number of weights to calculate
            
        Returns:
            Array of psi weights
        """
        psi = np.zeros(steps + max(self.p, self.q))
        psi[0] = 1
        
        for j in range(1, len(psi)):
            # AR contribution
            ar_sum = 0
            for i in range(min(j, self.p)):
                ar_sum += self.ar_params[i] * psi[j-i-1]
            
            # MA contribution
            ma_term = self.ma_params[j-1] if j <= self.q else 0
            
            psi[j] = ar_sum + ma_term
        
        return psi[:steps]
    
    def summary(self) -> Dict:
        """
        Get model summary.
        
        Returns:
            Dictionary with model information
        """
        if self.ar_params is None:
            return {"error": "Model not fitted"}
        
        return {
            "model": f"ARIMA({self.p},{self.d},{self.q})",
            "parameters": {
                "intercept": self.intercept,
                "ar": self.ar_params.tolist() if len(self.ar_params) > 0 else [],
                "ma": self.ma_params.tolist() if len(self.ma_params) > 0 else [],
                "sigma2": self.sigma2
            },
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "residual_stats": {
                "mean": np.mean(self.residuals),
                "std": np.std(self.residuals),
                "skewness": self._calculate_skewness(self.residuals),
                "kurtosis": self._calculate_kurtosis(self.residuals)
            }
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def plot_diagnostics(self):
        """
        Note: Plotting functionality would require matplotlib.
        This is a placeholder for diagnostic plots.
        """
        print("Diagnostic plots:")
        print("1. Residuals plot")
        print("2. ACF of residuals")
        print("3. PACF of residuals")
        print("4. Q-Q plot of residuals")
        print("\nImplement with matplotlib when needed.")

    def save(self, filepath: str) -> None:
        """
        Save model parameters to file.

        Args:
            filepath: Path to save the model (should end with .npz)
        """
        if self.ar_params is None:
            raise ValueError("Model must be fitted before saving")

        np.savez(filepath,
            p=self.p, d=self.d, q=self.q,
            ar_params=self.ar_params,
            ma_params=self.ma_params,
            intercept=self.intercept,
            sigma2=self.sigma2,
            residuals=self.residuals,
            fitted_values=self.fitted_values,
            data=self.data,
            differenced_data=self.differenced_data,
            aic=self.aic,
            bic=self.bic,
            log_likelihood=self.log_likelihood
        )

    @classmethod
    def load(cls, filepath: str) -> 'ARIMA':
        """
        Load model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded ARIMA model
        """
        data = np.load(filepath, allow_pickle=True)

        model = cls(p=int(data['p']), d=int(data['d']), q=int(data['q']))
        model.ar_params = data['ar_params']
        model.ma_params = data['ma_params']
        model.intercept = float(data['intercept'])
        model.sigma2 = float(data['sigma2'])
        model.residuals = data['residuals']
        model.fitted_values = data['fitted_values']
        model.data = data['data']
        model.differenced_data = data['differenced_data']
        model.aic = float(data['aic'])
        model.bic = float(data['bic'])
        model.log_likelihood = float(data['log_likelihood'])

        return model


class AutoARIMA:
    """
    Automatic ARIMA model selection using information criteria.
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5,
                 seasonal: bool = False, information_criterion: str = 'aic'):
        """
        Initialize AutoARIMA.
        
        Args:
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            seasonal: Whether to include seasonal components (not implemented)
            information_criterion: 'aic' or 'bic'
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.information_criterion = information_criterion
        
        self.best_model = None
        self.best_params = None
        self.results = []
        
    def fit(self, data: Union[np.ndarray, pd.Series]) -> ARIMA:
        """
        Find best ARIMA model by grid search.
        
        Args:
            data: Time series data
            
        Returns:
            Best fitted ARIMA model
        """
        best_score = np.inf
        
        # Grid search over all combinations
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    # Skip the (0,0,0) model
                    if p == 0 and d == 0 and q == 0:
                        continue
                    
                    try:
                        # Fit model
                        model = ARIMA(p=p, d=d, q=q)
                        model.fit(data)
                        
                        # Get score
                        if self.information_criterion == 'aic':
                            score = model.aic
                        else:
                            score = model.bic
                        
                        # Store results
                        self.results.append({
                            'order': (p, d, q),
                            'aic': model.aic,
                            'bic': model.bic,
                            'log_likelihood': model.log_likelihood
                        })
                        
                        # Update best model
                        if score < best_score:
                            best_score = score
                            self.best_model = model
                            self.best_params = (p, d, q)
                            
                    except Exception as e:
                        # Some models may fail to converge
                        continue
        
        if self.best_model is None:
            raise ValueError("No valid ARIMA model found")
        
        return self.best_model
    
    def summary(self) -> Dict:
        """Get summary of model selection."""
        if self.best_model is None:
            return {"error": "No model fitted"}
        
        return {
            "best_model": f"ARIMA{self.best_params}",
            "criterion": self.information_criterion,
            "best_score": getattr(self.best_model, self.information_criterion),
            "all_results": sorted(self.results, 
                                key=lambda x: x[self.information_criterion])[:10]
        }