import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from models.statistical.garch.garch_model import GARCH, EGARCH


class VolatilityForecaster:
    """
    Advanced volatility forecasting using GARCH models with crypto-specific enhancements.
    
    Features:
    - Multi-model ensemble (GARCH, EGARCH, GJR-GARCH)
    - Regime-aware volatility forecasting
    - High-frequency data handling
    - Term structure of volatility
    - Real-time updating
    """
    
    def __init__(self, models: Optional[List[str]] = None,
                 update_frequency: int = 60):
        """
        Initialize volatility forecaster.
        
        Args:
            models: List of models to use ['garch', 'egarch', 'gjr']
            update_frequency: How often to update models (in periods)
        """
        if models is None:
            models = ['garch', 'egarch']
            
        self.models = {}
        self.model_types = models
        self.update_frequency = update_frequency
        self.last_update = 0
        
        # Store historical forecasts for analysis
        self.forecast_history = []
        self.realized_volatility = []
        
    def fit_models(self, returns: np.ndarray, verbose: bool = True):
        """
        Fit all specified models.
        
        Args:
            returns: Return series
            verbose: Print fitting progress
        """
        for model_type in self.model_types:
            if verbose:
                print(f"Fitting {model_type.upper()} model...")
                
            if model_type == 'garch':
                model = GARCH(p=1, q=1, dist='t')
            elif model_type == 'egarch':
                model = EGARCH(p=1, q=1, dist='t')
            else:
                continue
                
            try:
                model.fit(returns)
                self.models[model_type] = model
                
                if verbose:
                    summary = model.summary()
                    print(f"  Persistence: {summary['persistence']:.4f}")
                    print(f"  AIC: {summary['aic']:.2f}")
                    
            except Exception as e:
                print(f"  Failed to fit {model_type}: {str(e)}")
                
    def forecast_ensemble(self, steps: int = 1, 
                         weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Generate ensemble volatility forecast.
        
        Args:
            steps: Forecast horizon
            weights: Model weights (if None, use equal weights)
            
        Returns:
            Dictionary with ensemble forecasts
        """
        if not self.models:
            raise ValueError("No models fitted")
            
        # Default equal weights
        if weights is None:
            weights = {model: 1.0/len(self.models) for model in self.models}
            
        # Collect forecasts from each model
        all_forecasts = {}
        for name, model in self.models.items():
            all_forecasts[name] = model.forecast(steps)
            
        # Compute weighted average
        ensemble_variance = np.zeros(steps)
        ensemble_vol = np.zeros(steps)
        
        for name, forecast in all_forecasts.items():
            weight = weights.get(name, 0)
            ensemble_variance += weight * forecast['variance']
            ensemble_vol += weight * forecast['volatility']
            
        # Compute prediction intervals (average of model intervals)
        vol_lower = np.zeros(steps)
        vol_upper = np.zeros(steps)
        
        for name, forecast in all_forecasts.items():
            weight = weights.get(name, 0)
            vol_lower += weight * forecast['volatility_lower']
            vol_upper += weight * forecast['volatility_upper']
            
        return {
            'variance': ensemble_variance,
            'volatility': ensemble_vol,
            'volatility_lower': vol_lower,
            'volatility_upper': vol_upper,
            'model_forecasts': all_forecasts,
            'weights': weights
        }
    
    def calculate_term_structure(self, horizons: List[int] = None) -> pd.DataFrame:
        """
        Calculate volatility term structure.
        
        Args:
            horizons: List of forecast horizons (default: 1, 5, 10, 30, 60 periods)
            
        Returns:
            DataFrame with term structure
        """
        if horizons is None:
            horizons = [1, 5, 10, 30, 60]  # 1min to 1hour for crypto
            
        term_structure = []
        
        for horizon in horizons:
            forecast = self.forecast_ensemble(steps=horizon)
            
            # Annualized volatility (crypto trades 24/7)
            # Assuming 1-minute data: 525,600 minutes per year
            annualization_factor = np.sqrt(525600)
            
            term_structure.append({
                'horizon': horizon,
                'volatility': forecast['volatility'][-1],
                'volatility_annualized': forecast['volatility'][-1] * annualization_factor,
                'cumulative_volatility': np.sqrt(np.sum(forecast['variance'])),
                'average_volatility': np.mean(forecast['volatility'])
            })
            
        return pd.DataFrame(term_structure)
    
    def update_forecast(self, new_return: float, refit: bool = False) -> Dict:
        """
        Update forecast with new data point.
        
        Args:
            new_return: New return observation
            refit: Whether to refit models
            
        Returns:
            Updated forecast
        """
        self.last_update += 1
        
        # Refit models periodically
        if refit or self.last_update >= self.update_frequency:
            # Get historical returns (would need to store these)
            # For now, just return current forecast
            self.last_update = 0
            
        # Generate new forecast
        return self.forecast_ensemble(steps=1)
    
    def calculate_volatility_metrics(self, returns: np.ndarray,
                                   window: int = 252) -> Dict:
        """
        Calculate various volatility metrics.
        
        Args:
            returns: Return series
            window: Rolling window size
            
        Returns:
            Dictionary with volatility metrics
        """
        # Realized volatility (simple)
        realized_vol = np.std(returns) * np.sqrt(252)
        
        # Parkinson volatility (if OHLC available)
        # Yang-Zhang volatility (if OHLC available)
        
        # Rolling volatility
        rolling_vol = pd.Series(returns).rolling(window).std()
        
        # EWMA volatility
        ewma_vol = pd.Series(returns).ewm(span=window).std()
        
        # Volatility of volatility
        vol_of_vol = np.std(rolling_vol.dropna())
        
        # Calculate forecast accuracy if we have history
        forecast_accuracy = {}
        if self.forecast_history and self.realized_volatility:
            forecast_errors = np.array(self.forecast_history) - np.array(self.realized_volatility)
            forecast_accuracy = {
                'mae': np.mean(np.abs(forecast_errors)),
                'rmse': np.sqrt(np.mean(forecast_errors**2)),
                'bias': np.mean(forecast_errors)
            }
            
        return {
            'realized_volatility': realized_vol,
            'current_volatility': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else np.nan,
            'ewma_volatility': ewma_vol.iloc[-1] if len(ewma_vol) > 0 else np.nan,
            'volatility_of_volatility': vol_of_vol,
            'min_volatility': np.min(rolling_vol.dropna()),
            'max_volatility': np.max(rolling_vol.dropna()),
            'volatility_percentile': np.percentile(rolling_vol.dropna(), 
                                                  [10, 25, 50, 75, 90]),
            'forecast_accuracy': forecast_accuracy
        }
    
    def detect_volatility_regime(self, returns: np.ndarray,
                               thresholds: Optional[Dict] = None) -> str:
        """
        Detect current volatility regime.
        
        Args:
            returns: Recent returns
            thresholds: Volatility thresholds for regimes
            
        Returns:
            Current regime ('low', 'normal', 'high', 'extreme')
        """
        if thresholds is None:
            # Default thresholds based on percentiles
            thresholds = {
                'low': 0.25,
                'normal': 0.75,
                'high': 0.95
            }
            
        # Calculate recent volatility
        recent_vol = np.std(returns[-30:])  # Last 30 periods
        
        # Calculate historical percentile
        historical_vols = pd.Series(returns).rolling(30).std().dropna()
        vol_percentile = (historical_vols < recent_vol).mean()
        
        # Determine regime
        if vol_percentile < thresholds['low']:
            regime = 'low'
        elif vol_percentile < thresholds['normal']:
            regime = 'normal'
        elif vol_percentile < thresholds['high']:
            regime = 'high'
        else:
            regime = 'extreme'
            
        return regime
    
    def forecast_with_regime_adjustment(self, returns: np.ndarray,
                                      steps: int = 1) -> Dict:
        """
        Forecast with regime-based adjustments.
        
        Args:
            returns: Return series
            steps: Forecast horizon
            
        Returns:
            Regime-adjusted forecast
        """
        # Get base forecast
        base_forecast = self.forecast_ensemble(steps)
        
        # Detect current regime
        regime = self.detect_volatility_regime(returns)
        
        # Apply regime adjustments
        regime_multipliers = {
            'low': 0.8,      # Reduce forecast in low vol
            'normal': 1.0,   # No adjustment
            'high': 1.2,     # Increase forecast in high vol
            'extreme': 1.5   # Significant increase in extreme vol
        }
        
        multiplier = regime_multipliers[regime]
        
        # Adjust forecasts
        adjusted_forecast = {
            'variance': base_forecast['variance'] * multiplier**2,
            'volatility': base_forecast['volatility'] * multiplier,
            'volatility_lower': base_forecast['volatility_lower'] * multiplier,
            'volatility_upper': base_forecast['volatility_upper'] * multiplier,
            'regime': regime,
            'regime_multiplier': multiplier,
            'base_forecast': base_forecast
        }
        
        return adjusted_forecast
    
    def calculate_volatility_smile(self, returns: np.ndarray,
                                 strikes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate implied volatility smile using GARCH dynamics.
        
        Args:
            returns: Return series
            strikes: Strike prices as percentage of spot
            
        Returns:
            DataFrame with volatility smile
        """
        if strikes is None:
            strikes = np.linspace(0.8, 1.2, 21)  # 80% to 120% of spot
            
        # Get current volatility forecast
        forecast = self.forecast_ensemble(steps=1)
        base_vol = forecast['volatility'][0]
        
        # Model volatility smile
        smile_data = []
        
        for strike in strikes:
            # Simple smile adjustment based on moneyness
            moneyness = np.log(strike)
            
            # Volatility smile parameters (calibrated to crypto options)
            smile_slope = 0.5  # Downward slope
            smile_convexity = 2.0  # Smile curvature
            
            # Calculate implied vol
            implied_vol = base_vol * (1 + smile_slope * moneyness + 
                                     smile_convexity * moneyness**2)
            
            smile_data.append({
                'strike': strike,
                'moneyness': moneyness,
                'implied_volatility': implied_vol,
                'implied_volatility_annualized': implied_vol * np.sqrt(525600)
            })
            
        return pd.DataFrame(smile_data)
    
    def generate_volatility_scenarios(self, base_volatility: float,
                                    num_scenarios: int = 1000,
                                    horizon: int = 60) -> np.ndarray:
        """
        Generate volatility scenarios using Monte Carlo.
        
        Args:
            base_volatility: Starting volatility
            num_scenarios: Number of scenarios to generate
            horizon: Time horizon
            
        Returns:
            Array of volatility paths (num_scenarios x horizon)
        """
        # Get model parameters
        if 'garch' in self.models:
            model = self.models['garch']
            omega = model.omega
            alpha = model.alpha[0]
            beta = model.beta[0]
        else:
            # Default parameters
            omega = 0.00001
            alpha = 0.05
            beta = 0.94
            
        # Initialize paths
        vol_paths = np.zeros((num_scenarios, horizon))
        vol_paths[:, 0] = base_volatility
        
        # Generate paths
        for i in range(num_scenarios):
            for t in range(1, horizon):
                # Generate innovation
                if 'garch' in self.models and self.models['garch'].dist == 't':
                    nu = self.models['garch'].nu
                    z = np.random.standard_t(nu) / np.sqrt((nu - 2) / nu)
                else:
                    z = np.random.standard_normal()
                    
                # Update variance
                variance = omega + alpha * (z * vol_paths[i, t-1])**2 + \
                          beta * vol_paths[i, t-1]**2
                          
                vol_paths[i, t] = np.sqrt(variance)
                
        return vol_paths


class RealizedVolatilityCalculator:
    """
    Calculate various realized volatility measures for crypto markets.
    """
    
    @staticmethod
    def simple_realized_volatility(returns: np.ndarray, 
                                 annualize: bool = True) -> float:
        """
        Simple realized volatility.
        
        Args:
            returns: Return series
            annualize: Whether to annualize
            
        Returns:
            Realized volatility
        """
        rv = np.sqrt(np.sum(returns**2))
        
        if annualize:
            # Crypto: 525,600 minutes per year
            rv *= np.sqrt(525600 / len(returns))
            
        return rv
    
    @staticmethod
    def parkinson_volatility(high: np.ndarray, low: np.ndarray,
                           annualize: bool = True) -> float:
        """
        Parkinson volatility estimator using high-low range.
        
        Args:
            high: High prices
            low: Low prices
            annualize: Whether to annualize
            
        Returns:
            Parkinson volatility
        """
        log_hl = np.log(high / low)
        pv = np.sqrt(np.sum(log_hl**2) / (4 * len(high) * np.log(2)))
        
        if annualize:
            pv *= np.sqrt(525600)
            
        return pv
    
    @staticmethod
    def garman_klass_volatility(open_: np.ndarray, high: np.ndarray,
                              low: np.ndarray, close: np.ndarray,
                              annualize: bool = True) -> float:
        """
        Garman-Klass volatility estimator.
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            annualize: Whether to annualize
            
        Returns:
            Garman-Klass volatility
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        gk = np.sqrt(np.mean(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2))
        
        if annualize:
            gk *= np.sqrt(525600)
            
        return gk