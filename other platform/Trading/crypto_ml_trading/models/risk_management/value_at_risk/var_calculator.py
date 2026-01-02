import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import norm, t as student_t
import warnings


class VaRCalculator:
    """
    Value at Risk (VaR) calculator for cryptocurrency portfolios.
    
    Implements multiple VaR methodologies:
    - Historical Simulation
    - Parametric (Variance-Covariance)
    - Monte Carlo Simulation
    - Conditional VaR (CVaR/Expected Shortfall)
    
    Optimized for crypto markets with:
    - Fat-tailed distributions
    - High volatility
    - 24/7 trading
    - Extreme events handling
    """
    
    def __init__(self, confidence_level: float = 0.95,
                 horizon: int = 1,
                 method: str = 'historical'):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            horizon: Time horizon in periods
            method: Default method ('historical', 'parametric', 'montecarlo')
        """
        self.confidence_level = confidence_level
        self.horizon = horizon
        self.method = method
        
        # Store results for analysis
        self.var_history = []
        self.backtest_results = []
        
    def calculate_var(self, returns: Union[np.ndarray, pd.Series],
                     method: Optional[str] = None,
                     portfolio_value: float = 1.0) -> Dict:
        """
        Calculate VaR using specified method.
        
        Args:
            returns: Historical returns
            method: Method to use (overrides default)
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with VaR metrics
        """
        if method is None:
            method = self.method
            
        # Convert to numpy if needed
        if isinstance(returns, pd.Series):
            returns = returns.values
            
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 20:
            warnings.warn("Insufficient data for reliable VaR calculation")
            
        # Calculate VaR based on method
        if method == 'historical':
            var_pct, cvar_pct = self._historical_var(returns)
        elif method == 'parametric':
            var_pct, cvar_pct = self._parametric_var(returns)
        elif method == 'montecarlo':
            var_pct, cvar_pct = self._monte_carlo_var(returns)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Scale for horizon (square root rule for multi-period)
        if self.horizon > 1:
            var_pct *= np.sqrt(self.horizon)
            cvar_pct *= np.sqrt(self.horizon)
            
        # Convert to portfolio value
        var_amount = var_pct * portfolio_value
        cvar_amount = cvar_pct * portfolio_value
        
        return {
            'var_percent': var_pct,
            'var_amount': var_amount,
            'cvar_percent': cvar_pct,
            'cvar_amount': cvar_amount,
            'confidence_level': self.confidence_level,
            'horizon': self.horizon,
            'method': method,
            'observations': len(returns)
        }
    
    def _historical_var(self, returns: np.ndarray) -> Tuple[float, float]:
        """
        Calculate VaR using historical simulation.
        
        Args:
            returns: Historical returns
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find VaR at confidence level
        index = int((1 - self.confidence_level) * len(returns))
        var = -sorted_returns[index]
        
        # Calculate CVaR (average of returns worse than VaR)
        tail_returns = sorted_returns[:index]
        cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var
        
        return var, cvar
    
    def _parametric_var(self, returns: np.ndarray) -> Tuple[float, float]:
        """
        Calculate parametric VaR assuming normal or t-distribution.
        
        Args:
            returns: Historical returns
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Test for normality (Jarque-Bera test)
        jb_stat = stats.jarque_bera(returns)[0]
        use_t_dist = jb_stat > 6  # Rough threshold
        
        if use_t_dist:
            # Fit t-distribution
            params = student_t.fit(returns)
            df = params[0]  # Degrees of freedom
            loc = params[1]  # Location
            scale = params[2]  # Scale
            
            # Calculate VaR
            var = -student_t.ppf(1 - self.confidence_level, df, loc, scale)
            
            # Calculate CVaR for t-distribution
            alpha = 1 - self.confidence_level
            t_score = student_t.ppf(alpha, df)
            pdf_value = student_t.pdf(t_score, df)
            cvar = -loc + scale * (df + t_score**2) / (df - 1) * pdf_value / alpha
            
        else:
            # Use normal distribution
            z_score = norm.ppf(1 - self.confidence_level)
            var = -(mean + z_score * std)
            
            # CVaR for normal distribution
            cvar = mean + std * norm.pdf(z_score) / (1 - self.confidence_level)
            
        return var, cvar
    
    def _monte_carlo_var(self, returns: np.ndarray,
                        n_simulations: int = 10000) -> Tuple[float, float]:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        # Fit distribution parameters
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Check for fat tails
        kurtosis = stats.kurtosis(returns)
        
        if kurtosis > 3:  # Fat-tailed
            # Use t-distribution
            params = student_t.fit(returns)
            df = params[0]
            
            # Generate scenarios
            scenarios = student_t.rvs(df, loc=mean, scale=std, size=n_simulations)
        else:
            # Use normal distribution
            scenarios = np.random.normal(mean, std, n_simulations)
            
        # Calculate VaR and CVaR from scenarios
        return self._historical_var(scenarios)
    
    def calculate_portfolio_var(self, returns: pd.DataFrame,
                              weights: Union[np.ndarray, pd.Series],
                              method: Optional[str] = None) -> Dict:
        """
        Calculate VaR for a portfolio of assets.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            method: VaR method
            
        Returns:
            Dictionary with portfolio VaR metrics
        """
        # Calculate portfolio returns
        if isinstance(weights, pd.Series):
            weights = weights.values
            
        portfolio_returns = returns @ weights
        
        # Calculate single-asset VaRs for comparison
        individual_vars = {}
        for asset in returns.columns:
            asset_var = self.calculate_var(returns[asset], method)
            individual_vars[asset] = asset_var['var_percent']
            
        # Calculate portfolio VaR
        portfolio_var = self.calculate_var(portfolio_returns, method)
        
        # Calculate diversification benefit
        weighted_sum_var = sum(w * var for w, var in 
                              zip(weights, individual_vars.values()))
        diversification_benefit = 1 - (portfolio_var['var_percent'] / weighted_sum_var)
        
        portfolio_var['individual_vars'] = individual_vars
        portfolio_var['diversification_benefit'] = diversification_benefit
        portfolio_var['weights'] = dict(zip(returns.columns, weights))
        
        return portfolio_var
    
    def stressed_var(self, returns: np.ndarray,
                    stress_period: Optional[Tuple[int, int]] = None,
                    stress_multiplier: float = 1.5) -> Dict:
        """
        Calculate stressed VaR using historical stress periods.
        
        Args:
            returns: Full historical returns
            stress_period: Indices of stress period (start, end)
            stress_multiplier: Multiplier for stress scenario
            
        Returns:
            Dictionary with stressed VaR metrics
        """
        if stress_period is None:
            # Find most volatile period
            window = min(60, len(returns) // 4)
            rolling_std = pd.Series(returns).rolling(window).std()
            stress_start = rolling_std.idxmax() - window // 2
            stress_end = stress_start + window
            stress_period = (max(0, stress_start), min(len(returns), stress_end))
            
        # Extract stress period returns
        stress_returns = returns[stress_period[0]:stress_period[1]]
        
        # Scale returns for stress
        stressed_returns = stress_returns * stress_multiplier
        
        # Calculate stressed VaR
        stressed_var = self.calculate_var(stressed_returns)
        stressed_var['stress_period'] = stress_period
        stressed_var['stress_multiplier'] = stress_multiplier
        
        return stressed_var
    
    def incremental_var(self, returns: pd.DataFrame,
                       weights: np.ndarray,
                       asset_index: int,
                       delta_weight: float = 0.01) -> float:
        """
        Calculate incremental VaR for a position change.
        
        Args:
            returns: Asset returns DataFrame
            weights: Current weights
            asset_index: Index of asset to change
            delta_weight: Change in weight
            
        Returns:
            Incremental VaR
        """
        # Current portfolio VaR
        current_var = self.calculate_portfolio_var(returns, weights)['var_percent']
        
        # New weights
        new_weights = weights.copy()
        new_weights[asset_index] += delta_weight
        new_weights /= new_weights.sum()  # Renormalize
        
        # New portfolio VaR
        new_var = self.calculate_portfolio_var(returns, new_weights)['var_percent']
        
        # Incremental VaR
        incremental_var = (new_var - current_var) / delta_weight
        
        return incremental_var
    
    def marginal_var(self, returns: pd.DataFrame,
                    weights: np.ndarray) -> pd.Series:
        """
        Calculate marginal VaR for each asset.
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            
        Returns:
            Series of marginal VaRs
        """
        marginal_vars = []
        
        for i in range(len(weights)):
            marginal_var = self.incremental_var(returns, weights, i)
            marginal_vars.append(marginal_var)
            
        return pd.Series(marginal_vars, index=returns.columns)
    
    def component_var(self, returns: pd.DataFrame,
                     weights: np.ndarray) -> pd.DataFrame:
        """
        Decompose portfolio VaR into component contributions.
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            
        Returns:
            DataFrame with component VaR analysis
        """
        # Calculate marginal VaRs
        marginal_vars = self.marginal_var(returns, weights)
        
        # Component VaR = weight * marginal VaR
        component_vars = weights * marginal_vars
        
        # Percentage contribution
        total_var = component_vars.sum()
        pct_contribution = component_vars / total_var
        
        return pd.DataFrame({
            'weight': weights,
            'marginal_var': marginal_vars,
            'component_var': component_vars,
            'pct_contribution': pct_contribution
        }, index=returns.columns)
    
    def backtest_var(self, returns: np.ndarray,
                    rolling_window: int = 252) -> pd.DataFrame:
        """
        Backtest VaR model on historical data.
        
        Args:
            returns: Historical returns
            rolling_window: Window for rolling VaR calculation
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        
        for i in range(rolling_window, len(returns)):
            # Calculate VaR using data up to time i
            window_returns = returns[i-rolling_window:i]
            var_result = self.calculate_var(window_returns)
            
            # Compare with actual return
            actual_return = returns[i]
            breach = actual_return < -var_result['var_percent']
            
            results.append({
                'date_index': i,
                'var': var_result['var_percent'],
                'actual_return': actual_return,
                'breach': breach,
                'breach_size': -actual_return - var_result['var_percent'] if breach else 0
            })
            
        backtest_df = pd.DataFrame(results)
        
        # Calculate backtest statistics
        breach_rate = backtest_df['breach'].mean()
        expected_breach_rate = 1 - self.confidence_level
        
        # Kupiec test (simplified)
        n_breaches = backtest_df['breach'].sum()
        n_obs = len(backtest_df)
        kupiec_stat = -2 * np.log((expected_breach_rate**n_breaches * 
                                   (1-expected_breach_rate)**(n_obs-n_breaches)) / 
                                  ((breach_rate**n_breaches * 
                                   (1-breach_rate)**(n_obs-n_breaches))))
        
        backtest_summary = {
            'breach_rate': breach_rate,
            'expected_breach_rate': expected_breach_rate,
            'n_breaches': n_breaches,
            'n_observations': n_obs,
            'kupiec_statistic': kupiec_stat,
            'model_adequate': abs(breach_rate - expected_breach_rate) < 0.05,
            'average_breach_size': backtest_df[backtest_df['breach']]['breach_size'].mean()
        }
        
        self.backtest_results = backtest_summary
        
        return backtest_df
    
    def calculate_expected_shortfall(self, returns: np.ndarray,
                                   threshold: Optional[float] = None) -> float:
        """
        Calculate Expected Shortfall (CVaR) with flexible threshold.
        
        Args:
            returns: Historical returns
            threshold: VaR threshold (calculated if not provided)
            
        Returns:
            Expected shortfall
        """
        if threshold is None:
            threshold, _ = self._historical_var(returns)
            threshold = -threshold  # Convert to return value
            
        # Get returns worse than threshold
        tail_returns = returns[returns <= threshold]
        
        if len(tail_returns) == 0:
            return abs(threshold)
            
        return -np.mean(tail_returns)
    
    def liquidity_adjusted_var(self, returns: np.ndarray,
                             bid_ask_spread: float,
                             daily_volume: float,
                             position_size: float) -> Dict:
        """
        Calculate liquidity-adjusted VaR.
        
        Args:
            returns: Historical returns
            bid_ask_spread: Average bid-ask spread
            daily_volume: Average daily volume
            position_size: Position size
            
        Returns:
            Dictionary with liquidity-adjusted VaR
        """
        # Base VaR
        base_var = self.calculate_var(returns)
        
        # Liquidity cost
        # Days to liquidate = position_size / (daily_volume * participation_rate)
        participation_rate = 0.1  # Don't want to be more than 10% of volume
        days_to_liquidate = position_size / (daily_volume * participation_rate)
        
        # Spread cost
        spread_cost = bid_ask_spread / 2  # Half-spread cost
        
        # Market impact (simplified square-root model)
        market_impact = 0.1 * np.sqrt(position_size / daily_volume)
        
        # Total liquidity cost
        liquidity_cost = spread_cost + market_impact
        
        # Adjusted VaR
        adjusted_var = base_var['var_percent'] + liquidity_cost
        
        return {
            'base_var': base_var['var_percent'],
            'liquidity_cost': liquidity_cost,
            'adjusted_var': adjusted_var,
            'days_to_liquidate': days_to_liquidate,
            'spread_cost': spread_cost,
            'market_impact': market_impact
        }