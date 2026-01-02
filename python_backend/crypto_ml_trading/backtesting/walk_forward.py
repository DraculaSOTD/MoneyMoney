import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtesting.engine import BacktestEngine, BacktestConfig


class WalkForwardAnalysis:
    """
    Walk-forward analysis for robust strategy testing.
    
    Features:
    - Rolling window optimization
    - Out-of-sample testing
    - Parameter stability analysis
    - Performance consistency tracking
    """
    
    def __init__(self, 
                 optimization_window: int = 30,  # days
                 test_window: int = 7,           # days
                 step_size: int = 7,             # days
                 min_train_size: int = 14):      # days
        """
        Initialize walk-forward analysis.
        
        Args:
            optimization_window: Training window size in days
            test_window: Testing window size in days
            step_size: Step size for rolling window in days
            min_train_size: Minimum training data required
        """
        self.optimization_window = optimization_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        
        self.results = []
        self.parameter_history = []
        
    def run(self, data: pd.DataFrame,
            model_builder: Callable,
            parameter_optimizer: Callable,
            signal_generator: Callable,
            backtest_config: Optional[BacktestConfig] = None) -> Dict:
        """
        Run walk-forward analysis.
        
        Args:
            data: Historical data
            model_builder: Function to build/train model
            parameter_optimizer: Function to optimize parameters
            signal_generator: Function to generate signals
            backtest_config: Backtesting configuration
            
        Returns:
            Dictionary with analysis results
        """
        if backtest_config is None:
            backtest_config = BacktestConfig()
            
        # Ensure data is sorted
        data = data.sort_values('timestamp')
        
        # Convert timestamps to dates for windowing
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        
        # Get date range
        start_date = data['date'].min()
        end_date = data['date'].max()
        
        # Initialize results
        all_trades = []
        all_equity_curves = []
        window_results = []
        
        # Walk-forward loop
        current_date = start_date + timedelta(days=self.optimization_window)
        
        while current_date + timedelta(days=self.test_window) <= end_date:
            # Define windows
            train_start = current_date - timedelta(days=self.optimization_window)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=self.test_window)
            
            # Get data for each window
            train_data = data[(data['date'] >= train_start) & (data['date'] < train_end)]
            test_data = data[(data['date'] >= test_start) & (data['date'] < test_end)]
            
            if len(train_data) < self.min_train_size * 1440:  # Assuming minute data
                current_date += timedelta(days=self.step_size)
                continue
                
            # Optimize parameters on training data
            optimal_params = parameter_optimizer(train_data)
            self.parameter_history.append({
                'window_start': train_start,
                'window_end': train_end,
                'parameters': optimal_params
            })
            
            # Build model with optimal parameters
            model = model_builder(train_data, optimal_params)
            
            # Create signal generator with trained model
            def model_signal_generator(data, positions):
                return signal_generator(data, positions, model)
            
            # Run backtest on test data
            engine = BacktestEngine(backtest_config)
            window_result = engine.run(
                test_data,
                model_signal_generator
            )
            
            # Store results
            window_results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'parameters': optimal_params,
                'performance': window_result
            })
            
            all_trades.extend(engine.trades)
            all_equity_curves.append(window_result['equity_curve'])
            
            # Move to next window
            current_date += timedelta(days=self.step_size)
            
        # Analyze results
        analysis = self._analyze_results(window_results, all_trades)
        
        return analysis
    
    def _analyze_results(self, window_results: List[Dict], 
                        all_trades: List) -> Dict:
        """Analyze walk-forward results."""
        if not window_results:
            return {'error': 'No valid windows found'}
            
        # Extract performance metrics
        returns = [w['performance'].get('total_return', 0) for w in window_results]
        sharpe_ratios = [w['performance'].get('sharpe_ratio', 0) for w in window_results]
        max_drawdowns = [w['performance'].get('max_drawdown', 0) for w in window_results]
        win_rates = [w['performance'].get('win_rate', 0) for w in window_results]
        
        # Calculate consistency metrics
        consistency_metrics = {
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'return_consistency': np.mean(returns) / (np.std(returns) + 1e-10),
            'positive_windows': sum(1 for r in returns if r > 0),
            'negative_windows': sum(1 for r in returns if r <= 0),
            'win_rate_stability': 1 - np.std(win_rates) / (np.mean(win_rates) + 1e-10)
        }
        
        # Parameter stability analysis
        param_stability = self._analyze_parameter_stability()
        
        # Overall performance
        overall_performance = self._calculate_overall_performance(window_results)
        
        # Risk analysis
        risk_metrics = {
            'avg_sharpe': np.mean(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.max(max_drawdowns),
            'drawdown_volatility': np.std(max_drawdowns)
        }
        
        return {
            'num_windows': len(window_results),
            'consistency_metrics': consistency_metrics,
            'parameter_stability': param_stability,
            'overall_performance': overall_performance,
            'risk_metrics': risk_metrics,
            'window_details': window_results,
            'parameter_history': self.parameter_history
        }
    
    def _analyze_parameter_stability(self) -> Dict:
        """Analyze how stable parameters are across windows."""
        if not self.parameter_history:
            return {}
            
        # Extract parameter values
        param_names = list(self.parameter_history[0]['parameters'].keys())
        param_series = {name: [] for name in param_names}
        
        for window in self.parameter_history:
            for name, value in window['parameters'].items():
                if isinstance(value, (int, float)):
                    param_series[name].append(value)
                    
        # Calculate stability metrics
        stability_metrics = {}
        
        for name, values in param_series.items():
            if values:
                values_array = np.array(values)
                stability_metrics[name] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'cv': np.std(values_array) / (np.mean(values_array) + 1e-10),
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                }
                
        return stability_metrics
    
    def _calculate_overall_performance(self, window_results: List[Dict]) -> Dict:
        """Calculate overall performance across all windows."""
        # Combine equity curves
        combined_returns = []
        
        for window in window_results:
            if 'equity_curve' in window['performance']:
                equity = window['performance']['equity_curve']
                if len(equity) > 1:
                    returns = np.diff(equity) / equity[:-1]
                    combined_returns.extend(returns)
                    
        if not combined_returns:
            return {}
            
        combined_returns = np.array(combined_returns)
        
        # Calculate metrics
        total_return = np.prod(1 + combined_returns) - 1
        sharpe_ratio = np.mean(combined_returns) / (np.std(combined_returns) + 1e-10) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (365 / (len(window_results) * self.test_window)) - 1,
            'overall_sharpe': sharpe_ratio,
            'return_volatility': np.std(combined_returns) * np.sqrt(252)
        }
    
    def optimize_window_sizes(self, data: pd.DataFrame,
                            model_builder: Callable,
                            parameter_optimizer: Callable,
                            signal_generator: Callable,
                            window_combinations: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """
        Optimize walk-forward window sizes.
        
        Args:
            data: Historical data
            model_builder: Model building function
            parameter_optimizer: Parameter optimization function
            signal_generator: Signal generation function
            window_combinations: List of (optimization_window, test_window) to try
            
        Returns:
            Dictionary with optimal window sizes
        """
        if window_combinations is None:
            # Default combinations to try
            window_combinations = [
                (20, 5), (30, 7), (30, 10),
                (45, 10), (60, 14), (90, 30)
            ]
            
        results = []
        
        for opt_window, test_window in window_combinations:
            # Create analyzer with these windows
            analyzer = WalkForwardAnalysis(
                optimization_window=opt_window,
                test_window=test_window,
                step_size=test_window
            )
            
            # Run analysis
            analysis = analyzer.run(
                data,
                model_builder,
                parameter_optimizer,
                signal_generator
            )
            
            results.append({
                'optimization_window': opt_window,
                'test_window': test_window,
                'ratio': opt_window / test_window,
                'performance': analysis.get('overall_performance', {}),
                'consistency': analysis.get('consistency_metrics', {})
            })
            
        # Find best combination
        best_result = max(results, 
                         key=lambda x: x['performance'].get('overall_sharpe', 0))
        
        return {
            'best_combination': (best_result['optimization_window'], 
                               best_result['test_window']),
            'all_results': results,
            'recommendation': self._make_window_recommendation(results)
        }
    
    def _make_window_recommendation(self, results: List[Dict]) -> str:
        """Make recommendation based on window optimization results."""
        if not results:
            return "Insufficient data for recommendation"
            
        # Analyze patterns
        ratios = [r['ratio'] for r in results]
        sharpes = [r['performance'].get('overall_sharpe', 0) for r in results]
        
        # Find optimal ratio
        best_ratio_idx = np.argmax(sharpes)
        best_ratio = ratios[best_ratio_idx]
        
        if best_ratio < 3:
            return "Short optimization window recommended - market conditions change rapidly"
        elif best_ratio < 6:
            return "Balanced window sizes recommended - good stability vs adaptability"
        else:
            return "Long optimization window recommended - stable market patterns"
            
    def plot_parameter_evolution(self):
        """
        Note: Plotting would require matplotlib.
        Placeholder for parameter evolution visualization.
        """
        print("Parameter Evolution Visualization:")
        print("1. Parameter values over time")
        print("2. Parameter stability heatmap")
        print("3. Performance vs parameter values")
        print("\nImplement with matplotlib when needed.")