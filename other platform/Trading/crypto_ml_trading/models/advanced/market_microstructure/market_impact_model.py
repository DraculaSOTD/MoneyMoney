"""
Market Impact Modeling for Cryptocurrency Trading.

Implements advanced market impact models for order execution optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class MarketImpactParameters:
    """Parameters for market impact models."""
    model_type: str
    parameters: Dict[str, float]
    confidence_interval: Tuple[float, float]
    r_squared: float
    last_updated: datetime


@dataclass
class ImpactPrediction:
    """Market impact prediction."""
    timestamp: datetime
    order_size: float
    predicted_impact: float
    confidence: float
    model_used: str
    components: Dict[str, float]
    risk_factors: Dict[str, float]


@dataclass
class ExecutionProfile:
    """Execution profile with impact analysis."""
    total_size: float
    execution_plan: List[Dict[str, Any]]
    total_impact: float
    execution_time: float
    risk_score: float
    cost_breakdown: Dict[str, float]


class MarketImpactModel:
    """
    Advanced market impact modeling for cryptocurrency trading.
    
    Features:
    - Multiple impact model implementations
    - Regime-dependent impact estimation
    - Real-time parameter calibration
    - Execution cost optimization
    - Risk-adjusted impact prediction
    - Cross-venue impact analysis
    """
    
    def __init__(self,
                 models: List[str] = None,
                 calibration_window: int = 1000,
                 min_trade_size: float = 0.001,
                 impact_horizon: int = 300):
        """
        Initialize market impact model.
        
        Args:
            models: List of models to use ('almgren_chriss', 'square_root', 'linear', 'barra')
            calibration_window: Number of observations for calibration
            min_trade_size: Minimum trade size to consider
            impact_horizon: Impact measurement horizon in seconds
        """
        self.models = models or ['square_root', 'linear', 'almgren_chriss']
        self.calibration_window = calibration_window
        self.min_trade_size = min_trade_size
        self.impact_horizon = impact_horizon
        
        # Model parameters
        self.model_params: Dict[str, MarketImpactParameters] = {}
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Market data
        self.trade_data: deque = deque(maxlen=2000)
        self.impact_observations: deque = deque(maxlen=1000)
        self.market_state_data: deque = deque(maxlen=1000)
        
        # Model state
        self.is_calibrated = False
        self.last_calibration = None
        self.current_regime = 'normal'
        
        # Initialize default parameters
        self._initialize_default_parameters()
    
    def _initialize_default_parameters(self) -> None:
        """Initialize default model parameters."""
        default_params = {
            'square_root': {
                'model_type': 'square_root',
                'parameters': {'gamma': 0.5, 'eta': 0.01, 'alpha': 0.1},
                'confidence_interval': (0.0, 1.0),
                'r_squared': 0.0,
                'last_updated': datetime.now()
            },
            'linear': {
                'model_type': 'linear',
                'parameters': {'beta': 0.1, 'fixed_cost': 0.001},
                'confidence_interval': (0.0, 1.0),
                'r_squared': 0.0,
                'last_updated': datetime.now()
            },
            'almgren_chriss': {
                'model_type': 'almgren_chriss',
                'parameters': {'sigma': 0.02, 'lambda': 0.001, 'eta': 0.01},
                'confidence_interval': (0.0, 1.0),
                'r_squared': 0.0,
                'last_updated': datetime.now()
            }
        }
        
        for model_name in self.models:
            if model_name in default_params:
                self.model_params[model_name] = MarketImpactParameters(**default_params[model_name])
    
    def add_trade_data(self,
                      timestamp: datetime,
                      trades: List[Dict[str, Any]],
                      market_state: Dict[str, Any]) -> None:
        """
        Add trade data for impact modeling.
        
        Args:
            timestamp: Data timestamp
            trades: List of trade records
            market_state: Current market state
        """
        # Store trade data
        for trade in trades:
            if trade.get('size', 0) >= self.min_trade_size:
                trade_record = {
                    'timestamp': timestamp,
                    'price': trade.get('price', 0),
                    'size': trade.get('size', 0),
                    'side': trade.get('side', 'unknown'),
                    'venue': trade.get('venue', 'default')
                }
                self.trade_data.append(trade_record)
        
        # Store market state
        market_record = {
            'timestamp': timestamp,
            'mid_price': market_state.get('mid_price', 0),
            'spread': market_state.get('spread', 0),
            'volume': market_state.get('volume', 0),
            'volatility': market_state.get('volatility', 0),
            'depth': market_state.get('depth', 0)
        }
        self.market_state_data.append(market_record)
        
        # Trigger calibration if enough data
        if len(self.trade_data) >= 100 and not self.is_calibrated:
            self.calibrate_models()
    
    def add_impact_observation(self,
                             order_size: float,
                             actual_impact: float,
                             execution_time: float,
                             market_conditions: Dict[str, Any]) -> None:
        """
        Add observed market impact for model validation.
        
        Args:
            order_size: Size of executed order
            actual_impact: Observed market impact
            execution_time: Time taken for execution
            market_conditions: Market conditions during execution
        """
        observation = {
            'timestamp': datetime.now(),
            'order_size': order_size,
            'actual_impact': actual_impact,
            'execution_time': execution_time,
            'market_conditions': market_conditions
        }
        
        self.impact_observations.append(observation)
        
        # Update model performance
        self._update_model_performance(observation)
    
    def _update_model_performance(self, observation: Dict[str, Any]) -> None:
        """Update model performance metrics."""
        order_size = observation['order_size']
        actual_impact = observation['actual_impact']
        market_conditions = observation['market_conditions']
        
        # Get predictions from each model
        for model_name in self.models:
            if model_name in self.model_params:
                predicted_impact = self._predict_single_model(
                    model_name, order_size, market_conditions
                )
                
                # Calculate error
                error = abs(predicted_impact - actual_impact)
                relative_error = error / max(abs(actual_impact), 0.001)
                
                # Update performance metrics
                perf = self.model_performance[model_name]
                
                if 'errors' not in perf:
                    perf['errors'] = deque(maxlen=100)
                    perf['relative_errors'] = deque(maxlen=100)
                
                perf['errors'].append(error)
                perf['relative_errors'].append(relative_error)
                
                # Calculate statistics
                perf['mae'] = np.mean(list(perf['errors']))
                perf['mape'] = np.mean(list(perf['relative_errors']))
                perf['rmse'] = np.sqrt(np.mean([e**2 for e in perf['errors']]))
    
    def calibrate_models(self) -> None:
        """Calibrate impact models using historical data."""
        if len(self.trade_data) < 50:
            return
        
        print(f"Calibrating market impact models with {len(self.trade_data)} trades...")
        
        # Prepare calibration data
        calibration_data = self._prepare_calibration_data()
        
        if not calibration_data:
            return
        
        # Calibrate each model
        for model_name in self.models:
            try:
                self._calibrate_single_model(model_name, calibration_data)
            except Exception as e:
                print(f"Failed to calibrate {model_name}: {e}")
        
        self.is_calibrated = True
        self.last_calibration = datetime.now()
        print("Market impact model calibration completed")
    
    def _prepare_calibration_data(self) -> List[Dict[str, Any]]:
        """Prepare data for model calibration."""
        calibration_data = []
        
        # Group trades by time windows to estimate impact
        trades = list(self.trade_data)
        market_states = list(self.market_state_data)
        
        for i, trade in enumerate(trades):
            # Find market state closest to trade time
            trade_time = trade['timestamp']
            
            # Find pre-trade market state
            pre_states = [ms for ms in market_states if ms['timestamp'] <= trade_time]
            if not pre_states:
                continue
            
            pre_state = pre_states[-1]
            pre_price = pre_state['mid_price']
            
            # Find post-trade market state (within impact horizon)
            horizon_end = trade_time + timedelta(seconds=self.impact_horizon)
            post_states = [ms for ms in market_states 
                          if trade_time < ms['timestamp'] <= horizon_end]
            
            if not post_states:
                continue
            
            # Calculate impact as price change
            post_price = post_states[-1]['mid_price']
            
            if pre_price > 0:
                raw_impact = (post_price - pre_price) / pre_price
                
                # Adjust for trade side
                if trade['side'] == 'sell':
                    raw_impact = -raw_impact
                
                # Only consider positive impacts (negative would be beneficial)
                if raw_impact > 0:
                    calibration_point = {
                        'trade_size': trade['size'],
                        'impact': raw_impact,
                        'spread': pre_state['spread'],
                        'volatility': pre_state.get('volatility', 0.01),
                        'depth': pre_state.get('depth', 100),
                        'volume': pre_state.get('volume', 1000)
                    }
                    calibration_data.append(calibration_point)
        
        return calibration_data
    
    def _calibrate_single_model(self, model_name: str, data: List[Dict[str, Any]]) -> None:
        """Calibrate a single impact model."""
        if not data or model_name not in self.model_params:
            return
        
        # Extract features
        sizes = np.array([d['trade_size'] for d in data])
        impacts = np.array([d['impact'] for d in data])
        spreads = np.array([d['spread'] for d in data])
        volatilities = np.array([d['volatility'] for d in data])
        depths = np.array([d['depth'] for d in data])
        
        if len(sizes) < 10:
            return
        
        # Calibrate based on model type
        if model_name == 'square_root':
            self._calibrate_square_root_model(sizes, impacts, spreads, depths)
        elif model_name == 'linear':
            self._calibrate_linear_model(sizes, impacts)
        elif model_name == 'almgren_chriss':
            self._calibrate_almgren_chriss_model(sizes, impacts, volatilities)
    
    def _calibrate_square_root_model(self,
                                   sizes: np.ndarray,
                                   impacts: np.ndarray,
                                   spreads: np.ndarray,
                                   depths: np.ndarray) -> None:
        """Calibrate square root impact model."""
        # Model: Impact = gamma * (size / depth)^alpha + eta * spread
        
        # Transform variables
        normalized_sizes = sizes / np.maximum(depths, 1.0)
        
        try:
            # Simple regression for alpha and gamma
            log_sizes = np.log(np.maximum(normalized_sizes, 1e-6))
            log_impacts = np.log(np.maximum(impacts, 1e-6))
            
            # Fit log-log regression
            if len(log_sizes) > 2:
                coeffs = np.polyfit(log_sizes, log_impacts, 1)
                alpha = coeffs[0]
                log_gamma = coeffs[1]
                gamma = np.exp(log_gamma)
                
                # Estimate eta from residuals
                predicted_log_impacts = np.polyval(coeffs, log_sizes)
                residuals = np.exp(log_impacts) - np.exp(predicted_log_impacts)
                
                if len(spreads) > 0 and np.std(spreads) > 0:
                    eta = np.corrcoef(residuals, spreads)[0, 1] * np.std(residuals) / np.std(spreads)
                    eta = max(0, min(1.0, eta))
                else:
                    eta = 0.01
                
                # Calculate R-squared
                ss_res = np.sum((log_impacts - predicted_log_impacts) ** 2)
                ss_tot = np.sum((log_impacts - np.mean(log_impacts)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Update parameters
                self.model_params['square_root'].parameters = {
                    'gamma': max(0.001, min(1.0, gamma)),
                    'alpha': max(0.1, min(1.0, alpha)),
                    'eta': eta
                }
                self.model_params['square_root'].r_squared = max(0, r_squared)
                self.model_params['square_root'].last_updated = datetime.now()
                
        except Exception as e:
            print(f"Square root model calibration failed: {e}")
    
    def _calibrate_linear_model(self, sizes: np.ndarray, impacts: np.ndarray) -> None:
        """Calibrate linear impact model."""
        # Model: Impact = beta * size + fixed_cost
        
        try:
            if len(sizes) > 2:
                # Linear regression
                coeffs = np.polyfit(sizes, impacts, 1)
                beta = coeffs[0]
                fixed_cost = coeffs[1]
                
                # Calculate R-squared
                predicted_impacts = np.polyval(coeffs, sizes)
                ss_res = np.sum((impacts - predicted_impacts) ** 2)
                ss_tot = np.sum((impacts - np.mean(impacts)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Update parameters
                self.model_params['linear'].parameters = {
                    'beta': max(0, beta),
                    'fixed_cost': max(0, fixed_cost)
                }
                self.model_params['linear'].r_squared = max(0, r_squared)
                self.model_params['linear'].last_updated = datetime.now()
                
        except Exception as e:
            print(f"Linear model calibration failed: {e}")
    
    def _calibrate_almgren_chriss_model(self,
                                      sizes: np.ndarray,
                                      impacts: np.ndarray,
                                      volatilities: np.ndarray) -> None:
        """Calibrate Almgren-Chriss impact model."""
        # Simplified Almgren-Chriss: Impact = lambda * size + eta * sigma * sqrt(size)
        
        try:
            # Create feature matrix
            linear_term = sizes
            sqrt_vol_term = volatilities * np.sqrt(sizes)
            
            # Multiple regression
            X = np.column_stack([linear_term, sqrt_vol_term])
            
            if X.shape[0] > X.shape[1]:
                coeffs = np.linalg.lstsq(X, impacts, rcond=None)[0]
                
                lambda_param = max(0, coeffs[0])
                eta_sigma = max(0, coeffs[1])
                
                # Estimate sigma from volatilities
                sigma = np.mean(volatilities) if len(volatilities) > 0 else 0.02
                eta = eta_sigma / sigma if sigma > 0 else 0.01
                
                # Calculate R-squared
                predicted_impacts = X @ coeffs
                ss_res = np.sum((impacts - predicted_impacts) ** 2)
                ss_tot = np.sum((impacts - np.mean(impacts)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Update parameters
                self.model_params['almgren_chriss'].parameters = {
                    'lambda': lambda_param,
                    'eta': eta,
                    'sigma': sigma
                }
                self.model_params['almgren_chriss'].r_squared = max(0, r_squared)
                self.model_params['almgren_chriss'].last_updated = datetime.now()
                
        except Exception as e:
            print(f"Almgren-Chriss model calibration failed: {e}")
    
    def predict_impact(self,
                      order_size: float,
                      market_conditions: Dict[str, Any],
                      model_ensemble: bool = True) -> ImpactPrediction:
        """
        Predict market impact for an order.
        
        Args:
            order_size: Size of the order
            market_conditions: Current market conditions
            model_ensemble: Whether to use ensemble prediction
            
        Returns:
            Impact prediction
        """
        timestamp = datetime.now()
        
        if model_ensemble:
            # Ensemble prediction
            predictions = {}
            weights = {}
            
            for model_name in self.models:
                if model_name in self.model_params:
                    pred = self._predict_single_model(model_name, order_size, market_conditions)
                    predictions[model_name] = pred
                    
                    # Weight by model performance
                    perf = self.model_performance.get(model_name, {})
                    accuracy = 1.0 - perf.get('mape', 0.5)  # Higher is better
                    r_squared = self.model_params[model_name].r_squared
                    
                    weights[model_name] = max(0.1, accuracy * r_squared)
            
            if predictions:
                # Weighted average
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weighted_impact = sum(
                        pred * weights[model] / total_weight
                        for model, pred in predictions.items()
                    )
                else:
                    weighted_impact = np.mean(list(predictions.values()))
                
                # Calculate confidence based on agreement
                pred_values = list(predictions.values())
                if len(pred_values) > 1:
                    std_dev = np.std(pred_values)
                    mean_pred = np.mean(pred_values)
                    confidence = max(0.1, 1.0 - (std_dev / max(abs(mean_pred), 0.001)))
                else:
                    confidence = 0.5
                
                best_model = max(weights.keys(), key=lambda k: weights[k])
            else:
                weighted_impact = 0.01  # Default
                confidence = 0.1
                best_model = 'none'
                predictions = {}
        
        else:
            # Use best performing model
            best_model = self._select_best_model()
            weighted_impact = self._predict_single_model(best_model, order_size, market_conditions)
            confidence = self.model_params[best_model].r_squared if best_model in self.model_params else 0.5
            predictions = {best_model: weighted_impact}
        
        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(order_size, market_conditions)
        
        return ImpactPrediction(
            timestamp=timestamp,
            order_size=order_size,
            predicted_impact=weighted_impact,
            confidence=confidence,
            model_used=best_model,
            components=predictions,
            risk_factors=risk_factors
        )
    
    def _predict_single_model(self,
                            model_name: str,
                            order_size: float,
                            market_conditions: Dict[str, Any]) -> float:
        """Predict impact using a single model."""
        if model_name not in self.model_params:
            return 0.01  # Default impact
        
        params = self.model_params[model_name].parameters
        
        if model_name == 'square_root':
            gamma = params.get('gamma', 0.5)
            alpha = params.get('alpha', 0.5)
            eta = params.get('eta', 0.01)
            
            depth = market_conditions.get('depth', 100)
            spread = market_conditions.get('spread', 0.001)
            
            normalized_size = order_size / max(depth, 1.0)
            impact = gamma * (normalized_size ** alpha) + eta * spread
            
        elif model_name == 'linear':
            beta = params.get('beta', 0.1)
            fixed_cost = params.get('fixed_cost', 0.001)
            
            impact = beta * order_size + fixed_cost
            
        elif model_name == 'almgren_chriss':
            lambda_param = params.get('lambda', 0.001)
            eta = params.get('eta', 0.01)
            sigma = params.get('sigma', 0.02)
            
            volatility = market_conditions.get('volatility', sigma)
            
            impact = lambda_param * order_size + eta * volatility * np.sqrt(order_size)
            
        else:
            impact = 0.01  # Default
        
        return max(0, impact)
    
    def _select_best_model(self) -> str:
        """Select the best performing model."""
        if not self.model_performance:
            return self.models[0] if self.models else 'square_root'
        
        best_model = None
        best_score = -1
        
        for model_name in self.models:
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]
                
                # Combined score: accuracy and R-squared
                accuracy = 1.0 - perf.get('mape', 1.0)
                r_squared = self.model_params[model_name].r_squared
                
                score = 0.7 * accuracy + 0.3 * r_squared
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model or self.models[0]
    
    def _calculate_risk_factors(self,
                              order_size: float,
                              market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk factors for impact prediction."""
        risk_factors = {}
        
        # Size risk (larger orders have higher uncertainty)
        avg_volume = market_conditions.get('volume', 1000)
        size_ratio = order_size / max(avg_volume, 1.0)
        risk_factors['size_risk'] = min(1.0, size_ratio)
        
        # Volatility risk
        volatility = market_conditions.get('volatility', 0.02)
        risk_factors['volatility_risk'] = min(1.0, volatility * 50)
        
        # Liquidity risk
        depth = market_conditions.get('depth', 100)
        spread = market_conditions.get('spread', 0.001)
        
        liquidity_score = depth / max(spread * 1000, 1.0)
        risk_factors['liquidity_risk'] = max(0.0, 1.0 - liquidity_score / 1000)
        
        # Model uncertainty
        if self.model_performance:
            avg_mape = np.mean([
                perf.get('mape', 0.5) for perf in self.model_performance.values()
            ])
            risk_factors['model_uncertainty'] = min(1.0, avg_mape)
        else:
            risk_factors['model_uncertainty'] = 0.5
        
        return risk_factors
    
    def optimize_execution(self,
                         total_size: float,
                         max_time: float,
                         market_conditions: Dict[str, Any],
                         risk_aversion: float = 0.5) -> ExecutionProfile:
        """
        Optimize order execution to minimize total cost.
        
        Args:
            total_size: Total order size
            max_time: Maximum execution time (seconds)
            market_conditions: Current market conditions
            risk_aversion: Risk aversion parameter (0-1)
            
        Returns:
            Optimal execution profile
        """
        # Simple TWAP-based optimization
        # In practice, this would use more sophisticated algorithms
        
        min_slice = total_size * 0.01  # Minimum 1% slice
        max_slice = total_size * 0.2   # Maximum 20% slice
        
        # Determine optimal slice size and timing
        volatility = market_conditions.get('volatility', 0.02)
        depth = market_conditions.get('depth', 100)
        
        # Risk-adjusted slice size
        base_slice = total_size / 10  # Start with 10 slices
        
        # Adjust for market conditions
        if volatility > 0.03:  # High volatility
            slice_size = min(base_slice, max_slice * 0.5)  # Smaller slices
        elif depth < total_size * 0.1:  # Low liquidity
            slice_size = min(base_slice, min_slice * 10)  # Much smaller slices
        else:
            slice_size = base_slice
        
        slice_size = max(min_slice, min(max_slice, slice_size))
        
        # Calculate number of slices
        num_slices = max(1, int(np.ceil(total_size / slice_size)))
        actual_slice_size = total_size / num_slices
        
        # Calculate timing
        slice_interval = max_time / num_slices
        
        # Create execution plan
        execution_plan = []
        cumulative_size = 0
        cumulative_impact = 0
        
        for i in range(num_slices):
            slice_timestamp = i * slice_interval
            
            # Adjust slice size for last slice
            if i == num_slices - 1:
                current_slice_size = total_size - cumulative_size
            else:
                current_slice_size = actual_slice_size
            
            # Predict impact for this slice
            slice_prediction = self.predict_impact(current_slice_size, market_conditions)
            
            slice_plan = {
                'slice_number': i + 1,
                'size': current_slice_size,
                'timing': slice_timestamp,
                'predicted_impact': slice_prediction.predicted_impact,
                'confidence': slice_prediction.confidence
            }
            
            execution_plan.append(slice_plan)
            cumulative_size += current_slice_size
            cumulative_impact += slice_prediction.predicted_impact
        
        # Calculate total costs
        total_impact = cumulative_impact / num_slices  # Average impact per slice
        
        # Add timing risk
        timing_risk = volatility * np.sqrt(max_time / 3600)  # Scale by execution time
        
        # Add market risk
        market_risk = risk_aversion * volatility * np.sqrt(total_size / depth)
        
        total_cost = total_impact + timing_risk + market_risk
        
        # Calculate risk score
        impact_uncertainty = np.mean([plan['confidence'] for plan in execution_plan])
        risk_score = 1.0 - impact_uncertainty
        
        cost_breakdown = {
            'market_impact': total_impact,
            'timing_risk': timing_risk,
            'market_risk': market_risk,
            'total_cost': total_cost
        }
        
        return ExecutionProfile(
            total_size=total_size,
            execution_plan=execution_plan,
            total_impact=total_impact,
            execution_time=max_time,
            risk_score=risk_score,
            cost_breakdown=cost_breakdown
        )
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of impact models."""
        summary = {
            'is_calibrated': self.is_calibrated,
            'last_calibration': self.last_calibration.isoformat() if self.last_calibration else None,
            'num_observations': len(self.impact_observations),
            'num_trades': len(self.trade_data),
            'models': {}
        }
        
        for model_name in self.models:
            if model_name in self.model_params:
                model_info = {
                    'parameters': self.model_params[model_name].parameters,
                    'r_squared': self.model_params[model_name].r_squared,
                    'last_updated': self.model_params[model_name].last_updated.isoformat()
                }
                
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]
                    model_info['performance'] = {
                        'mae': perf.get('mae', 0),
                        'mape': perf.get('mape', 0),
                        'rmse': perf.get('rmse', 0),
                        'num_predictions': len(perf.get('errors', []))
                    }
                
                summary['models'][model_name] = model_info
        
        return summary