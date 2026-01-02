"""
Quantile Loss Functions for TFT Probabilistic Forecasting.

Implements various loss functions for quantile regression and uncertainty estimation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class QuantileLoss:
    """
    Quantile loss for probabilistic forecasting.
    
    Enables prediction of uncertainty intervals and risk-aware forecasting.
    """
    
    def __init__(self, quantiles: List[float] = None):
        """
        Initialize quantile loss.
        
        Args:
            quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
        """
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        
        # Validate quantiles
        for q in self.quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantiles must be between 0 and 1, got {q}")
                
    def compute_loss(self, predictions: Dict[str, np.ndarray],
                    targets: np.ndarray,
                    sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute quantile loss for all quantiles.
        
        Args:
            predictions: Dictionary with quantile predictions
            targets: True values (batch_size, seq_len)
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary with loss values
        """
        losses = {}
        total_loss = 0.0
        
        for i, q in enumerate(self.quantiles):
            pred_key = f'q{int(q*100)}'
            if pred_key not in predictions:
                raise KeyError(f"Missing prediction for quantile {q}")
                
            pred = predictions[pred_key]
            loss = self._quantile_loss(pred, targets, q, sample_weights)
            
            losses[f'loss_{pred_key}'] = loss
            total_loss += loss
            
        losses['total_loss'] = total_loss
        losses['avg_loss'] = total_loss / len(self.quantiles)
        
        return losses
    
    def _quantile_loss(self, predictions: np.ndarray,
                      targets: np.ndarray,
                      quantile: float,
                      sample_weights: Optional[np.ndarray] = None) -> float:
        """
        Compute quantile loss for a single quantile.
        
        Args:
            predictions: Predicted values
            targets: True values
            quantile: Quantile level
            sample_weights: Optional sample weights
            
        Returns:
            Quantile loss value
        """
        # Calculate errors
        errors = targets - predictions
        
        # Quantile loss formula
        loss = np.where(
            errors >= 0,
            quantile * errors,
            (quantile - 1) * errors
        )
        
        # Apply sample weights if provided
        if sample_weights is not None:
            loss = loss * sample_weights
            
        return np.mean(loss)
    
    def compute_gradients(self, predictions: Dict[str, np.ndarray],
                         targets: np.ndarray,
                         sample_weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute gradients for quantile loss.
        
        Args:
            predictions: Dictionary with quantile predictions
            targets: True values
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary with gradients for each quantile
        """
        gradients = {}
        
        for i, q in enumerate(self.quantiles):
            pred_key = f'q{int(q*100)}'
            pred = predictions[pred_key]
            
            # Gradient computation
            errors = targets - pred
            grad = np.where(errors >= 0, -q, -(q - 1))
            
            # Apply sample weights
            if sample_weights is not None:
                grad = grad * sample_weights
                
            gradients[pred_key] = grad
            
        return gradients
    
    def evaluate_prediction_intervals(self, predictions: Dict[str, np.ndarray],
                                    targets: np.ndarray) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage and quality.
        
        Args:
            predictions: Dictionary with quantile predictions
            targets: True values
            
        Returns:
            Dictionary with interval metrics
        """
        metrics = {}
        
        # Sort quantiles for interval analysis
        sorted_quantiles = sorted(self.quantiles)
        
        # Calculate coverage for different intervals
        for i in range(len(sorted_quantiles) - 1):
            lower_q = sorted_quantiles[i]
            upper_q = sorted_quantiles[-(i+1)]
            
            if lower_q >= upper_q:
                continue
                
            lower_key = f'q{int(lower_q*100)}'
            upper_key = f'q{int(upper_q*100)}'
            
            if lower_key in predictions and upper_key in predictions:
                lower_pred = predictions[lower_key]
                upper_pred = predictions[upper_key]
                
                # Calculate coverage
                in_interval = (targets >= lower_pred) & (targets <= upper_pred)
                coverage = np.mean(in_interval)
                
                # Expected coverage
                expected_coverage = upper_q - lower_q
                
                # Interval width
                interval_width = np.mean(upper_pred - lower_pred)
                
                interval_name = f'interval_{int(lower_q*100)}_{int(upper_q*100)}'
                metrics[f'coverage_{interval_name}'] = coverage
                metrics[f'expected_coverage_{interval_name}'] = expected_coverage
                metrics[f'coverage_error_{interval_name}'] = abs(coverage - expected_coverage)
                metrics[f'avg_width_{interval_name}'] = interval_width
                
        # Median absolute error (if 0.5 quantile available)
        if 'q50' in predictions:
            mae = np.mean(np.abs(targets - predictions['q50']))
            metrics['median_absolute_error'] = mae
            
        return metrics
    
    def compute_crps(self, predictions: Dict[str, np.ndarray],
                    targets: np.ndarray) -> float:
        """
        Compute Continuous Ranked Probability Score (CRPS).
        
        CRPS measures the quality of probabilistic forecasts.
        
        Args:
            predictions: Dictionary with quantile predictions
            targets: True values
            
        Returns:
            CRPS value
        """
        # Sort quantiles and predictions
        sorted_items = sorted([(q, f'q{int(q*100)}') for q in self.quantiles])
        
        crps_values = []
        
        for t_idx in range(targets.shape[0]):
            for seq_idx in range(targets.shape[1]):
                target = targets[t_idx, seq_idx]
                
                # Extract quantiles and predictions for this point
                quantile_preds = []
                for q, key in sorted_items:
                    if key in predictions:
                        quantile_preds.append((q, predictions[key][t_idx, seq_idx]))\n                \n                if len(quantile_preds) < 2:\n                    continue\n                    \n                # Compute CRPS using quantile integration\n                crps = self._crps_from_quantiles(quantile_preds, target)\n                crps_values.append(crps)\n                \n        return np.mean(crps_values) if crps_values else float('inf')\n    \n    def _crps_from_quantiles(self, quantile_preds: List[Tuple[float, float]],\n                           target: float) -> float:\n        \"\"\"Compute CRPS from quantile predictions.\"\"\"\n        crps = 0.0\n        \n        # Sort by quantile level\n        quantile_preds.sort()\n        \n        for i in range(len(quantile_preds) - 1):\n            q1, pred1 = quantile_preds[i]\n            q2, pred2 = quantile_preds[i + 1]\n            \n            # Contribution to CRPS\n            if target <= pred1:\n                # Target is below both quantiles\n                crps += (q2 - q1) * (pred1 - target) * (1 - 0)\n            elif target >= pred2:\n                # Target is above both quantiles\n                crps += (q2 - q1) * (target - pred2) * (1 - 1)\n            else:\n                # Target is between quantiles\n                # Split into two parts\n                crps += (q2 - q1) * (\n                    (pred1 - target) * (1 - q1) +\n                    (target - pred1) * (q2 - q1) / (pred2 - pred1)\n                )\n                \n        return crps\n    \n    def compute_pinball_loss(self, predictions: Dict[str, np.ndarray],\n                           targets: np.ndarray) -> Dict[str, float]:\n        \"\"\"Compute pinball loss (alternative name for quantile loss).\"\"\"\n        return self.compute_loss(predictions, targets)\n    \n    def get_uncertainty_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:\n        \"\"\"\n        Extract uncertainty metrics from quantile predictions.\n        \n        Args:\n            predictions: Dictionary with quantile predictions\n            \n        Returns:\n            Dictionary with uncertainty metrics\n        \"\"\"\n        metrics = {}\n        \n        # Find median prediction\n        if 'q50' in predictions:\n            metrics['median'] = predictions['q50']\n        elif len(self.quantiles) > 0:\n            # Use middle quantile as approximation\n            mid_idx = len(self.quantiles) // 2\n            mid_q = sorted(self.quantiles)[mid_idx]\n            mid_key = f'q{int(mid_q*100)}'\n            if mid_key in predictions:\n                metrics['median'] = predictions[mid_key]\n                \n        # Calculate prediction intervals\n        sorted_quantiles = sorted(self.quantiles)\n        \n        if len(sorted_quantiles) >= 2:\n            # Interquartile range\n            if 'q25' in predictions and 'q75' in predictions:\n                metrics['iqr'] = predictions['q75'] - predictions['q25']\n                \n            # 90% prediction interval\n            lower_keys = [f'q{int(q*100)}' for q in sorted_quantiles if q <= 0.1]\n            upper_keys = [f'q{int(q*100)}' for q in sorted_quantiles if q >= 0.9]\n            \n            if lower_keys and upper_keys:\n                lower_key = lower_keys[-1]  # Closest to 0.1\n                upper_key = upper_keys[0]   # Closest to 0.9\n                \n                if lower_key in predictions and upper_key in predictions:\n                    metrics['prediction_interval_90'] = (\n                        predictions[upper_key] - predictions[lower_key]\n                    )\n                    \n        # Coefficient of variation (if we have std approximation)\n        if 'iqr' in metrics and 'median' in metrics:\n            # Approximate std as IQR / 1.35 (for normal distribution)\n            approx_std = metrics['iqr'] / 1.35\n            metrics['coefficient_of_variation'] = (\n                approx_std / (np.abs(metrics['median']) + 1e-8)\n            )\n            \n        return metrics\n\n\nclass AdaptiveQuantileLoss:\n    \"\"\"\n    Adaptive quantile loss that adjusts weights based on prediction quality.\n    \"\"\"\n    \n    def __init__(self, quantiles: List[float] = None,\n                 adaptation_rate: float = 0.01):\n        \"\"\"\n        Initialize adaptive quantile loss.\n        \n        Args:\n            quantiles: List of quantiles to predict\n            adaptation_rate: Rate of weight adaptation\n        \"\"\"\n        self.base_loss = QuantileLoss(quantiles)\n        self.quantiles = self.base_loss.quantiles\n        self.adaptation_rate = adaptation_rate\n        \n        # Initialize adaptive weights\n        self.quantile_weights = {f'q{int(q*100)}': 1.0 for q in self.quantiles}\n        self.performance_history = {f'q{int(q*100)}': [] for q in self.quantiles}\n        \n    def compute_loss(self, predictions: Dict[str, np.ndarray],\n                    targets: np.ndarray,\n                    sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:\n        \"\"\"\n        Compute adaptive quantile loss.\n        \"\"\"\n        # Compute base losses\n        base_losses = self.base_loss.compute_loss(predictions, targets, sample_weights)\n        \n        # Apply adaptive weights\n        weighted_losses = {}\n        total_weighted_loss = 0.0\n        \n        for i, q in enumerate(self.quantiles):\n            loss_key = f'loss_q{int(q*100)}'\n            weight_key = f'q{int(q*100)}'\n            \n            if loss_key in base_losses:\n                weighted_loss = base_losses[loss_key] * self.quantile_weights[weight_key]\n                weighted_losses[loss_key] = weighted_loss\n                total_weighted_loss += weighted_loss\n                \n                # Update performance history\n                self.performance_history[weight_key].append(base_losses[loss_key])\n                \n        weighted_losses['total_loss'] = total_weighted_loss\n        weighted_losses['avg_loss'] = total_weighted_loss / len(self.quantiles)\n        \n        # Update weights based on recent performance\n        self._update_weights()\n        \n        return weighted_losses\n    \n    def _update_weights(self, window_size: int = 100):\n        \"\"\"Update quantile weights based on recent performance.\"\"\"\n        if len(self.performance_history[f'q{int(self.quantiles[0]*100)}']) < 10:\n            return  # Not enough history\n            \n        # Calculate recent performance for each quantile\n        recent_performance = {}\n        \n        for q in self.quantiles:\n            key = f'q{int(q*100)}'\n            recent_losses = self.performance_history[key][-window_size:]\n            recent_performance[key] = np.mean(recent_losses)\n            \n        # Update weights (lower recent loss = higher weight)\n        max_performance = max(recent_performance.values())\n        \n        for key in self.quantile_weights:\n            if key in recent_performance:\n                # Inverse relationship: better performance (lower loss) gets higher weight\n                relative_performance = recent_performance[key] / (max_performance + 1e-8)\n                target_weight = 2.0 - relative_performance  # Range: [1, 2]\n                \n                # Exponential moving average update\n                self.quantile_weights[key] = (\n                    (1 - self.adaptation_rate) * self.quantile_weights[key] +\n                    self.adaptation_rate * target_weight\n                )\n                \n        # Normalize weights\n        total_weight = sum(self.quantile_weights.values())\n        for key in self.quantile_weights:\n            self.quantile_weights[key] /= total_weight\n    \n    def get_weight_history(self) -> Dict[str, List[float]]:\n        \"\"\"Get history of weight adaptations.\"\"\"\n        return {\n            'current_weights': self.quantile_weights.copy(),\n            'performance_history': {\n                k: v.copy() for k, v in self.performance_history.items()\n            }\n        }"