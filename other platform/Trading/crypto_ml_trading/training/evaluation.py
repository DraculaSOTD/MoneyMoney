"""
Model Evaluation
Comprehensive model evaluation with trading-specific metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a model."""
    classification_metrics: Dict
    trading_metrics: Dict
    confidence_calibration: Dict
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'classification_metrics': self.classification_metrics,
            'trading_metrics': self.trading_metrics,
            'confidence_calibration': self.confidence_calibration,
            'confusion_matrix': self.confusion_matrix.tolist() if isinstance(self.confusion_matrix, np.ndarray) else self.confusion_matrix,
            'feature_importance': dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])  # Top 20 features
        }


class ModelEvaluator:
    """Comprehensive model evaluation with trading-specific metrics."""

    # Trading simulation parameters
    INITIAL_CAPITAL = 10000.0
    TRADING_FEE = 0.001  # 0.1%
    RISK_FREE_RATE = 0.02  # 2% annual

    def __init__(self, initial_capital: float = None):
        if initial_capital:
            self.INITIAL_CAPITAL = initial_capital

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        prices: Optional[pd.Series] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> EvaluationReport:
        """
        Perform comprehensive model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            prices: Price series for trading simulation (optional)
            feature_importance: Feature importance scores (optional)

        Returns:
            EvaluationReport with all metrics
        """
        logger.info("Evaluating model performance...")

        # Classification metrics
        class_metrics = self.calculate_classification_metrics(y_true, y_pred, y_prob)

        # Trading metrics (if prices provided)
        if prices is not None:
            trading_metrics = self.calculate_trading_metrics(y_pred, prices)
        else:
            trading_metrics = {}

        # Confidence calibration (if probabilities provided)
        if y_prob is not None:
            calibration = self.calculate_confidence_calibration(y_true, y_pred, y_prob)
        else:
            calibration = {}

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        report = EvaluationReport(
            classification_metrics=class_metrics,
            trading_metrics=trading_metrics,
            confidence_calibration=calibration,
            confusion_matrix=cm,
            feature_importance=feature_importance or {},
            predictions=y_pred,
            probabilities=y_prob,
            labels=y_true
        )

        logger.info(
            f"Evaluation complete: accuracy={class_metrics.get('accuracy', 0):.4f}, "
            f"f1={class_metrics.get('f1_weighted', 0):.4f}"
        )

        return report

    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            Dictionary of classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

        # Per-class metrics
        n_classes = len(np.unique(y_true))
        for i in range(n_classes):
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)
            metrics[f'precision_class_{i}'] = precision_score(binary_true, binary_pred, zero_division=0)
            metrics[f'recall_class_{i}'] = recall_score(binary_true, binary_pred, zero_division=0)
            metrics[f'f1_class_{i}'] = f1_score(binary_true, binary_pred, zero_division=0)

        # AUC-ROC (if probabilities available)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multiclass - use OvR
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate AUC-ROC: {e}")

        return metrics

    def calculate_trading_metrics(
        self,
        predictions: np.ndarray,
        prices: pd.Series
    ) -> Dict:
        """
        Calculate trading-specific metrics based on model predictions.

        Simulates trading based on predictions:
        - 0 = Sell/Short
        - 1 = Buy/Long
        - 2 = Hold

        Args:
            predictions: Model predictions
            prices: Price series

        Returns:
            Dictionary of trading metrics
        """
        if len(predictions) != len(prices):
            logger.warning("Predictions and prices length mismatch")
            return {}

        # Calculate returns
        returns = prices.pct_change().fillna(0)

        # Generate positions from predictions
        # Buy = 1, Sell = -1, Hold = 0
        position_map = {0: -1, 1: 1, 2: 0}
        positions = np.array([position_map.get(p, 0) for p in predictions])

        # Calculate strategy returns (shifted by 1 to avoid lookahead)
        strategy_returns = returns.values[1:] * positions[:-1]

        # Apply trading fees (on position changes)
        position_changes = np.abs(np.diff(positions))
        fees = position_changes * self.TRADING_FEE

        # Net returns
        net_returns = strategy_returns - fees

        # Calculate equity curve
        equity_curve = self.INITIAL_CAPITAL * (1 + np.cumsum(net_returns))

        # Calculate metrics
        metrics = {}

        # Total return
        final_value = equity_curve[-1] if len(equity_curve) > 0 else self.INITIAL_CAPITAL
        metrics['total_return'] = (final_value - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL

        # Annualized return (assuming 1-minute data)
        n_periods = len(net_returns)
        periods_per_year = 525600  # Minutes per year
        if n_periods > 0:
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (periods_per_year / n_periods) - 1

        # Sharpe ratio
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(pd.Series(net_returns))

        # Sortino ratio
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(pd.Series(net_returns))

        # Maximum drawdown
        metrics['max_drawdown'] = self.calculate_max_drawdown(pd.Series(equity_curve))

        # Profit factor
        metrics['profit_factor'] = self.calculate_profit_factor(pd.Series(net_returns))

        # Win rate
        winning_trades = np.sum(net_returns > 0)
        total_trades = np.sum(positions[:-1] != 0)
        metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0

        # Average trade
        if total_trades > 0:
            metrics['avg_trade_return'] = np.mean(net_returns[positions[:-1] != 0])
            metrics['avg_win'] = np.mean(net_returns[net_returns > 0]) if np.sum(net_returns > 0) > 0 else 0
            metrics['avg_loss'] = np.mean(net_returns[net_returns < 0]) if np.sum(net_returns < 0) > 0 else 0

        # Trade statistics
        metrics['total_trades'] = int(total_trades)
        metrics['winning_trades'] = int(winning_trades)
        metrics['losing_trades'] = int(total_trades - winning_trades)

        return metrics

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.RISK_FREE_RATE

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Convert to per-minute risk-free rate
        rf_per_minute = (1 + risk_free_rate) ** (1 / 525600) - 1

        excess_returns = returns - rf_per_minute
        sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Annualize
        sharpe_annual = sharpe * np.sqrt(525600)

        return float(sharpe_annual)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.RISK_FREE_RATE

        if len(returns) == 0:
            return 0.0

        rf_per_minute = (1 + risk_free_rate) ** (1 / 525600) - 1
        excess_returns = returns - rf_per_minute

        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0

        if downside_std == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_std

        # Annualize
        sortino_annual = sortino * np.sqrt(525600)

        return float(sortino_annual)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve series

        Returns:
            Maximum drawdown as a decimal (negative)
        """
        if len(equity_curve) == 0:
            return 0.0

        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        return float(max_drawdown)

    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        Args:
            returns: Returns series

        Returns:
            Profit factor
        """
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())

        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0

        return float(gross_profits / gross_losses)

    def calculate_confidence_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        Calculate confidence calibration metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            Dictionary of calibration metrics
        """
        # Get confidence (max probability)
        if y_prob.ndim == 1:
            confidence = y_prob
        else:
            confidence = np.max(y_prob, axis=1)

        # Correct predictions
        correct = (y_true == y_pred).astype(int)

        # Binned calibration
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        calibration_data = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                bin_accuracy = np.mean(correct[in_bin])
                bin_confidence = np.mean(confidence[in_bin])
                calibration_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'bin_size': int(bin_size),
                    'accuracy': float(bin_accuracy),
                    'confidence': float(bin_confidence),
                    'gap': float(abs(bin_accuracy - bin_confidence))
                })

        # Expected Calibration Error (ECE)
        total_samples = len(y_true)
        ece = sum(
            (d['bin_size'] / total_samples) * d['gap']
            for d in calibration_data
        )

        # Average confidence
        avg_confidence = float(np.mean(confidence))

        # Confidence-accuracy correlation
        correlation = float(np.corrcoef(confidence, correct)[0, 1]) if len(confidence) > 1 else 0.0

        return {
            'ece': ece,
            'average_confidence': avg_confidence,
            'confidence_accuracy_correlation': correlation,
            'bins': calibration_data
        }


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    prices: Optional[pd.Series] = None
) -> EvaluationReport:
    """
    Convenience function to evaluate model predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        prices: Price series for trading simulation

    Returns:
        EvaluationReport
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate(y_true, y_pred, y_prob, prices)
