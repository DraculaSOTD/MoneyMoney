"""
Model Comparison Framework
Compares new models against deployed models with statistical testing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from database.models import ProfileModel, TradingProfile
from crypto_ml_trading.training.evaluation import ModelEvaluator, EvaluationReport

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Results from model comparison."""
    new_accuracy: float
    deployed_accuracy: float
    accuracy_diff: float
    new_sharpe: float
    deployed_sharpe: float
    sharpe_diff: float
    is_significant: bool
    p_value: float
    should_deploy: bool
    reason: str
    detailed_comparison: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'new_accuracy': self.new_accuracy,
            'deployed_accuracy': self.deployed_accuracy,
            'accuracy_diff': self.accuracy_diff,
            'new_sharpe': self.new_sharpe,
            'deployed_sharpe': self.deployed_sharpe,
            'sharpe_diff': self.sharpe_diff,
            'is_significant': self.is_significant,
            'p_value': self.p_value,
            'should_deploy': self.should_deploy,
            'reason': self.reason,
            'detailed_comparison': self.detailed_comparison
        }


class ModelComparator:
    """Compares new model against deployed model."""

    # Significance level for statistical tests
    SIGNIFICANCE_LEVEL = 0.05
    # Minimum improvement required (in absolute terms)
    MIN_ACCURACY_IMPROVEMENT = 0.01  # 1%
    MIN_SHARPE_IMPROVEMENT = 0.1

    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session
        self.evaluator = ModelEvaluator()

    def compare(
        self,
        new_model_metrics: EvaluationReport,
        deployed_model_id: Optional[int] = None,
        deployed_predictions: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        test_prices: Optional[pd.Series] = None
    ) -> ComparisonResult:
        """
        Compare new model against deployed model.

        Args:
            new_model_metrics: Evaluation report for new model
            deployed_model_id: ID of deployed model (to load from DB)
            deployed_predictions: Predictions from deployed model
            test_labels: True test labels
            test_prices: Price series for trading metrics

        Returns:
            ComparisonResult with comparison details
        """
        logger.info("Comparing new model against deployed model...")

        # Get new model metrics
        new_accuracy = new_model_metrics.classification_metrics.get('accuracy', 0)
        new_sharpe = new_model_metrics.trading_metrics.get('sharpe_ratio', 0)

        # If no deployed model, always deploy if above threshold
        if deployed_predictions is None and deployed_model_id is None:
            logger.info("No deployed model found - recommending deployment")
            return ComparisonResult(
                new_accuracy=new_accuracy,
                deployed_accuracy=0,
                accuracy_diff=new_accuracy,
                new_sharpe=new_sharpe,
                deployed_sharpe=0,
                sharpe_diff=new_sharpe,
                is_significant=True,
                p_value=0.0,
                should_deploy=new_accuracy >= 0.55,  # Minimum threshold
                reason="No deployed model - deploying if above minimum threshold"
            )

        # Get deployed model predictions if needed
        if deployed_predictions is None and deployed_model_id is not None:
            deployed_predictions = self._load_deployed_predictions(deployed_model_id)

        if deployed_predictions is None or test_labels is None:
            logger.warning("Cannot compare - missing deployed predictions or test labels")
            return ComparisonResult(
                new_accuracy=new_accuracy,
                deployed_accuracy=0,
                accuracy_diff=new_accuracy,
                new_sharpe=new_sharpe,
                deployed_sharpe=0,
                sharpe_diff=new_sharpe,
                is_significant=False,
                p_value=1.0,
                should_deploy=False,
                reason="Cannot compare - missing data"
            )

        # Calculate deployed model accuracy
        deployed_accuracy = float(np.mean(deployed_predictions == test_labels))

        # Calculate deployed sharpe if prices provided
        if test_prices is not None:
            deployed_trading = self.evaluator.calculate_trading_metrics(
                deployed_predictions, test_prices
            )
            deployed_sharpe = deployed_trading.get('sharpe_ratio', 0)
        else:
            deployed_sharpe = 0

        # Run statistical significance test
        is_significant, p_value = self._statistical_significance_test(
            new_model_metrics.predictions,
            deployed_predictions,
            test_labels
        )

        # Calculate differences
        accuracy_diff = new_accuracy - deployed_accuracy
        sharpe_diff = new_sharpe - deployed_sharpe

        # Determine if should deploy
        should_deploy, reason = self._should_deploy(
            new_accuracy=new_accuracy,
            deployed_accuracy=deployed_accuracy,
            new_sharpe=new_sharpe,
            deployed_sharpe=deployed_sharpe,
            is_significant=is_significant,
            p_value=p_value
        )

        # Detailed comparison
        detailed = self._detailed_comparison(
            new_model_metrics,
            deployed_predictions,
            test_labels,
            test_prices
        )

        result = ComparisonResult(
            new_accuracy=new_accuracy,
            deployed_accuracy=deployed_accuracy,
            accuracy_diff=accuracy_diff,
            new_sharpe=new_sharpe,
            deployed_sharpe=deployed_sharpe,
            sharpe_diff=sharpe_diff,
            is_significant=is_significant,
            p_value=p_value,
            should_deploy=should_deploy,
            reason=reason,
            detailed_comparison=detailed
        )

        logger.info(
            f"Comparison complete: new_acc={new_accuracy:.4f}, "
            f"deployed_acc={deployed_accuracy:.4f}, diff={accuracy_diff:+.4f}, "
            f"significant={is_significant}, should_deploy={should_deploy}"
        )

        return result

    def _load_deployed_predictions(self, model_id: int) -> Optional[np.ndarray]:
        """
        Load predictions from deployed model.

        In practice, this would load the model and run inference.
        For now, returns None (caller should provide predictions).
        """
        logger.warning(
            f"Loading deployed model {model_id} predictions not implemented - "
            "provide predictions directly"
        )
        return None

    def _statistical_significance_test(
        self,
        new_predictions: np.ndarray,
        deployed_predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Perform statistical significance test using McNemar's test.

        McNemar's test is appropriate for comparing two classifiers on the same data.

        Args:
            new_predictions: Predictions from new model
            deployed_predictions: Predictions from deployed model
            true_labels: True labels

        Returns:
            Tuple of (is_significant, p_value)
        """
        if len(new_predictions) != len(deployed_predictions):
            logger.warning("Prediction arrays have different lengths")
            return False, 1.0

        # Create contingency table for McNemar's test
        # Count cases where models disagree
        new_correct = new_predictions == true_labels
        deployed_correct = deployed_predictions == true_labels

        # b: deployed correct, new wrong
        # c: new correct, deployed wrong
        b = np.sum(deployed_correct & ~new_correct)
        c = np.sum(new_correct & ~deployed_correct)

        # McNemar's test with continuity correction
        if b + c == 0:
            return False, 1.0

        # Chi-square statistic
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        is_significant = p_value < self.SIGNIFICANCE_LEVEL

        logger.debug(f"McNemar's test: b={b}, c={c}, chi2={chi2:.4f}, p={p_value:.4f}")

        return is_significant, float(p_value)

    def _should_deploy(
        self,
        new_accuracy: float,
        deployed_accuracy: float,
        new_sharpe: float,
        deployed_sharpe: float,
        is_significant: bool,
        p_value: float
    ) -> Tuple[bool, str]:
        """
        Determine if new model should be deployed.

        Criteria:
        1. Improvement must be statistically significant (p < 0.05)
        2. Accuracy improvement must be >= MIN_ACCURACY_IMPROVEMENT
        3. Sharpe ratio should not decrease significantly

        Args:
            new_accuracy: New model accuracy
            deployed_accuracy: Deployed model accuracy
            new_sharpe: New model Sharpe ratio
            deployed_sharpe: Deployed model Sharpe ratio
            is_significant: Whether improvement is significant
            p_value: P-value from significance test

        Returns:
            Tuple of (should_deploy, reason)
        """
        accuracy_diff = new_accuracy - deployed_accuracy
        sharpe_diff = new_sharpe - deployed_sharpe

        # Check minimum accuracy
        if new_accuracy < 0.55:
            return False, f"New model accuracy ({new_accuracy:.4f}) below minimum threshold (0.55)"

        # Check if improvement is significant
        if not is_significant:
            return False, f"Improvement not statistically significant (p={p_value:.4f})"

        # Check minimum improvement
        if accuracy_diff < self.MIN_ACCURACY_IMPROVEMENT:
            return False, f"Accuracy improvement ({accuracy_diff:+.4f}) below threshold ({self.MIN_ACCURACY_IMPROVEMENT})"

        # Check Sharpe ratio doesn't decrease significantly
        if sharpe_diff < -self.MIN_SHARPE_IMPROVEMENT:
            return False, f"Sharpe ratio decreased significantly ({sharpe_diff:+.4f})"

        # All checks passed
        reason = (
            f"New model is better: accuracy {new_accuracy:.4f} vs {deployed_accuracy:.4f} "
            f"(+{accuracy_diff:.4f}, p={p_value:.4f}), "
            f"Sharpe {new_sharpe:.4f} vs {deployed_sharpe:.4f}"
        )
        return True, reason

    def _detailed_comparison(
        self,
        new_metrics: EvaluationReport,
        deployed_predictions: np.ndarray,
        test_labels: np.ndarray,
        test_prices: Optional[pd.Series]
    ) -> Dict:
        """
        Create detailed comparison between models.
        """
        detailed = {}

        # Per-class comparison
        classes = np.unique(test_labels)
        class_comparison = {}

        for cls in classes:
            cls_mask = test_labels == cls
            new_cls_acc = np.mean(new_metrics.predictions[cls_mask] == test_labels[cls_mask])
            deployed_cls_acc = np.mean(deployed_predictions[cls_mask] == test_labels[cls_mask])
            class_comparison[int(cls)] = {
                'new_accuracy': float(new_cls_acc),
                'deployed_accuracy': float(deployed_cls_acc),
                'diff': float(new_cls_acc - deployed_cls_acc)
            }

        detailed['per_class'] = class_comparison

        # Agreement analysis
        agreement = np.mean(new_metrics.predictions == deployed_predictions)
        detailed['model_agreement'] = float(agreement)

        # Cases where new model is correct and deployed is wrong
        new_wins = np.sum(
            (new_metrics.predictions == test_labels) &
            (deployed_predictions != test_labels)
        )
        deployed_wins = np.sum(
            (deployed_predictions == test_labels) &
            (new_metrics.predictions != test_labels)
        )
        detailed['new_model_wins'] = int(new_wins)
        detailed['deployed_model_wins'] = int(deployed_wins)

        return detailed


def compare_models(
    new_predictions: np.ndarray,
    deployed_predictions: np.ndarray,
    true_labels: np.ndarray,
    prices: Optional[pd.Series] = None
) -> ComparisonResult:
    """
    Convenience function to compare two models.

    Args:
        new_predictions: Predictions from new model
        deployed_predictions: Predictions from deployed model
        true_labels: True labels
        prices: Optional price series for trading metrics

    Returns:
        ComparisonResult
    """
    comparator = ModelComparator()
    evaluator = ModelEvaluator()

    # Evaluate new model
    new_metrics = evaluator.evaluate(
        y_true=true_labels,
        y_pred=new_predictions,
        prices=prices
    )

    return comparator.compare(
        new_model_metrics=new_metrics,
        deployed_predictions=deployed_predictions,
        test_labels=true_labels,
        test_prices=prices
    )
