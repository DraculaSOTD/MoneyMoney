"""
Walk-Forward Validator
Implements walk-forward validation for time series models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Represents a single fold in walk-forward validation."""
    fold_number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'fold_number': self.fold_number,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'train_size': self.train_size,
            'test_size': self.test_size
        }


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    n_folds: int
    mean_accuracy: float
    std_accuracy: float
    mean_sharpe: float
    std_sharpe: float
    fold_results: List[Dict]
    aggregate_metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'n_folds': self.n_folds,
            'mean_accuracy': self.mean_accuracy,
            'std_accuracy': self.std_accuracy,
            'mean_sharpe': self.mean_sharpe,
            'std_sharpe': self.std_sharpe,
            'fold_results': self.fold_results,
            'aggregate_metrics': self.aggregate_metrics
        }


class WalkForwardValidator:
    """Implements walk-forward validation for time series."""

    def __init__(
        self,
        n_folds: int = 5,
        min_train_size: int = 50000,
        gap_size: int = 1440,  # 24 hours at 1-minute intervals
        expanding_window: bool = True
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_folds: Number of validation folds
            min_train_size: Minimum training set size
            gap_size: Gap between train and test to prevent leakage
            expanding_window: If True, use expanding window; if False, use sliding window
        """
        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.gap_size = gap_size
        self.expanding_window = expanding_window

    def create_folds(
        self,
        data_length: int,
        test_size: Optional[int] = None
    ) -> List[WalkForwardFold]:
        """
        Create walk-forward validation folds.

        For 500K+ samples with 5 folds:
        - Each test period: ~50K samples (~35 days)
        - Gap between train/test: 1440 samples (24 hours)
        - Expanding training window

        Args:
            data_length: Total number of samples
            test_size: Size of each test set (auto-calculated if None)

        Returns:
            List of WalkForwardFold objects
        """
        if test_size is None:
            # Calculate test size to fit n_folds
            # Leave room for min_train_size, gaps, and all test sets
            available_for_test = data_length - self.min_train_size - (self.n_folds * self.gap_size)
            test_size = available_for_test // self.n_folds

        if test_size <= 0:
            raise ValueError(
                f"Insufficient data: need at least {self.min_train_size + self.n_folds * (test_size + self.gap_size)} samples"
            )

        folds = []
        test_end = data_length

        for fold_num in range(self.n_folds - 1, -1, -1):
            test_start = test_end - test_size
            train_end = test_start - self.gap_size

            if self.expanding_window:
                # Expanding window: always start from the beginning
                train_start = 0
            else:
                # Sliding window: fixed training size
                train_start = max(0, train_end - self.min_train_size)

            # Ensure minimum training size
            if train_end - train_start < self.min_train_size:
                if fold_num == 0:
                    logger.warning(
                        f"Fold {fold_num} has insufficient training data: "
                        f"{train_end - train_start} < {self.min_train_size}"
                    )
                break

            fold = WalkForwardFold(
                fold_number=fold_num,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_size=train_end - train_start,
                test_size=test_end - test_start
            )
            folds.append(fold)

            # Move to next fold
            test_end = test_start

        # Reverse to get chronological order
        folds.reverse()

        logger.info(f"Created {len(folds)} walk-forward folds")
        for fold in folds:
            logger.debug(
                f"Fold {fold.fold_number}: train[{fold.train_start}:{fold.train_end}] "
                f"({fold.train_size}), test[{fold.test_start}:{fold.test_end}] ({fold.test_size})"
            )

        return folds

    def validate(
        self,
        model_class: Any,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict,
        train_func: Optional[Callable] = None,
        eval_func: Optional[Callable] = None,
        prices: Optional[pd.Series] = None
    ) -> WalkForwardResult:
        """
        Perform walk-forward validation.

        Args:
            model_class: Model class to instantiate
            X: Feature array
            y: Label array
            params: Model parameters
            train_func: Optional custom training function(model, X_train, y_train) -> model
            eval_func: Optional custom evaluation function(model, X_test, y_test) -> Dict
            prices: Optional price series for trading metrics

        Returns:
            WalkForwardResult with validation results
        """
        data_length = len(X)
        folds = self.create_folds(data_length)

        if not folds:
            raise ValueError("No valid folds could be created")

        fold_results = []
        accuracies = []
        sharpe_ratios = []

        logger.info(f"Starting walk-forward validation with {len(folds)} folds")

        for fold in folds:
            logger.info(f"Processing fold {fold.fold_number + 1}/{len(folds)}")

            # Split data
            X_train = X[fold.train_start:fold.train_end]
            y_train = y[fold.train_start:fold.train_end]
            X_test = X[fold.test_start:fold.test_end]
            y_test = y[fold.test_start:fold.test_end]

            try:
                # Create and train model
                model = model_class(**params)

                if train_func:
                    model = train_func(model, X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                # Evaluate
                if eval_func:
                    metrics = eval_func(model, X_test, y_test)
                else:
                    y_pred = model.predict(X_test)
                    accuracy = np.mean(y_pred == y_test)
                    metrics = {'accuracy': accuracy}

                # Calculate trading metrics if prices provided
                if prices is not None:
                    test_prices = prices.iloc[fold.test_start:fold.test_end]
                    if len(test_prices) == len(y_pred):
                        trading_metrics = self._calculate_trading_metrics(y_pred, test_prices)
                        metrics.update(trading_metrics)

                # Store results
                fold_result = {
                    'fold': fold.fold_number,
                    **fold.to_dict(),
                    'metrics': metrics
                }
                fold_results.append(fold_result)

                # Track key metrics
                accuracies.append(metrics.get('accuracy', 0))
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))

                logger.info(
                    f"Fold {fold.fold_number + 1} complete: "
                    f"accuracy={metrics.get('accuracy', 0):.4f}, "
                    f"sharpe={metrics.get('sharpe_ratio', 0):.4f}"
                )

            except Exception as e:
                logger.error(f"Error in fold {fold.fold_number}: {e}")
                fold_results.append({
                    'fold': fold.fold_number,
                    **fold.to_dict(),
                    'error': str(e)
                })

        # Aggregate metrics
        aggregate = self.aggregate_metrics(fold_results)

        result = WalkForwardResult(
            n_folds=len(folds),
            mean_accuracy=float(np.mean(accuracies)) if accuracies else 0,
            std_accuracy=float(np.std(accuracies)) if accuracies else 0,
            mean_sharpe=float(np.mean(sharpe_ratios)) if sharpe_ratios else 0,
            std_sharpe=float(np.std(sharpe_ratios)) if sharpe_ratios else 0,
            fold_results=fold_results,
            aggregate_metrics=aggregate
        )

        logger.info(
            f"Walk-forward validation complete: "
            f"mean_accuracy={result.mean_accuracy:.4f} (+/- {result.std_accuracy:.4f}), "
            f"mean_sharpe={result.mean_sharpe:.4f}"
        )

        return result

    def aggregate_metrics(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all folds.

        Args:
            fold_results: Results from each fold

        Returns:
            Dictionary of aggregated metrics
        """
        # Collect all metrics
        all_metrics = {}
        for result in fold_results:
            if 'metrics' not in result:
                continue
            for key, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # Calculate aggregates
        aggregate = {}
        for key, values in all_metrics.items():
            if values:
                aggregate[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }

        return aggregate

    def _calculate_trading_metrics(
        self,
        predictions: np.ndarray,
        prices: pd.Series
    ) -> Dict:
        """
        Calculate basic trading metrics for a fold.

        Args:
            predictions: Model predictions
            prices: Price series

        Returns:
            Dictionary of trading metrics
        """
        from crypto_ml_trading.training.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()
        return evaluator.calculate_trading_metrics(predictions, prices)


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.

    Implements purging (gap between train/test) and embargo
    (gap after test set in subsequent folds) to prevent data leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 1440,
        embargo_gap: int = 1440
    ):
        """
        Initialize PurgedKFold.

        Args:
            n_splits: Number of folds
            purge_gap: Gap between train and test sets
            embargo_gap: Gap after test set for subsequent folds
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature array

        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits

        splits = []

        for fold in range(self.n_splits):
            # Test set for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples

            # Training indices (everything except test + gaps)
            train_indices = []

            # Add samples before test (with purge gap)
            if test_start - self.purge_gap > 0:
                train_indices.extend(range(0, test_start - self.purge_gap))

            # Add samples after test (with embargo gap)
            if test_end + self.embargo_gap < n_samples:
                train_indices.extend(range(test_end + self.embargo_gap, n_samples))

            test_indices = list(range(test_start, test_end))

            splits.append((np.array(train_indices), np.array(test_indices)))

        return splits
