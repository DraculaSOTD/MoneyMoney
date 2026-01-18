"""
Concept Drift Detector
Detects feature and prediction distribution changes over time.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Report of drift detection analysis."""
    feature_drifts: Dict[str, float]  # Feature name -> drift score
    prediction_drift: float
    overall_drift_score: float
    drifted_features: List[str]
    recommendation: str  # "retrain", "monitor", "ok"
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'feature_drifts': self.feature_drifts,
            'prediction_drift': self.prediction_drift,
            'overall_drift_score': self.overall_drift_score,
            'drifted_features': self.drifted_features,
            'recommendation': self.recommendation,
            'details': self.details
        }


class ConceptDriftDetector:
    """Detects feature and prediction distribution changes."""

    # Thresholds
    KL_DIVERGENCE_THRESHOLD = 0.1  # KL divergence threshold for drift
    PSI_THRESHOLD = 0.2  # Population Stability Index threshold
    FEATURE_DRIFT_THRESHOLD = 0.1  # Per-feature drift threshold
    OVERALL_DRIFT_THRESHOLD = 0.15  # Overall drift threshold

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        n_bins: int = 10
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference feature data (e.g., training data)
            reference_predictions: Reference predictions
            n_bins: Number of bins for distribution comparison
        """
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self.n_bins = n_bins

        # Pre-compute reference distributions
        self._reference_distributions: Dict[str, np.ndarray] = {}
        self._prediction_distribution: Optional[np.ndarray] = None

        if reference_data is not None:
            self._compute_reference_distributions()

        if reference_predictions is not None:
            self._compute_prediction_distribution()

    def _compute_reference_distributions(self) -> None:
        """Compute histograms for reference data."""
        for col in self.reference_data.columns:
            if not np.issubdtype(self.reference_data[col].dtype, np.number):
                continue

            values = self.reference_data[col].dropna().values
            if len(values) == 0:
                continue

            # Compute histogram
            hist, bin_edges = np.histogram(values, bins=self.n_bins, density=True)
            self._reference_distributions[col] = {
                'hist': hist,
                'bin_edges': bin_edges,
                'min': values.min(),
                'max': values.max()
            }

    def _compute_prediction_distribution(self) -> None:
        """Compute histogram for reference predictions."""
        if self.reference_predictions is None:
            return

        hist, bin_edges = np.histogram(
            self.reference_predictions,
            bins=self.n_bins,
            density=True
        )
        self._prediction_distribution = {
            'hist': hist,
            'bin_edges': bin_edges
        }

    def set_reference(
        self,
        reference_data: pd.DataFrame,
        reference_predictions: Optional[np.ndarray] = None
    ) -> None:
        """
        Set reference data for drift comparison.

        Args:
            reference_data: Reference feature data
            reference_predictions: Reference predictions
        """
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
        self._compute_reference_distributions()
        if reference_predictions is not None:
            self._compute_prediction_distribution()

    def detect_feature_drift(
        self,
        new_data: pd.DataFrame
    ) -> DriftReport:
        """
        Detect drift in feature distributions.

        Args:
            new_data: New data to compare against reference

        Returns:
            DriftReport with drift analysis
        """
        if not self._reference_distributions:
            raise ValueError("No reference data set. Call set_reference() first.")

        feature_drifts = {}
        drifted_features = []

        for col in new_data.columns:
            if col not in self._reference_distributions:
                continue

            ref = self._reference_distributions[col]
            values = new_data[col].dropna().values

            if len(values) == 0:
                continue

            # Compute PSI (Population Stability Index)
            psi = self.calculate_psi(
                expected=ref['hist'],
                actual=values,
                bin_edges=ref['bin_edges']
            )

            feature_drifts[col] = psi

            if psi > self.FEATURE_DRIFT_THRESHOLD:
                drifted_features.append(col)

        # Calculate overall drift score
        if feature_drifts:
            overall_drift = np.mean(list(feature_drifts.values()))
        else:
            overall_drift = 0.0

        # Determine recommendation
        recommendation = self._get_recommendation(
            overall_drift,
            len(drifted_features),
            len(feature_drifts)
        )

        report = DriftReport(
            feature_drifts=feature_drifts,
            prediction_drift=0.0,  # Set by detect_prediction_drift
            overall_drift_score=overall_drift,
            drifted_features=drifted_features,
            recommendation=recommendation,
            details={
                'total_features_checked': len(feature_drifts),
                'drifted_feature_count': len(drifted_features),
                'drift_threshold': self.FEATURE_DRIFT_THRESHOLD
            }
        )

        logger.info(
            f"Feature drift detection: overall={overall_drift:.4f}, "
            f"drifted={len(drifted_features)}/{len(feature_drifts)}, "
            f"recommendation={recommendation}"
        )

        return report

    def detect_prediction_drift(
        self,
        new_predictions: np.ndarray
    ) -> DriftReport:
        """
        Detect drift in prediction distributions.

        Args:
            new_predictions: New predictions to compare

        Returns:
            DriftReport with prediction drift analysis
        """
        if self._prediction_distribution is None:
            raise ValueError("No reference predictions set. Call set_reference() first.")

        # Compute PSI for predictions
        psi = self.calculate_psi(
            expected=self._prediction_distribution['hist'],
            actual=new_predictions,
            bin_edges=self._prediction_distribution['bin_edges']
        )

        # Also calculate KL divergence
        kl_div = self._calculate_kl_divergence(
            self._prediction_distribution['hist'],
            new_predictions
        )

        # Determine recommendation
        if psi > self.PSI_THRESHOLD:
            recommendation = "retrain"
        elif psi > self.PSI_THRESHOLD / 2:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        report = DriftReport(
            feature_drifts={},
            prediction_drift=psi,
            overall_drift_score=psi,
            drifted_features=[],
            recommendation=recommendation,
            details={
                'psi': psi,
                'kl_divergence': kl_div,
                'psi_threshold': self.PSI_THRESHOLD
            }
        )

        logger.info(
            f"Prediction drift detection: PSI={psi:.4f}, KL={kl_div:.4f}, "
            f"recommendation={recommendation}"
        )

        return report

    def calculate_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Calculate KL divergence between two distributions.

        KL(P || Q) = sum(P * log(P / Q))

        Args:
            p: Reference distribution
            q: Comparison distribution
            epsilon: Small value to avoid log(0)

        Returns:
            KL divergence value
        """
        # Normalize
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)

        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q

        # Add epsilon to avoid division by zero
        p = np.clip(p, epsilon, None)
        q = np.clip(q, epsilon, None)

        # Calculate KL divergence
        kl = np.sum(p * np.log(p / q))

        return float(kl)

    def _calculate_kl_divergence(
        self,
        reference_hist: np.ndarray,
        new_data: np.ndarray
    ) -> float:
        """Calculate KL divergence between reference and new data."""
        # Compute histogram for new data with same bins
        new_hist, _ = np.histogram(
            new_data,
            bins=self.n_bins,
            density=True
        )

        return self.calculate_kl_divergence(reference_hist, new_hist)

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bin_edges: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))

        Args:
            expected: Expected distribution (histogram or raw values)
            actual: Actual values
            bin_edges: Bin edges for histogram (if expected is histogram)

        Returns:
            PSI value
        """
        epsilon = 1e-10

        # If expected is a histogram, compute histogram for actual with same bins
        if bin_edges is not None:
            actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
            expected_hist = expected
        else:
            # Both are raw values, compute histograms
            all_values = np.concatenate([expected.flatten(), actual.flatten()])
            bin_edges = np.histogram_bin_edges(all_values, bins=self.n_bins)

            expected_hist, _ = np.histogram(expected, bins=bin_edges, density=True)
            actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)

        # Normalize to proportions
        expected_pct = expected_hist / (expected_hist.sum() + epsilon)
        actual_pct = actual_hist / (actual_hist.sum() + epsilon)

        # Add epsilon to avoid division by zero
        expected_pct = np.clip(expected_pct, epsilon, None)
        actual_pct = np.clip(actual_pct, epsilon, None)

        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)

    def _get_recommendation(
        self,
        overall_drift: float,
        n_drifted: int,
        n_total: int
    ) -> str:
        """
        Get recommendation based on drift analysis.

        Args:
            overall_drift: Overall drift score
            n_drifted: Number of drifted features
            n_total: Total number of features

        Returns:
            Recommendation string
        """
        drift_ratio = n_drifted / n_total if n_total > 0 else 0

        if overall_drift > self.OVERALL_DRIFT_THRESHOLD or drift_ratio > 0.3:
            return "retrain"
        elif overall_drift > self.OVERALL_DRIFT_THRESHOLD / 2 or drift_ratio > 0.1:
            return "monitor"
        else:
            return "ok"

    def should_retrain(self, drift_report: DriftReport) -> bool:
        """
        Determine if model should be retrained based on drift report.

        Args:
            drift_report: DriftReport from drift detection

        Returns:
            True if model should be retrained
        """
        return drift_report.recommendation == "retrain"

    def detect_all_drift(
        self,
        new_data: pd.DataFrame,
        new_predictions: Optional[np.ndarray] = None
    ) -> DriftReport:
        """
        Detect both feature and prediction drift.

        Args:
            new_data: New feature data
            new_predictions: New predictions (optional)

        Returns:
            Combined DriftReport
        """
        # Feature drift
        feature_report = self.detect_feature_drift(new_data)

        # Prediction drift if predictions provided
        if new_predictions is not None and self._prediction_distribution is not None:
            pred_report = self.detect_prediction_drift(new_predictions)
            prediction_drift = pred_report.prediction_drift
        else:
            prediction_drift = 0.0

        # Combined score
        combined_drift = (
            feature_report.overall_drift_score * 0.7 +
            prediction_drift * 0.3
        )

        # Update recommendation based on combined drift
        if combined_drift > self.OVERALL_DRIFT_THRESHOLD:
            recommendation = "retrain"
        elif combined_drift > self.OVERALL_DRIFT_THRESHOLD / 2:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        return DriftReport(
            feature_drifts=feature_report.feature_drifts,
            prediction_drift=prediction_drift,
            overall_drift_score=combined_drift,
            drifted_features=feature_report.drifted_features,
            recommendation=recommendation,
            details={
                'feature_drift_score': feature_report.overall_drift_score,
                'prediction_drift_score': prediction_drift,
                'combined_drift_score': combined_drift
            }
        )


def detect_drift(
    reference_data: pd.DataFrame,
    new_data: pd.DataFrame,
    reference_predictions: Optional[np.ndarray] = None,
    new_predictions: Optional[np.ndarray] = None
) -> DriftReport:
    """
    Convenience function to detect drift.

    Args:
        reference_data: Reference feature data
        new_data: New feature data
        reference_predictions: Reference predictions
        new_predictions: New predictions

    Returns:
        DriftReport
    """
    detector = ConceptDriftDetector(
        reference_data=reference_data,
        reference_predictions=reference_predictions
    )

    return detector.detect_all_drift(new_data, new_predictions)
