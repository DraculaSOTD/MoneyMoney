"""
Ensemble Anomaly Detector combining multiple anomaly detection methods.

Integrates LOF, Isolation Forest, and Statistical methods for robust anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.anomaly_detection.lof_detector import LocalOutlierFactor
from models.advanced.anomaly_detection.isolation_forest import IsolationForest
from models.advanced.anomaly_detection.statistical_detector import StatisticalAnomalyDetector, AnomalyResult


@dataclass
class EnsembleAnomalyResult:
    """Result from ensemble anomaly detection."""
    is_anomaly: bool
    ensemble_score: float
    confidence: float
    method_scores: Dict[str, float]
    method_predictions: Dict[str, bool]
    consensus_level: float
    severity: str
    explanation: str
    individual_results: Dict[str, Any]


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods.
    
    Features:
    - Combines LOF, Isolation Forest, and Statistical methods
    - Weighted voting based on method performance
    - Adaptive thresholds based on data characteristics
    - Confidence estimation and uncertainty quantification
    - Feature-wise anomaly attribution
    - Real-time adaptation capabilities
    """
    
    def __init__(self,
                 methods: List[str] = None,
                 weights: Optional[Dict[str, float]] = None,
                 voting_strategy: str = 'weighted_average',
                 confidence_threshold: float = 0.6,
                 adaptation_rate: float = 0.01):
        """
        Initialize ensemble anomaly detector.
        
        Args:
            methods: List of methods to include in ensemble
            weights: Weights for each method
            voting_strategy: Strategy for combining predictions
            confidence_threshold: Minimum confidence for anomaly classification
            adaptation_rate: Rate of weight adaptation
        """
        self.methods = methods or ['lof', 'isolation_forest', 'statistical']
        self.voting_strategy = voting_strategy
        self.confidence_threshold = confidence_threshold
        self.adaptation_rate = adaptation_rate
        
        # Initialize method weights
        if weights is None:
            self.weights = {method: 1.0 / len(self.methods) for method in self.methods}
        else:
            self.weights = weights.copy()
            
        # Initialize detectors
        self.detectors = {}
        self._initialize_detectors()
        
        # Performance tracking
        self.method_performance = {method: [] for method in self.methods}
        self.training_data = None
        
        # Ensemble statistics
        self.ensemble_threshold = 0.5
        self.is_fitted = False
        
    def _initialize_detectors(self):
        """Initialize individual anomaly detectors."""
        if 'lof' in self.methods:
            self.detectors['lof'] = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1
            )
            
        if 'isolation_forest' in self.methods:
            self.detectors['isolation_forest'] = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                max_samples=256
            )
            
        if 'statistical' in self.methods:
            self.detectors['statistical'] = StatisticalAnomalyDetector(
                methods=['z_score', 'modified_z_score', 'iqr'],
                z_threshold=3.0
            )
            
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'EnsembleAnomalyDetector':
        """
        Fit ensemble on training data.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Optional labels for supervised adaptation
            
        Returns:
            Self
        """
        X = np.array(X)
        self.training_data = X.copy()
        
        print(f"Fitting ensemble with {len(self.methods)} methods on {X.shape[0]} samples")
        
        # Fit each detector
        for method, detector in self.detectors.items():\n            try:\n                print(f\"Fitting {method}...\")\n                if method == 'statistical':\n                    # Statistical methods don't need explicit fitting\n                    pass\n                else:\n                    detector.fit(X)\n                print(f\"  {method} fitted successfully\")\n            except Exception as e:\n                print(f\"Error fitting {method}: {e}\")\n                # Remove failed method\n                self.methods.remove(method)\n                if method in self.weights:\n                    del self.weights[method]\n                    \n        # Renormalize weights\n        if self.weights:\n            total_weight = sum(self.weights.values())\n            for method in self.weights:\n                self.weights[method] /= total_weight\n                \n        # Compute ensemble threshold on training data\n        if len(self.methods) > 0:\n            train_scores = self._compute_ensemble_scores(X)\n            self.ensemble_threshold = np.percentile(train_scores, 90)  # Top 10% as anomalies\n            \n        self.is_fitted = True\n        return self\n    \n    def predict(self, X: np.ndarray) -> np.ndarray:\n        \"\"\"Predict anomalies for new data.\"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Ensemble must be fitted before prediction\")\n            \n        results = self.detect_anomalies(X)\n        return np.array([1 if not result.is_anomaly else -1 for result in results])\n    \n    def detect_anomalies(self, X: np.ndarray) -> List[EnsembleAnomalyResult]:\n        \"\"\"Detect anomalies using ensemble approach.\"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Ensemble must be fitted before detection\")\n            \n        X = np.array(X)\n        n_samples = X.shape[0]\n        \n        # Get predictions from each method\n        method_results = {}\n        \n        for method in self.methods:\n            try:\n                method_results[method] = self._get_method_results(method, X)\n            except Exception as e:\n                print(f\"Error in {method}: {e}\")\n                # Skip this method\n                continue\n                \n        # Combine results\n        ensemble_results = []\n        \n        for i in range(n_samples):\n            result = self._combine_predictions(i, method_results)\n            ensemble_results.append(result)\n            \n        return ensemble_results\n    \n    def _get_method_results(self, method: str, X: np.ndarray) -> Dict[str, np.ndarray]:\n        \"\"\"Get results from individual method.\"\"\"\n        detector = self.detectors[method]\n        \n        if method == 'lof':\n            scores = detector.decision_function(X)\n            predictions = detector.predict(X)\n            return {\n                'scores': scores,\n                'predictions': predictions == -1,  # Convert to boolean\n                'details': [detector.explain_anomaly(x.reshape(1, -1)) for x in X[:5]]  # Sample explanations\n            }\n            \n        elif method == 'isolation_forest':\n            scores = detector.decision_function(X)\n            predictions = detector.predict(X)\n            return {\n                'scores': 1 - scores,  # Invert scores (higher = more anomalous)\n                'predictions': predictions == -1,\n                'details': [detector.explain_anomaly(x) for x in X[:5]]\n            }\n            \n        elif method == 'statistical':\n            # For multivariate data, apply to each feature and combine\n            if X.ndim == 1 or X.shape[1] == 1:\n                data = X.flatten() if X.ndim > 1 else X\n                stat_results = detector.detect_anomalies(data, method='ensemble')\n                scores = np.array([r.score for r in stat_results])\n                predictions = np.array([r.is_anomaly for r in stat_results])\n            else:\n                # Multivariate: apply to each feature and combine\n                feature_scores = []\n                feature_predictions = []\n                \n                for feature_idx in range(X.shape[1]):\n                    feature_data = X[:, feature_idx]\n                    stat_results = detector.detect_anomalies(feature_data, method='ensemble')\n                    feature_scores.append([r.score for r in stat_results])\n                    feature_predictions.append([r.is_anomaly for r in stat_results])\n                    \n                # Combine across features\n                scores = np.mean(feature_scores, axis=0)\n                predictions = np.any(feature_predictions, axis=0)\n                \n            return {\n                'scores': scores,\n                'predictions': predictions,\n                'details': []  # Statistical method details\n            }\n            \n        else:\n            raise ValueError(f\"Unknown method: {method}\")\n    \n    def _combine_predictions(self, sample_idx: int, method_results: Dict[str, Dict]) -> EnsembleAnomalyResult:\n        \"\"\"Combine predictions from all methods for a single sample.\"\"\"\n        method_scores = {}\n        method_predictions = {}\n        \n        # Extract scores and predictions for this sample\n        for method, results in method_results.items():\n            if sample_idx < len(results['scores']):\n                method_scores[method] = float(results['scores'][sample_idx])\n                method_predictions[method] = bool(results['predictions'][sample_idx])\n            else:\n                method_scores[method] = 0.0\n                method_predictions[method] = False\n                \n        # Apply voting strategy\n        ensemble_score, is_anomaly, confidence = self._apply_voting_strategy(\n            method_scores, method_predictions\n        )\n        \n        # Calculate consensus level\n        consensus_level = self._calculate_consensus(method_predictions)\n        \n        # Determine severity\n        severity = self._determine_severity(ensemble_score, consensus_level)\n        \n        # Generate explanation\n        explanation = self._generate_explanation(\n            method_scores, method_predictions, is_anomaly\n        )\n        \n        return EnsembleAnomalyResult(\n            is_anomaly=is_anomaly,\n            ensemble_score=ensemble_score,\n            confidence=confidence,\n            method_scores=method_scores,\n            method_predictions=method_predictions,\n            consensus_level=consensus_level,\n            severity=severity,\n            explanation=explanation,\n            individual_results={method: results for method, results in method_results.items()}\n        )\n    \n    def _apply_voting_strategy(self, \n                              method_scores: Dict[str, float],\n                              method_predictions: Dict[str, bool]) -> Tuple[float, bool, float]:\n        \"\"\"Apply voting strategy to combine method results.\"\"\"\n        if self.voting_strategy == 'majority':\n            # Simple majority voting\n            votes = sum(method_predictions.values())\n            is_anomaly = votes > len(method_predictions) / 2\n            ensemble_score = votes / len(method_predictions)\n            confidence = abs(ensemble_score - 0.5) * 2  # Distance from 0.5\n            \n        elif self.voting_strategy == 'weighted_average':\n            # Weighted average of scores\n            weighted_score = 0.0\n            total_weight = 0.0\n            \n            for method, score in method_scores.items():\n                weight = self.weights.get(method, 1.0)\n                weighted_score += weight * score\n                total_weight += weight\n                \n            if total_weight > 0:\n                ensemble_score = weighted_score / total_weight\n            else:\n                ensemble_score = 0.5\n                \n            is_anomaly = ensemble_score > self.ensemble_threshold\n            confidence = min(1.0, abs(ensemble_score - self.ensemble_threshold) / self.ensemble_threshold)\n            \n        elif self.voting_strategy == 'max':\n            # Maximum score across methods\n            ensemble_score = max(method_scores.values()) if method_scores else 0.0\n            is_anomaly = any(method_predictions.values())\n            confidence = ensemble_score\n            \n        elif self.voting_strategy == 'consensus':\n            # Require consensus (all methods agree)\n            is_anomaly = all(method_predictions.values())\n            ensemble_score = np.mean(list(method_scores.values())) if method_scores else 0.0\n            confidence = ensemble_score if is_anomaly else 1 - ensemble_score\n            \n        else:\n            raise ValueError(f\"Unknown voting strategy: {self.voting_strategy}\")\n            \n        return ensemble_score, is_anomaly, confidence\n    \n    def _calculate_consensus(self, method_predictions: Dict[str, bool]) -> float:\n        \"\"\"Calculate consensus level among methods.\"\"\"\n        if not method_predictions:\n            return 0.0\n            \n        # Count agreements\n        positive_votes = sum(method_predictions.values())\n        total_votes = len(method_predictions)\n        \n        # Consensus is how far from 50-50 split\n        vote_ratio = positive_votes / total_votes\n        consensus = abs(vote_ratio - 0.5) * 2\n        \n        return consensus\n    \n    def _determine_severity(self, ensemble_score: float, consensus_level: float) -> str:\n        \"\"\"Determine anomaly severity.\"\"\"\n        if ensemble_score > 0.8 and consensus_level > 0.8:\n            return 'critical'\n        elif ensemble_score > 0.7 and consensus_level > 0.6:\n            return 'high'\n        elif ensemble_score > 0.6 and consensus_level > 0.4:\n            return 'medium'\n        elif ensemble_score > 0.5:\n            return 'low'\n        else:\n            return 'normal'\n    \n    def _generate_explanation(self,\n                            method_scores: Dict[str, float],\n                            method_predictions: Dict[str, bool],\n                            is_anomaly: bool) -> str:\n        \"\"\"Generate human-readable explanation.\"\"\"\n        if not is_anomaly:\n            return \"No anomaly detected - all methods indicate normal behavior\"\n            \n        # Find which methods detected anomaly\n        detecting_methods = [method for method, pred in method_predictions.items() if pred]\n        \n        if len(detecting_methods) == len(method_predictions):\n            explanation = f\"Strong anomaly detected by all methods ({', '.join(detecting_methods)})\"\n        elif len(detecting_methods) > len(method_predictions) / 2:\n            explanation = f\"Anomaly detected by majority of methods: {', '.join(detecting_methods)}\"\n        else:\n            explanation = f\"Anomaly detected by: {', '.join(detecting_methods)}\"\n            \n        # Add score information\n        max_score_method = max(method_scores, key=method_scores.get)\n        max_score = method_scores[max_score_method]\n        \n        explanation += f\". Highest anomaly score: {max_score:.3f} ({max_score_method})\"\n        \n        return explanation\n    \n    def _compute_ensemble_scores(self, X: np.ndarray) -> np.ndarray:\n        \"\"\"Compute ensemble scores for threshold determination.\"\"\"\n        method_results = {}\n        \n        for method in self.methods:\n            try:\n                method_results[method] = self._get_method_results(method, X)\n            except Exception as e:\n                print(f\"Error computing scores for {method}: {e}\")\n                continue\n                \n        scores = []\n        for i in range(X.shape[0]):\n            method_scores = {}\n            for method, results in method_results.items():\n                if i < len(results['scores']):\n                    method_scores[method] = results['scores'][i]\n                    \n            # Weighted average\n            weighted_score = 0.0\n            total_weight = 0.0\n            \n            for method, score in method_scores.items():\n                weight = self.weights.get(method, 1.0)\n                weighted_score += weight * score\n                total_weight += weight\n                \n            if total_weight > 0:\n                ensemble_score = weighted_score / total_weight\n            else:\n                ensemble_score = 0.0\n                \n            scores.append(ensemble_score)\n            \n        return np.array(scores)\n    \n    def update_weights(self, true_labels: np.ndarray, predictions: List[EnsembleAnomalyResult]):\n        \"\"\"Update method weights based on performance.\"\"\"\n        if len(true_labels) != len(predictions):\n            raise ValueError(\"Length mismatch between labels and predictions\")\n            \n        # Calculate performance for each method\n        method_accuracies = {}\n        \n        for method in self.methods:\n            correct = 0\n            total = 0\n            \n            for i, (true_label, pred) in enumerate(zip(true_labels, predictions)):\n                if method in pred.method_predictions:\n                    method_pred = pred.method_predictions[method]\n                    true_anomaly = true_label == -1\n                    \n                    if method_pred == true_anomaly:\n                        correct += 1\n                    total += 1\n                    \n            if total > 0:\n                accuracy = correct / total\n                method_accuracies[method] = accuracy\n                self.method_performance[method].append(accuracy)\n                \n        # Update weights using exponential moving average\n        if method_accuracies:\n            max_accuracy = max(method_accuracies.values())\n            \n            for method in self.methods:\n                if method in method_accuracies:\n                    # Relative performance\n                    relative_perf = method_accuracies[method] / (max_accuracy + 1e-8)\n                    \n                    # Update weight\n                    current_weight = self.weights.get(method, 1.0)\n                    new_weight = (1 - self.adaptation_rate) * current_weight + self.adaptation_rate * relative_perf\n                    self.weights[method] = new_weight\n                    \n            # Renormalize weights\n            total_weight = sum(self.weights.values())\n            if total_weight > 0:\n                for method in self.weights:\n                    self.weights[method] /= total_weight\n    \n    def get_feature_importance(self, X: np.ndarray) -> Dict[str, np.ndarray]:\n        \"\"\"Get feature importance from each method.\"\"\"\n        importance_scores = {}\n        \n        if 'isolation_forest' in self.detectors:\n            iso_importance = self.detectors['isolation_forest'].feature_importance(X)\n            importance_scores['isolation_forest'] = iso_importance\n            \n        # For LOF and statistical methods, compute based on contribution analysis\n        if len(importance_scores) == 0:\n            # Default uniform importance\n            n_features = X.shape[1] if X.ndim > 1 else 1\n            importance_scores['uniform'] = np.ones(n_features) / n_features\n            \n        return importance_scores\n    \n    def explain_sample(self, x: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict:\n        \"\"\"Provide detailed explanation for a single sample.\"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Ensemble must be fitted before explanation\")\n            \n        x = np.array(x).reshape(1, -1)\n        results = self.detect_anomalies(x)\n        \n        if not results:\n            return {'error': 'No results generated'}\n            \n        result = results[0]\n        \n        explanation = {\n            'is_anomaly': result.is_anomaly,\n            'ensemble_score': result.ensemble_score,\n            'confidence': result.confidence,\n            'severity': result.severity,\n            'consensus_level': result.consensus_level,\n            'explanation': result.explanation,\n            'method_breakdown': {\n                method: {\n                    'score': result.method_scores.get(method, 0),\n                    'prediction': result.method_predictions.get(method, False),\n                    'weight': self.weights.get(method, 0)\n                }\n                for method in self.methods\n            }\n        }\n        \n        # Add feature analysis if available\n        if feature_names and len(feature_names) == x.shape[1]:\n            try:\n                feature_importance = self.get_feature_importance(x)\n                explanation['feature_analysis'] = {\n                    method: {\n                        'feature_importance': {\n                            feature_names[i]: float(importance[i])\n                            for i in range(len(feature_names))\n                            if i < len(importance)\n                        }\n                    }\n                    for method, importance in feature_importance.items()\n                }\n            except Exception as e:\n                explanation['feature_analysis'] = {'error': str(e)}\n                \n        return explanation\n    \n    def get_ensemble_stats(self) -> Dict:\n        \"\"\"Get ensemble statistics and performance metrics.\"\"\"\n        stats = {\n            'methods': self.methods,\n            'weights': self.weights.copy(),\n            'voting_strategy': self.voting_strategy,\n            'ensemble_threshold': self.ensemble_threshold,\n            'confidence_threshold': self.confidence_threshold,\n            'is_fitted': self.is_fitted\n        }\n        \n        if self.training_data is not None:\n            stats['training_data_shape'] = self.training_data.shape\n            \n        # Method performance history\n        if any(self.method_performance.values()):\n            stats['method_performance'] = {\n                method: {\n                    'recent_accuracy': perf[-10:] if perf else [],\n                    'avg_accuracy': np.mean(perf) if perf else 0,\n                    'std_accuracy': np.std(perf) if perf else 0\n                }\n                for method, perf in self.method_performance.items()\n            }\n            \n        return stats\n    \n    def save_model(self, filepath: str):\n        \"\"\"Save ensemble model to file.\"\"\"\n        import pickle\n        \n        model_data = {\n            'detectors': self.detectors,\n            'weights': self.weights,\n            'methods': self.methods,\n            'voting_strategy': self.voting_strategy,\n            'confidence_threshold': self.confidence_threshold,\n            'ensemble_threshold': self.ensemble_threshold,\n            'method_performance': dict(self.method_performance),\n            'is_fitted': self.is_fitted\n        }\n        \n        with open(filepath, 'wb') as f:\n            pickle.dump(model_data, f)\n            \n    def load_model(self, filepath: str):\n        \"\"\"Load ensemble model from file.\"\"\"\n        import pickle\n        \n        with open(filepath, 'rb') as f:\n            model_data = pickle.load(f)\n            \n        self.detectors = model_data['detectors']\n        self.weights = model_data['weights']\n        self.methods = model_data['methods']\n        self.voting_strategy = model_data['voting_strategy']\n        self.confidence_threshold = model_data['confidence_threshold']\n        self.ensemble_threshold = model_data['ensemble_threshold']\n        self.method_performance = model_data['method_performance']\n        self.is_fitted = model_data['is_fitted']"