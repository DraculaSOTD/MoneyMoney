"""
Local Outlier Factor (LOF) Implementation for Anomaly Detection.

Implements LOF algorithm for detecting local density-based anomalies in trading data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import heapq
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class LocalOutlierFactor:
    """
    Local Outlier Factor (LOF) for anomaly detection.
    
    LOF compares the local density of a point with the local densities of its neighbors.
    Points that have substantially lower density than their neighbors are considered outliers.
    
    Features:
    - Density-based anomaly detection
    - Handles local anomalies well
    - Robust to global density variations
    - Configurable neighborhood size
    - Real-time scoring capability
    """
    
    def __init__(self,
                 n_neighbors: int = 20,
                 contamination: float = 0.1,
                 distance_metric: str = 'euclidean',
                 leaf_size: int = 30):
        """
        Initialize LOF detector.
        
        Args:
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of outliers
            distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            leaf_size: Leaf size for efficient neighbor search
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.distance_metric = distance_metric
        self.leaf_size = leaf_size
        
        # Training data storage
        self.training_data = None
        self.n_samples = 0
        self.n_features = 0
        
        # Precomputed values
        self.knn_distances = None
        self.knn_indices = None
        self.lrd_values = None
        self.lof_scores = None
        self.threshold = None
        
        # Distance function
        self.distance_fn = self._get_distance_function()
        
    def _get_distance_function(self):
        """Get distance function based on metric."""
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance
        elif self.distance_metric == 'cosine':
            return self._cosine_distance
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Manhattan distance."""
        return np.sum(np.abs(x1 - x2))
    
    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Cosine distance."""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0
            
        similarity = dot_product / (norm_x1 * norm_x2)
        return 1.0 - similarity
    
    def fit(self, X: np.ndarray) -> 'LocalOutlierFactor':
        """
        Fit LOF detector on training data.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Self
        """
        X = np.array(X)
        self.training_data = X.copy()
        self.n_samples, self.n_features = X.shape
        
        if self.n_neighbors >= self.n_samples:
            self.n_neighbors = max(1, self.n_samples - 1)
            
        print(f"Fitting LOF with {self.n_samples} samples, {self.n_neighbors} neighbors")
        
        # Compute k-nearest neighbors
        self.knn_distances, self.knn_indices = self._compute_knn(X)
        
        # Compute local reachability density
        self.lrd_values = self._compute_lrd(X)
        
        # Compute LOF scores
        self.lof_scores = self._compute_lof(X)
        
        # Determine threshold based on contamination
        self.threshold = np.percentile(self.lof_scores, 100 * (1 - self.contamination))
        
        return self
    
    def _compute_knn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute k-nearest neighbors for all points."""
        n_samples = X.shape[0]
        knn_distances = np.zeros((n_samples, self.n_neighbors))
        knn_indices = np.zeros((n_samples, self.n_neighbors), dtype=int)
        
        for i in range(n_samples):
            # Compute distances to all other points
            distances = []
            for j in range(n_samples):
                if i != j:
                    dist = self.distance_fn(X[i], X[j])
                    distances.append((dist, j))
                    
            # Sort by distance and take k nearest
            distances.sort()
            
            # Handle case where we have fewer points than n_neighbors
            k = min(self.n_neighbors, len(distances))
            
            for idx in range(k):
                knn_distances[i, idx] = distances[idx][0]
                knn_indices[i, idx] = distances[idx][1]
                
            # Fill remaining with the farthest neighbor
            if k < self.n_neighbors:
                for idx in range(k, self.n_neighbors):
                    knn_distances[i, idx] = distances[k-1][0] if distances else 0
                    knn_indices[i, idx] = distances[k-1][1] if distances else i
                    
        return knn_distances, knn_indices
    
    def _compute_lrd(self, X: np.ndarray) -> np.ndarray:
        """Compute Local Reachability Density for all points."""
        n_samples = X.shape[0]
        lrd_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get k-distance of point i
            k_distance_i = self.knn_distances[i, -1]  # Distance to k-th neighbor
            
            # Compute reachability distances to neighbors
            reachability_distances = []
            
            for j in range(self.n_neighbors):
                neighbor_idx = self.knn_indices[i, j]
                
                # k-distance of neighbor
                k_distance_neighbor = self.knn_distances[neighbor_idx, -1]
                
                # Distance from i to neighbor
                direct_distance = self.knn_distances[i, j]
                
                # Reachability distance
                reachability_dist = max(direct_distance, k_distance_neighbor)
                reachability_distances.append(reachability_dist)
                
            # Local reachability density
            if len(reachability_distances) > 0:
                avg_reachability = np.mean(reachability_distances)
                if avg_reachability > 0:
                    lrd_values[i] = 1.0 / avg_reachability
                else:
                    lrd_values[i] = float('inf')
            else:
                lrd_values[i] = 0.0
                
        return lrd_values
    
    def _compute_lof(self, X: np.ndarray) -> np.ndarray:
        """Compute Local Outlier Factor for all points."""
        n_samples = X.shape[0]
        lof_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get LRD of neighbors
            neighbor_lrds = []
            
            for j in range(self.n_neighbors):
                neighbor_idx = self.knn_indices[i, j]
                neighbor_lrds.append(self.lrd_values[neighbor_idx])
                
            # LOF is ratio of average neighbor LRD to own LRD
            if self.lrd_values[i] > 0 and len(neighbor_lrds) > 0:
                avg_neighbor_lrd = np.mean(neighbor_lrds)
                lof_scores[i] = avg_neighbor_lrd / self.lrd_values[i]
            else:
                lof_scores[i] = 1.0
                
        return lof_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies for new data.
        
        Args:
            X: Data to predict (n_samples, n_features)
            
        Returns:
            Anomaly predictions (1 for normal, -1 for anomaly)
        """
        if self.training_data is None:
            raise ValueError("Model must be fitted before prediction")
            
        lof_scores = self.decision_function(X)
        return np.where(lof_scores > self.threshold, -1, 1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute LOF scores for new data.
        
        Args:
            X: Data to score (n_samples, n_features)
            
        Returns:
            LOF scores
        """
        if self.training_data is None:
            raise ValueError("Model must be fitted before scoring")
            
        X = np.array(X)
        n_test_samples = X.shape[0]
        lof_scores = np.zeros(n_test_samples)
        
        for i in range(n_test_samples):
            lof_scores[i] = self._compute_lof_single(X[i])
            
        return lof_scores
    
    def _compute_lof_single(self, x: np.ndarray) -> float:
        """Compute LOF score for a single point."""
        # Find k-nearest neighbors in training data
        distances = []
        for j in range(self.n_samples):
            dist = self.distance_fn(x, self.training_data[j])
            distances.append((dist, j))
            
        distances.sort()
        k = min(self.n_neighbors, len(distances))
        
        # Get k-distance
        k_distance = distances[k-1][0]
        
        # Compute reachability distances to neighbors
        reachability_distances = []
        neighbor_indices = []
        
        for idx in range(k):
            neighbor_idx = distances[idx][1]
            neighbor_indices.append(neighbor_idx)
            
            # k-distance of neighbor from training
            neighbor_k_distance = self.knn_distances[neighbor_idx, -1]
            
            # Reachability distance
            reachability_dist = max(distances[idx][0], neighbor_k_distance)
            reachability_distances.append(reachability_dist)
            
        # Local reachability density
        if len(reachability_distances) > 0:
            avg_reachability = np.mean(reachability_distances)
            if avg_reachability > 0:
                lrd = 1.0 / avg_reachability
            else:
                lrd = float('inf')
        else:
            return 1.0
            
        # LOF computation
        neighbor_lrds = [self.lrd_values[idx] for idx in neighbor_indices]
        
        if lrd > 0 and len(neighbor_lrds) > 0:
            avg_neighbor_lrd = np.mean(neighbor_lrds)
            lof_score = avg_neighbor_lrd / lrd
        else:
            lof_score = 1.0
            
        return lof_score
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and predict anomalies in one step.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Anomaly predictions (1 for normal, -1 for anomaly)
        """
        self.fit(X)
        return np.where(self.lof_scores > self.threshold, -1, 1)
    
    def get_anomaly_details(self, X: np.ndarray) -> List[Dict]:
        """
        Get detailed anomaly information.
        
        Args:
            X: Data to analyze
            
        Returns:
            List of anomaly details for each point
        """
        lof_scores = self.decision_function(X)
        predictions = self.predict(X)
        
        details = []
        for i, (score, pred) in enumerate(zip(lof_scores, predictions)):
            detail = {
                'index': i,
                'lof_score': float(score),
                'is_anomaly': pred == -1,
                'anomaly_strength': max(0, score - 1.0),  # How much above normal
                'confidence': min(1.0, abs(score - self.threshold) / self.threshold)
            }
            
            # Add interpretation
            if score > 2.0:
                detail['severity'] = 'high'
                detail['interpretation'] = 'Strong outlier - significantly different from neighbors'
            elif score > 1.5:
                detail['severity'] = 'medium'
                detail['interpretation'] = 'Moderate outlier - somewhat different from neighbors'
            elif score > 1.2:
                detail['severity'] = 'low'
                detail['interpretation'] = 'Weak outlier - slightly different from neighbors'
            else:
                detail['severity'] = 'normal'
                detail['interpretation'] = 'Normal point - similar to neighbors'
                
            details.append(detail)
            
        return details
    
    def explain_anomaly(self, x: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict:
        """
        Explain why a point is considered anomalous.
        
        Args:
            x: Single data point to explain
            feature_names: Names of features
            
        Returns:
            Explanation dictionary
        """
        if self.training_data is None:
            raise ValueError("Model must be fitted before explanation")
            
        # Compute LOF score
        lof_score = self._compute_lof_single(x)
        
        # Find nearest neighbors
        distances = []
        for j in range(self.n_samples):
            dist = self.distance_fn(x, self.training_data[j])
            distances.append((dist, j))
            
        distances.sort()
        k = min(self.n_neighbors, len(distances))
        
        # Analyze feature contributions
        feature_contributions = np.zeros(self.n_features)
        
        for idx in range(k):
            neighbor_idx = distances[idx][1]
            neighbor = self.training_data[neighbor_idx]
            
            # Feature-wise distances
            feature_diffs = np.abs(x - neighbor)
            feature_contributions += feature_diffs
            
        # Normalize by number of neighbors
        feature_contributions /= k
        
        # Create explanation
        explanation = {
            'lof_score': float(lof_score),
            'is_anomaly': lof_score > self.threshold,
            'threshold': float(self.threshold),
            'nearest_neighbors': [
                {
                    'distance': float(distances[i][0]),
                    'index': int(distances[i][1])
                }
                for i in range(min(5, k))  # Top 5 neighbors
            ],
            'feature_contributions': feature_contributions.tolist()
        }
        
        if feature_names:
            explanation['top_contributing_features'] = [
                {
                    'feature': feature_names[i],
                    'contribution': float(feature_contributions[i])
                }
                for i in np.argsort(feature_contributions)[::-1][:5]
            ]
            
        return explanation
    
    def update_model(self, X_new: np.ndarray, retrain_threshold: float = 0.1):
        """
        Update model with new data.
        
        Args:
            X_new: New training data
            retrain_threshold: Fraction of new data to trigger full retrain
        """
        if self.training_data is None:
            self.fit(X_new)
            return
            
        # Add new data to training set
        old_size = self.training_data.shape[0]
        self.training_data = np.vstack([self.training_data, X_new])
        new_size = self.training_data.shape[0]
        
        # Decide whether to retrain
        if (new_size - old_size) / old_size > retrain_threshold:
            print(f"Retraining LOF model with {new_size} samples")
            self.fit(self.training_data)
        else:
            # Incremental update (simplified)
            print(f"Incremental update with {new_size - old_size} new samples")
            self.n_samples = new_size
            
    def get_model_stats(self) -> Dict:
        """Get model statistics and diagnostics."""
        if self.training_data is None:
            return {}
            
        stats = {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_neighbors': self.n_neighbors,
            'contamination': self.contamination,
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'distance_metric': self.distance_metric
        }
        
        if self.lof_scores is not None:
            stats.update({
                'mean_lof_score': float(np.mean(self.lof_scores)),
                'std_lof_score': float(np.std(self.lof_scores)),
                'min_lof_score': float(np.min(self.lof_scores)),
                'max_lof_score': float(np.max(self.lof_scores)),
                'median_lof_score': float(np.median(self.lof_scores))
            })
            
        if self.lrd_values is not None:
            stats.update({
                'mean_lrd': float(np.mean(self.lrd_values[self.lrd_values != float('inf')])),
                'std_lrd': float(np.std(self.lrd_values[self.lrd_values != float('inf')]))
            })
            
        return stats