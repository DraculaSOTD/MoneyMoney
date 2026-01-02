"""
Isolation Forest Implementation for Anomaly Detection.

Implements Isolation Forest algorithm for detecting anomalies in high-dimensional data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


class IsolationTree:
    """
    Single Isolation Tree for the Isolation Forest.
    """
    
    def __init__(self, max_depth: int = 10):
        """
        Initialize Isolation Tree.
        
        Args:
            max_depth: Maximum depth of the tree
        """
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X: np.ndarray) -> 'IsolationTree':
        """Build the isolation tree."""
        self.root = self._build_tree(X, depth=0)
        return self
        
    def _build_tree(self, X: np.ndarray, depth: int) -> Dict[str, Any]:
        """Recursively build the tree."""
        n_samples, n_features = X.shape
        
        # Terminal conditions
        if depth >= self.max_depth or n_samples <= 1:
            return {
                'type': 'leaf',
                'size': n_samples,
                'depth': depth
            }
            
        # Randomly select feature and split value
        feature_idx = np.random.randint(0, n_features)
        feature_values = X[:, feature_idx]
        
        # Handle case where all values are the same
        if np.all(feature_values == feature_values[0]):
            return {
                'type': 'leaf',
                'size': n_samples,
                'depth': depth
            }
            
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        
        # Random split value
        split_value = np.random.uniform(min_val, max_val)
        
        # Split data
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        # Handle edge cases
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {
                'type': 'leaf',
                'size': n_samples,
                'depth': depth
            }
            
        # Recursive building
        left_child = self._build_tree(X[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], depth + 1)
        
        return {
            'type': 'internal',
            'feature_idx': feature_idx,
            'split_value': split_value,
            'left': left_child,
            'right': right_child,
            'depth': depth
        }
    
    def path_length(self, x: np.ndarray) -> float:
        """Compute path length for a single sample."""
        if self.root is None:
            return 0.0
            
        return self._traverse(x, self.root)
    
    def _traverse(self, x: np.ndarray, node: Dict[str, Any]) -> float:
        """Traverse tree to compute path length."""
        if node['type'] == 'leaf':
            # Add adjustment for incomplete trees
            size = node['size']
            if size <= 1:
                return node['depth']
            else:
                # Average path length in BST with 'size' nodes
                return node['depth'] + self._average_path_length(size)
                
        # Internal node
        feature_val = x[node['feature_idx']]
        if feature_val < node['split_value']:
            return self._traverse(x, node['left'])
        else:
            return self._traverse(x, node['right'])
            
    def _average_path_length(self, n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0.0
        elif n == 2:
            return 1.0
        else:
            # Harmonic number approximation
            return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


class IsolationForest:
    """
    Isolation Forest for anomaly detection.
    
    Isolation Forest isolates anomalies by randomly selecting features and split values.
    Anomalies are isolated closer to the root of the tree.
    
    Features:
    - Efficient for high-dimensional data
    - Linear time complexity
    - No assumptions about data distribution
    - Good performance on various anomaly types
    - Minimal parameter tuning required
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_samples: int = 256,
                 contamination: float = 0.1,
                 max_features: float = 1.0,
                 bootstrap: bool = False,
                 random_state: Optional[int] = None):
        """
        Initialize Isolation Forest.
        
        Args:
            n_estimators: Number of isolation trees
            max_samples: Maximum number of samples for each tree
            contamination: Expected proportion of outliers
            max_features: Fraction of features to use for each tree
            bootstrap: Whether to use bootstrap sampling
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Model components
        self.trees: List[IsolationTree] = []
        self.feature_indices: List[np.ndarray] = []
        self.training_data = None
        self.threshold = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
            
    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """
        Fit Isolation Forest on training data.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Self
        """
        X = np.array(X)
        self.training_data = X.copy()
        n_samples, n_features = X.shape
        
        # Determine actual max_samples
        actual_max_samples = min(self.max_samples, n_samples)
        
        # Determine number of features per tree
        n_features_per_tree = max(1, int(self.max_features * n_features))
        
        print(f"Fitting Isolation Forest with {self.n_estimators} trees, "
              f"{actual_max_samples} samples per tree, {n_features_per_tree} features per tree")
        
        # Build trees
        self.trees = []
        self.feature_indices = []
        
        for i in range(self.n_estimators):
            # Sample data for this tree
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, actual_max_samples, replace=True)
            else:
                sample_indices = np.random.choice(n_samples, actual_max_samples, replace=False)
                
            # Sample features for this tree
            feature_indices = np.random.choice(n_features, n_features_per_tree, replace=False)
            
            # Get subset of data
            X_sample = X[sample_indices][:, feature_indices]
            
            # Build tree
            tree = IsolationTree(max_depth=int(np.ceil(np.log2(actual_max_samples))))
            tree.fit(X_sample)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_indices)
            
        # Compute threshold based on contamination
        scores = self.decision_function(X)
        self.threshold = np.percentile(scores, 100 * self.contamination)
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Args:
            X: Data to score (n_samples, n_features)
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.trees:
            raise ValueError("Model must be fitted before scoring")
            
        X = np.array(X)
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        # Compute average path length across all trees
        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feature_idx]
            
            for j in range(n_samples):
                path_length = tree.path_length(X_subset[j])
                scores[j] += path_length
                
        # Average and normalize
        scores /= len(self.trees)
        
        # Convert to anomaly score (higher = more anomalous)
        # Normalization factor based on average path length in BST
        n_train_samples = min(self.max_samples, self.training_data.shape[0])
        normalization_factor = self._average_path_length(n_train_samples)
        
        if normalization_factor > 0:
            normalized_scores = scores / normalization_factor
            # Convert to anomaly score: 2^(-s)
            anomaly_scores = np.power(2, -normalized_scores)
        else:
            anomaly_scores = np.ones_like(scores)
            
        return anomaly_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies for new data.
        
        Args:
            X: Data to predict (n_samples, n_features)
            
        Returns:
            Anomaly predictions (1 for normal, -1 for anomaly)
        """
        scores = self.decision_function(X)
        return np.where(scores >= self.threshold, 1, -1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and predict anomalies in one step.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Anomaly predictions (1 for normal, -1 for anomaly)
        """
        self.fit(X)
        return self.predict(X)
    
    def _average_path_length(self, n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0.0
        elif n == 2:
            return 1.0
        else:
            return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
    
    def get_anomaly_details(self, X: np.ndarray) -> List[Dict]:
        """
        Get detailed anomaly information.
        
        Args:
            X: Data to analyze
            
        Returns:
            List of anomaly details for each point
        """
        scores = self.decision_function(X)
        predictions = self.predict(X)
        
        details = []
        for i, (score, pred) in enumerate(zip(scores, predictions)):
            detail = {
                'index': i,
                'anomaly_score': float(score),
                'is_anomaly': pred == -1,
                'threshold': float(self.threshold),
                'anomaly_strength': max(0, self.threshold - score),
                'confidence': min(1.0, abs(score - self.threshold) / max(self.threshold, 1e-8))
            }
            
            # Add interpretation
            if score < 0.5:
                detail['severity'] = 'high'
                detail['interpretation'] = 'Strong anomaly - very isolated from normal data'
            elif score < 0.6:
                detail['severity'] = 'medium'
                detail['interpretation'] = 'Moderate anomaly - somewhat isolated'
            elif score < 0.7:
                detail['severity'] = 'low'
                detail['interpretation'] = 'Weak anomaly - slightly isolated'
            else:
                detail['severity'] = 'normal'
                detail['interpretation'] = 'Normal point - not isolated'
                
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
        if not self.trees:
            raise ValueError("Model must be fitted before explanation")
            
        x = np.array(x)
        
        # Compute anomaly score
        anomaly_score = self.decision_function(x.reshape(1, -1))[0]
        
        # Analyze feature contributions
        feature_contributions = np.zeros(len(x))
        feature_usage_count = np.zeros(len(x))
        
        # Track path lengths per feature
        for tree, feature_idx in zip(self.trees, self.feature_indices):
            x_subset = x[feature_idx]
            path_length = tree.path_length(x_subset)
            
            # Distribute path length contribution across used features
            for idx in feature_idx:
                feature_contributions[idx] += path_length
                feature_usage_count[idx] += 1
                
        # Average contributions
        mask = feature_usage_count > 0
        feature_contributions[mask] /= feature_usage_count[mask]
        
        # Normalize contributions
        if np.sum(feature_contributions) > 0:
            feature_contributions /= np.sum(feature_contributions)
            
        explanation = {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': anomaly_score < self.threshold,
            'threshold': float(self.threshold),
            'average_path_length': float(np.mean([
                tree.path_length(x[feature_idx]) 
                for tree, feature_idx in zip(self.trees, self.feature_indices)
            ])),
            'feature_contributions': feature_contributions.tolist()
        }
        
        if feature_names:
            explanation['top_contributing_features'] = [
                {
                    'feature': feature_names[i],
                    'contribution': float(feature_contributions[i]),
                    'usage_frequency': float(feature_usage_count[i] / len(self.trees))
                }
                for i in np.argsort(feature_contributions)[::-1][:5]
                if feature_contributions[i] > 0
            ]
            
        return explanation
    
    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute feature importance based on usage in splits.
        
        Args:
            X: Data to analyze
            
        Returns:
            Feature importance scores
        """
        if not self.trees:
            raise ValueError("Model must be fitted first")
            
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        # Count feature usage across all trees
        for feature_idx in self.feature_indices:
            for idx in feature_idx:
                importance[idx] += 1
                
        # Normalize
        importance /= len(self.trees)
        
        return importance
    
    def get_model_stats(self) -> Dict:
        """Get model statistics and diagnostics."""
        stats = {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'threshold': float(self.threshold) if self.threshold is not None else None
        }
        
        if self.training_data is not None:
            stats.update({
                'n_training_samples': self.training_data.shape[0],
                'n_features': self.training_data.shape[1]
            })
            
        if self.trees:
            # Analyze tree statistics
            tree_depths = []
            for tree in self.trees:
                depth = self._get_tree_depth(tree.root)
                tree_depths.append(depth)
                
            stats.update({
                'avg_tree_depth': float(np.mean(tree_depths)),
                'min_tree_depth': int(np.min(tree_depths)),
                'max_tree_depth': int(np.max(tree_depths)),
                'std_tree_depth': float(np.std(tree_depths))
            })
            
        return stats
    
    def _get_tree_depth(self, node: Dict[str, Any]) -> int:
        """Get maximum depth of a tree."""
        if node is None or node['type'] == 'leaf':
            return node.get('depth', 0) if node else 0
            
        left_depth = self._get_tree_depth(node.get('left'))
        right_depth = self._get_tree_depth(node.get('right'))
        
        return max(left_depth, right_depth)
    
    def update_threshold(self, contamination: float):
        """
        Update the anomaly threshold.
        
        Args:
            contamination: New contamination rate
        """
        if self.training_data is None:
            raise ValueError("Model must be fitted first")
            
        self.contamination = contamination
        scores = self.decision_function(self.training_data)
        self.threshold = np.percentile(scores, 100 * contamination)
        
    def get_outlier_samples(self, X: np.ndarray, return_indices: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get outlier samples from the dataset.
        
        Args:
            X: Data to analyze
            return_indices: Whether to return indices
            
        Returns:
            Outlier samples and optionally their indices
        """
        predictions = self.predict(X)
        outlier_mask = predictions == -1
        outlier_samples = X[outlier_mask]
        
        if return_indices:
            outlier_indices = np.where(outlier_mask)[0]
            return outlier_samples, outlier_indices
        else:
            return outlier_samples, None