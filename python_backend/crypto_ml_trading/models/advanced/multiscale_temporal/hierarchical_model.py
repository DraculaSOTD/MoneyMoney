"""
Hierarchical Temporal Modeling for Multi-Scale Analysis.

Implements hierarchical time series models that capture dependencies across multiple time scales.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class HierarchicalNode:
    """Represents a node in the hierarchical temporal model."""
    level: int
    scale: int
    parent: Optional['HierarchicalNode']
    children: List['HierarchicalNode']
    time_series: np.ndarray
    features: Dict[str, float]
    model_params: Dict[str, Any]
    forecasts: Dict[int, np.ndarray]  # horizon -> predictions


@dataclass
class HierarchicalStructure:
    """Defines the hierarchical structure of the temporal model."""
    levels: List[int]  # Scales at each level
    aggregation_methods: Dict[int, str]  # level -> method
    reconciliation_method: str
    coherency_constraints: List[str]


@dataclass
class CrossScaleDependency:
    """Represents dependency between different scales."""
    source_scale: int
    target_scale: int
    dependency_type: str
    strength: float
    lag: int
    parameters: Dict[str, float]


class HierarchicalTemporalModel:
    """
    Hierarchical temporal model for multi-scale time series analysis.
    
    Features:
    - Multi-level temporal hierarchy
    - Scale-dependent modeling
    - Cross-scale dependency modeling
    - Hierarchical forecasting with reconciliation
    - Bottom-up and top-down approaches
    - Optimal combination forecasting
    - Coherency enforcement
    """
    
    def __init__(self,
                 structure: Optional[HierarchicalStructure] = None,
                 base_model_type: str = 'ar',
                 reconciliation_method: str = 'optimal_combination',
                 max_hierarchy_levels: int = 5):
        """
        Initialize hierarchical temporal model.
        
        Args:
            structure: Hierarchical structure definition
            base_model_type: Base model type for each node
            reconciliation_method: Forecast reconciliation method
            max_hierarchy_levels: Maximum number of hierarchy levels
        """
        self.structure = structure or self._create_default_structure()
        self.base_model_type = base_model_type
        self.reconciliation_method = reconciliation_method
        self.max_hierarchy_levels = max_hierarchy_levels
        
        # Model components
        self.hierarchy: Dict[int, List[HierarchicalNode]] = defaultdict(list)
        self.root_node: Optional[HierarchicalNode] = None
        self.leaf_nodes: List[HierarchicalNode] = []
        
        # Cross-scale dependencies
        self.dependencies: List[CrossScaleDependency] = []
        self.dependency_matrix: Optional[np.ndarray] = None
        
        # Model state
        self.is_fitted = False
        self.aggregation_matrices: Dict[int, np.ndarray] = {}
        self.reconciliation_matrix: Optional[np.ndarray] = None
        
    def _create_default_structure(self) -> HierarchicalStructure:
        """Create default hierarchical structure."""
        return HierarchicalStructure(
            levels=[1, 2, 4, 8, 16],  # Default scales
            aggregation_methods={
                0: 'sum',  # Bottom level
                1: 'mean',
                2: 'mean',
                3: 'mean',
                4: 'sum'   # Top level
            },
            reconciliation_method='optimal_combination',
            coherency_constraints=['sum_coherent', 'non_negative']
        )
    
    def build_hierarchy(self, base_series: np.ndarray) -> None:
        """
        Build hierarchical structure from base time series.
        
        Args:
            base_series: Base time series data
        """
        self.hierarchy.clear()
        self.leaf_nodes.clear()
        
        # Create leaf nodes (finest resolution)
        finest_scale = min(self.structure.levels)
        leaf_series = self._decompose_to_scale(base_series, finest_scale)
        
        # Create leaf nodes
        for i, series in enumerate(leaf_series):
            node = HierarchicalNode(
                level=0,
                scale=finest_scale,
                parent=None,
                children=[],
                time_series=series,
                features={},
                model_params={},
                forecasts={}
            )
            self.hierarchy[0].append(node)
            self.leaf_nodes.append(node)
        
        # Build upper levels
        for level in range(1, len(self.structure.levels)):
            scale = self.structure.levels[level]
            aggregation_method = self.structure.aggregation_methods.get(level, 'mean')
            
            # Create nodes at this level
            lower_level_nodes = self.hierarchy[level - 1]
            upper_level_nodes = self._create_upper_level(
                lower_level_nodes, scale, aggregation_method, level
            )
            
            self.hierarchy[level] = upper_level_nodes
        
        # Set root node
        if self.hierarchy:
            top_level = max(self.hierarchy.keys())
            if self.hierarchy[top_level]:
                self.root_node = self.hierarchy[top_level][0]
    
    def _decompose_to_scale(self, series: np.ndarray, scale: int) -> List[np.ndarray]:
        """Decompose series to specified scale."""
        if scale == 1:
            return [series]
        
        # Simple decomposition: split into segments
        segment_length = len(series) // scale
        if segment_length < 1:
            return [series]
        
        segments = []
        for i in range(scale):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i < scale - 1 else len(series)
            segments.append(series[start_idx:end_idx])
        
        return segments
    
    def _create_upper_level(self, 
                           lower_nodes: List[HierarchicalNode],
                           scale: int,
                           aggregation_method: str,
                           level: int) -> List[HierarchicalNode]:
        """Create upper level nodes from lower level."""
        if not lower_nodes:
            return []
        
        # Group lower nodes for aggregation
        grouping_factor = 2  # Each upper node aggregates 2 lower nodes
        upper_nodes = []
        
        for i in range(0, len(lower_nodes), grouping_factor):
            group = lower_nodes[i:i + grouping_factor]
            
            # Aggregate time series
            aggregated_series = self._aggregate_series(
                [node.time_series for node in group], 
                aggregation_method
            )
            
            # Create upper node
            upper_node = HierarchicalNode(
                level=level,
                scale=scale,
                parent=None,
                children=group,
                time_series=aggregated_series,
                features={},
                model_params={},
                forecasts={}
            )
            
            # Set parent relationships
            for child in group:
                child.parent = upper_node
            
            upper_nodes.append(upper_node)
        
        return upper_nodes
    
    def _aggregate_series(self, series_list: List[np.ndarray], method: str) -> np.ndarray:
        """Aggregate multiple time series."""
        if not series_list:
            return np.array([])
        
        # Ensure all series have same length
        min_length = min(len(s) for s in series_list)
        trimmed_series = [s[:min_length] for s in series_list]
        
        if method == 'sum':
            return np.sum(trimmed_series, axis=0)
        elif method == 'mean':
            return np.mean(trimmed_series, axis=0)
        elif method == 'median':
            return np.median(trimmed_series, axis=0)
        elif method == 'max':
            return np.max(trimmed_series, axis=0)
        elif method == 'min':
            return np.min(trimmed_series, axis=0)
        else:
            return np.mean(trimmed_series, axis=0)  # Default to mean
    
    def fit(self, base_series: np.ndarray) -> None:
        """
        Fit hierarchical temporal model.
        
        Args:
            base_series: Base time series data
        """
        # Build hierarchy
        self.build_hierarchy(base_series)
        
        # Fit base models at each node
        self._fit_node_models()
        
        # Learn cross-scale dependencies
        self._learn_dependencies()
        
        # Build aggregation matrices
        self._build_aggregation_matrices()
        
        # Build reconciliation matrix
        self._build_reconciliation_matrix()
        
        self.is_fitted = True
    
    def _fit_node_models(self) -> None:
        """Fit base models at each hierarchical node."""
        for level_nodes in self.hierarchy.values():
            for node in level_nodes:
                self._fit_single_node_model(node)
    
    def _fit_single_node_model(self, node: HierarchicalNode) -> None:
        """Fit model for a single node."""
        if len(node.time_series) < 3:
            return
        
        # Extract features
        node.features = self._extract_node_features(node.time_series)
        
        # Fit model based on type
        if self.base_model_type == 'ar':
            node.model_params = self._fit_ar_model(node.time_series)
        elif self.base_model_type == 'linear_trend':
            node.model_params = self._fit_linear_trend(node.time_series)
        elif self.base_model_type == 'seasonal':
            node.model_params = self._fit_seasonal_model(node.time_series)
        else:
            # Default: simple mean model
            node.model_params = {'mean': np.mean(node.time_series)}
    
    def _extract_node_features(self, series: np.ndarray) -> Dict[str, float]:
        """Extract features for a node's time series."""
        if len(series) < 2:
            return {}
        
        features = {
            'mean': np.mean(series),
            'std': np.std(series),
            'min': np.min(series),
            'max': np.max(series),
            'trend': np.polyfit(range(len(series)), series, 1)[0],
            'variance': np.var(series),
            'skewness': self._calculate_skewness(series),
            'kurtosis': self._calculate_kurtosis(series)
        }
        
        # Autocorrelation features
        if len(series) > 2:
            features['autocorr_lag1'] = np.corrcoef(series[:-1], series[1:])[0, 1]
        
        return features
    
    def _calculate_skewness(self, series: np.ndarray) -> float:
        """Calculate skewness of series."""
        if len(series) < 3 or np.std(series) == 0:
            return 0.0
        
        mean_val = np.mean(series)
        std_val = np.std(series)
        skewness = np.mean(((series - mean_val) / std_val) ** 3)
        
        return skewness
    
    def _calculate_kurtosis(self, series: np.ndarray) -> float:
        """Calculate kurtosis of series."""
        if len(series) < 4 or np.std(series) == 0:
            return 0.0
        
        mean_val = np.mean(series)
        std_val = np.std(series)
        kurtosis = np.mean(((series - mean_val) / std_val) ** 4) - 3
        
        return kurtosis
    
    def _fit_ar_model(self, series: np.ndarray, max_order: int = 3) -> Dict[str, Any]:
        """Fit autoregressive model."""
        if len(series) < max_order + 2:
            return {'coefficients': [np.mean(series)], 'order': 0}
        
        best_order = 1
        best_coeffs = [np.mean(series)]
        best_mse = float('inf')
        
        for order in range(1, min(max_order + 1, len(series) // 2)):
            try:
                # Prepare data
                X = np.column_stack([series[order-i-1:-i-1] for i in range(order)])
                y = series[order:]
                
                if len(X) == 0 or len(y) == 0:
                    continue
                
                # Fit linear regression
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Calculate MSE
                predictions = X @ coeffs
                mse = np.mean((y - predictions) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_order = order
                    best_coeffs = coeffs
                    
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        return {
            'coefficients': best_coeffs,
            'order': best_order,
            'mse': best_mse
        }
    
    def _fit_linear_trend(self, series: np.ndarray) -> Dict[str, Any]:
        """Fit linear trend model."""
        if len(series) < 2:
            return {'slope': 0, 'intercept': np.mean(series)}
        
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series, 1)
        
        return {
            'slope': coeffs[0],
            'intercept': coeffs[1],
            'r_squared': np.corrcoef(x, series)[0, 1] ** 2
        }
    
    def _fit_seasonal_model(self, series: np.ndarray, period: int = 24) -> Dict[str, Any]:
        """Fit seasonal model."""
        if len(series) < period * 2:
            # Fall back to linear trend
            return self._fit_linear_trend(series)
        
        # Simple seasonal decomposition
        seasonal_pattern = np.zeros(period)
        
        for i in range(period):
            positions = np.arange(i, len(series), period)
            if len(positions) > 1:
                seasonal_pattern[i] = np.mean(series[positions])
        
        # Detrend
        seasonal_extended = np.tile(seasonal_pattern, len(series) // period + 1)[:len(series)]
        detrended = series - seasonal_extended
        
        # Fit trend to detrended series
        trend_params = self._fit_linear_trend(detrended)
        
        return {
            'seasonal_pattern': seasonal_pattern,
            'trend_slope': trend_params['slope'],
            'trend_intercept': trend_params['intercept'],
            'period': period
        }
    
    def _learn_dependencies(self) -> None:
        """Learn cross-scale dependencies."""
        self.dependencies.clear()
        
        # Analyze dependencies between different levels
        for level1 in self.hierarchy:
            for level2 in self.hierarchy:
                if level1 != level2:
                    self._analyze_level_dependencies(level1, level2)
        
        # Build dependency matrix
        self._build_dependency_matrix()
    
    def _analyze_level_dependencies(self, level1: int, level2: int) -> None:
        """Analyze dependencies between two levels."""
        nodes1 = self.hierarchy[level1]
        nodes2 = self.hierarchy[level2]
        
        for node1 in nodes1:
            for node2 in nodes2:
                dependency = self._measure_dependency(node1, node2)
                
                if dependency['strength'] > 0.3:  # Significant dependency threshold
                    dep = CrossScaleDependency(
                        source_scale=node1.scale,
                        target_scale=node2.scale,
                        dependency_type=dependency['type'],
                        strength=dependency['strength'],
                        lag=dependency['lag'],
                        parameters=dependency['parameters']
                    )
                    self.dependencies.append(dep)
    
    def _measure_dependency(self, node1: HierarchicalNode, node2: HierarchicalNode) -> Dict[str, Any]:
        """Measure dependency between two nodes."""
        series1 = node1.time_series
        series2 = node2.time_series
        
        # Align series lengths
        min_length = min(len(series1), len(series2))
        if min_length < 3:
            return {'strength': 0, 'type': 'none', 'lag': 0, 'parameters': {}}
        
        series1 = series1[:min_length]
        series2 = series2[:min_length]
        
        # Test different types of dependencies
        correlations = []
        
        # Linear correlation
        linear_corr = abs(np.corrcoef(series1, series2)[0, 1])
        correlations.append(('linear', linear_corr, 0, {}))
        
        # Lagged correlations
        for lag in range(1, min(10, min_length // 2)):
            if len(series1) > lag and len(series2) > lag:
                lagged_corr = abs(np.corrcoef(series1[:-lag], series2[lag:])[0, 1])
                correlations.append(('lagged', lagged_corr, lag, {}))
        
        # Find best dependency
        best_dep = max(correlations, key=lambda x: x[1])
        
        return {
            'strength': best_dep[1],
            'type': best_dep[0],
            'lag': best_dep[2],
            'parameters': best_dep[3]
        }
    
    def _build_dependency_matrix(self) -> None:
        """Build dependency matrix from learned dependencies."""
        if not self.dependencies:
            return
        
        # Get all unique scales
        scales = set()
        for dep in self.dependencies:
            scales.add(dep.source_scale)
            scales.add(dep.target_scale)
        
        scales = sorted(list(scales))
        n_scales = len(scales)
        
        if n_scales == 0:
            return
        
        # Create dependency matrix
        self.dependency_matrix = np.zeros((n_scales, n_scales))
        
        scale_to_idx = {scale: i for i, scale in enumerate(scales)}
        
        for dep in self.dependencies:
            source_idx = scale_to_idx[dep.source_scale]
            target_idx = scale_to_idx[dep.target_scale]
            self.dependency_matrix[source_idx, target_idx] = dep.strength
    
    def _build_aggregation_matrices(self) -> None:
        """Build aggregation matrices for each level."""
        for level in range(1, len(self.hierarchy)):
            upper_nodes = self.hierarchy[level]
            lower_nodes = self.hierarchy[level - 1]
            
            if not upper_nodes or not lower_nodes:
                continue
            
            # Create aggregation matrix
            n_upper = len(upper_nodes)
            n_lower = len(lower_nodes)
            
            agg_matrix = np.zeros((n_upper, n_lower))
            
            for i, upper_node in enumerate(upper_nodes):
                for j, lower_node in enumerate(lower_nodes):
                    if lower_node in upper_node.children:
                        # Equal weight for children
                        agg_matrix[i, j] = 1.0 / len(upper_node.children)
            
            self.aggregation_matrices[level] = agg_matrix
    
    def _build_reconciliation_matrix(self) -> None:
        """Build reconciliation matrix for coherent forecasting."""
        if not self.hierarchy or len(self.hierarchy) < 2:
            return
        
        # For simplicity, use bottom-up reconciliation matrix
        # In practice, this would implement optimal combination
        
        bottom_level = min(self.hierarchy.keys())
        top_level = max(self.hierarchy.keys())
        
        n_bottom = len(self.hierarchy[bottom_level])
        n_total = sum(len(nodes) for nodes in self.hierarchy.values())
        
        if n_bottom == 0 or n_total == 0:
            return
        
        # Simple bottom-up matrix
        self.reconciliation_matrix = np.zeros((n_total, n_bottom))
        
        # Bottom level maps directly
        for i in range(n_bottom):
            self.reconciliation_matrix[i, i] = 1.0
        
        # Upper levels aggregate from bottom
        row_idx = n_bottom
        for level in range(bottom_level + 1, top_level + 1):
            if level not in self.hierarchy:
                continue
            
            for node in self.hierarchy[level]:
                # Find corresponding bottom-level nodes
                bottom_indices = self._get_bottom_level_indices(node)
                for idx in bottom_indices:
                    if idx < n_bottom:
                        self.reconciliation_matrix[row_idx, idx] = 1.0 / len(bottom_indices)
                row_idx += 1
    
    def _get_bottom_level_indices(self, node: HierarchicalNode) -> List[int]:
        """Get indices of bottom-level nodes that contribute to this node."""
        if node.level == 0:  # Bottom level
            # Find index in bottom level
            bottom_nodes = self.hierarchy[0]
            try:
                return [bottom_nodes.index(node)]
            except ValueError:
                return []
        
        # Recursively get from children
        indices = []
        for child in node.children:
            indices.extend(self._get_bottom_level_indices(child))
        
        return indices
    
    def forecast(self, horizons: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Generate hierarchical forecasts.
        
        Args:
            horizons: Forecast horizons
            
        Returns:
            Hierarchical forecasts by horizon and level
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = {}
        
        for horizon in horizons:
            level_forecasts = {}
            
            # Generate forecasts at each level
            for level, nodes in self.hierarchy.items():
                node_forecasts = []
                
                for node in nodes:
                    forecast = self._forecast_single_node(node, horizon)
                    node_forecasts.append(forecast)
                
                level_forecasts[f'level_{level}'] = np.array(node_forecasts)
            
            # Apply reconciliation
            if self.reconciliation_method == 'bottom_up':
                level_forecasts = self._reconcile_bottom_up(level_forecasts)
            elif self.reconciliation_method == 'top_down':
                level_forecasts = self._reconcile_top_down(level_forecasts)
            elif self.reconciliation_method == 'optimal_combination':
                level_forecasts = self._reconcile_optimal_combination(level_forecasts)
            
            forecasts[horizon] = level_forecasts
        
        return forecasts
    
    def _forecast_single_node(self, node: HierarchicalNode, horizon: int) -> np.ndarray:
        """Generate forecast for a single node."""
        if not node.model_params:
            # Default: repeat last value
            if len(node.time_series) > 0:
                return np.full(horizon, node.time_series[-1])
            else:
                return np.zeros(horizon)
        
        # Forecast based on model type
        if self.base_model_type == 'ar':
            return self._forecast_ar_node(node, horizon)
        elif self.base_model_type == 'linear_trend':
            return self._forecast_trend_node(node, horizon)
        elif self.base_model_type == 'seasonal':
            return self._forecast_seasonal_node(node, horizon)
        else:
            # Default: mean forecast
            mean_value = node.model_params.get('mean', 0)
            return np.full(horizon, mean_value)
    
    def _forecast_ar_node(self, node: HierarchicalNode, horizon: int) -> np.ndarray:
        """Forecast using AR model."""
        coeffs = node.model_params.get('coefficients', [])
        order = node.model_params.get('order', 0)
        
        if order == 0 or not coeffs:
            # Fall back to mean
            return np.full(horizon, np.mean(node.time_series))
        
        # Initialize with last values
        series = node.time_series.copy()
        forecasts = []
        
        for _ in range(horizon):
            if len(series) >= order:
                # Use last 'order' values
                recent_values = series[-order:]
                forecast = np.dot(coeffs, recent_values[::-1])  # Reverse order
            else:
                forecast = np.mean(series)
            
            forecasts.append(forecast)
            series = np.append(series, forecast)
        
        return np.array(forecasts)
    
    def _forecast_trend_node(self, node: HierarchicalNode, horizon: int) -> np.ndarray:
        """Forecast using linear trend."""
        slope = node.model_params.get('slope', 0)
        intercept = node.model_params.get('intercept', 0)
        
        last_time = len(node.time_series) - 1
        future_times = np.arange(last_time + 1, last_time + 1 + horizon)
        
        forecasts = slope * future_times + intercept
        
        return forecasts
    
    def _forecast_seasonal_node(self, node: HierarchicalNode, horizon: int) -> np.ndarray:
        """Forecast using seasonal model."""
        seasonal_pattern = node.model_params.get('seasonal_pattern', [])
        period = node.model_params.get('period', 24)
        trend_slope = node.model_params.get('trend_slope', 0)
        trend_intercept = node.model_params.get('trend_intercept', 0)
        
        if not seasonal_pattern:
            # Fall back to trend
            return self._forecast_trend_node(node, horizon)
        
        forecasts = []
        last_time = len(node.time_series) - 1
        
        for h in range(horizon):
            future_time = last_time + 1 + h
            
            # Trend component
            trend = trend_slope * future_time + trend_intercept
            
            # Seasonal component
            seasonal_idx = h % period
            seasonal = seasonal_pattern[seasonal_idx] if seasonal_idx < len(seasonal_pattern) else 0
            
            forecasts.append(trend + seasonal)
        
        return np.array(forecasts)
    
    def _reconcile_bottom_up(self, level_forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply bottom-up reconciliation."""
        reconciled = level_forecasts.copy()
        
        # Bottom level stays the same
        if 'level_0' in reconciled:
            bottom_forecasts = reconciled['level_0']
            
            # Aggregate to upper levels
            for level in range(1, len(self.hierarchy)):
                level_key = f'level_{level}'
                if level in self.aggregation_matrices:
                    agg_matrix = self.aggregation_matrices[level]
                    reconciled[level_key] = agg_matrix @ bottom_forecasts.T
        
        return reconciled
    
    def _reconcile_top_down(self, level_forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply top-down reconciliation."""
        reconciled = level_forecasts.copy()
        
        # Start from top level and disaggregate down
        max_level = max([int(k.split('_')[1]) for k in level_forecasts.keys()])
        top_key = f'level_{max_level}'
        
        if top_key in reconciled:
            # Simple proportional disaggregation
            # In practice, this would use historical proportions
            for level in range(max_level - 1, -1, -1):
                level_key = f'level_{level}'
                # Simplified: equal disaggregation
                # Real implementation would use proper proportions
                pass
        
        return reconciled
    
    def _reconcile_optimal_combination(self, level_forecasts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply optimal combination reconciliation."""
        if self.reconciliation_matrix is None:
            # Fall back to bottom-up
            return self._reconcile_bottom_up(level_forecasts)
        
        # Stack all forecasts
        all_forecasts = []
        for level in sorted([int(k.split('_')[1]) for k in level_forecasts.keys()]):
            level_key = f'level_{level}'
            if level_key in level_forecasts:
                all_forecasts.append(level_forecasts[level_key].flatten())
        
        if not all_forecasts:
            return level_forecasts
        
        stacked_forecasts = np.concatenate(all_forecasts)
        
        # Apply reconciliation matrix
        try:
            reconciled_forecasts = self.reconciliation_matrix @ stacked_forecasts[:self.reconciliation_matrix.shape[1]]
            
            # Unstack back to levels
            reconciled = {}
            start_idx = 0
            for level in sorted([int(k.split('_')[1]) for k in level_forecasts.keys()]):
                level_key = f'level_{level}'
                if level_key in level_forecasts:
                    level_size = level_forecasts[level_key].size
                    reconciled[level_key] = reconciled_forecasts[start_idx:start_idx + level_size].reshape(level_forecasts[level_key].shape)
                    start_idx += level_size
            
            return reconciled
            
        except (ValueError, IndexError):
            # Fall back to bottom-up if reconciliation fails
            return self._reconcile_bottom_up(level_forecasts)
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get summary of hierarchical structure."""
        summary = {
            'num_levels': len(self.hierarchy),
            'total_nodes': sum(len(nodes) for nodes in self.hierarchy.values()),
            'structure': self.structure.__dict__,
            'dependencies': len(self.dependencies),
            'is_fitted': self.is_fitted
        }
        
        # Add level details
        level_details = {}
        for level, nodes in self.hierarchy.items():
            level_details[level] = {
                'num_nodes': len(nodes),
                'scale': nodes[0].scale if nodes else None,
                'avg_series_length': np.mean([len(node.time_series) for node in nodes]) if nodes else 0
            }
        
        summary['levels'] = level_details
        
        return summary
    
    def get_cross_scale_dependencies(self) -> List[Dict[str, Any]]:
        """Get cross-scale dependency information."""
        return [
            {
                'source_scale': dep.source_scale,
                'target_scale': dep.target_scale,
                'type': dep.dependency_type,
                'strength': dep.strength,
                'lag': dep.lag,
                'parameters': dep.parameters
            }
            for dep in self.dependencies
        ]
    
    def update_node_forecasts(self, node_forecasts: Dict[str, Dict[int, np.ndarray]]) -> None:
        """Update stored forecasts for nodes."""
        for level, nodes in self.hierarchy.items():
            level_key = f'level_{level}'
            if level_key in node_forecasts:
                for i, node in enumerate(nodes):
                    for horizon, forecasts in node_forecasts[level_key].items():
                        if i < len(forecasts):
                            node.forecasts[horizon] = forecasts[i]