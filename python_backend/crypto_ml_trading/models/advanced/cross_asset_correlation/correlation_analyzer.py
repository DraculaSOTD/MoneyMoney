"""
Cross-Asset Correlation Analysis.

Implements comprehensive correlation analysis across multiple asset classes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.matrix_operations import MatrixOperations


@dataclass
class AssetData:
    """Asset price and return data."""
    symbol: str
    asset_class: str
    prices: np.ndarray
    returns: np.ndarray
    timestamps: List[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationMatrix:
    """Correlation matrix with metadata."""
    timestamp: datetime
    correlation_matrix: np.ndarray
    asset_symbols: List[str]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    market_regime: str


@dataclass
class CorrelationBreakdown:
    """Detailed correlation breakdown."""
    timestamp: datetime
    pairwise_correlations: Dict[Tuple[str, str], float]
    asset_class_correlations: Dict[Tuple[str, str], float]
    concentration_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]


class CorrelationAnalyzer:
    """
    Advanced cross-asset correlation analysis.
    
    Features:
    - Multi-asset correlation calculation and monitoring
    - Rolling correlation analysis with multiple windows
    - Correlation regime detection and classification
    - Asset class correlation aggregation
    - Correlation stability and concentration metrics
    - Principal component analysis of correlations
    """
    
    def __init__(self,
                 correlation_windows: List[int] = None,
                 min_observations: int = 30,
                 asset_classes: List[str] = None,
                 correlation_threshold: float = 0.05):
        """
        Initialize correlation analyzer.
        
        Args:
            correlation_windows: List of window sizes for rolling correlations
            min_observations: Minimum observations required for correlation
            asset_classes: List of asset classes to analyze
            correlation_threshold: Threshold for significant correlation changes
        """
        self.correlation_windows = correlation_windows or [30, 60, 120, 252]
        self.min_observations = min_observations
        self.asset_classes = asset_classes or [
            'crypto', 'equity', 'bond', 'commodity', 'fx', 'volatility'
        ]
        self.correlation_threshold = correlation_threshold
        
        # Asset data storage
        self.asset_data: Dict[str, AssetData] = {}
        self.asset_class_mapping: Dict[str, str] = {}
        
        # Correlation matrices
        self.correlation_matrices: Dict[int, deque] = {
            window: deque(maxlen=500) for window in self.correlation_windows
        }
        
        # Analysis results
        self.correlation_breakdowns: deque = deque(maxlen=200)
        self.regime_history: deque = deque(maxlen=100)
        
        # Principal component analysis
        self.pca_results: Dict[int, Dict] = {}
        
    def add_asset_data(self,
                      symbol: str,
                      asset_class: str,
                      prices: np.ndarray,
                      timestamps: List[datetime],
                      metadata: Optional[Dict] = None) -> None:
        """
        Add asset data for correlation analysis.
        
        Args:
            symbol: Asset symbol
            asset_class: Asset class category
            prices: Price series
            timestamps: Corresponding timestamps
            metadata: Additional asset metadata
        """
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Store asset data
        self.asset_data[symbol] = AssetData(
            symbol=symbol,
            asset_class=asset_class,
            prices=prices,
            returns=returns,
            timestamps=timestamps,
            metadata=metadata or {}
        )
        
        self.asset_class_mapping[symbol] = asset_class
        
    def update_asset_data(self,
                         symbol: str,
                         new_price: float,
                         timestamp: datetime) -> None:
        """
        Update existing asset data with new price point.
        
        Args:
            symbol: Asset symbol
            new_price: New price
            timestamp: Price timestamp
        """
        if symbol not in self.asset_data:
            return
        
        asset = self.asset_data[symbol]
        
        # Update prices and returns
        asset.prices = np.append(asset.prices, new_price)
        if len(asset.prices) > 1:
            new_return = (new_price - asset.prices[-2]) / asset.prices[-2]
            asset.returns = np.append(asset.returns, new_return)
        
        asset.timestamps.append(timestamp)
        
        # Trigger correlation update
        self._update_correlations(timestamp)
    
    def _update_correlations(self, timestamp: datetime) -> None:
        """Update correlation matrices for all windows."""
        if len(self.asset_data) < 2:
            return
        
        for window in self.correlation_windows:
            corr_matrix = self._calculate_correlation_matrix(window, timestamp)
            if corr_matrix is not None:
                self.correlation_matrices[window].append(corr_matrix)
        
        # Update analysis
        self._update_correlation_analysis(timestamp)
    
    def _calculate_correlation_matrix(self,
                                    window: int,
                                    timestamp: datetime) -> Optional[CorrelationMatrix]:
        """Calculate correlation matrix for given window."""
        # Collect return data for all assets
        return_data = {}
        min_length = float('inf')
        
        for symbol, asset in self.asset_data.items():
            if len(asset.returns) >= window:
                returns = asset.returns[-window:]
                return_data[symbol] = returns
                min_length = min(min_length, len(returns))
        
        if len(return_data) < 2 or min_length < self.min_observations:
            return None
        
        # Align return series
        symbols = list(return_data.keys())
        return_matrix = np.column_stack([
            return_data[symbol][-min_length:] for symbol in symbols
        ])
        
        # Calculate correlation matrix
        try:
            correlation_matrix = np.corrcoef(return_matrix.T)
            
            # Handle NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            
            # Condition number
            condition_number = np.max(eigenvalues) / np.max(np.min(eigenvalues), 1e-10)
            
            # Market regime classification
            market_regime = self._classify_market_regime(correlation_matrix, eigenvalues)
            
            return CorrelationMatrix(
                timestamp=timestamp,
                correlation_matrix=correlation_matrix,
                asset_symbols=symbols,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                condition_number=condition_number,
                market_regime=market_regime
            )
            
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return None
    
    def _classify_market_regime(self,
                              correlation_matrix: np.ndarray,
                              eigenvalues: np.ndarray) -> str:
        """Classify market regime based on correlation structure."""
        # Calculate average correlation (excluding diagonal)
        n = correlation_matrix.shape[0]
        if n < 2:
            return 'insufficient_data'
        
        off_diagonal = correlation_matrix[np.triu_indices(n, k=1)]
        avg_correlation = np.mean(off_diagonal)
        
        # Calculate correlation concentration (largest eigenvalue ratio)
        if len(eigenvalues) > 1:
            eigenvalue_concentration = eigenvalues[-1] / np.sum(eigenvalues)
        else:
            eigenvalue_concentration = 1.0
        
        # Regime classification
        if avg_correlation > 0.7 and eigenvalue_concentration > 0.7:
            return 'crisis'  # High correlation, concentrated risk
        elif avg_correlation > 0.5:
            return 'risk_off'  # High correlation
        elif avg_correlation < 0.2:
            return 'risk_on'  # Low correlation, diversification benefits
        elif eigenvalue_concentration > 0.6:
            return 'concentrated'  # Concentrated risk factors
        else:
            return 'normal'
    
    def _update_correlation_analysis(self, timestamp: datetime) -> None:
        """Update comprehensive correlation analysis."""
        # Use medium-term window for analysis
        analysis_window = 60
        
        if analysis_window not in self.correlation_matrices:
            return
        
        if not self.correlation_matrices[analysis_window]:
            return
        
        latest_corr = self.correlation_matrices[analysis_window][-1]
        
        # Calculate pairwise correlations
        pairwise_correlations = self._extract_pairwise_correlations(latest_corr)
        
        # Calculate asset class correlations
        asset_class_correlations = self._calculate_asset_class_correlations(latest_corr)
        
        # Calculate concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(latest_corr)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(analysis_window)
        
        breakdown = CorrelationBreakdown(
            timestamp=timestamp,
            pairwise_correlations=pairwise_correlations,
            asset_class_correlations=asset_class_correlations,
            concentration_metrics=concentration_metrics,
            stability_metrics=stability_metrics
        )
        
        self.correlation_breakdowns.append(breakdown)
        
        # Update PCA analysis
        self._update_pca_analysis(analysis_window, latest_corr)
    
    def _extract_pairwise_correlations(self,
                                     corr_matrix: CorrelationMatrix) -> Dict[Tuple[str, str], float]:
        """Extract pairwise correlations from matrix."""
        pairwise = {}
        symbols = corr_matrix.asset_symbols
        matrix = corr_matrix.correlation_matrix
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pair = (symbols[i], symbols[j])
                correlation = matrix[i, j]
                pairwise[pair] = correlation
        
        return pairwise
    
    def _calculate_asset_class_correlations(self,
                                          corr_matrix: CorrelationMatrix) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between asset classes."""
        symbols = corr_matrix.asset_symbols
        matrix = corr_matrix.correlation_matrix
        
        # Group assets by class
        class_assets = defaultdict(list)
        for i, symbol in enumerate(symbols):
            asset_class = self.asset_class_mapping.get(symbol, 'unknown')
            class_assets[asset_class].append(i)
        
        # Calculate average correlations between classes
        class_correlations = {}
        
        for class1, indices1 in class_assets.items():
            for class2, indices2 in class_assets.items():
                if class1 >= class2:  # Avoid duplicates
                    continue
                
                correlations = []
                for i in indices1:
                    for j in indices2:
                        correlations.append(matrix[i, j])
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    class_correlations[(class1, class2)] = avg_correlation
        
        return class_correlations
    
    def _calculate_concentration_metrics(self,
                                       corr_matrix: CorrelationMatrix) -> Dict[str, float]:
        """Calculate correlation concentration metrics."""
        eigenvalues = corr_matrix.eigenvalues
        matrix = corr_matrix.correlation_matrix
        
        # Eigenvalue concentration
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            eigenvalue_concentration = eigenvalues[-1] / total_variance
            top3_concentration = np.sum(eigenvalues[-3:]) / total_variance
        else:
            eigenvalue_concentration = 1.0
            top3_concentration = 1.0
        
        # Average correlation
        n = matrix.shape[0]
        if n > 1:
            off_diagonal = matrix[np.triu_indices(n, k=1)]
            avg_correlation = np.mean(off_diagonal)
            max_correlation = np.max(off_diagonal)
            min_correlation = np.min(off_diagonal)
        else:
            avg_correlation = 0.0
            max_correlation = 0.0
            min_correlation = 0.0
        
        # Condition number (measure of multicollinearity)
        condition_number = corr_matrix.condition_number
        
        return {
            'eigenvalue_concentration': eigenvalue_concentration,
            'top3_eigenvalue_concentration': top3_concentration,
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'min_correlation': min_correlation,
            'condition_number': condition_number,
            'effective_dimension': total_variance / eigenvalues[-1] if eigenvalues[-1] > 0 else n
        }
    
    def _calculate_stability_metrics(self, window: int) -> Dict[str, float]:
        """Calculate correlation stability metrics."""
        if window not in self.correlation_matrices or len(self.correlation_matrices[window]) < 2:
            return {}
        
        recent_matrices = list(self.correlation_matrices[window])[-10:]  # Last 10 observations
        
        if len(recent_matrices) < 2:
            return {}
        
        # Calculate correlation of correlations (stability measure)
        correlation_series = []
        
        for i in range(len(recent_matrices) - 1):
            matrix1 = recent_matrices[i].correlation_matrix
            matrix2 = recent_matrices[i + 1].correlation_matrix
            
            # Extract upper triangular correlations
            n = matrix1.shape[0]
            if matrix2.shape[0] != n:
                continue  # Skip if different dimensions
            
            corr1 = matrix1[np.triu_indices(n, k=1)]
            corr2 = matrix2[np.triu_indices(n, k=1)]
            
            # Calculate correlation between correlation vectors
            if len(corr1) > 1 and np.std(corr1) > 0 and np.std(corr2) > 0:
                stability = np.corrcoef(corr1, corr2)[0, 1]
                correlation_series.append(stability)
        
        if correlation_series:
            avg_stability = np.mean(correlation_series)
            stability_volatility = np.std(correlation_series)
        else:
            avg_stability = 0.0
            stability_volatility = 0.0
        
        # Calculate average eigenvalue stability
        eigenvalue_series = [m.eigenvalues for m in recent_matrices]
        eigenvalue_volatility = 0.0
        
        if len(eigenvalue_series) > 1:
            # Calculate volatility of largest eigenvalue
            largest_eigenvalues = [ev[-1] for ev in eigenvalue_series]
            if len(largest_eigenvalues) > 1:
                eigenvalue_volatility = np.std(largest_eigenvalues) / np.mean(largest_eigenvalues)
        
        return {
            'correlation_stability': avg_stability,
            'stability_volatility': stability_volatility,
            'eigenvalue_volatility': eigenvalue_volatility,
            'num_observations': len(correlation_series)
        }
    
    def _update_pca_analysis(self,
                           window: int,
                           corr_matrix: CorrelationMatrix) -> None:
        """Update principal component analysis."""
        eigenvalues = corr_matrix.eigenvalues
        eigenvectors = corr_matrix.eigenvectors
        symbols = corr_matrix.asset_symbols
        
        # Calculate explained variance ratios
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance if total_variance > 0 else eigenvalues
        
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(explained_variance_ratio[::-1])[::-1]
        
        # Factor loadings (correlations between assets and principal components)
        factor_loadings = {}
        for i, symbol in enumerate(symbols):
            loadings = {}
            for j in range(min(3, len(eigenvalues))):  # Top 3 components
                loading = eigenvectors[i, -(j+1)] * np.sqrt(eigenvalues[-(j+1)])
                loadings[f'PC{j+1}'] = loading
            factor_loadings[symbol] = loadings
        
        self.pca_results[window] = {
            'timestamp': corr_matrix.timestamp,
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'factor_loadings': factor_loadings,
            'first_pc_variance': explained_variance_ratio[-1] if len(explained_variance_ratio) > 0 else 0,
            'effective_rank': np.sum(explained_variance_ratio > 0.01)  # Components explaining >1% variance
        }
    
    def get_correlation_matrix(self, window: int = 60) -> Optional[CorrelationMatrix]:
        """Get latest correlation matrix for specified window."""
        if window in self.correlation_matrices and self.correlation_matrices[window]:
            return self.correlation_matrices[window][-1]
        return None
    
    def get_asset_correlation(self,
                            asset1: str,
                            asset2: str,
                            window: int = 60) -> Optional[float]:
        """Get correlation between two specific assets."""
        corr_matrix = self.get_correlation_matrix(window)
        
        if corr_matrix is None:
            return None
        
        try:
            idx1 = corr_matrix.asset_symbols.index(asset1)
            idx2 = corr_matrix.asset_symbols.index(asset2)
            return corr_matrix.correlation_matrix[idx1, idx2]
        except ValueError:
            return None
    
    def get_asset_class_correlation(self,
                                  class1: str,
                                  class2: str) -> Optional[float]:
        """Get average correlation between two asset classes."""
        if not self.correlation_breakdowns:
            return None
        
        latest_breakdown = self.correlation_breakdowns[-1]
        
        # Try both orders
        for pair in [(class1, class2), (class2, class1)]:
            if pair in latest_breakdown.asset_class_correlations:
                return latest_breakdown.asset_class_correlations[pair]
        
        return None
    
    def get_correlation_regime(self, window: int = 60) -> Optional[str]:
        """Get current correlation regime."""
        corr_matrix = self.get_correlation_matrix(window)
        return corr_matrix.market_regime if corr_matrix else None
    
    def get_top_correlations(self,
                           window: int = 60,
                           top_n: int = 10,
                           asset_filter: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """Get top N correlations."""
        if not self.correlation_breakdowns:
            return []
        
        latest_breakdown = self.correlation_breakdowns[-1]
        pairwise = latest_breakdown.pairwise_correlations
        
        # Filter if requested
        if asset_filter:
            filtered_pairs = {
                pair: corr for pair, corr in pairwise.items()
                if pair[0] in asset_filter or pair[1] in asset_filter
            }
        else:
            filtered_pairs = pairwise
        
        # Sort by absolute correlation
        sorted_correlations = sorted(
            filtered_pairs.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return [(pair[0], pair[1], corr) for pair, corr in sorted_correlations[:top_n]]
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get comprehensive correlation summary."""
        summary = {
            'assets_tracked': len(self.asset_data),
            'asset_classes': list(set(self.asset_class_mapping.values())),
            'correlation_windows': self.correlation_windows
        }
        
        # Latest correlation breakdown
        if self.correlation_breakdowns:
            latest_breakdown = self.correlation_breakdowns[-1]
            summary['latest_analysis'] = {
                'timestamp': latest_breakdown.timestamp.isoformat(),
                'concentration_metrics': latest_breakdown.concentration_metrics,
                'stability_metrics': latest_breakdown.stability_metrics,
                'num_pairs': len(latest_breakdown.pairwise_correlations)
            }
        
        # Current regime
        current_regime = self.get_correlation_regime()
        if current_regime:
            summary['current_regime'] = current_regime
        
        # PCA summary
        if 60 in self.pca_results:
            pca = self.pca_results[60]
            summary['pca_analysis'] = {
                'first_pc_variance': pca['first_pc_variance'],
                'effective_rank': pca['effective_rank'],
                'top3_variance': np.sum(pca['explained_variance_ratio'][-3:]) if len(pca['explained_variance_ratio']) >= 3 else 0
            }
        
        return summary
    
    def detect_correlation_breakpoints(self,
                                     window: int = 60,
                                     lookback: int = 20) -> List[Dict[str, Any]]:
        """Detect structural breaks in correlations."""
        if window not in self.correlation_matrices:
            return []
        
        matrices = list(self.correlation_matrices[window])
        if len(matrices) < lookback + 5:
            return []
        
        breakpoints = []
        
        # Analyze recent correlation changes
        recent_matrices = matrices[-lookback:]
        
        for i in range(5, len(recent_matrices)):
            current_matrix = recent_matrices[i]
            prev_matrix = recent_matrices[i-1]
            
            # Calculate matrix difference
            if (current_matrix.correlation_matrix.shape == prev_matrix.correlation_matrix.shape):
                matrix_diff = current_matrix.correlation_matrix - prev_matrix.correlation_matrix
                max_change = np.max(np.abs(matrix_diff))
                
                # Regime change detection
                regime_change = current_matrix.market_regime != prev_matrix.market_regime
                
                if max_change > self.correlation_threshold or regime_change:
                    breakpoint = {
                        'timestamp': current_matrix.timestamp,
                        'max_correlation_change': max_change,
                        'regime_change': regime_change,
                        'old_regime': prev_matrix.market_regime,
                        'new_regime': current_matrix.market_regime,
                        'eigenvalue_change': abs(current_matrix.eigenvalues[-1] - prev_matrix.eigenvalues[-1])
                    }
                    breakpoints.append(breakpoint)
        
        return breakpoints
    
    def get_diversification_ratio(self) -> Optional[float]:
        """Calculate portfolio diversification ratio."""
        corr_matrix = self.get_correlation_matrix()
        
        if corr_matrix is None:
            return None
        
        n_assets = len(corr_matrix.asset_symbols)
        if n_assets < 2:
            return None
        
        # Equal-weighted portfolio
        weights = np.ones(n_assets) / n_assets
        
        # Portfolio variance
        portfolio_var = weights.T @ corr_matrix.correlation_matrix @ weights
        
        # Average individual variance (assuming unit variance)
        individual_var = 1.0
        
        # Diversification ratio
        diversification_ratio = (weights.T @ np.ones(n_assets) * np.sqrt(individual_var)) / np.sqrt(portfolio_var)
        
        return diversification_ratio