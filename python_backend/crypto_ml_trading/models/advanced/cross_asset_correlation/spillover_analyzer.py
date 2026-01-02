"""
Cross-Market Spillover Analysis.

Implements spillover effect analysis between different markets and asset classes.
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
class SpilloverIndex:
    """Spillover index measurement."""
    timestamp: datetime
    total_spillover: float
    directional_spillovers: Dict[str, float]  # From each asset
    net_spillovers: Dict[str, float]  # Net spillover for each asset
    pairwise_spillovers: Dict[Tuple[str, str], float]
    spillover_matrix: np.ndarray
    asset_symbols: List[str]


@dataclass
class SpilloverEvent:
    """Detected spillover event."""
    timestamp: datetime
    event_type: str  # 'shock', 'contagion', 'flight_to_quality'
    source_markets: List[str]
    affected_markets: List[str]
    intensity: float
    duration: Optional[float] = None
    description: str = ""


@dataclass
class ContagionMetrics:
    """Contagion analysis metrics."""
    timestamp: datetime
    contagion_probability: float
    transmission_speed: float
    amplification_factor: float
    vulnerability_scores: Dict[str, float]
    systemic_risk_indicator: float


class SpilloverAnalyzer:
    """
    Cross-market spillover and contagion analysis.
    
    Features:
    - Diebold-Yilmaz spillover index calculation
    - VAR-based spillover decomposition
    - Shock transmission analysis
    - Contagion detection and measurement
    - Flight-to-quality identification
    - Network-based spillover analysis
    - Real-time spillover monitoring
    """
    
    def __init__(self,
                 var_lags: int = 5,
                 forecast_horizon: int = 10,
                 spillover_window: int = 100,
                 shock_threshold: float = 2.0):
        """
        Initialize spillover analyzer.
        
        Args:
            var_lags: Number of lags for VAR model
            forecast_horizon: Forecast horizon for spillover decomposition
            spillover_window: Rolling window for spillover calculation
            shock_threshold: Threshold for shock detection (in standard deviations)
        """
        self.var_lags = var_lags
        self.forecast_horizon = forecast_horizon
        self.spillover_window = spillover_window
        self.shock_threshold = shock_threshold
        
        # Data storage
        self.asset_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.asset_symbols: List[str] = []
        self.market_categories: Dict[str, str] = {}  # Asset -> market category mapping
        
        # Analysis results
        self.spillover_indices: deque = deque(maxlen=200)
        self.spillover_events: deque = deque(maxlen=100)
        self.contagion_metrics: deque = deque(maxlen=100)
        
        # VAR model components
        self.var_coefficients: Optional[np.ndarray] = None
        self.var_residuals: Optional[np.ndarray] = None
        self.var_sigma: Optional[np.ndarray] = None
        
        # Network analysis
        self.spillover_network: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    def add_market_data(self,
                       symbol: str,
                       market_category: str,
                       returns: np.ndarray,
                       timestamps: List[datetime]) -> None:
        """
        Add market data for spillover analysis.
        
        Args:
            symbol: Asset/market symbol
            market_category: Market category (e.g., 'crypto', 'equity', 'bond')
            returns: Return series
            timestamps: Corresponding timestamps
        """
        if symbol not in self.asset_symbols:
            self.asset_symbols.append(symbol)
        
        self.market_categories[symbol] = market_category
        
        # Store data
        for i, (timestamp, return_val) in enumerate(zip(timestamps, returns)):
            self.asset_data[symbol].append({
                'timestamp': timestamp,
                'return': return_val
            })
        
        # Update analysis if we have enough data
        if len(self.asset_data[symbol]) >= self.spillover_window:
            self._update_spillover_analysis()
    
    def update_real_time(self,
                        market_data: Dict[str, float],
                        timestamp: datetime) -> Optional[SpilloverIndex]:
        """
        Update spillover analysis with real-time data.
        
        Args:
            market_data: Dictionary of symbol -> return
            timestamp: Data timestamp
            
        Returns:
            Updated spillover index if available
        """
        # Add new data points
        for symbol, return_val in market_data.items():
            if symbol not in self.asset_symbols:
                self.asset_symbols.append(symbol)
                
            self.asset_data[symbol].append({
                'timestamp': timestamp,
                'return': return_val
            })
        
        # Check for shocks
        self._detect_market_shocks(market_data, timestamp)
        
        # Update spillover analysis
        return self._update_spillover_analysis()
    
    def _detect_market_shocks(self,
                            market_data: Dict[str, float],
                            timestamp: datetime) -> None:
        """Detect market shocks in real-time data."""
        for symbol, return_val in market_data.items():
            if len(self.asset_data[symbol]) < 30:
                continue
            
            # Calculate recent volatility
            recent_returns = [obs['return'] for obs in list(self.asset_data[symbol])[-30:]]
            volatility = np.std(recent_returns)
            
            # Check for shock
            if abs(return_val) > self.shock_threshold * volatility:
                shock_event = SpilloverEvent(
                    timestamp=timestamp,
                    event_type='shock',
                    source_markets=[symbol],
                    affected_markets=[],  # Will be determined by spillover analysis
                    intensity=abs(return_val) / volatility,
                    description=f"Market shock detected in {symbol}"
                )
                self.spillover_events.append(shock_event)
    
    def _update_spillover_analysis(self) -> Optional[SpilloverIndex]:
        """Update spillover index calculation."""
        if len(self.asset_symbols) < 2:
            return None
        
        # Check if we have enough data for all assets
        min_length = min(len(self.asset_data[symbol]) for symbol in self.asset_symbols)
        if min_length < self.spillover_window:
            return None
        
        # Prepare return matrix
        return_matrix = self._prepare_return_matrix()
        
        if return_matrix is None:
            return None
        
        # Estimate VAR model
        self._estimate_var_model(return_matrix)
        
        # Calculate spillover index
        spillover_index = self._calculate_spillover_index()
        
        if spillover_index:
            self.spillover_indices.append(spillover_index)
            
            # Analyze for contagion
            self._analyze_contagion(spillover_index)
            
            # Update spillover network
            self._update_spillover_network(spillover_index)
        
        return spillover_index
    
    def _prepare_return_matrix(self) -> Optional[np.ndarray]:
        """Prepare aligned return matrix for VAR estimation."""
        # Get common time period
        min_length = min(len(self.asset_data[symbol]) for symbol in self.asset_symbols)
        
        if min_length < self.spillover_window:
            return None
        
        # Extract returns for the window
        return_matrix = np.zeros((self.spillover_window, len(self.asset_symbols)))
        
        for j, symbol in enumerate(self.asset_symbols):
            recent_data = list(self.asset_data[symbol])[-self.spillover_window:]
            returns = [obs['return'] for obs in recent_data]
            return_matrix[:, j] = returns
        
        return return_matrix
    
    def _estimate_var_model(self, return_matrix: np.ndarray) -> None:
        """Estimate VAR model for spillover analysis."""
        T, n = return_matrix.shape
        
        # Create lagged matrix
        Y = return_matrix[self.var_lags:, :]  # Dependent variables
        X = np.ones((T - self.var_lags, 1))  # Constant term
        
        # Add lagged variables
        for lag in range(1, self.var_lags + 1):
            X_lag = return_matrix[self.var_lags - lag:-lag, :]
            X = np.concatenate([X, X_lag], axis=1)
        
        # OLS estimation for each equation
        try:
            self.var_coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
            
            # Calculate residuals
            Y_fitted = X @ self.var_coefficients
            self.var_residuals = Y - Y_fitted
            
            # Residual covariance matrix
            self.var_sigma = np.cov(self.var_residuals.T)
            
        except Exception as e:
            print(f"VAR estimation failed: {e}")
            # Use identity matrices as fallback
            self.var_coefficients = np.zeros((1 + n * self.var_lags, n))
            self.var_residuals = np.random.randn(T - self.var_lags, n) * 0.01
            self.var_sigma = np.eye(n) * 0.01
    
    def _calculate_spillover_index(self) -> Optional[SpilloverIndex]:
        """Calculate Diebold-Yilmaz spillover index."""
        if (self.var_coefficients is None or 
            self.var_residuals is None or 
            self.var_sigma is None):
            return None
        
        n = len(self.asset_symbols)
        
        # Generalized impulse response functions
        ma_coefficients = self._calculate_ma_representation()
        
        # Forecast error variance decomposition
        fevd_matrix = self._calculate_fevd(ma_coefficients)
        
        # Spillover calculations
        total_spillover = self._calculate_total_spillover(fevd_matrix)
        directional_spillovers = self._calculate_directional_spillovers(fevd_matrix)
        net_spillovers = self._calculate_net_spillovers(fevd_matrix)
        pairwise_spillovers = self._calculate_pairwise_spillovers(fevd_matrix)
        
        return SpilloverIndex(
            timestamp=datetime.now(),
            total_spillover=total_spillover,
            directional_spillovers=directional_spillovers,
            net_spillovers=net_spillovers,
            pairwise_spillovers=pairwise_spillovers,
            spillover_matrix=fevd_matrix,
            asset_symbols=self.asset_symbols.copy()
        )
    
    def _calculate_ma_representation(self) -> np.ndarray:
        """Calculate MA representation from VAR coefficients."""
        n = len(self.asset_symbols)
        
        # Initialize MA coefficient matrices
        ma_coeffs = np.zeros((self.forecast_horizon, n, n))
        
        # MA(0) coefficient (identity for structural shocks)
        ma_coeffs[0] = np.eye(n)
        
        # Calculate MA coefficients recursively
        var_coeffs_matrices = self.var_coefficients[1:].reshape(self.var_lags, n, n)
        
        for h in range(1, self.forecast_horizon):
            ma_coeffs[h] = 0
            for j in range(min(h, self.var_lags)):
                if h - j - 1 >= 0:
                    ma_coeffs[h] += ma_coeffs[h - j - 1] @ var_coeffs_matrices[j]
        
        return ma_coeffs
    
    def _calculate_fevd(self, ma_coefficients: np.ndarray) -> np.ndarray:
        """Calculate forecast error variance decomposition."""
        n = len(self.asset_symbols)
        
        # Cholesky decomposition of residual covariance
        try:
            P = np.linalg.cholesky(self.var_sigma)
        except:
            # If Cholesky fails, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(self.var_sigma)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
            P = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Calculate generalized impulse responses
        impulse_responses = np.zeros((self.forecast_horizon, n, n))
        
        for h in range(self.forecast_horizon):
            impulse_responses[h] = ma_coefficients[h] @ P
        
        # Calculate FEVD
        fevd = np.zeros((n, n))
        
        for i in range(n):
            # Forecast error variance for variable i
            total_variance = 0
            variance_contributions = np.zeros(n)
            
            for h in range(self.forecast_horizon):
                for j in range(n):
                    contribution = impulse_responses[h, i, j] ** 2
                    variance_contributions[j] += contribution
                    total_variance += contribution
            
            # Normalize to get proportions
            if total_variance > 0:
                fevd[i, :] = variance_contributions / total_variance
            else:
                fevd[i, i] = 1.0  # Self-contribution = 100%
        
        return fevd
    
    def _calculate_total_spillover(self, fevd_matrix: np.ndarray) -> float:
        """Calculate total spillover index."""
        n = len(self.asset_symbols)
        
        # Total spillover = sum of off-diagonal elements / sum of all elements
        off_diagonal_sum = np.sum(fevd_matrix) - np.trace(fevd_matrix)
        total_sum = np.sum(fevd_matrix)
        
        return (off_diagonal_sum / total_sum * 100) if total_sum > 0 else 0.0
    
    def _calculate_directional_spillovers(self, fevd_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate directional spillover indices."""
        directional = {}
        
        for i, symbol in enumerate(self.asset_symbols):
            # Spillover FROM symbol i TO others
            spillover_from = np.sum(fevd_matrix[:, i]) - fevd_matrix[i, i]
            total_variance = np.sum(fevd_matrix[:, i])
            
            directional[symbol] = (spillover_from / total_variance * 100) if total_variance > 0 else 0.0
        
        return directional
    
    def _calculate_net_spillovers(self, fevd_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate net spillover indices."""
        net_spillovers = {}
        
        for i, symbol in enumerate(self.asset_symbols):
            # Spillover FROM symbol i TO others
            spillover_to_others = np.sum(fevd_matrix[:, i]) - fevd_matrix[i, i]
            
            # Spillover FROM others TO symbol i
            spillover_from_others = np.sum(fevd_matrix[i, :]) - fevd_matrix[i, i]
            
            # Net spillover
            net_spillovers[symbol] = spillover_to_others - spillover_from_others
        
        return net_spillovers
    
    def _calculate_pairwise_spillovers(self, fevd_matrix: np.ndarray) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise spillover indices."""
        pairwise = {}
        
        for i, symbol_i in enumerate(self.asset_symbols):
            for j, symbol_j in enumerate(self.asset_symbols):
                if i != j:
                    # Spillover from j to i
                    spillover = fevd_matrix[i, j] * 100
                    pairwise[(symbol_j, symbol_i)] = spillover
        
        return pairwise
    
    def _analyze_contagion(self, spillover_index: SpilloverIndex) -> None:
        """Analyze for contagion effects."""
        # Calculate contagion probability based on spillover patterns
        total_spillover = spillover_index.total_spillover
        
        # High spillover suggests potential contagion
        contagion_prob = min(1.0, total_spillover / 50.0)  # Normalize to 0-1
        
        # Calculate transmission speed (change in spillovers)
        transmission_speed = 0.0
        if len(self.spillover_indices) > 1:
            prev_spillover = self.spillover_indices[-2].total_spillover
            transmission_speed = abs(total_spillover - prev_spillover)
        
        # Amplification factor (how much spillovers exceed normal levels)
        if len(self.spillover_indices) > 10:
            avg_spillover = np.mean([si.total_spillover for si in list(self.spillover_indices)[-10:]])
            amplification_factor = total_spillover / avg_spillover if avg_spillover > 0 else 1.0
        else:
            amplification_factor = 1.0
        
        # Vulnerability scores (assets most susceptible to spillovers)
        vulnerability_scores = {}
        for symbol in self.asset_symbols:
            # High incoming spillover = high vulnerability
            if symbol in spillover_index.net_spillovers:
                net_spillover = spillover_index.net_spillovers[symbol]
                vulnerability_scores[symbol] = max(0, -net_spillover / 10.0)  # Negative net = vulnerable
        
        # Systemic risk indicator
        systemic_risk = min(1.0, contagion_prob * amplification_factor)
        
        contagion_metrics = ContagionMetrics(
            timestamp=spillover_index.timestamp,
            contagion_probability=contagion_prob,
            transmission_speed=transmission_speed,
            amplification_factor=amplification_factor,
            vulnerability_scores=vulnerability_scores,
            systemic_risk_indicator=systemic_risk
        )
        
        self.contagion_metrics.append(contagion_metrics)
        
        # Detect contagion events
        if contagion_prob > 0.7 and amplification_factor > 1.5:
            # Identify source and affected markets
            net_spillovers = spillover_index.net_spillovers
            source_markets = [symbol for symbol, net in net_spillovers.items() if net > 5.0]
            affected_markets = [symbol for symbol, net in net_spillovers.items() if net < -5.0]
            
            if source_markets and affected_markets:
                contagion_event = SpilloverEvent(
                    timestamp=spillover_index.timestamp,
                    event_type='contagion',
                    source_markets=source_markets,
                    affected_markets=affected_markets,
                    intensity=systemic_risk,
                    description=f"Contagion detected from {source_markets} to {affected_markets}"
                )
                self.spillover_events.append(contagion_event)
    
    def _update_spillover_network(self, spillover_index: SpilloverIndex) -> None:
        """Update spillover network representation."""
        # Clear old network
        self.spillover_network.clear()
        
        # Build network from pairwise spillovers
        for (source, target), spillover in spillover_index.pairwise_spillovers.items():
            if spillover > 1.0:  # Only significant spillovers
                if source not in self.spillover_network:
                    self.spillover_network[source] = {}
                self.spillover_network[source][target] = spillover
    
    def detect_flight_to_quality(self) -> List[SpilloverEvent]:
        """Detect flight-to-quality events."""
        if not self.spillover_indices or len(self.spillover_indices) < 2:
            return []
        
        events = []
        current_spillover = self.spillover_indices[-1]
        
        # Look for patterns indicating flight to quality
        # - High spillovers from risky to safe assets
        # - Low spillovers from safe to risky assets
        
        safe_havens = []
        risky_assets = []
        
        for symbol in self.asset_symbols:
            market_category = self.market_categories.get(symbol, 'unknown')
            
            if market_category in ['bond', 'gold', 'usd']:
                safe_havens.append(symbol)
            elif market_category in ['crypto', 'equity']:
                risky_assets.append(symbol)
        
        if safe_havens and risky_assets:
            # Calculate average spillover from risky to safe
            risky_to_safe_spillover = 0.0
            count = 0
            
            for risky in risky_assets:
                for safe in safe_havens:
                    key = (risky, safe)
                    if key in current_spillover.pairwise_spillovers:
                        risky_to_safe_spillover += current_spillover.pairwise_spillovers[key]
                        count += 1
            
            if count > 0:
                avg_spillover = risky_to_safe_spillover / count
                
                # Threshold for flight-to-quality
                if avg_spillover > 15.0:  # High spillover threshold
                    flight_event = SpilloverEvent(
                        timestamp=current_spillover.timestamp,
                        event_type='flight_to_quality',
                        source_markets=risky_assets,
                        affected_markets=safe_havens,
                        intensity=avg_spillover / 20.0,  # Normalize
                        description="Flight-to-quality pattern detected"
                    )
                    events.append(flight_event)
                    self.spillover_events.append(flight_event)
        
        return events
    
    def get_spillover_summary(self) -> Dict[str, Any]:
        """Get comprehensive spillover analysis summary."""
        summary = {
            'assets_analyzed': len(self.asset_symbols),
            'market_categories': dict(set(self.market_categories.items())),
            'analysis_window': self.spillover_window
        }
        
        if self.spillover_indices:
            latest_spillover = self.spillover_indices[-1]
            summary['latest_spillover'] = {
                'timestamp': latest_spillover.timestamp.isoformat(),
                'total_spillover': latest_spillover.total_spillover,
                'top_spillover_sources': sorted(
                    latest_spillover.directional_spillovers.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3],
                'most_vulnerable': sorted(
                    latest_spillover.net_spillovers.items(),
                    key=lambda x: x[1]
                )[:3]
            }
        
        if self.contagion_metrics:
            latest_contagion = self.contagion_metrics[-1]
            summary['contagion_analysis'] = {
                'contagion_probability': latest_contagion.contagion_probability,
                'systemic_risk_indicator': latest_contagion.systemic_risk_indicator,
                'transmission_speed': latest_contagion.transmission_speed
            }
        
        # Recent events
        recent_events = list(self.spillover_events)[-5:]
        summary['recent_events'] = [
            {
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type,
                'intensity': event.intensity,
                'description': event.description
            }
            for event in recent_events
        ]
        
        return summary
    
    def get_market_vulnerability_ranking(self) -> List[Tuple[str, float]]:
        """Get ranking of markets by vulnerability to spillovers."""
        if not self.contagion_metrics:
            return []
        
        latest_metrics = self.contagion_metrics[-1]
        vulnerability_scores = latest_metrics.vulnerability_scores
        
        return sorted(vulnerability_scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_spillover_network_metrics(self) -> Dict[str, Any]:
        """Get network-based spillover metrics."""
        if not self.spillover_network:
            return {}
        
        # Calculate network centrality measures
        in_degree = defaultdict(float)
        out_degree = defaultdict(float)
        
        for source, targets in self.spillover_network.items():
            out_degree[source] = sum(targets.values())
            for target, weight in targets.items():
                in_degree[target] += weight
        
        # Most central nodes (highest total spillover activity)
        centrality = {}
        for node in self.asset_symbols:
            centrality[node] = in_degree[node] + out_degree[node]
        
        return {
            'network_density': len(self.spillover_network) / len(self.asset_symbols) if self.asset_symbols else 0,
            'most_central_assets': sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5],
            'spillover_transmitters': sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:3],
            'spillover_receivers': sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:3]
        }