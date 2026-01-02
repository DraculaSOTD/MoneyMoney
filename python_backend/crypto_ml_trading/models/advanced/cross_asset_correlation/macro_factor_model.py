"""
Macro Factor Model for Cross-Asset Analysis.

Implements factor models linking cryptocurrency markets to macro-economic factors.
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
class MacroFactor:
    """Macro-economic factor definition."""
    name: str
    category: str  # 'monetary', 'economic', 'sentiment', 'volatility'
    description: str
    data: deque = field(default_factory=lambda: deque(maxlen=1000))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorExposure:
    """Factor exposure measurement."""
    timestamp: datetime
    asset_symbol: str
    factor_loadings: Dict[str, float]
    factor_contributions: Dict[str, float]
    idiosyncratic_risk: float
    systematic_risk: float
    r_squared: float


@dataclass
class FactorRegimeChange:
    """Factor regime change detection."""
    timestamp: datetime
    factor_name: str
    old_regime: str
    new_regime: str
    confidence: float
    impact_assessment: Dict[str, float]


class MacroFactorModel:
    """
    Macro factor model for cross-asset correlation analysis.
    
    Features:
    - Multi-factor model estimation
    - Dynamic factor loading analysis
    - Regime change detection in factor relationships
    - Risk attribution and decomposition
    - Macro scenario analysis
    - Factor forecasting and stress testing
    """
    
    def __init__(self,
                 estimation_window: int = 252,
                 factor_categories: List[str] = None,
                 regime_detection_window: int = 60):
        """
        Initialize macro factor model.
        
        Args:
            estimation_window: Window for factor model estimation
            factor_categories: Categories of macro factors to include
            regime_detection_window: Window for regime change detection
        """
        self.estimation_window = estimation_window
        self.factor_categories = factor_categories or [
            'monetary', 'economic', 'sentiment', 'volatility', 'geopolitical'
        ]
        self.regime_detection_window = regime_detection_window
        
        # Factor definitions
        self.macro_factors: Dict[str, MacroFactor] = {}
        self.asset_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Model components
        self.factor_loadings: Dict[str, Dict[str, float]] = defaultdict(dict)  # asset -> factor -> loading
        self.factor_model_results: Dict[str, Dict] = {}  # asset -> model results
        
        # Analysis results
        self.factor_exposures: deque = deque(maxlen=500)
        self.regime_changes: deque = deque(maxlen=100)
        
        # Model state
        self.is_fitted = False
        self.last_estimation: Optional[datetime] = None
        
        # Initialize common macro factors
        self._initialize_common_factors()
    
    def _initialize_common_factors(self) -> None:
        """Initialize common macro-economic factors."""
        factor_definitions = [
            # Monetary factors
            ('fed_funds_rate', 'monetary', 'Federal funds rate'),
            ('10y_treasury', 'monetary', '10-year Treasury yield'),
            ('yield_curve_slope', 'monetary', '10Y-2Y yield spread'),
            ('dollar_index', 'monetary', 'US Dollar Index (DXY)'),
            ('real_rates', 'monetary', 'Real interest rates'),
            
            # Economic factors
            ('gdp_growth', 'economic', 'GDP growth rate'),
            ('inflation_rate', 'economic', 'Consumer Price Index'),
            ('unemployment', 'economic', 'Unemployment rate'),
            ('pmi_manufacturing', 'economic', 'Manufacturing PMI'),
            ('consumer_confidence', 'economic', 'Consumer confidence index'),
            
            # Market sentiment factors
            ('vix', 'sentiment', 'VIX volatility index'),
            ('put_call_ratio', 'sentiment', 'Put/call ratio'),
            ('high_yield_spreads', 'sentiment', 'High yield credit spreads'),
            ('risk_parity_factor', 'sentiment', 'Risk parity performance'),
            
            # Volatility factors
            ('realized_vol_equity', 'volatility', 'Realized volatility of equity markets'),
            ('realized_vol_bonds', 'volatility', 'Realized volatility of bond markets'),
            ('vol_of_vol', 'volatility', 'Volatility of volatility'),
            
            # Geopolitical factors
            ('geopolitical_risk', 'geopolitical', 'Geopolitical risk index'),
            ('policy_uncertainty', 'geopolitical', 'Economic policy uncertainty')
        ]
        
        for name, category, description in factor_definitions:
            self.macro_factors[name] = MacroFactor(
                name=name,
                category=category,
                description=description
            )
    
    def add_factor_data(self,
                       factor_name: str,
                       values: np.ndarray,
                       timestamps: List[datetime],
                       metadata: Optional[Dict] = None) -> None:
        """
        Add macro factor data.
        
        Args:
            factor_name: Name of the macro factor
            values: Factor values
            timestamps: Corresponding timestamps
            metadata: Additional metadata
        """
        if factor_name not in self.macro_factors:
            # Create new factor if not exists
            self.macro_factors[factor_name] = MacroFactor(
                name=factor_name,
                category='custom',
                description=f'Custom factor: {factor_name}',
                metadata=metadata or {}
            )
        
        factor = self.macro_factors[factor_name]
        
        # Add data points
        for timestamp, value in zip(timestamps, values):
            factor.data.append({
                'timestamp': timestamp,
                'value': value
            })
    
    def add_asset_returns(self,
                         symbol: str,
                         returns: np.ndarray,
                         timestamps: List[datetime]) -> None:
        """
        Add asset return data for factor analysis.
        
        Args:
            symbol: Asset symbol
            returns: Return series
            timestamps: Corresponding timestamps
        """
        for timestamp, return_val in zip(timestamps, returns):
            self.asset_returns[symbol].append({
                'timestamp': timestamp,
                'return': return_val
            })
    
    def estimate_factor_model(self,
                            target_assets: Optional[List[str]] = None,
                            factor_subset: Optional[List[str]] = None) -> None:
        """
        Estimate factor model for specified assets.
        
        Args:
            target_assets: Assets to analyze (default: all available)
            factor_subset: Subset of factors to use (default: all available)
        """
        if target_assets is None:
            target_assets = list(self.asset_returns.keys())
        
        if factor_subset is None:
            factor_subset = [name for name, factor in self.macro_factors.items() 
                           if len(factor.data) >= self.estimation_window]
        
        # Estimate model for each asset
        for asset in target_assets:
            if len(self.asset_returns[asset]) >= self.estimation_window:
                model_result = self._estimate_single_asset_model(asset, factor_subset)
                self.factor_model_results[asset] = model_result
        
        self.is_fitted = True
        self.last_estimation = datetime.now()
    
    def _estimate_single_asset_model(self,
                                   asset: str,
                                   factor_names: List[str]) -> Dict[str, Any]:
        """Estimate factor model for a single asset."""
        # Prepare data
        asset_data, factor_data = self._prepare_regression_data(asset, factor_names)
        
        if asset_data is None or factor_data is None:
            return {'error': 'Insufficient data'}
        
        # Run factor regression
        try:
            # Add constant term
            X = np.column_stack([np.ones(len(factor_data)), factor_data])
            y = asset_data
            
            # OLS estimation
            coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Calculate fitted values and residuals
            y_fitted = X @ coefficients
            residuals = y - y_fitted
            
            # Model statistics
            alpha = coefficients[0]  # Intercept
            factor_loadings = coefficients[1:]  # Factor loadings
            
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Standard errors (simplified)
            mse = ss_res / max(1, len(y) - len(coefficients))
            var_coeff = mse * np.linalg.pinv(X.T @ X)
            std_errors = np.sqrt(np.diag(var_coeff))
            
            # T-statistics
            t_stats = coefficients / std_errors
            
            # Risk decomposition
            systematic_variance = np.var(y_fitted)
            idiosyncratic_variance = np.var(residuals)
            total_variance = systematic_variance + idiosyncratic_variance
            
            # Factor contributions to risk
            factor_contributions = {}
            for i, factor_name in enumerate(factor_names):
                loading = factor_loadings[i]
                factor_var = np.var(factor_data[:, i])
                contribution = (loading ** 2) * factor_var
                factor_contributions[factor_name] = contribution / total_variance if total_variance > 0 else 0
            
            # Store factor loadings
            loading_dict = {}
            for i, factor_name in enumerate(factor_names):
                loading_dict[factor_name] = factor_loadings[i]
            
            self.factor_loadings[asset] = loading_dict
            
            return {
                'alpha': alpha,
                'factor_loadings': loading_dict,
                'factor_contributions': factor_contributions,
                'r_squared': r_squared,
                'systematic_risk': systematic_variance / total_variance if total_variance > 0 else 0,
                'idiosyncratic_risk': idiosyncratic_variance / total_variance if total_variance > 0 else 1,
                'residuals': residuals,
                't_statistics': dict(zip(['alpha'] + factor_names, t_stats)),
                'standard_errors': dict(zip(['alpha'] + factor_names, std_errors))
            }
            
        except Exception as e:
            return {'error': f'Estimation failed: {str(e)}'}
    
    def _prepare_regression_data(self,
                               asset: str,
                               factor_names: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare aligned data for factor regression."""
        # Get asset returns
        asset_data_list = list(self.asset_returns[asset])[-self.estimation_window:]
        
        if len(asset_data_list) < self.estimation_window:
            return None, None
        
        # Get factor data aligned with asset data
        asset_timestamps = [obs['timestamp'] for obs in asset_data_list]
        asset_returns = [obs['return'] for obs in asset_data_list]
        
        factor_matrix = []
        
        for factor_name in factor_names:
            factor = self.macro_factors[factor_name]
            factor_series = []
            
            for timestamp in asset_timestamps:
                # Find closest factor observation
                closest_factor_obs = self._find_closest_factor_observation(factor, timestamp)
                
                if closest_factor_obs is not None:
                    factor_series.append(closest_factor_obs['value'])
                else:
                    factor_series.append(0.0)  # Default value
            
            factor_matrix.append(factor_series)
        
        if not factor_matrix:
            return None, None
        
        factor_data = np.column_stack(factor_matrix)
        asset_data = np.array(asset_returns)
        
        return asset_data, factor_data
    
    def _find_closest_factor_observation(self,
                                       factor: MacroFactor,
                                       target_timestamp: datetime) -> Optional[Dict]:
        """Find closest factor observation to target timestamp."""
        if not factor.data:
            return None
        
        closest_obs = None
        min_time_diff = float('inf')
        
        for obs in factor.data:
            time_diff = abs((obs['timestamp'] - target_timestamp).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_obs = obs
        
        # Only use observation if within reasonable time window (e.g., 7 days)
        if min_time_diff <= 7 * 24 * 3600:  # 7 days in seconds
            return closest_obs
        
        return None
    
    def update_factor_exposures(self, timestamp: Optional[datetime] = None) -> None:
        """Update factor exposures for all assets."""
        if not self.is_fitted:
            return
        
        timestamp = timestamp or datetime.now()
        
        for asset in self.factor_model_results:
            model_result = self.factor_model_results[asset]
            
            if 'error' in model_result:
                continue
            
            exposure = FactorExposure(
                timestamp=timestamp,
                asset_symbol=asset,
                factor_loadings=model_result['factor_loadings'],
                factor_contributions=model_result['factor_contributions'],
                idiosyncratic_risk=model_result['idiosyncratic_risk'],
                systematic_risk=model_result['systematic_risk'],
                r_squared=model_result['r_squared']
            )
            
            self.factor_exposures.append(exposure)
    
    def detect_regime_changes(self) -> List[FactorRegimeChange]:
        """Detect regime changes in factor relationships."""
        if len(self.factor_exposures) < self.regime_detection_window * 2:
            return []
        
        regime_changes = []
        
        # Group exposures by asset
        asset_exposures = defaultdict(list)
        for exposure in self.factor_exposures:
            asset_exposures[exposure.asset_symbol].append(exposure)
        
        for asset, exposures in asset_exposures.items():
            if len(exposures) < self.regime_detection_window * 2:
                continue
            
            # Analyze each factor for regime changes
            for factor_name in self.macro_factors:
                if factor_name in exposures[0].factor_loadings:
                    regime_change = self._detect_factor_regime_change(
                        asset, factor_name, exposures
                    )
                    if regime_change:
                        regime_changes.append(regime_change)
                        self.regime_changes.append(regime_change)
        
        return regime_changes
    
    def _detect_factor_regime_change(self,
                                   asset: str,
                                   factor_name: str,
                                   exposures: List[FactorExposure]) -> Optional[FactorRegimeChange]:
        """Detect regime change for a specific factor."""
        # Extract factor loadings over time
        loadings = [exp.factor_loadings.get(factor_name, 0) for exp in exposures]
        
        if len(loadings) < self.regime_detection_window * 2:
            return None
        
        # Split into recent and historical periods
        recent_loadings = loadings[-self.regime_detection_window:]
        historical_loadings = loadings[-2*self.regime_detection_window:-self.regime_detection_window]
        
        # Calculate statistics for each period
        recent_mean = np.mean(recent_loadings)
        historical_mean = np.mean(historical_loadings)
        
        recent_std = np.std(recent_loadings)
        historical_std = np.std(historical_loadings)
        
        # Test for structural break (simplified)
        mean_change = abs(recent_mean - historical_mean)
        pooled_std = np.sqrt((recent_std**2 + historical_std**2) / 2)
        
        # Z-test for difference in means
        if pooled_std > 0:
            z_score = mean_change / (pooled_std * np.sqrt(2 / self.regime_detection_window))
            confidence = min(1.0, z_score / 2.0)  # Convert to 0-1 scale
        else:
            confidence = 0.0
        
        # Threshold for regime change detection
        if confidence > 0.8:  # High confidence threshold
            # Classify regimes
            old_regime = self._classify_factor_regime(historical_mean, factor_name)
            new_regime = self._classify_factor_regime(recent_mean, factor_name)
            
            if old_regime != new_regime:
                # Assess impact
                impact_assessment = self._assess_regime_change_impact(
                    asset, factor_name, historical_mean, recent_mean
                )
                
                return FactorRegimeChange(
                    timestamp=exposures[-1].timestamp,
                    factor_name=factor_name,
                    old_regime=old_regime,
                    new_regime=new_regime,
                    confidence=confidence,
                    impact_assessment=impact_assessment
                )
        
        return None
    
    def _classify_factor_regime(self, loading: float, factor_name: str) -> str:
        """Classify factor regime based on loading magnitude."""
        abs_loading = abs(loading)
        
        if abs_loading < 0.1:
            return 'neutral'
        elif abs_loading < 0.3:
            return 'low_sensitivity'
        elif abs_loading < 0.7:
            return 'moderate_sensitivity'
        else:
            return 'high_sensitivity'
    
    def _assess_regime_change_impact(self,
                                   asset: str,
                                   factor_name: str,
                                   old_loading: float,
                                   new_loading: float) -> Dict[str, float]:
        """Assess impact of regime change."""
        # Calculate change in sensitivity
        sensitivity_change = abs(new_loading - old_loading)
        
        # Calculate change in risk attribution
        # (simplified - would need factor volatility for accurate calculation)
        factor_vol = 0.02  # Assumed factor volatility
        old_contribution = (old_loading * factor_vol) ** 2
        new_contribution = (new_loading * factor_vol) ** 2
        risk_change = abs(new_contribution - old_contribution)
        
        return {
            'sensitivity_change': sensitivity_change,
            'risk_contribution_change': risk_change,
            'directional_change': 1.0 if (new_loading > 0) != (old_loading > 0) else 0.0
        }
    
    def perform_scenario_analysis(self,
                                scenario_shocks: Dict[str, float],
                                target_assets: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Perform scenario analysis with factor shocks.
        
        Args:
            scenario_shocks: Dictionary of factor_name -> shock magnitude
            target_assets: Assets to analyze (default: all)
            
        Returns:
            Dictionary of asset -> impact metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scenario analysis")
        
        if target_assets is None:
            target_assets = list(self.factor_model_results.keys())
        
        scenario_results = {}
        
        for asset in target_assets:
            if asset not in self.factor_loadings:
                continue
            
            loadings = self.factor_loadings[asset]
            
            # Calculate impact from each factor shock
            factor_impacts = {}
            total_impact = 0.0
            
            for factor_name, shock in scenario_shocks.items():
                if factor_name in loadings:
                    loading = loadings[factor_name]
                    impact = loading * shock
                    factor_impacts[factor_name] = impact
                    total_impact += impact
            
            # Additional metrics
            model_result = self.factor_model_results.get(asset, {})
            systematic_risk = model_result.get('systematic_risk', 0.5)
            
            # Scale impact by systematic risk (idiosyncratic component unaffected)
            scaled_impact = total_impact * systematic_risk
            
            scenario_results[asset] = {
                'total_impact': total_impact,
                'scaled_impact': scaled_impact,
                'factor_impacts': factor_impacts,
                'systematic_risk': systematic_risk
            }
        
        return scenario_results
    
    def get_factor_attribution(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get factor attribution for a specific asset."""
        if asset not in self.factor_model_results:
            return None
        
        model_result = self.factor_model_results[asset]
        
        if 'error' in model_result:
            return {'error': model_result['error']}
        
        return {
            'factor_loadings': model_result['factor_loadings'],
            'factor_contributions': model_result['factor_contributions'],
            'systematic_risk': model_result['systematic_risk'],
            'idiosyncratic_risk': model_result['idiosyncratic_risk'],
            'alpha': model_result['alpha'],
            'r_squared': model_result['r_squared']
        }
    
    def get_factor_summary(self) -> Dict[str, Any]:
        """Get comprehensive factor model summary."""
        summary = {
            'model_status': 'fitted' if self.is_fitted else 'not_fitted',
            'assets_analyzed': len(self.factor_model_results),
            'factors_available': len(self.macro_factors),
            'last_estimation': self.last_estimation.isoformat() if self.last_estimation else None
        }
        
        # Factor category breakdown
        category_counts = defaultdict(int)
        for factor in self.macro_factors.values():
            category_counts[factor.category] += 1
        summary['factors_by_category'] = dict(category_counts)
        
        # Model performance summary
        if self.factor_model_results:
            r_squared_values = []
            systematic_risk_values = []
            
            for asset, result in self.factor_model_results.items():
                if 'r_squared' in result:
                    r_squared_values.append(result['r_squared'])
                if 'systematic_risk' in result:
                    systematic_risk_values.append(result['systematic_risk'])
            
            if r_squared_values:
                summary['model_performance'] = {
                    'avg_r_squared': np.mean(r_squared_values),
                    'avg_systematic_risk': np.mean(systematic_risk_values),
                    'best_fit_asset': max(self.factor_model_results.items(), 
                                        key=lambda x: x[1].get('r_squared', 0))[0]
                }
        
        # Recent regime changes
        if self.regime_changes:
            recent_changes = list(self.regime_changes)[-5:]
            summary['recent_regime_changes'] = [
                {
                    'timestamp': change.timestamp.isoformat(),
                    'factor': change.factor_name,
                    'old_regime': change.old_regime,
                    'new_regime': change.new_regime,
                    'confidence': change.confidence
                }
                for change in recent_changes
            ]
        
        return summary
    
    def get_top_factor_exposures(self,
                               asset: str,
                               top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top factor exposures for an asset."""
        if asset not in self.factor_loadings:
            return []
        
        loadings = self.factor_loadings[asset]
        
        # Sort by absolute loading magnitude
        sorted_loadings = sorted(
            loadings.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_loadings[:top_n]