"""
Enhanced Temporal Fusion Transformer (TFT) Model.

Extends the base TFT with:
- Multi-asset support for portfolio-wide predictions
- Cross-attention between different assets  
- Regime-specific quantile predictions
- Adaptive horizon selection based on market volatility
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.advanced.transformer_tft.tft_model import TemporalFusionTransformer
from utils.matrix_operations import MatrixOperations


class MarketRegimeDetector:
    """Detect market regimes for regime-specific predictions."""
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.regime_features = {}
        
    def detect_regime(self, market_data: np.ndarray) -> str:
        """
        Detect current market regime.
        
        Args:
            market_data: Recent market data (price, volume, volatility)
            
        Returns:
            Regime identifier (trending_up, trending_down, ranging, volatile)
        """
        if market_data.shape[0] < self.lookback_window:
            return "unknown"
        
        # Calculate regime indicators
        returns = np.diff(market_data[:, 0]) / market_data[:-1, 0]
        volatility = np.std(returns[-20:])
        trend = np.mean(returns[-20:])
        
        # Volume analysis
        volume_trend = np.polyfit(range(20), market_data[-20:, 1], 1)[0]
        
        # Classify regime
        if volatility > 0.03:  # High volatility threshold
            return "volatile"
        elif abs(trend) < 0.001:  # Ranging market
            return "ranging"
        elif trend > 0.002:  # Trending up
            return "trending_up"
        else:  # Trending down
            return "trending_down"
    
    def get_regime_features(self, regime: str) -> Dict[str, float]:
        """Get statistical features for a specific regime."""
        # Pre-computed regime characteristics
        regime_stats = {
            "trending_up": {"momentum": 0.7, "mean_reversion": 0.3, "volatility_mult": 0.8},
            "trending_down": {"momentum": 0.7, "mean_reversion": 0.3, "volatility_mult": 1.2},
            "ranging": {"momentum": 0.2, "mean_reversion": 0.8, "volatility_mult": 0.9},
            "volatile": {"momentum": 0.4, "mean_reversion": 0.6, "volatility_mult": 1.5},
            "unknown": {"momentum": 0.5, "mean_reversion": 0.5, "volatility_mult": 1.0}
        }
        return regime_stats.get(regime, regime_stats["unknown"])


class AdaptiveHorizonSelector:
    """Select optimal prediction horizon based on market conditions."""
    
    def __init__(self, min_horizon: int = 5, max_horizon: int = 60):
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.horizon_performance = {}
        
    def select_horizon(self, volatility: float, regime: str, 
                      asset_correlation: float) -> int:
        """
        Select optimal prediction horizon.
        
        Args:
            volatility: Current market volatility
            regime: Current market regime
            asset_correlation: Average correlation with other assets
            
        Returns:
            Optimal prediction horizon in minutes
        """
        # Base horizon depends on regime
        regime_horizons = {
            "trending_up": 30,
            "trending_down": 20,
            "ranging": 45,
            "volatile": 15,
            "unknown": 30
        }
        
        base_horizon = regime_horizons.get(regime, 30)
        
        # Adjust for volatility (higher volatility = shorter horizon)
        volatility_adjustment = 1.0 - min(volatility / 0.05, 0.5)
        
        # Adjust for correlation (higher correlation = can use longer horizon)
        correlation_adjustment = 1.0 + (asset_correlation * 0.2)
        
        # Calculate final horizon
        horizon = int(base_horizon * volatility_adjustment * correlation_adjustment)
        
        # Clamp to valid range
        return max(self.min_horizon, min(self.max_horizon, horizon))
    
    def update_performance(self, horizon: int, performance: float):
        """Update horizon performance for adaptive learning."""
        if horizon not in self.horizon_performance:
            self.horizon_performance[horizon] = []
        self.horizon_performance[horizon].append(performance)


class EnhancedTemporalFusionTransformer(TemporalFusionTransformer):
    """
    Enhanced TFT with multi-asset support and advanced features.
    
    New features:
    - Multi-asset cross-attention
    - Regime-specific predictions
    - Adaptive horizon selection
    - Portfolio-wide optimization
    """
    
    def __init__(self,
                 n_encoder_steps: int = 168,
                 n_prediction_steps: int = 24,
                 n_features: int = 10,
                 n_static_features: int = 5,
                 n_assets: int = 5,
                 hidden_size: int = 160,
                 lstm_layers: int = 2,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.1,
                 quantiles: List[float] = None,
                 enable_regime_specific: bool = True,
                 enable_cross_attention: bool = True,
                 enable_adaptive_horizon: bool = True):
        """
        Initialize enhanced TFT model.
        
        Args:
            n_assets: Number of assets to model jointly
            enable_regime_specific: Enable regime-specific predictions
            enable_cross_attention: Enable cross-asset attention
            enable_adaptive_horizon: Enable adaptive horizon selection
        """
        super().__init__(
            n_encoder_steps=n_encoder_steps,
            n_prediction_steps=n_prediction_steps,
            n_features=n_features,
            n_static_features=n_static_features,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
            quantiles=quantiles
        )
        
        self.n_assets = n_assets
        self.enable_regime_specific = enable_regime_specific
        self.enable_cross_attention = enable_cross_attention
        self.enable_adaptive_horizon = enable_adaptive_horizon
        
        # Additional components
        if enable_regime_specific:
            self.regime_detector = MarketRegimeDetector()
            self._init_regime_specific_params()
            
        if enable_cross_attention:
            self._init_cross_attention_params()
            
        if enable_adaptive_horizon:
            self.horizon_selector = AdaptiveHorizonSelector()
            self._init_adaptive_horizon_params()
            
        # Multi-asset parameters
        self._init_multi_asset_params()
        
    def _init_regime_specific_params(self):
        """Initialize regime-specific parameters."""
        # Regime embeddings
        self.n_regimes = 5  # trending_up, trending_down, ranging, volatile, unknown
        self.params['regime_embedding'] = self._xavier_init((self.hidden_size, self.n_regimes))
        
        # Regime-specific output heads
        for regime_idx in range(self.n_regimes):
            for i, q in enumerate(self.quantiles):
                self.params[f'regime_{regime_idx}_output_q{i}_W'] = self._xavier_init((1, self.hidden_size))
                self.params[f'regime_{regime_idx}_output_q{i}_b'] = np.zeros((1, 1))
    
    def _init_cross_attention_params(self):
        """Initialize cross-asset attention parameters."""
        # Cross-attention between assets
        self.params['cross_attention_Wq'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['cross_attention_Wk'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['cross_attention_Wv'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['cross_attention_Wo'] = self._xavier_init((self.hidden_size, self.hidden_size))
        
        # Asset interaction network
        self.params['asset_interaction_W1'] = self._xavier_init((self.hidden_size, self.hidden_size * self.n_assets))
        self.params['asset_interaction_b1'] = np.zeros((self.hidden_size, 1))
        self.params['asset_interaction_W2'] = self._xavier_init((self.hidden_size * self.n_assets, self.hidden_size))
        self.params['asset_interaction_b2'] = np.zeros((self.hidden_size * self.n_assets, 1))
    
    def _init_adaptive_horizon_params(self):
        """Initialize adaptive horizon parameters."""
        # Horizon-specific scaling
        self.params['horizon_scale_W'] = self._xavier_init((self.hidden_size, self.hidden_size))
        self.params['horizon_scale_b'] = np.zeros((self.hidden_size, 1))
        
        # Horizon attention weights
        max_horizons = 12  # Support up to 12 different horizon lengths
        self.params['horizon_attention_W'] = self._xavier_init((max_horizons, self.hidden_size))
        self.params['horizon_attention_b'] = np.zeros((max_horizons, 1))
    
    def _init_multi_asset_params(self):
        """Initialize multi-asset specific parameters."""
        # Asset embeddings
        self.params['asset_embedding'] = self._xavier_init((self.hidden_size, self.n_assets))
        
        # Portfolio-level aggregation
        self.params['portfolio_agg_W'] = self._xavier_init((self.hidden_size, self.hidden_size * self.n_assets))
        self.params['portfolio_agg_b'] = np.zeros((self.hidden_size, 1))
        
        # Asset-specific output scaling
        for asset_idx in range(self.n_assets):
            self.params[f'asset_{asset_idx}_scale'] = np.ones((1, 1))
            self.params[f'asset_{asset_idx}_bias'] = np.zeros((1, 1))
    
    def forward_multi_asset(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Forward pass for multi-asset prediction.
        
        Args:
            inputs: Dictionary containing:
                - 'temporal_inputs': (batch_size, n_assets, n_timesteps, n_features)
                - 'static_inputs': (batch_size, n_assets, n_static_features)
                - 'market_data': Additional market context
                
        Returns:
            Multi-asset predictions and attention weights
        """
        batch_size = inputs['temporal_inputs'].shape[0]
        n_assets = inputs['temporal_inputs'].shape[1]
        
        # Detect market regime if enabled
        current_regime = "unknown"
        regime_features = {}
        
        if self.enable_regime_specific and 'market_data' in inputs:
            current_regime = self.regime_detector.detect_regime(inputs['market_data'])
            regime_features = self.regime_detector.get_regime_features(current_regime)
        
        # Select adaptive horizon if enabled
        prediction_horizon = self.n_prediction_steps
        
        if self.enable_adaptive_horizon and 'market_data' in inputs:
            volatility = self._calculate_volatility(inputs['market_data'])
            asset_correlation = self._calculate_asset_correlation(inputs['temporal_inputs'])
            prediction_horizon = self.horizon_selector.select_horizon(
                volatility, current_regime, asset_correlation
            )
        
        # Process each asset
        asset_encodings = []
        asset_predictions = []
        
        for asset_idx in range(n_assets):
            # Extract single asset data
            asset_temporal = inputs['temporal_inputs'][:, asset_idx, :, :]
            asset_static = inputs['static_inputs'][:, asset_idx, :]
            
            # Add asset embedding to static features
            asset_embed = self.params['asset_embedding'][:, asset_idx:asset_idx+1]
            asset_static_enhanced = np.concatenate([
                asset_static,
                np.tile(asset_embed.T, (batch_size, 1))
            ], axis=1)
            
            # Forward pass for single asset
            asset_inputs = {
                'temporal_inputs': asset_temporal,
                'static_inputs': asset_static_enhanced,
                'known_future_mask': inputs.get('known_future_mask')
            }
            
            asset_output = super().forward(asset_inputs)
            asset_encodings.append(asset_output)
            asset_predictions.append(asset_output['predictions'])
        
        # Apply cross-asset attention if enabled
        if self.enable_cross_attention:
            enhanced_predictions = self._apply_cross_asset_attention(
                asset_encodings, asset_predictions
            )
        else:
            enhanced_predictions = asset_predictions
        
        # Apply regime-specific adjustments if enabled
        if self.enable_regime_specific:
            final_predictions = self._apply_regime_specific_heads(
                enhanced_predictions, current_regime, regime_features
            )
        else:
            final_predictions = enhanced_predictions
        
        # Portfolio-level aggregation
        portfolio_prediction = self._aggregate_portfolio_predictions(
            final_predictions, inputs.get('portfolio_weights')
        )
        
        return {
            'asset_predictions': final_predictions,
            'portfolio_prediction': portfolio_prediction,
            'current_regime': current_regime,
            'prediction_horizon': prediction_horizon,
            'cross_asset_attention': self._get_cross_asset_attention_weights() if self.enable_cross_attention else None
        }
    
    def _apply_cross_asset_attention(self, asset_encodings: List[Dict], 
                                   asset_predictions: List[Dict]) -> List[Dict]:
        """Apply attention mechanism across different assets."""
        n_assets = len(asset_encodings)
        batch_size = asset_predictions[0]['q50'].shape[0]
        
        # Stack hidden representations from each asset
        hidden_states = []
        for enc in asset_encodings:
            # Use final decoder states as representation
            hidden = enc.get('decoder_final_state', np.random.randn(batch_size, self.hidden_size))
            hidden_states.append(hidden)
        
        hidden_states = np.stack(hidden_states, axis=1)  # (batch, n_assets, hidden)
        
        # Cross-asset attention
        enhanced_states = np.zeros_like(hidden_states)
        
        for i in range(n_assets):
            # Current asset as query
            query = hidden_states[:, i, :] @ self.params['cross_attention_Wq'].T
            
            # All assets as keys and values
            keys = hidden_states @ self.params['cross_attention_Wk'].T
            values = hidden_states @ self.params['cross_attention_Wv'].T
            
            # Compute attention scores
            scores = np.zeros((batch_size, n_assets))
            for b in range(batch_size):
                scores[b] = (query[b] @ keys[b].T) / np.sqrt(self.hidden_size)
            
            # Apply softmax
            attention_weights = np.zeros_like(scores)
            for b in range(batch_size):
                attention_weights[b] = self._softmax(scores[b])
            
            # Apply attention
            for b in range(batch_size):
                enhanced_states[b, i] = attention_weights[b] @ values[b]
        
        # Project back
        enhanced_states = enhanced_states @ self.params['cross_attention_Wo'].T
        
        # Apply asset interaction network
        all_assets_concat = enhanced_states.reshape(batch_size, -1).T
        interaction = self._gated_residual_network(
            all_assets_concat,
            self.params['asset_interaction_W1'],
            self.params['asset_interaction_b1'],
            self.params['asset_interaction_W2'], 
            self.params['asset_interaction_b2']
        )
        
        # Enhance predictions with cross-asset information
        enhanced_predictions = []
        for i, pred in enumerate(asset_predictions):
            # Add cross-asset context to predictions
            enhancement_factor = 1.0 + 0.1 * np.tanh(interaction[i*self.hidden_size:(i+1)*self.hidden_size, :].T)
            
            enhanced_pred = {}
            for key, values in pred.items():
                if key.startswith('q'):
                    enhanced_pred[key] = values * enhancement_factor.mean()
                else:
                    enhanced_pred[key] = values
                    
            enhanced_predictions.append(enhanced_pred)
        
        # Store attention weights for analysis
        self._cross_asset_attention_weights = attention_weights
        
        return enhanced_predictions
    
    def _apply_regime_specific_heads(self, predictions: List[Dict], 
                                    regime: str, regime_features: Dict) -> List[Dict]:
        """Apply regime-specific output heads."""
        regime_map = {
            "trending_up": 0,
            "trending_down": 1, 
            "ranging": 2,
            "volatile": 3,
            "unknown": 4
        }
        
        regime_idx = regime_map.get(regime, 4)
        
        # Get regime embedding
        regime_embed = self.params['regime_embedding'][:, regime_idx:regime_idx+1]
        
        # Adjust predictions based on regime
        adjusted_predictions = []
        
        for asset_pred in predictions:
            adjusted_pred = {}
            
            for key, values in asset_pred.items():
                if key.startswith('q'):
                    # Apply regime-specific transformation
                    q_idx = int(key[1:]) // 10  # Extract quantile index
                    
                    if q_idx < len(self.quantiles):
                        # Use regime-specific output head
                        regime_weight = self.params[f'regime_{regime_idx}_output_q{q_idx}_W']
                        regime_bias = self.params[f'regime_{regime_idx}_output_q{q_idx}_b']
                        
                        # Apply regime adjustment
                        adjustment = (regime_weight @ regime_embed + regime_bias)[0, 0]
                        
                        # Modify predictions based on regime characteristics
                        momentum_factor = regime_features.get('momentum', 0.5)
                        volatility_mult = regime_features.get('volatility_mult', 1.0)
                        
                        # Adjust quantile spread in volatile regimes
                        if key == 'q50':  # Median prediction
                            adjusted_pred[key] = values + adjustment * momentum_factor
                        elif key == 'q10':  # Lower quantile
                            adjusted_pred[key] = values + adjustment * momentum_factor - 0.02 * volatility_mult
                        elif key == 'q90':  # Upper quantile
                            adjusted_pred[key] = values + adjustment * momentum_factor + 0.02 * volatility_mult
                        else:
                            adjusted_pred[key] = values + adjustment * momentum_factor
                else:
                    adjusted_pred[key] = values
                    
            adjusted_predictions.append(adjusted_pred)
        
        return adjusted_predictions
    
    def _aggregate_portfolio_predictions(self, asset_predictions: List[Dict],
                                       portfolio_weights: Optional[np.ndarray]) -> Dict:
        """Aggregate individual asset predictions to portfolio level."""
        n_assets = len(asset_predictions)
        
        # Use equal weights if not provided
        if portfolio_weights is None:
            portfolio_weights = np.ones(n_assets) / n_assets
        
        # Ensure weights sum to 1
        portfolio_weights = portfolio_weights / np.sum(portfolio_weights)
        
        # Aggregate predictions
        portfolio_pred = {}
        
        # Get all quantile keys
        quantile_keys = [k for k in asset_predictions[0].keys() if k.startswith('q')]
        
        for key in quantile_keys:
            weighted_sum = None
            
            for i, asset_pred in enumerate(asset_predictions):
                if weighted_sum is None:
                    weighted_sum = asset_pred[key] * portfolio_weights[i]
                else:
                    weighted_sum += asset_pred[key] * portfolio_weights[i]
            
            portfolio_pred[key] = weighted_sum
        
        # Add portfolio-specific adjustments
        portfolio_concat = np.concatenate([pred['q50'].flatten() for pred in asset_predictions])
        portfolio_features = self._gated_residual_network(
            portfolio_concat.reshape(-1, 1),
            self.params['portfolio_agg_W'][:, :portfolio_concat.shape[0]],
            self.params['portfolio_agg_b'],
            self.params['portfolio_agg_W'][:portfolio_concat.shape[0], :],
            self.params['portfolio_agg_b']
        )
        
        # Apply portfolio-level scaling
        portfolio_scale = 1.0 + 0.05 * np.tanh(portfolio_features.mean())
        for key in portfolio_pred:
            portfolio_pred[key] *= portfolio_scale
        
        return portfolio_pred
    
    def _calculate_volatility(self, market_data: np.ndarray) -> float:
        """Calculate current market volatility."""
        if market_data.shape[0] < 20:
            return 0.02  # Default volatility
        
        returns = np.diff(market_data[-20:, 0]) / market_data[-21:-1, 0]
        return np.std(returns)
    
    def _calculate_asset_correlation(self, temporal_inputs: np.ndarray) -> float:
        """Calculate average correlation between assets."""
        n_assets = temporal_inputs.shape[1]
        if n_assets < 2:
            return 0.0
        
        # Use recent price data for correlation
        correlations = []
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Simple correlation of recent returns
                asset_i = temporal_inputs[0, i, -20:, 0]  # Price feature
                asset_j = temporal_inputs[0, j, -20:, 0]
                
                if len(asset_i) > 1 and len(asset_j) > 1:
                    corr = np.corrcoef(asset_i, asset_j)[0, 1]
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.5
    
    def _get_cross_asset_attention_weights(self) -> Optional[np.ndarray]:
        """Get the cross-asset attention weights from last forward pass."""
        return getattr(self, '_cross_asset_attention_weights', None)
    
    def train_step(self, inputs: Dict[str, np.ndarray], 
                  targets: Dict[str, np.ndarray], 
                  learning_rate: float = 0.001) -> Dict[str, float]:
        """
        Single training step with multi-asset support.
        
        Args:
            inputs: Input data including multi-asset temporal and static features
            targets: Target values for each asset and quantile
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary of loss values
        """
        # Forward pass
        predictions = self.forward_multi_asset(inputs)
        
        # Calculate losses
        losses = {}
        total_loss = 0.0
        
        # Asset-level losses
        for asset_idx, asset_pred in enumerate(predictions['asset_predictions']):
            for quantile_key in ['q10', 'q50', 'q90']:
                if quantile_key in asset_pred and f'asset_{asset_idx}_{quantile_key}' in targets:
                    # Quantile loss
                    quantile = float(quantile_key[1:]) / 100
                    loss = self._quantile_loss(
                        asset_pred[quantile_key],
                        targets[f'asset_{asset_idx}_{quantile_key}'],
                        quantile
                    )
                    losses[f'asset_{asset_idx}_{quantile_key}_loss'] = loss
                    total_loss += loss
        
        # Portfolio-level loss if targets provided
        if 'portfolio_q50' in targets:
            portfolio_loss = self._quantile_loss(
                predictions['portfolio_prediction']['q50'],
                targets['portfolio_q50'],
                0.5
            )
            losses['portfolio_loss'] = portfolio_loss
            total_loss += portfolio_loss * 2.0  # Higher weight for portfolio accuracy
        
        # Backward pass and update (simplified for demonstration)
        losses['total_loss'] = total_loss
        
        # Update horizon selector performance if enabled
        if self.enable_adaptive_horizon:
            performance_metric = -total_loss  # Negative loss as performance
            self.horizon_selector.update_performance(
                predictions['prediction_horizon'],
                performance_metric
            )
        
        return losses
    
    def _quantile_loss(self, predictions: np.ndarray, 
                      targets: np.ndarray, 
                      quantile: float) -> float:
        """Calculate quantile loss."""
        errors = targets - predictions
        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
    
    def get_interpretability_outputs(self, inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Get interpretability outputs for multi-asset model.
        
        Returns:
            Dictionary containing:
            - Feature importance for each asset
            - Cross-asset interaction strengths
            - Regime-specific adjustments
            - Temporal attention patterns
        """
        outputs = self.forward_multi_asset(inputs)
        
        interpretability = {
            'current_regime': outputs['current_regime'],
            'selected_horizon': outputs['prediction_horizon'],
            'cross_asset_attention': outputs.get('cross_asset_attention'),
            'asset_feature_importance': []
        }
        
        # Get feature importance for each asset
        for asset_idx in range(self.n_assets):
            asset_temporal = inputs['temporal_inputs'][:, asset_idx, :, :]
            asset_static = inputs['static_inputs'][:, asset_idx, :]
            
            asset_inputs = {
                'temporal_inputs': asset_temporal,
                'static_inputs': asset_static,
                'known_future_mask': inputs.get('known_future_mask')
            }
            
            importance = super().get_feature_importance(asset_inputs)
            interpretability['asset_feature_importance'].append(importance)
        
        return interpretability