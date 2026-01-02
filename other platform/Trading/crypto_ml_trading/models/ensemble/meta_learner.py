import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.matrix_operations import MatrixOperations


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    model_name: str
    action: str  # buy, sell, hold
    confidence: float
    features: Dict[str, float]
    metadata: Dict[str, Any]


class MetaLearner:
    """
    Meta-learning ensemble for combining multiple model predictions.
    
    Features:
    - Dynamic weight adjustment based on performance
    - Model correlation analysis
    - Confidence-based weighting
    - Online learning capabilities
    - Stacking with neural network meta-model
    """
    
    def __init__(self, 
                 model_names: List[str],
                 learning_rate: float = 0.01,
                 lookback_window: int = 100):
        """
        Initialize meta-learner.
        
        Args:
            model_names: List of base model names
            learning_rate: Learning rate for weight updates
            lookback_window: Window for performance tracking
        """
        self.model_names = model_names
        self.learning_rate = learning_rate
        self.lookback_window = lookback_window
        
        # Model weights (initialized uniformly)
        self.model_weights = {name: 1.0 / len(model_names) for name in model_names}
        
        # Performance tracking
        self.performance_history = {name: [] for name in model_names}
        self.prediction_history = []
        
        # Meta-model parameters (stacking)
        self.meta_model_params = self._initialize_meta_model()
        
        # Model correlation matrix
        self.correlation_matrix = np.eye(len(model_names))
        
    def _initialize_meta_model(self) -> Dict[str, np.ndarray]:
        """Initialize neural network meta-model."""
        n_models = len(self.model_names)
        n_features = n_models * 4  # For each model: action_probs(3) + confidence(1)
        hidden_size = 32
        
        params = {}
        
        # Hidden layer
        params['W1'] = self._xavier_init((hidden_size, n_features))
        params['b1'] = np.zeros((hidden_size, 1))
        
        # Output layer (3 actions + confidence)
        params['W2'] = self._xavier_init((4, hidden_size))
        params['b2'] = np.zeros((4, 1))
        
        return params
    
    def _xavier_init(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier weight initialization."""
        fan_in, fan_out = shape[1], shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def combine_predictions(self, predictions: List[ModelPrediction],
                          market_state: Optional[Dict] = None) -> Dict:
        """
        Combine multiple model predictions into ensemble prediction.
        
        Args:
            predictions: List of model predictions
            market_state: Current market conditions
            
        Returns:
            Ensemble prediction
        """
        # Store prediction for history
        self.prediction_history.append({
            'predictions': predictions,
            'market_state': market_state
        })
        
        # Method 1: Weighted voting
        weighted_vote = self._weighted_voting(predictions)
        
        # Method 2: Stacking with meta-model
        stacked_prediction = self._stacking_prediction(predictions)
        
        # Method 3: Dynamic blending based on market conditions
        if market_state:
            dynamic_prediction = self._dynamic_blending(predictions, market_state)
        else:
            dynamic_prediction = weighted_vote
            
        # Combine methods
        final_prediction = self._combine_ensemble_methods(
            weighted_vote, stacked_prediction, dynamic_prediction
        )
        
        # Add uncertainty estimation
        final_prediction['uncertainty'] = self._estimate_uncertainty(predictions)
        
        return final_prediction
    
    def _weighted_voting(self, predictions: List[ModelPrediction]) -> Dict:
        """Weighted voting ensemble."""
        action_scores = {'buy': 0, 'sell': 0, 'hold': 0}
        total_weight = 0
        
        for pred in predictions:
            weight = self.model_weights[pred.model_name] * pred.confidence
            action_scores[pred.action] += weight
            total_weight += weight
            
        # Normalize scores
        for action in action_scores:
            action_scores[action] /= total_weight if total_weight > 0 else 1
            
        # Get final action
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action]
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'action_probabilities': action_scores,
            'method': 'weighted_voting'
        }
    
    def _stacking_prediction(self, predictions: List[ModelPrediction]) -> Dict:
        """Stacking ensemble with meta-model."""
        # Prepare input features
        features = []
        
        for pred in predictions:
            # One-hot encode action
            action_encoding = [0, 0, 0]
            if pred.action == 'buy':
                action_encoding[0] = 1
            elif pred.action == 'hold':
                action_encoding[1] = 1
            else:  # sell
                action_encoding[2] = 1
                
            # Add confidence
            features.extend(action_encoding + [pred.confidence])
            
        features = np.array(features).reshape(-1, 1)
        
        # Forward pass through meta-model
        hidden = np.maximum(0, self.meta_model_params['W1'] @ features + 
                          self.meta_model_params['b1'])
        output = self.meta_model_params['W2'] @ hidden + self.meta_model_params['b2']
        
        # Apply softmax to first 3 outputs (actions)
        action_logits = output[:3, 0]
        action_probs = self._softmax(action_logits)
        
        # Sigmoid for confidence
        confidence = 1 / (1 + np.exp(-output[3, 0]))
        
        # Get final action
        action_idx = np.argmax(action_probs)
        actions = ['buy', 'hold', 'sell']
        
        return {
            'action': actions[action_idx],
            'confidence': float(confidence),
            'action_probabilities': {
                'buy': float(action_probs[0]),
                'hold': float(action_probs[1]),
                'sell': float(action_probs[2])
            },
            'method': 'stacking'
        }
    
    def _dynamic_blending(self, predictions: List[ModelPrediction],
                         market_state: Dict) -> Dict:
        """Dynamic blending based on market conditions."""
        # Adjust weights based on market regime
        regime = market_state.get('regime', 'neutral')
        volatility = market_state.get('volatility', 0.02)
        
        adjusted_weights = self.model_weights.copy()
        
        # Boost certain models in specific regimes
        if regime == 'bull':
            # Boost trend-following models
            for pred in predictions:
                if 'momentum' in pred.model_name.lower() or 'trend' in pred.model_name.lower():
                    adjusted_weights[pred.model_name] *= 1.5
                    
        elif regime == 'bear':
            # Boost risk-averse models
            for pred in predictions:
                if 'risk' in pred.model_name.lower() or 'garch' in pred.model_name.lower():
                    adjusted_weights[pred.model_name] *= 1.5
                    
        # Adjust for volatility
        if volatility > 0.05:  # High volatility
            # Reduce confidence in all predictions
            confidence_factor = 0.7
        else:
            confidence_factor = 1.0
            
        # Weighted combination with adjustments
        action_scores = {'buy': 0, 'sell': 0, 'hold': 0}
        total_weight = 0
        
        for pred in predictions:
            weight = adjusted_weights[pred.model_name] * pred.confidence * confidence_factor
            action_scores[pred.action] += weight
            total_weight += weight
            
        # Normalize
        for action in action_scores:
            action_scores[action] /= total_weight if total_weight > 0 else 1
            
        final_action = max(action_scores, key=action_scores.get)
        
        return {
            'action': final_action,
            'confidence': action_scores[final_action] * confidence_factor,
            'action_probabilities': action_scores,
            'method': 'dynamic_blending',
            'regime_adjustment': regime,
            'volatility_adjustment': confidence_factor
        }
    
    def _combine_ensemble_methods(self, weighted: Dict, stacked: Dict,
                                 dynamic: Dict) -> Dict:
        """Combine different ensemble methods."""
        # Average the action probabilities
        combined_probs = {}
        
        for action in ['buy', 'hold', 'sell']:
            probs = [
                weighted['action_probabilities'].get(action, 0),
                stacked['action_probabilities'].get(action, 0),
                dynamic['action_probabilities'].get(action, 0)
            ]
            combined_probs[action] = np.mean(probs)
            
        # Get final action
        final_action = max(combined_probs, key=combined_probs.get)
        
        # Average confidence
        avg_confidence = np.mean([
            weighted['confidence'],
            stacked['confidence'],
            dynamic['confidence']
        ])
        
        return {
            'action': final_action,
            'confidence': avg_confidence,
            'action_probabilities': combined_probs,
            'ensemble_methods': {
                'weighted_voting': weighted,
                'stacking': stacked,
                'dynamic_blending': dynamic
            }
        }
    
    def _estimate_uncertainty(self, predictions: List[ModelPrediction]) -> float:
        """Estimate prediction uncertainty."""
        # Action disagreement
        actions = [p.action for p in predictions]
        action_counts = {a: actions.count(a) for a in set(actions)}
        
        # Entropy of action distribution
        total = len(actions)
        action_entropy = -sum(
            (count/total) * np.log(count/total + 1e-10)
            for count in action_counts.values()
        )
        
        # Confidence variance
        confidences = [p.confidence for p in predictions]
        confidence_var = np.var(confidences)
        
        # Combined uncertainty (higher is more uncertain)
        uncertainty = action_entropy * 0.5 + confidence_var * 0.5
        
        return float(uncertainty)
    
    def update_weights(self, predictions: List[ModelPrediction],
                      actual_return: float):
        """
        Update model weights based on performance.
        
        Args:
            predictions: Model predictions
            actual_return: Actual market return
        """
        # Update performance history
        for pred in predictions:
            # Calculate individual model performance
            if pred.action == 'buy' and actual_return > 0:
                performance = pred.confidence * actual_return
            elif pred.action == 'sell' and actual_return < 0:
                performance = pred.confidence * abs(actual_return)
            elif pred.action == 'hold' and abs(actual_return) < 0.001:
                performance = pred.confidence * 0.001
            else:
                performance = -pred.confidence * abs(actual_return)
                
            self.performance_history[pred.model_name].append(performance)
            
        # Update weights using exponential moving average of performance
        for model_name in self.model_names:
            if len(self.performance_history[model_name]) > 0:
                recent_performance = self.performance_history[model_name][-self.lookback_window:]
                avg_performance = np.mean(recent_performance)
                
                # Update weight
                self.model_weights[model_name] *= (1 + self.learning_rate * avg_performance)
                
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        for model_name in self.model_names:
            self.model_weights[model_name] /= total_weight
            
        # Update correlation matrix
        self._update_correlation_matrix()
        
    def _update_correlation_matrix(self):
        """Update model correlation matrix."""
        if len(self.prediction_history) < 50:
            return
            
        # Extract recent predictions
        recent_predictions = self.prediction_history[-self.lookback_window:]
        
        # Build action matrix
        n_models = len(self.model_names)
        action_matrix = np.zeros((len(recent_predictions), n_models))
        
        for i, record in enumerate(recent_predictions):
            for j, model_name in enumerate(self.model_names):
                # Find prediction for this model
                for pred in record['predictions']:
                    if pred.model_name == model_name:
                        # Encode action as number
                        if pred.action == 'buy':
                            action_matrix[i, j] = 1
                        elif pred.action == 'sell':
                            action_matrix[i, j] = -1
                        else:  # hold
                            action_matrix[i, j] = 0
                        break
                        
        # Calculate correlation
        self.correlation_matrix = np.corrcoef(action_matrix.T)
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_model_diagnostics(self) -> Dict:
        """Get diagnostics for model performance."""
        diagnostics = {
            'model_weights': self.model_weights.copy(),
            'correlation_matrix': self.correlation_matrix.tolist(),
            'performance_summary': {}
        }
        
        # Performance summary for each model
        for model_name in self.model_names:
            if self.performance_history[model_name]:
                recent_perf = self.performance_history[model_name][-self.lookback_window:]
                diagnostics['performance_summary'][model_name] = {
                    'avg_performance': np.mean(recent_perf),
                    'performance_std': np.std(recent_perf),
                    'win_rate': sum(1 for p in recent_perf if p > 0) / len(recent_perf),
                    'total_predictions': len(self.performance_history[model_name])
                }
                
        return diagnostics


class EnsembleOrchestrator:
    """
    Orchestrates multiple models and meta-learning ensemble.
    
    Manages the entire prediction pipeline from individual models
    to final ensemble prediction.
    """
    
    def __init__(self, models: Dict[str, Any], meta_learner: MetaLearner):
        """
        Initialize orchestrator.
        
        Args:
            models: Dictionary of initialized models
            meta_learner: Meta-learning ensemble
        """
        self.models = models
        self.meta_learner = meta_learner
        
        # Model-specific preprocessing
        self.preprocessing_configs = self._setup_preprocessing()
        
    def _setup_preprocessing(self) -> Dict:
        """Setup model-specific preprocessing configurations."""
        return {
            'arima': {'lookback': 100, 'differencing': True},
            'garch': {'lookback': 100, 'returns': True},
            'hmm': {'lookback': 200, 'features': ['returns', 'volume']},
            'gru': {'sequence_length': 60, 'normalize': True},
            'cnn': {'image_size': 64, 'methods': ['gasf', 'gadf']},
            'ppo': {'lookback': 60, 'full_state': True},
            'sentiment': {'aggregate_window': 60, 'min_confidence': 0.5}
        }
    
    def generate_predictions(self, market_data: pd.DataFrame,
                           sentiment_data: Optional[List[Dict]] = None) -> Dict:
        """
        Generate predictions from all models and combine.
        
        Args:
            market_data: Recent market data
            sentiment_data: Recent sentiment analyses
            
        Returns:
            Final ensemble prediction
        """
        predictions = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'sentiment' and sentiment_data:
                    pred = self._get_sentiment_prediction(model, sentiment_data)
                else:
                    pred = self._get_model_prediction(model_name, model, market_data)
                    
                if pred:
                    predictions.append(pred)
                    
            except Exception as e:
                print(f"Error getting prediction from {model_name}: {e}")
                continue
                
        # Get market state for dynamic blending
        market_state = self._analyze_market_state(market_data)
        
        # Combine predictions
        ensemble_prediction = self.meta_learner.combine_predictions(
            predictions, market_state
        )
        
        # Add model-specific insights
        ensemble_prediction['model_predictions'] = {
            p.model_name: {
                'action': p.action,
                'confidence': p.confidence,
                'features': p.features
            } for p in predictions
        }
        
        return ensemble_prediction
    
    def _get_model_prediction(self, model_name: str, model: Any,
                            market_data: pd.DataFrame) -> Optional[ModelPrediction]:
        """Get prediction from individual model."""
        config = self.preprocessing_configs.get(model_name, {})
        
        # Model-specific prediction logic
        if model_name == 'arima':
            # ARIMA prediction
            returns = market_data['returns'].dropna().values[-config['lookback']:]
            forecast = model.predict(steps=5)
            
            action = 'buy' if np.mean(forecast) > 0.001 else 'sell' if np.mean(forecast) < -0.001 else 'hold'
            confidence = min(0.8, abs(np.mean(forecast)) * 100)
            
            features = {
                'forecast_mean': float(np.mean(forecast)),
                'forecast_std': float(np.std(forecast))
            }
            
        elif model_name == 'garch':
            # GARCH volatility prediction
            returns = market_data['returns'].dropna().values[-config['lookback']:]
            vol_forecast = model.forecast(steps=5)
            current_vol = vol_forecast['volatility'][0]
            future_vol = np.mean(vol_forecast['volatility'])
            
            # High volatility = opportunity
            if future_vol > current_vol * 1.2:
                action = 'buy'  # Volatility expansion
                confidence = 0.6
            else:
                action = 'hold'
                confidence = 0.5
                
            features = {
                'current_volatility': float(current_vol),
                'forecast_volatility': float(future_vol)
            }
            
        else:
            # Default prediction
            action = 'hold'
            confidence = 0.5
            features = {}
            
        return ModelPrediction(
            model_name=model_name,
            action=action,
            confidence=confidence,
            features=features,
            metadata={'timestamp': market_data.index[-1]}
        )
    
    def _get_sentiment_prediction(self, model: Any,
                                sentiment_data: List[Dict]) -> Optional[ModelPrediction]:
        """Get prediction from sentiment model."""
        if not sentiment_data:
            return None
            
        # Aggregate recent sentiment
        aggregated = model.aggregate_sentiment(sentiment_data)
        signal = model.generate_trading_signal(aggregated)
        
        return ModelPrediction(
            model_name='sentiment',
            action=signal['action'],
            confidence=signal['confidence'],
            features={
                'sentiment_score': signal['sentiment_score'],
                'sentiment_trend': signal['sentiment_trend'],
                'data_points': signal['data_points']
            },
            metadata={'reasoning': signal['reasoning']}
        )
    
    def _analyze_market_state(self, market_data: pd.DataFrame) -> Dict:
        """Analyze current market state."""
        returns = market_data['returns'].dropna()
        
        # Calculate metrics
        volatility = returns.rolling(20).std().iloc[-1]
        momentum = returns.rolling(10).mean().iloc[-1]
        
        # Determine regime (simplified)
        if momentum > 0.002 and volatility < 0.03:
            regime = 'bull'
        elif momentum < -0.002 and volatility > 0.04:
            regime = 'bear'
        else:
            regime = 'neutral'
            
        return {
            'regime': regime,
            'volatility': float(volatility),
            'momentum': float(momentum),
            'volume_trend': float(market_data['volume'].pct_change().rolling(10).mean().iloc[-1])
        }
    
    def update_ensemble(self, predictions: List[ModelPrediction],
                       actual_return: float):
        """Update ensemble weights based on performance."""
        self.meta_learner.update_weights(predictions, actual_return)