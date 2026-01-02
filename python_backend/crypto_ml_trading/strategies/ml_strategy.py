"""
ML-based Trading Strategy
Integrates trained ML models with the backtesting system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import logging
from datetime import datetime

# Import system components
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
from features.ml_feature_engineering import MLFeatureEngineering
from data.preprocessing import AdvancedPreprocessor
from models.risk_management.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class MLTradingStrategy:
    """
    Trading strategy that uses ML model predictions.
    Integrates with the backtesting engine's signal generator interface.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 model_name: str,
                 feature_config: Dict,
                 risk_config: Optional[Dict] = None,
                 preprocessor: Optional[AdvancedPreprocessor] = None,
                 sequence_length: int = 50):
        """
        Initialize ML trading strategy.
        
        Args:
            model: Trained PyTorch model
            model_name: Name of the model (lstm, gru, cnn_lstm)
            feature_config: Configuration for feature engineering
            risk_config: Risk management configuration
            preprocessor: Fitted preprocessor from training
            sequence_length: Sequence length for time series models
        """
        self.model = model
        self.model_name = model_name
        self.feature_config = feature_config
        self.sequence_length = sequence_length
        
        # Initialize components
        self.ml_engineer = MLFeatureEngineering()
        self.preprocessor = preprocessor or AdvancedPreprocessor()
        self.risk_manager = RiskManager(risk_config or self._default_risk_config())
        
        # Model should be in eval mode
        self.model.eval()
        
        # Track performance
        self.prediction_history = []
        self.signal_history = []
        
    def _default_risk_config(self) -> Dict:
        """Default risk management configuration."""
        return {
            'max_position_size': 0.2,  # 20% of capital
            'stop_loss_pct': 0.02,     # 2% stop loss
            'take_profit_pct': 0.04,   # 4% take profit
            'max_correlation': 0.7,     # Max correlation between positions
            'risk_free_rate': 0.02,    # 2% annual risk-free rate
            'target_volatility': 0.15  # 15% target volatility
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw OHLCV data.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        # Ensure we have enough data
        if len(data) < self.sequence_length + 50:  # Need extra for indicators
            logger.warning(f"Insufficient data: {len(data)} rows")
            return None
        
        # 1. Add technical indicators
        data_with_indicators = EnhancedTechnicalIndicators.compute_all_indicators(
            data.copy(), self.feature_config.get('indicators', {})
        )
        
        # 2. Create ML features
        data_with_features = self.ml_engineer.prepare_ml_features(
            data_with_indicators, self.feature_config
        )
        
        # 3. Apply preprocessing (scaling, etc.)
        if hasattr(self.preprocessor, 'transform'):
            # Use fitted preprocessor from training
            data_processed = self.preprocessor.transform(data_with_features)
        else:
            # Fit and transform if not fitted
            data_processed = self.preprocessor.preprocess(data_with_features)
        
        return data_processed
    
    def get_model_prediction(self, features: pd.DataFrame) -> Tuple[int, np.ndarray]:
        """
        Get model prediction for the latest data point.
        
        Args:
            features: Prepared features DataFrame
            
        Returns:
            Tuple of (predicted_class, confidence_scores)
        """
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        # Ensure we have features
        feature_cols = [col for col in feature_cols if col in features.columns]
        
        # Extract the last sequence
        if len(features) < self.sequence_length:
            logger.warning(f"Insufficient data for sequence: {len(features)}")
            return 1, np.array([0.33, 0.33, 0.34])  # Default: hold
        
        # Get the last sequence
        sequence_data = features[feature_cols].iloc[-self.sequence_length:].values
        
        # Reshape for model input
        X = torch.FloatTensor(sequence_data).unsqueeze(0)  # Add batch dimension
        
        # Special handling for CNN-LSTM
        if self.model_name == 'cnn_lstm':
            X = X.unsqueeze(1)  # Add channel dimension
        
        # Get prediction
        with torch.no_grad():
            output = self.model(X)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    
    def generate_signal(self, 
                       historical_data: pd.DataFrame,
                       current_positions: Dict,
                       account_info: Optional[Dict] = None) -> Dict:
        """
        Generate trading signal based on ML model prediction.
        
        This is the main interface with the backtesting engine.
        
        Args:
            historical_data: Historical OHLCV data
            current_positions: Current open positions
            account_info: Account information (balance, etc.)
            
        Returns:
            Trading signal dictionary
        """
        # Default signal (no action)
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'expected_return': 0.0,
            'win_rate': 0.5,
            'predicted_class': 1,  # Hold
            'probabilities': [0.33, 0.33, 0.34],
            'stop_loss': None,
            'take_profit': None,
            'position_size': 0.0
        }
        
        try:
            # Prepare features
            features = self.prepare_features(historical_data)
            if features is None or len(features) < self.sequence_length:
                return signal
            
            # Get model prediction
            predicted_class, probabilities = self.get_model_prediction(features)
            
            # Convert prediction to trading action
            # 0: Sell, 1: Buy, 2: Hold
            if predicted_class == 0:
                action = 'sell'
            elif predicted_class == 1:
                action = 'buy'
            else:
                action = 'hold'
            
            # Calculate confidence (difference between top two probabilities)
            sorted_probs = np.sort(probabilities)[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]
            
            # Estimate expected return based on historical performance
            # This is a simplified approach - in practice, you might want to
            # track actual performance by prediction confidence
            expected_return = (probabilities[1] - probabilities[0]) * 0.02  # 2% base
            
            # Estimate win rate based on confidence
            win_rate = 0.5 + (confidence * 0.3)  # Scale confidence to win rate
            
            # Get current price for risk calculations
            current_price = historical_data['close'].iloc[-1]
            
            # Calculate position size using risk manager
            if account_info:
                position_info = self.risk_manager.calculate_position_size(
                    signal_strength=confidence,
                    expected_return=expected_return,
                    win_rate=win_rate,
                    current_price=current_price,
                    account_balance=account_info.get('balance', 10000),
                    existing_positions=current_positions
                )
                
                position_size = position_info['position_size']
                stop_loss = position_info.get('stop_loss')
                take_profit = position_info.get('take_profit')
            else:
                # Simple position sizing based on confidence
                position_size = confidence * 0.1  # Max 10% of capital
                
                # Simple stop loss and take profit
                if action == 'buy':
                    stop_loss = current_price * 0.98  # 2% stop loss
                    take_profit = current_price * 1.04  # 4% take profit
                elif action == 'sell':
                    stop_loss = current_price * 1.02
                    take_profit = current_price * 0.96
                else:
                    stop_loss = None
                    take_profit = None
            
            # Update signal
            signal.update({
                'action': action,
                'confidence': confidence,
                'expected_return': expected_return,
                'win_rate': win_rate,
                'predicted_class': predicted_class,
                'probabilities': probabilities.tolist(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'timestamp': historical_data.index[-1] if isinstance(historical_data.index, pd.DatetimeIndex) else datetime.now()
            })
            
            # Track signal
            self.signal_history.append(signal.copy())
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return signal
    
    def update_performance(self, trade_result: Dict):
        """
        Update strategy performance based on trade results.
        
        Args:
            trade_result: Result of executed trade
        """
        # Track prediction accuracy
        if 'actual_return' in trade_result and self.signal_history:
            last_signal = self.signal_history[-1]
            
            # Check if prediction was correct
            actual_direction = 'buy' if trade_result['actual_return'] > 0 else 'sell'
            predicted_direction = last_signal['action']
            
            self.prediction_history.append({
                'timestamp': trade_result.get('timestamp'),
                'predicted': predicted_direction,
                'actual': actual_direction,
                'correct': predicted_direction == actual_direction,
                'confidence': last_signal['confidence'],
                'return': trade_result['actual_return']
            })
    
    def get_performance_summary(self) -> Dict:
        """Get strategy performance summary."""
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'confident_accuracy': 0.0
            }
        
        # Calculate metrics
        df = pd.DataFrame(self.prediction_history)
        
        total_predictions = len(df)
        accuracy = df['correct'].mean()
        avg_confidence = df['confidence'].mean()
        
        # Accuracy for high confidence predictions
        high_conf_mask = df['confidence'] > 0.7
        confident_accuracy = df[high_conf_mask]['correct'].mean() if high_conf_mask.any() else 0.0
        
        return {
            'total_predictions': total_predictions,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'confident_accuracy': confident_accuracy,
            'confidence_distribution': df['confidence'].describe().to_dict(),
            'return_by_confidence': df.groupby(pd.cut(df['confidence'], bins=5))['return'].mean().to_dict()
        }


class EnsembleMLStrategy(MLTradingStrategy):
    """
    Ensemble strategy that combines predictions from multiple ML models.
    """
    
    def __init__(self,
                 models: Dict[str, torch.nn.Module],
                 feature_config: Dict,
                 risk_config: Optional[Dict] = None,
                 preprocessor: Optional[AdvancedPreprocessor] = None,
                 sequence_length: int = 50,
                 ensemble_method: str = 'weighted_average'):
        """
        Initialize ensemble ML strategy.
        
        Args:
            models: Dictionary of model_name -> model
            feature_config: Configuration for feature engineering
            risk_config: Risk management configuration
            preprocessor: Fitted preprocessor from training
            sequence_length: Sequence length for time series models
            ensemble_method: Method for combining predictions
        """
        # Initialize base class with first model (for compatibility)
        first_model_name = list(models.keys())[0]
        super().__init__(
            models[first_model_name],
            first_model_name,
            feature_config,
            risk_config,
            preprocessor,
            sequence_length
        )
        
        self.models = models
        self.ensemble_method = ensemble_method
        
        # Model weights (can be optimized based on performance)
        self.model_weights = {name: 1.0 / len(models) for name in models}
        
    def get_ensemble_prediction(self, features: pd.DataFrame) -> Tuple[int, np.ndarray]:
        """
        Get ensemble prediction from all models.
        
        Args:
            features: Prepared features DataFrame
            
        Returns:
            Tuple of (predicted_class, confidence_scores)
        """
        all_probabilities = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            self.model = model
            self.model_name = model_name
            
            _, probabilities = self.get_model_prediction(features)
            all_probabilities.append(probabilities)
        
        # Combine predictions
        all_probabilities = np.array(all_probabilities)
        
        if self.ensemble_method == 'weighted_average':
            # Weighted average of probabilities
            weights = np.array([self.model_weights[name] for name in self.models])
            ensemble_probs = np.average(all_probabilities, axis=0, weights=weights)
            
        elif self.ensemble_method == 'voting':
            # Majority voting
            predictions = np.argmax(all_probabilities, axis=1)
            ensemble_class = np.bincount(predictions).argmax()
            # Create probability distribution
            ensemble_probs = np.zeros(all_probabilities.shape[1])
            ensemble_probs[ensemble_class] = 1.0
            
        else:  # Default to simple average
            ensemble_probs = np.mean(all_probabilities, axis=0)
        
        # Normalize probabilities
        ensemble_probs = ensemble_probs / ensemble_probs.sum()
        
        predicted_class = np.argmax(ensemble_probs)
        
        return predicted_class, ensemble_probs
    
    def generate_signal(self,
                       historical_data: pd.DataFrame,
                       current_positions: Dict,
                       account_info: Optional[Dict] = None) -> Dict:
        """
        Generate trading signal based on ensemble prediction.
        """
        # Prepare base signal
        signal = super().generate_signal(historical_data, current_positions, account_info)
        
        try:
            # Get features
            features = self.prepare_features(historical_data)
            if features is None or len(features) < self.sequence_length:
                return signal
            
            # Get ensemble prediction
            predicted_class, probabilities = self.get_ensemble_prediction(features)
            
            # Update signal with ensemble results
            signal['predicted_class'] = predicted_class
            signal['probabilities'] = probabilities.tolist()
            signal['ensemble_method'] = self.ensemble_method
            signal['num_models'] = len(self.models)
            
            # Recalculate action and confidence
            if predicted_class == 0:
                signal['action'] = 'sell'
            elif predicted_class == 1:
                signal['action'] = 'buy'
            else:
                signal['action'] = 'hold'
            
            # Enhanced confidence calculation for ensemble
            sorted_probs = np.sort(probabilities)[::-1]
            signal['confidence'] = sorted_probs[0] - sorted_probs[1]
            
            # Agreement score (how much models agree)
            model_predictions = []
            for model_name, model in self.models.items():
                self.model = model
                self.model_name = model_name
                pred_class, _ = self.get_model_prediction(features)
                model_predictions.append(pred_class)
            
            agreement_score = np.mean([p == predicted_class for p in model_predictions])
            signal['model_agreement'] = agreement_score
            
            # Adjust confidence based on agreement
            signal['confidence'] *= agreement_score
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
        
        return signal
    
    def update_model_weights(self, performance_by_model: Dict[str, float]):
        """
        Update model weights based on individual performance.
        
        Args:
            performance_by_model: Dictionary of model_name -> performance_score
        """
        # Normalize scores to weights
        total_score = sum(performance_by_model.values())
        if total_score > 0:
            self.model_weights = {
                name: score / total_score 
                for name, score in performance_by_model.items()
            }
        
        logger.info(f"Updated model weights: {self.model_weights}")


def create_ml_signal_generator(model_or_models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
                              feature_config: Dict,
                              preprocessor: Optional[AdvancedPreprocessor] = None,
                              **kwargs):
    """
    Factory function to create a signal generator for backtesting.
    
    Args:
        model_or_models: Single model or dictionary of models
        feature_config: Feature engineering configuration
        preprocessor: Fitted preprocessor
        **kwargs: Additional arguments for strategy
        
    Returns:
        Signal generator function for backtesting engine
    """
    # Create strategy
    if isinstance(model_or_models, dict):
        strategy = EnsembleMLStrategy(
            model_or_models,
            feature_config,
            preprocessor=preprocessor,
            **kwargs
        )
    else:
        model_name = kwargs.pop('model_name', 'unknown')
        strategy = MLTradingStrategy(
            model_or_models,
            model_name,
            feature_config,
            preprocessor=preprocessor,
            **kwargs
        )
    
    # Return signal generator function
    def signal_generator(historical_data: pd.DataFrame,
                        positions: Dict = None,
                        **extra_args) -> Dict:
        """Signal generator for backtesting engine."""
        return strategy.generate_signal(
            historical_data,
            positions or {},
            extra_args.get('account_info')
        )
    
    # Attach strategy for access to performance metrics
    signal_generator.strategy = strategy
    
    return signal_generator