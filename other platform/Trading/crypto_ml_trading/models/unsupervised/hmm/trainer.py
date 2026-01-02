import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from .hmm_model import HiddenMarkovModel, RegimeDetector
from .regime_analyzer import RegimeAnalyzer


class HMMTrainer:
    """
    Trainer for Hidden Markov Model with cryptocurrency-specific features.
    
    Features:
    - Multi-feature regime detection
    - Cross-validation for model selection
    - Stability analysis
    - Online regime detection
    - Integration with trading strategies
    """
    
    def __init__(self, n_states: Optional[int] = None,
                 feature_set: str = 'standard'):
        """
        Initialize HMM trainer.
        
        Args:
            n_states: Number of hidden states (None for auto-selection)
            feature_set: Feature set to use ('minimal', 'standard', 'full')
        """
        self.n_states = n_states
        self.feature_set = feature_set
        self.model = None
        self.detector = None
        self.analyzer = None
        self.feature_config = self._get_feature_config()
        
    def _get_feature_config(self) -> Dict:
        """Get feature configuration based on feature set."""
        configs = {
            'minimal': {
                'features': ['returns', 'volatility'],
                'volatility_window': 20,
                'volume_features': False
            },
            'standard': {
                'features': ['returns', 'volatility', 'volume_change', 'spread'],
                'volatility_window': 20,
                'volume_features': True,
                'spread_window': 10
            },
            'full': {
                'features': ['returns', 'volatility', 'volume_change', 'spread',
                           'momentum', 'rsi', 'order_flow'],
                'volatility_window': 20,
                'volume_features': True,
                'spread_window': 10,
                'momentum_window': 14,
                'rsi_window': 14
            }
        }
        
        return configs.get(self.feature_set, configs['standard'])
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM training.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Feature matrix
        """
        features = []
        
        # Returns
        returns = data['close'].pct_change().fillna(0).values
        features.append(returns.reshape(-1, 1))
        
        # Volatility
        volatility = self._compute_volatility(
            returns, 
            window=self.feature_config['volatility_window']
        )
        features.append(volatility.reshape(-1, 1))
        
        # Volume features
        if self.feature_config.get('volume_features', False) and 'volume' in data:
            volume_change = data['volume'].pct_change().fillna(0).values
            features.append(volume_change.reshape(-1, 1))
            
            # Volume-weighted average price deviation
            if 'high' in data and 'low' in data:
                vwap = (data['high'] + data['low'] + data['close']) / 3
                vwap_deviation = (data['close'] - vwap) / vwap
                features.append(vwap_deviation.values.reshape(-1, 1))
                
        # Spread (high-low range)
        if 'spread' in self.feature_config['features'] and 'high' in data and 'low' in data:
            spread = (data['high'] - data['low']) / data['close']
            spread_ma = spread.rolling(self.feature_config.get('spread_window', 10)).mean()
            features.append(spread_ma.fillna(spread.mean()).values.reshape(-1, 1))
            
        # Advanced features for 'full' feature set
        if self.feature_set == 'full':
            # Momentum
            if 'momentum' in self.feature_config['features']:
                momentum = self._compute_momentum(
                    data['close'].values,
                    window=self.feature_config.get('momentum_window', 14)
                )
                features.append(momentum.reshape(-1, 1))
                
            # RSI
            if 'rsi' in self.feature_config['features']:
                rsi = self._compute_rsi(
                    data['close'].values,
                    window=self.feature_config.get('rsi_window', 14)
                )
                features.append(rsi.reshape(-1, 1))
                
            # Order flow (buy/sell pressure)
            if 'order_flow' in self.feature_config['features'] and 'volume' in data:
                order_flow = self._compute_order_flow(data)
                features.append(order_flow.reshape(-1, 1))
                
        # Combine features
        X = np.hstack(features)
        
        # Handle NaN values
        X = self._handle_nan_values(X)
        
        return X
    
    def train(self, data: pd.DataFrame, 
             validation_split: float = 0.2,
             cv_folds: int = 3) -> Dict:
        """
        Train HMM model with cross-validation.
        
        Args:
            data: Training data
            validation_split: Validation data fraction
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results
        """
        # Prepare features
        X = self.prepare_features(data)
        
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        X_train = X[:-n_val]
        X_val = X[-n_val:]
        
        # Initialize detector
        self.detector = RegimeDetector(min_states=2, max_states=5)
        
        # Select number of states if not specified
        if self.n_states is None:
            print("Selecting optimal number of states...")
            self.n_states = self.detector.select_n_states(X_train, criterion='bic')
            print(f"Selected {self.n_states} states")
            
        # Train model with cross-validation
        cv_scores = []
        
        if cv_folds > 1:
            fold_size = len(X_train) // cv_folds
            
            for fold in range(cv_folds):
                # Create fold indices
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else len(X_train)
                
                # Split fold data
                fold_train_idx = np.concatenate([
                    np.arange(0, val_start),
                    np.arange(val_end, len(X_train))
                ])
                fold_val_idx = np.arange(val_start, val_end)
                
                # Train model
                fold_model = HiddenMarkovModel(
                    n_states=self.n_states,
                    n_features=X.shape[1]
                )
                fold_model.fit(X_train[fold_train_idx])
                
                # Evaluate
                fold_score = fold_model.score(X_train[fold_val_idx])
                cv_scores.append(fold_score)
                
        # Train final model on all training data
        self.model = HiddenMarkovModel(
            n_states=self.n_states,
            n_features=X.shape[1]
        )
        self.model.fit(X_train, verbose=True)
        
        # Evaluate on validation set
        val_score = self.model.score(X_val)
        
        # Set up analyzer
        self.detector.best_model = self.model
        self.detector.n_states = self.n_states
        returns = data['close'].pct_change().fillna(0).values
        self.detector._label_regimes(returns[:-n_val])
        
        self.analyzer = RegimeAnalyzer(self.detector)
        
        # Analyze regimes
        regime_analysis = self.analyzer.analyze_regimes(data)
        
        return {
            'n_states': self.n_states,
            'train_score': self.model.score(X_train),
            'val_score': val_score,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores) if cv_scores else None,
            'cv_std': np.std(cv_scores) if cv_scores else None,
            'regime_analysis': regime_analysis,
            'convergence_iter': self.model.n_iter_performed,
            'model_params': {
                'initial_prob': self.model.initial_prob.tolist(),
                'transition_prob': self.model.transition_prob.tolist(),
                'means': self.model.means.tolist(),
                'covars': self.model.covars.tolist()
            }
        }
    
    def predict_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime for new data.
        
        Args:
            data: New data for prediction
            
        Returns:
            DataFrame with regime predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        # Prepare features
        X = self.prepare_features(data)
        
        # Predict states
        states = self.model.predict(X)
        state_probs = self.model.predict_proba(X)
        
        # Convert to regimes
        regimes = [self.detector.regime_labels.get(s, f'state_{s}') for s in states]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': data.index,
            'regime': regimes,
            'regime_confidence': np.max(state_probs, axis=1)
        })
        
        # Add regime probabilities
        for state, label in self.detector.regime_labels.items():
            results[f'{label}_prob'] = state_probs[:, state]
            
        return results
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on regime.
        
        Args:
            data: Market data
            
        Returns:
            DataFrame with trading signals
        """
        # Get regime predictions
        regime_predictions = self.predict_regime(data)
        
        # Get regime-specific parameters
        signals = []
        
        for idx, row in regime_predictions.iterrows():
            regime = row['regime']
            confidence = row['regime_confidence']
            
            # Get strategy recommendation
            params = self.analyzer.get_regime_specific_parameters(regime)
            
            # Generate signal based on regime
            signal = self._generate_regime_signal(
                regime, confidence, params, data.iloc[idx]
            )
            
            signals.append(signal)
            
        # Create signals DataFrame
        signals_df = pd.DataFrame(signals, index=data.index)
        
        # Add regime information
        signals_df = pd.concat([signals_df, regime_predictions.set_index('timestamp')], axis=1)
        
        return signals_df
    
    def _generate_regime_signal(self, regime: str, confidence: float,
                              params: Dict, market_data: pd.Series) -> Dict:
        """Generate trading signal based on regime."""
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'size_multiplier': 1.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
        
        # Adjust signal based on regime
        if regime == 'bull' and confidence > 0.7:
            signal['action'] = 'buy'
            signal['confidence'] = confidence
            signal['size_multiplier'] = 1.5
            signal['stop_loss_pct'] = 0.015
            signal['take_profit_pct'] = 0.06
            
        elif regime == 'bear' and confidence > 0.7:
            signal['action'] = 'sell'
            signal['confidence'] = confidence
            signal['size_multiplier'] = 0.5
            signal['stop_loss_pct'] = 0.01
            signal['take_profit_pct'] = 0.03
            
        elif regime == 'sideways':
            # Mean reversion strategy
            signal['action'] = 'hold'  # Would implement mean reversion logic
            signal['confidence'] = confidence * 0.5
            signal['size_multiplier'] = 0.7
            
        return signal
    
    def online_update(self, new_data: pd.DataFrame, 
                     update_threshold: int = 100) -> bool:
        """
        Update model with new data (online learning).
        
        Args:
            new_data: New market data
            update_threshold: Minimum new samples before updating
            
        Returns:
            Whether model was updated
        """
        if len(new_data) < update_threshold:
            return False
            
        # Prepare new features
        X_new = self.prepare_features(new_data)
        
        # Simple online update: retrain on recent data
        # In practice, would implement incremental learning
        self.model.fit(X_new, n_iter=20, verbose=False)
        
        return True
    
    # Utility methods
    
    def _compute_volatility(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Compute rolling volatility."""
        volatility = np.zeros_like(returns)
        
        for i in range(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i])
            
        # Fill initial values
        if window < len(returns):
            volatility[:window] = volatility[window]
            
        return volatility * np.sqrt(252)  # Annualized
    
    def _compute_momentum(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Compute price momentum."""
        momentum = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            momentum[i] = (prices[i] - prices[i-window]) / prices[i-window]
            
        return momentum
    
    def _compute_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Compute RSI."""
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:window] = 50  # Neutral
        rsi[window] = 100 - 100 / (1 + rs)
        
        for i in range(window + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta
                
            up = (up * (window - 1) + up_val) / window
            down = (down * (window - 1) + down_val) / window
            
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
            
        return rsi / 100  # Normalize to [0, 1]
    
    def _compute_order_flow(self, data: pd.DataFrame) -> np.ndarray:
        """Compute order flow indicator."""
        # Simplified order flow: price movement weighted by volume
        price_change = data['close'].pct_change().fillna(0)
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
        
        order_flow = price_change * volume_ratio
        
        # Smooth with moving average
        order_flow_ma = order_flow.rolling(5).mean().fillna(0)
        
        return order_flow_ma.values
    
    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """Handle NaN values in features."""
        # Forward fill then backward fill
        df = pd.DataFrame(X)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN, fill with zeros
        df = df.fillna(0)
        
        return df.values