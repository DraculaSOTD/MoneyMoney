import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.special import logsumexp
import warnings


class HiddenMarkovModel:
    """
    Hidden Markov Model for cryptocurrency market regime detection.
    
    Features:
    - Multiple hidden states representing market regimes
    - Gaussian emissions for continuous observations
    - Baum-Welch algorithm for parameter estimation
    - Viterbi algorithm for state decoding
    - Forward-backward algorithm for state probabilities
    """
    
    def __init__(self, n_states: int = 3, n_features: int = 1,
                 covariance_type: str = 'diag'):
        """
        Initialize HMM model.
        
        Args:
            n_states: Number of hidden states (market regimes)
            n_features: Number of observation features
            covariance_type: Type of covariance ('full', 'diag', 'spherical')
        """
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type
        
        # Model parameters
        self.initial_prob = None
        self.transition_prob = None
        self.means = None
        self.covars = None
        
        # Training state
        self.is_fitted = False
        self.n_iter_performed = 0
        self.monitor_history = []
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize model parameters with sensible defaults."""
        # Initial state probabilities (uniform)
        self.initial_prob = np.ones(self.n_states) / self.n_states
        
        # Transition matrix (slight preference for staying in same state)
        self.transition_prob = np.eye(self.n_states) * 0.7
        off_diagonal = (1 - 0.7) / (self.n_states - 1)
        self.transition_prob[self.transition_prob == 0] = off_diagonal
        
        # Emission parameters (random initialization)
        self.means = np.random.randn(self.n_states, self.n_features)
        
        if self.covariance_type == 'full':
            self.covars = np.array([np.eye(self.n_features) for _ in range(self.n_states)])
        elif self.covariance_type == 'diag':
            self.covars = np.ones((self.n_states, self.n_features))
        else:  # spherical
            self.covars = np.ones(self.n_states)
            
    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood of observations under each state.
        
        Args:
            X: Observations (n_samples, n_features)
            
        Returns:
            Log likelihood matrix (n_samples, n_states)
        """
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_states))
        
        for i in range(self.n_states):
            if self.covariance_type == 'full':
                # Multivariate Gaussian with full covariance
                diff = X - self.means[i]
                inv_cov = np.linalg.inv(self.covars[i])
                log_prob[:, i] = -0.5 * (
                    np.sum((diff @ inv_cov) * diff, axis=1) +
                    np.log(np.linalg.det(self.covars[i])) +
                    self.n_features * np.log(2 * np.pi)
                )
            elif self.covariance_type == 'diag':
                # Diagonal covariance
                diff = X - self.means[i]
                log_prob[:, i] = -0.5 * (
                    np.sum(diff**2 / self.covars[i], axis=1) +
                    np.sum(np.log(self.covars[i])) +
                    self.n_features * np.log(2 * np.pi)
                )
            else:  # spherical
                # Spherical covariance
                diff = X - self.means[i]
                log_prob[:, i] = -0.5 * (
                    np.sum(diff**2, axis=1) / self.covars[i] +
                    self.n_features * np.log(self.covars[i]) +
                    self.n_features * np.log(2 * np.pi)
                )
                
        return log_prob
    
    def _forward(self, log_prob: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm for HMM.
        
        Args:
            log_prob: Log observation probabilities (n_samples, n_states)
            
        Returns:
            Tuple of (forward probabilities, log likelihood)
        """
        n_samples = log_prob.shape[0]
        log_alpha = np.zeros((n_samples, self.n_states))
        
        # Initialize
        log_alpha[0] = np.log(self.initial_prob) + log_prob[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + np.log(self.transition_prob[:, j])
                ) + log_prob[t, j]
                
        # Total log likelihood
        log_likelihood = logsumexp(log_alpha[-1])
        
        return log_alpha, log_likelihood
    
    def _backward(self, log_prob: np.ndarray) -> np.ndarray:
        """
        Backward algorithm for HMM.
        
        Args:
            log_prob: Log observation probabilities (n_samples, n_states)
            
        Returns:
            Backward probabilities
        """
        n_samples = log_prob.shape[0]
        log_beta = np.zeros((n_samples, self.n_states))
        
        # Initialize (log(1) = 0)
        log_beta[-1] = 0
        
        # Backward pass
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.transition_prob[i]) + 
                    log_prob[t + 1] + 
                    log_beta[t + 1]
                )
                
        return log_beta
    
    def _compute_posteriors(self, log_alpha: np.ndarray, 
                          log_beta: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities of states.
        
        Args:
            log_alpha: Forward probabilities
            log_beta: Backward probabilities
            
        Returns:
            State posterior probabilities (n_samples, n_states)
        """
        log_gamma = log_alpha + log_beta
        # Normalize
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)
    
    def _compute_xi(self, log_alpha: np.ndarray, log_beta: np.ndarray,
                   log_prob: np.ndarray) -> np.ndarray:
        """
        Compute xi (transition posteriors).
        
        Args:
            log_alpha: Forward probabilities
            log_beta: Backward probabilities
            log_prob: Log observation probabilities
            
        Returns:
            Transition posteriors (n_samples-1, n_states, n_states)
        """
        n_samples = log_prob.shape[0]
        log_xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        
        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[t, i, j] = (
                        log_alpha[t, i] + 
                        np.log(self.transition_prob[i, j]) +
                        log_prob[t + 1, j] + 
                        log_beta[t + 1, j]
                    )
                    
            # Normalize
            log_xi[t] -= logsumexp(log_xi[t])
            
        return np.exp(log_xi)
    
    def fit(self, X: np.ndarray, n_iter: int = 100, 
            tol: float = 1e-4, verbose: bool = False) -> 'HiddenMarkovModel':
        """
        Fit HMM using Baum-Welch algorithm.
        
        Args:
            X: Observations (n_samples, n_features)
            n_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Print convergence info
            
        Returns:
            Self
        """
        # Ensure proper shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        prev_log_likelihood = -np.inf
        
        for iteration in range(n_iter):
            # E-step
            log_prob = self._compute_log_likelihood(X)
            log_alpha, log_likelihood = self._forward(log_prob)
            log_beta = self._backward(log_prob)
            
            gamma = self._compute_posteriors(log_alpha, log_beta)
            xi = self._compute_xi(log_alpha, log_beta, log_prob)
            
            # M-step
            # Update initial probabilities
            self.initial_prob = gamma[0]
            
            # Update transition probabilities
            self.transition_prob = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
            
            # Update emission parameters
            for i in range(self.n_states):
                weight = gamma[:, i]
                weight_sum = np.sum(weight)
                
                # Update means
                self.means[i] = np.sum(weight[:, np.newaxis] * X, axis=0) / weight_sum
                
                # Update covariances
                diff = X - self.means[i]
                if self.covariance_type == 'full':
                    self.covars[i] = (diff.T @ (weight[:, np.newaxis] * diff)) / weight_sum
                    # Add small value for numerical stability
                    self.covars[i] += np.eye(self.n_features) * 1e-6
                elif self.covariance_type == 'diag':
                    self.covars[i] = np.sum(weight[:, np.newaxis] * diff**2, axis=0) / weight_sum
                    self.covars[i] = np.maximum(self.covars[i], 1e-6)
                else:  # spherical
                    self.covars[i] = np.sum(weight[:, np.newaxis] * diff**2) / (weight_sum * self.n_features)
                    self.covars[i] = max(self.covars[i], 1e-6)
                    
            # Check convergence
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.4f}")
                
            if abs(log_likelihood - prev_log_likelihood) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            prev_log_likelihood = log_likelihood
            self.monitor_history.append(log_likelihood)
            
        self.is_fitted = True
        self.n_iter_performed = iteration + 1
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Find most likely state sequence using Viterbi algorithm.
        
        Args:
            X: Observations (n_samples, n_features)
            
        Returns:
            Most likely state sequence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples = X.shape[0]
        
        # Compute log probabilities
        log_prob = self._compute_log_likelihood(X)
        
        # Viterbi algorithm
        log_delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialize
        log_delta[0] = np.log(self.initial_prob) + log_prob[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_states):
                temp = log_delta[t-1] + np.log(self.transition_prob[:, j])
                psi[t, j] = np.argmax(temp)
                log_delta[t, j] = temp[psi[t, j]] + log_prob[t, j]
                
        # Backward pass
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
            
        return states
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities for each state.
        
        Args:
            X: Observations (n_samples, n_features)
            
        Returns:
            State probabilities (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        log_prob = self._compute_log_likelihood(X)
        log_alpha, _ = self._forward(log_prob)
        log_beta = self._backward(log_prob)
        
        return self._compute_posteriors(log_alpha, log_beta)
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute average log-likelihood of observations.
        
        Args:
            X: Observations
            
        Returns:
            Average log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        log_prob = self._compute_log_likelihood(X)
        _, log_likelihood = self._forward(log_prob)
        
        return log_likelihood / X.shape[0]
    
    def sample(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (observations, states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
            
        states = np.zeros(n_samples, dtype=int)
        observations = np.zeros((n_samples, self.n_features))
        
        # Sample initial state
        states[0] = np.random.choice(self.n_states, p=self.initial_prob)
        
        # Sample states
        for t in range(1, n_samples):
            states[t] = np.random.choice(
                self.n_states, 
                p=self.transition_prob[states[t-1]]
            )
            
        # Sample observations
        for t in range(n_samples):
            state = states[t]
            
            if self.covariance_type == 'full':
                observations[t] = np.random.multivariate_normal(
                    self.means[state], self.covars[state]
                )
            elif self.covariance_type == 'diag':
                observations[t] = np.random.normal(
                    self.means[state], np.sqrt(self.covars[state])
                )
            else:  # spherical
                observations[t] = np.random.normal(
                    self.means[state], np.sqrt(self.covars[state])
                )
                
        return observations, states


class RegimeDetector:
    """
    Market regime detection using Hidden Markov Models.
    
    Specialized for cryptocurrency markets with features like:
    - Automatic state number selection
    - Regime labeling (bull, bear, sideways)
    - Regime change detection
    - Confidence estimation
    """
    
    def __init__(self, min_states: int = 2, max_states: int = 5):
        """
        Initialize regime detector.
        
        Args:
            min_states: Minimum number of regimes to test
            max_states: Maximum number of regimes to test
        """
        self.min_states = min_states
        self.max_states = max_states
        self.best_model = None
        self.n_states = None
        self.regime_labels = None
        
    def select_n_states(self, X: np.ndarray, criterion: str = 'bic') -> int:
        """
        Select optimal number of states using information criterion.
        
        Args:
            X: Observations
            criterion: 'aic' or 'bic'
            
        Returns:
            Optimal number of states
        """
        scores = []
        
        for n_states in range(self.min_states, self.max_states + 1):
            model = HiddenMarkovModel(n_states=n_states, n_features=X.shape[1])
            model.fit(X, verbose=False)
            
            # Calculate information criterion
            log_likelihood = model.score(X) * X.shape[0]
            n_params = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * X.shape[1]
            
            if criterion == 'aic':
                score = -2 * log_likelihood + 2 * n_params
            else:  # bic
                score = -2 * log_likelihood + n_params * np.log(X.shape[0])
                
            scores.append((n_states, score, model))
            
        # Select best model
        best_idx = np.argmin([s[1] for s in scores])
        self.n_states = scores[best_idx][0]
        self.best_model = scores[best_idx][2]
        
        return self.n_states
    
    def fit(self, returns: np.ndarray, volumes: Optional[np.ndarray] = None,
            auto_select: bool = True) -> 'RegimeDetector':
        """
        Fit regime detection model.
        
        Args:
            returns: Return series
            volumes: Optional volume series
            auto_select: Automatically select number of states
            
        Returns:
            Self
        """
        # Prepare features
        features = [returns.reshape(-1, 1)]
        
        # Add rolling volatility
        volatility = self._compute_rolling_volatility(returns, window=20)
        features.append(volatility.reshape(-1, 1))
        
        # Add volume if provided
        if volumes is not None:
            volume_change = np.concatenate([[0], np.diff(np.log(volumes + 1))])
            features.append(volume_change.reshape(-1, 1))
            
        X = np.hstack(features)
        
        # Remove NaN values
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        
        # Select number of states
        if auto_select:
            self.select_n_states(X)
        else:
            if self.n_states is None:
                self.n_states = 3
            self.best_model = HiddenMarkovModel(
                n_states=self.n_states, 
                n_features=X.shape[1]
            )
            self.best_model.fit(X)
            
        # Label regimes
        self._label_regimes(returns[valid_idx])
        
        return self
    
    def _compute_rolling_volatility(self, returns: np.ndarray, 
                                  window: int = 20) -> np.ndarray:
        """Compute rolling volatility."""
        volatility = np.zeros_like(returns)
        
        for i in range(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i])
            
        # Fill initial values
        volatility[:window] = volatility[window]
        
        return volatility
    
    def _label_regimes(self, returns: np.ndarray):
        """Label regimes based on their characteristics."""
        # Get average return for each state
        states = self.best_model.predict(returns.reshape(-1, 1))
        
        regime_stats = {}
        for state in range(self.n_states):
            state_returns = returns[states == state]
            if len(state_returns) > 0:
                regime_stats[state] = {
                    'mean_return': np.mean(state_returns),
                    'volatility': np.std(state_returns),
                    'frequency': len(state_returns) / len(returns)
                }
                
        # Sort by mean return
        sorted_states = sorted(regime_stats.keys(), 
                             key=lambda x: regime_stats[x]['mean_return'])
        
        # Assign labels
        self.regime_labels = {}
        if self.n_states == 2:
            self.regime_labels[sorted_states[0]] = 'bear'
            self.regime_labels[sorted_states[1]] = 'bull'
        elif self.n_states == 3:
            self.regime_labels[sorted_states[0]] = 'bear'
            self.regime_labels[sorted_states[1]] = 'sideways'
            self.regime_labels[sorted_states[2]] = 'bull'
        else:
            # For more states, use numeric labels
            for i, state in enumerate(sorted_states):
                self.regime_labels[state] = f'regime_{i}'
                
    def predict_regime(self, returns: np.ndarray, 
                      volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict market regime.
        
        Args:
            returns: Return series
            volumes: Optional volume series
            
        Returns:
            Regime predictions
        """
        # Prepare features (same as in fit)
        features = [returns.reshape(-1, 1)]
        
        volatility = self._compute_rolling_volatility(returns, window=20)
        features.append(volatility.reshape(-1, 1))
        
        if volumes is not None:
            volume_change = np.concatenate([[0], np.diff(np.log(volumes + 1))])
            features.append(volume_change.reshape(-1, 1))
            
        X = np.hstack(features)
        
        # Predict states
        states = self.best_model.predict(X)
        
        # Convert to regime labels
        regimes = np.array([self.regime_labels.get(s, f'regime_{s}') 
                           for s in states])
        
        return regimes
    
    def predict_regime_proba(self, returns: np.ndarray,
                           volumes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Get regime probabilities.
        
        Args:
            returns: Return series
            volumes: Optional volume series
            
        Returns:
            Dictionary of regime probabilities
        """
        # Prepare features
        features = [returns.reshape(-1, 1)]
        
        volatility = self._compute_rolling_volatility(returns, window=20)
        features.append(volatility.reshape(-1, 1))
        
        if volumes is not None:
            volume_change = np.concatenate([[0], np.diff(np.log(volumes + 1))])
            features.append(volume_change.reshape(-1, 1))
            
        X = np.hstack(features)
        
        # Get state probabilities
        state_probs = self.best_model.predict_proba(X)
        
        # Convert to regime probabilities
        regime_probs = {}
        for state, label in self.regime_labels.items():
            regime_probs[label] = state_probs[:, state]
            
        return regime_probs
    
    def detect_regime_changes(self, returns: np.ndarray,
                            volumes: Optional[np.ndarray] = None,
                            threshold: float = 0.8) -> List[Dict]:
        """
        Detect regime changes with confidence.
        
        Args:
            returns: Return series
            volumes: Optional volume series
            threshold: Confidence threshold for regime change
            
        Returns:
            List of regime change events
        """
        regimes = self.predict_regime(returns, volumes)
        regime_probs = self.predict_regime_proba(returns, volumes)
        
        changes = []
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                # Get confidence in new regime
                new_regime_conf = regime_probs[regimes[i]][i]
                
                if new_regime_conf >= threshold:
                    changes.append({
                        'index': i,
                        'from_regime': regimes[i-1],
                        'to_regime': regimes[i],
                        'confidence': new_regime_conf,
                        'transition_probs': {
                            regime: regime_probs[regime][i] 
                            for regime in self.regime_labels.values()
                        }
                    })
                    
        return changes