"""
Multi-Model Machine Learning Network for Cryptocurrency Trading
Main entry point for the trading system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data.data_loader import DataLoader, FileDataSource, create_synthetic_data
from features.technical_indicators import TechnicalIndicators
from models.statistical.arima.arima_model import ARIMA, AutoARIMA
from models.statistical.arima.parameter_estimator import ARIMAParameterEstimator
from models.statistical.arima.utils import ARIMAUtils
from models.statistical.garch.garch_model import GARCH
from models.statistical.garch.volatility_forecaster import VolatilityForecaster
from models.statistical.garch.utils import GARCHUtils
from models.risk_management.risk_manager import RiskManager
from models.unsupervised.hmm.hmm_model import RegimeDetector
from models.unsupervised.hmm.regime_analyzer import RegimeAnalyzer
from models.unsupervised.hmm.trainer import HMMTrainer


class CryptoMLTradingSystem:
    """Main orchestrator for the multi-model trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trading system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.data_loader = None
        self.current_data = None
        self.risk_manager = None
        self.regime_analyzer = None
        self.current_regime = None
        
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "data": {
                "source": "file",
                "data_dir": "./data/historical",
                "symbols": ["BTCUSDT"],
                "interval": "1m",
                "lookback_days": 30
            },
            "models": {
                "arima": {
                    "enabled": True,
                    "max_p": 5,
                    "max_d": 2,
                    "max_q": 5,
                    "refit_interval": 1440  # Refit every day
                },
                "garch": {
                    "enabled": True,
                    "p": 1,
                    "q": 1
                },
                "gru": {
                    "enabled": True,
                    "hidden_size": 128,
                    "num_layers": 3
                },
                "hmm": {
                    "enabled": True,
                    "n_states": None,  # Auto-select
                    "feature_set": "standard",
                    "update_interval": 1440  # Update regime detection daily
                }
            },
            "trading": {
                "position_sizing": "kelly",
                "max_position": 0.2,
                "stop_loss": 0.02,
                "take_profit": 0.03
            },
            "risk_management": {
                "max_drawdown": 0.15,
                "var_confidence": 0.95,
                "max_portfolio_risk": 0.05,
                "kelly_fraction": 0.25
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                self._merge_configs(default_config, loaded_config)
                
        return default_config
    
    def _merge_configs(self, default: dict, loaded: dict):
        """Recursively merge loaded config into default config."""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def initialize_data_source(self):
        """Initialize data source based on configuration."""
        source_type = self.config["data"]["source"]
        
        if source_type == "file":
            data_dir = self.config["data"]["data_dir"]
            data_source = FileDataSource(data_dir)
        else:
            raise ValueError(f"Unknown data source: {source_type}")
        
        self.data_loader = DataLoader(data_source)
        
    def load_historical_data(self, symbol: str, days_back: int = 30):
        """Load historical data for training."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        print(f"Loading data for {symbol} from {start_time} to {end_time}")
        
        try:
            # Try loading from data source
            df = self.data_loader.load_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval=self.config["data"]["interval"]
            )
        except FileNotFoundError:
            print(f"Data file not found for {symbol}. Creating synthetic data for demonstration...")
            df = create_synthetic_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval=self.config["data"]["interval"]
            )
            # Save synthetic data
            if isinstance(self.data_loader.data_source, FileDataSource):
                self.data_loader.data_source.save_data(df, symbol, self.config["data"]["interval"])
        
        # Calculate technical indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        self.current_data = df
        return df
    
    def train_arima_model(self, data: pd.DataFrame):
        """Train ARIMA model on historical data."""
        if not self.config["models"]["arima"]["enabled"]:
            return
        
        print("\nTraining ARIMA model...")
        
        # Use log returns for stationarity
        prices = data['close'].values
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Check stationarity
        stationarity = ARIMAUtils.check_stationarity(log_returns)
        print(f"Stationarity check: {stationarity['recommendation']}")
        
        # Estimate parameters using Box-Jenkins
        params = ARIMAParameterEstimator.box_jenkins_identification(
            log_returns,
            max_p=self.config["models"]["arima"]["max_p"],
            max_d=self.config["models"]["arima"]["max_d"],
            max_q=self.config["models"]["arima"]["max_q"]
        )
        print(f"Suggested ARIMA order: {params['suggested_order']}")
        
        # Fit model
        if self.config["models"]["arima"].get("auto_select", True):
            # Use AutoARIMA for automatic selection
            auto_arima = AutoARIMA(
                max_p=self.config["models"]["arima"]["max_p"],
                max_d=self.config["models"]["arima"]["max_d"],
                max_q=self.config["models"]["arima"]["max_q"]
            )
            model = auto_arima.fit(log_returns)
            print(f"AutoARIMA selected: {auto_arima.best_params}")
        else:
            # Use suggested parameters
            p, d, q = params['suggested_order']
            model = ARIMA(p=p, d=d, q=q)
            model.fit(log_returns)
        
        self.models['arima'] = model
        
        # Print model summary
        summary = model.summary()
        print(f"\nARIMA Model Summary:")
        print(f"Model: {summary['model']}")
        print(f"AIC: {summary['aic']:.4f}")
        print(f"BIC: {summary['bic']:.4f}")
        print(f"Log-likelihood: {summary['log_likelihood']:.4f}")
        
        # Make sample predictions
        predictions, lower, upper = model.predict(steps=60, return_conf_int=True)
        print(f"\nNext 60-minute forecast:")
        print(f"Mean prediction: {np.mean(predictions):.6f}")
        print(f"Prediction std: {np.std(predictions):.6f}")
        
        return model
    
    def train_garch_model(self, data: pd.DataFrame):
        """Train GARCH model for volatility prediction."""
        if not self.config["models"]["garch"]["enabled"]:
            return
        
        print("\nTraining GARCH model...")
        
        # Use returns for GARCH
        returns = data['returns'].dropna().values
        
        # Test for ARCH effects
        arch_test = GARCHUtils.test_arch_effects(returns)
        print(f"ARCH effects test: {arch_test['recommendation']}")
        
        # Fit GARCH model
        garch_model = GARCH(
            p=self.config["models"]["garch"]["p"],
            q=self.config["models"]["garch"]["q"],
            dist='t'  # Student's t for crypto
        )
        garch_model.fit(returns)
        
        self.models['garch'] = garch_model
        
        # Print summary
        summary = garch_model.summary()
        print(f"\nGARCH Model Summary:")
        print(f"Model: {summary['model']}")
        print(f"Persistence: {summary['persistence']:.4f}")
        print(f"Unconditional variance: {summary['unconditional_variance']:.6f}")
        print(f"AIC: {summary['aic']:.4f}")
        
        # Volatility forecast
        vol_forecast = garch_model.forecast(steps=60)
        print(f"\nVolatility forecast (next hour):")
        print(f"Current volatility: {vol_forecast['volatility'][0]:.6f}")
        print(f"Average volatility: {np.mean(vol_forecast['volatility']):.6f}")
        
        # Calculate VaR
        var_95, cvar_95 = garch_model.calculate_var(confidence_level=0.95)
        print(f"\nRisk metrics:")
        print(f"VaR (95%): {var_95:.4%}")
        print(f"CVaR (95%): {cvar_95:.4%}")
        
        return garch_model
    
    def train_hmm_model(self, data: pd.DataFrame):
        """Train Hidden Markov Model for regime detection."""
        if not self.config["models"]["hmm"]["enabled"]:
            return
        
        print("\nTraining Hidden Markov Model for regime detection...")
        
        # Initialize HMM trainer
        hmm_trainer = HMMTrainer(
            n_states=self.config["models"]["hmm"]["n_states"],
            feature_set=self.config["models"]["hmm"]["feature_set"]
        )
        
        # Train model
        results = hmm_trainer.train(data, validation_split=0.2, cv_folds=3)
        
        # Store model and analyzer
        self.models['hmm'] = hmm_trainer
        self.regime_analyzer = hmm_trainer.analyzer
        
        # Print results
        print(f"\nHMM Model Summary:")
        print(f"Number of regimes: {results['n_states']}")
        print(f"Train score: {results['train_score']:.4f}")
        print(f"Validation score: {results['val_score']:.4f}")
        if results.get('cv_mean'):
            print(f"Cross-validation score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        # Current regime analysis
        regime_analysis = results['regime_analysis']
        self.current_regime = regime_analysis['current_regime']
        
        print(f"\nCurrent Market Regime: {self.current_regime}")
        print(f"Regime Confidence: {regime_analysis['regime_probability'][self.current_regime]:.2%}")
        
        # Print regime statistics
        print("\nRegime Statistics:")
        for regime, stats in regime_analysis['regime_statistics'].items():
            print(f"\n{regime.upper()} regime:")
            print(f"  Mean return: {stats['mean_return']*100:.3f}%")
            print(f"  Volatility: {stats['volatility']*100:.2f}%")
            print(f"  Sharpe ratio: {stats['sharpe_ratio']:.2f}")
            print(f"  Frequency: {stats['frequency']:.1%}")
            print(f"  Max drawdown: {stats['max_drawdown']:.2%}")
        
        # Print transition matrix
        print("\nRegime Transition Probabilities:")
        print(regime_analysis['transition_matrix'])
        
        return hmm_trainer
    
    def generate_signals(self, current_price: float, current_time: datetime, 
                        current_data: Optional[pd.DataFrame] = None) -> dict:
        """Generate trading signals from all models."""
        signals = {}
        
        # HMM regime-based signal adjustment
        regime_recommendations = None
        if 'hmm' in self.models and current_data is not None:
            # Get current regime
            regime_pred = self.models['hmm'].predict_regime(current_data.tail(100))
            current_regime = regime_pred.iloc[-1]['regime']
            regime_confidence = regime_pred.iloc[-1]['regime_confidence']
            
            # Get regime recommendations
            if self.regime_analyzer:
                regime_probs = self.models['hmm'].detector.predict_regime_proba(
                    current_data['returns'].dropna().values[-100:],
                    current_data.get('volume', pd.Series()).values[-100:] if 'volume' in current_data else None
                )
                regime_recommendations = self.regime_analyzer._generate_strategy_recommendations(
                    current_regime, regime_probs, len(regime_probs[current_regime])-1
                )
            
            signals['hmm'] = {
                'action': 'hold',
                'confidence': regime_confidence,
                'regime': current_regime,
                'regime_recommendations': regime_recommendations
            }
        
        # Adjust position sizing based on regime
        position_multiplier = 1.0
        if regime_recommendations:
            position_multiplier = regime_recommendations['position_sizing'].get('base_size', 1.0)
        
        # ARIMA signal
        if 'arima' in self.models:
            arima_pred = self.models['arima'].predict(steps=5)
            arima_return = np.mean(arima_pred)
            
            # Convert log return to price prediction
            predicted_price = current_price * np.exp(arima_return)
            
            if predicted_price > current_price * 1.001:  # 0.1% threshold
                signals['arima'] = {'action': 'buy', 'confidence': 0.7}
            elif predicted_price < current_price * 0.999:
                signals['arima'] = {'action': 'sell', 'confidence': 0.7}
            else:
                signals['arima'] = {'action': 'hold', 'confidence': 0.5}
            
            # Add expected metrics for risk management
            signals['arima']['expected_return'] = arima_return
            signals['arima']['win_rate'] = 0.55  # Placeholder
            signals['arima']['win_loss_ratio'] = 1.5  # Placeholder
            
            # Adjust confidence based on regime
            if current_regime == 'bull' and signals['arima']['action'] == 'buy':
                signals['arima']['confidence'] *= 1.2
            elif current_regime == 'bear' and signals['arima']['action'] == 'sell':
                signals['arima']['confidence'] *= 1.2
            elif current_regime == 'sideways':
                signals['arima']['confidence'] *= 0.8
        
        # GARCH signal (volatility-based)
        if 'garch' in self.models:
            vol_forecast = self.models['garch'].forecast(steps=5)
            current_vol = vol_forecast['volatility'][0]
            future_vol = np.mean(vol_forecast['volatility'])
            
            # High volatility = opportunity for trading
            # Low volatility = stay out
            if future_vol > current_vol * 1.1:  # Rising volatility
                # Use ARIMA direction with higher confidence
                if 'arima' in signals:
                    signals['garch'] = {
                        'action': signals['arima']['action'],
                        'confidence': min(0.9, future_vol / current_vol * 0.5),
                        'volatility': future_vol
                    }
                else:
                    signals['garch'] = {'action': 'hold', 'confidence': 0.3}
            else:
                signals['garch'] = {'action': 'hold', 'confidence': 0.6}
        
        # Placeholder for other models
        # signals['gru'] = ...
        
        return signals
    
    def combine_signals(self, signals: dict) -> dict:
        """Combine signals from multiple models using ensemble logic."""
        # Get regime recommendations if available
        regime_recommendations = None
        if 'hmm' in signals and signals['hmm'].get('regime_recommendations'):
            regime_recommendations = signals['hmm']['regime_recommendations']
        
        # Simple voting mechanism for now
        actions = []
        confidences = []
        
        for model, signal in signals.items():
            actions.append(signal['action'])
            confidences.append(signal['confidence'])
        
        # Weighted voting
        action_scores = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for action, confidence in zip(actions, confidences):
            action_scores[action] += confidence
        
        # Select action with highest score
        final_action = max(action_scores, key=action_scores.get)
        final_confidence = action_scores[final_action] / sum(confidences)
        
        # Apply regime-based adjustments
        if regime_recommendations:
            # Adjust confidence based on regime alignment
            if signals.get('hmm', {}).get('regime') == 'bear' and final_action == 'buy':
                final_confidence *= 0.7  # Reduce confidence for contrarian trades
            elif signals.get('hmm', {}).get('regime') == 'bull' and final_action == 'sell':
                final_confidence *= 0.7
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'model_signals': signals,
            'regime': signals.get('hmm', {}).get('regime', 'unknown'),
            'position_multiplier': regime_recommendations['position_sizing']['base_size'] if regime_recommendations else 1.0
        }
    
    def calculate_position_size(self, signal: dict, current_price: float, 
                              portfolio_value: float = 100000) -> dict:
        """Calculate position size using risk management system."""
        if self.risk_manager is None:
            self.initialize_risk_manager()
        
        # Prepare market data
        market_data = {
            'price': current_price,
            'volatility': self.models['garch'].conditional_variance[-1]**0.5 if 'garch' in self.models else 0.02,
            'returns_history': self.current_data['returns'].dropna().values[-100:] if self.current_data is not None else np.array([])
        }
        
        # Calculate position size with risk management
        position_details = self.risk_manager.calculate_position_size(
            signal=signal,
            market_data=market_data,
            portfolio_value=portfolio_value
        )
        
        return position_details
    
    def initialize_risk_manager(self):
        """Initialize risk management system."""
        self.risk_manager = RiskManager(
            max_position_size=self.config["trading"]["max_position"],
            max_portfolio_risk=self.config["risk_management"]["max_portfolio_risk"],
            max_drawdown=self.config["risk_management"]["max_drawdown"],
            var_confidence=self.config["risk_management"]["var_confidence"],
            kelly_fraction=self.config["risk_management"]["kelly_fraction"]
        )
    
    def run_backtest(self, start_date: datetime, end_date: datetime):
        """Run backtest on historical data."""
        print("\nRunning backtest...")
        # Placeholder for backtesting logic
        print("Backtesting framework to be implemented")
        
    def run_live_trading(self):
        """Run live trading loop."""
        print("\nStarting live trading simulation...")
        
        # Simulate live trading with most recent data
        if self.current_data is None:
            raise ValueError("No data loaded. Call load_historical_data first.")
        
        # Use last 100 candles for simulation
        recent_data = self.current_data.tail(100)
        
        for i in range(len(recent_data) - 1):
            current_candle = recent_data.iloc[i]
            current_price = current_candle['close']
            current_time = current_candle['timestamp']
            
            # Generate signals
            signals = self.generate_signals(current_price, current_time, recent_data[:i+1])
            
            # Combine signals
            final_signal = self.combine_signals(signals)
            
            # Calculate position size with risk management
            position_details = self.calculate_position_size(
                final_signal, 
                current_price,
                portfolio_value=100000  # Example portfolio value
            )
            
            # Log decision
            if final_signal['action'] != 'hold' and position_details['position_size_pct'] > 0:
                print(f"\nTime: {current_time}")
                print(f"Price: ${current_price:.2f}")
                print(f"Signal: {final_signal['action'].upper()}")
                print(f"Confidence: {final_signal['confidence']:.2%}")
                print(f"Regime: {final_signal.get('regime', 'unknown').upper()}")
                print(f"Position Size: {position_details['position_size_pct']:.2%}")
                print(f"Kelly Size: {position_details['kelly_size']:.2%}")
                print(f"Position VaR: {position_details['position_var']:.2%}")
                print(f"Risk Budget Used: {position_details['risk_budget_used']:.1%}")
                
                # Calculate stop loss and take profit
                volatility = self.models['garch'].conditional_variance[-1]**0.5 if 'garch' in self.models else 0.02
                stop_loss = self.risk_manager.calculate_stop_loss(
                    entry_price=current_price,
                    position_size=position_details['position_size_pct'],
                    volatility=volatility
                )
                
                take_profit = self.risk_manager.calculate_take_profit(
                    entry_price=current_price,
                    stop_loss=stop_loss['stop_price'],
                    risk_reward_ratio=2.0
                )
                
                print(f"Stop Loss: ${stop_loss['stop_price']:.2f} ({stop_loss['stop_distance_pct']:.1f}%)")
                print(f"Take Profit: ${take_profit['take_profit_price']:.2f} ({take_profit['reward_distance_pct']:.1f}%)")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Multi-Model ML Network for Cryptocurrency Trading")
    print("=" * 60)
    
    # Initialize system
    system = CryptoMLTradingSystem()
    
    # Initialize data source
    system.initialize_data_source()
    
    # Load historical data
    symbol = system.config["data"]["symbols"][0]
    df = system.load_historical_data(symbol, days_back=7)  # Use 7 days for demo
    
    print(f"\nLoaded {len(df)} candles of {symbol} data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Train models
    system.train_arima_model(df)
    system.train_garch_model(df)
    system.train_hmm_model(df)
    
    # Initialize risk management
    system.initialize_risk_manager()
    
    # Show risk management capabilities
    print("\n" + "=" * 60)
    print("Risk Management System Initialized")
    print(f"Max Position Size: {system.config['trading']['max_position']:.0%}")
    print(f"Max Portfolio Risk (VaR): {system.config['risk_management']['max_portfolio_risk']:.0%}")
    print(f"Max Drawdown: {system.config['risk_management']['max_drawdown']:.0%}")
    print(f"Kelly Fraction: {system.config['risk_management']['kelly_fraction']:.0%}")
    
    # Run simulation
    print("\n" + "=" * 60)
    system.run_live_trading()
    
    print("\n" + "=" * 60)
    print("Trading simulation complete!")
    print("\nNext steps:")
    print("1. Implement CNN for chart pattern recognition")
    print("2. Implement PPO reinforcement learning")
    print("3. Implement Sentiment Analysis Transformer")
    print("4. Create Meta-Learner Ensemble")
    print("5. Add real-time data connection")
    print("6. Implement production deployment features")


if __name__ == "__main__":
    main()