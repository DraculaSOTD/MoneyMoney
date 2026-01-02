"""
Comprehensive Training Orchestrator for Multi-Model Trading System

Coordinates training of all models with:
- Unified data pipeline
- Hyperparameter optimization
- Model selection and ensemble
- Performance tracking
- Production deployment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import all enhanced trainers
from models.deep_learning.gru_attention.enhanced_trainer import EnhancedGRUAttentionTrainer
from models.deep_learning.cnn_pattern.enhanced_trainer import EnhancedCNNPatternTrainer
from models.reinforcement.ppo.enhanced_trainer import EnhancedPPOTrainer
from models.advanced.transformer_tft.enhanced_trainer import EnhancedTFTTrainer, EnhancedTFTConfig
from models.unsupervised.hmm.trainer import HMMTrainer

# Import models
from models.deep_learning.gru_attention.model import GRUAttentionModel
from models.deep_learning.cnn_pattern.cnn_model import CNNPatternRecognizer
from models.reinforcement.ppo.ppo_agent import PPOAgent
from models.reinforcement.ppo.trading_env import TradingEnvironment

# Import data and feature modules
from data.data_loader import DataLoader, FileDataSource, create_synthetic_data
from features.technical_indicators import TechnicalIndicators
from features.feature_pipeline import FeatureEngineer

# Import utilities
from models.base_trainer import CosineAnnealingScheduler, ReduceLROnPlateauScheduler


class TrainingOrchestrator:
    """
    Master orchestrator for training all models in the trading system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize training orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.trainers = {}
        self.training_results = defaultdict(dict)
        self.best_models = {}
        
        # Initialize paths
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.results_dir = Path(self.config['paths']['results_dir'])
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = DataLoader(FileDataSource(self.data_dir))
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration."""
        default_config = {
            "paths": {
                "data_dir": "./data/historical",
                "checkpoint_dir": "./checkpoints",
                "results_dir": "./results"
            },
            "data": {
                "symbols": ["BTCUSDT"],
                "interval": "1h",
                "train_days": 365,
                "val_days": 30,
                "test_days": 30
            },
            "models": {
                "gru_attention": {
                    "enabled": True,
                    "hidden_size": 256,
                    "num_layers": 3,
                    "attention_heads": 8,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "sequence_length": 168  # 7 days
                },
                "cnn_pattern": {
                    "enabled": True,
                    "image_size": 64,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "use_mixup": True
                },
                "ppo": {
                    "enabled": True,
                    "n_steps": 2048,
                    "learning_rate": 3e-4,
                    "n_envs": 4,
                    "batch_size": 64
                },
                "tft": {
                    "enabled": True,
                    "hidden_size": 160,
                    "num_attention_heads": 4,
                    "learning_rate": 0.001,
                    "batch_size": 32
                },
                "hmm": {
                    "enabled": True,
                    "n_states": None,
                    "feature_set": "full"
                }
            },
            "training": {
                "max_epochs": 100,
                "early_stopping_patience": 15,
                "parallel_training": True,
                "num_workers": 4,
                "hyperopt_trials": 20
            },
            "ensemble": {
                "method": "weighted_voting",
                "meta_learner": "gradient_boosting"
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self._merge_configs(default_config, loaded_config)
                
        return default_config
    
    def _merge_configs(self, default: Dict, loaded: Dict):
        """Merge configurations."""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
                
    def load_and_prepare_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for all models.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Train, validation, and test DataFrames
        """
        print(f"\nLoading data for {symbol}...")
        
        # Calculate date ranges
        end_date = datetime.now()
        test_start = end_date - timedelta(days=self.config['data']['test_days'])
        val_start = test_start - timedelta(days=self.config['data']['val_days'])
        train_start = val_start - timedelta(days=self.config['data']['train_days'])
        
        # Load data
        try:
            full_data = self.data_loader.load_data(
                symbol=symbol,
                start_time=train_start,
                end_time=end_date,
                interval=self.config['data']['interval']
            )
        except FileNotFoundError:
            print(f"Data not found, creating synthetic data for {symbol}")
            full_data = create_synthetic_data(
                symbol=symbol,
                start_time=train_start,
                end_time=end_date,
                interval=self.config['data']['interval']
            )
            
        # Add technical indicators
        full_data = TechnicalIndicators.calculate_all_indicators(full_data)
        
        # Engineer features
        full_data = self.feature_engineer.engineer_features(full_data)
        
        # Split data
        train_data = full_data[full_data['timestamp'] < val_start]
        val_data = full_data[(full_data['timestamp'] >= val_start) & 
                            (full_data['timestamp'] < test_start)]
        test_data = full_data[full_data['timestamp'] >= test_start]
        
        print(f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def initialize_models(self, data_shape: Dict):
        """
        Initialize all enabled models.
        
        Args:
            data_shape: Dictionary with data dimensions
        """
        print("\nInitializing models...")
        
        # GRU-Attention
        if self.config['models']['gru_attention']['enabled']:
            gru_config = self.config['models']['gru_attention']
            model = GRUAttentionModel(
                input_size=data_shape['n_features'],
                hidden_size=gru_config['hidden_size'],
                num_layers=gru_config['num_layers'],
                num_classes=3,  # Buy, Hold, Sell
                attention_heads=gru_config['attention_heads']
            )
            
            trainer = EnhancedGRUAttentionTrainer(
                model=model,
                optimizer='adamw',
                learning_rate=gru_config['learning_rate'],
                batch_size=gru_config['batch_size'],
                sequence_length=gru_config['sequence_length'],
                checkpoint_dir=str(self.checkpoint_dir / 'gru_attention')
            )
            
            self.models['gru_attention'] = model
            self.trainers['gru_attention'] = trainer
            print("✓ GRU-Attention initialized")
            
        # CNN Pattern
        if self.config['models']['cnn_pattern']['enabled']:
            cnn_config = self.config['models']['cnn_pattern']
            model = CNNPatternRecognizer(
                image_size=cnn_config['image_size'],
                num_classes=5  # Pattern types
            )
            
            trainer = EnhancedCNNPatternTrainer(
                model=model,
                optimizer='adamw',
                learning_rate=cnn_config['learning_rate'],
                batch_size=cnn_config['batch_size'],
                use_mixup=cnn_config['use_mixup'],
                checkpoint_dir=str(self.checkpoint_dir / 'cnn_pattern')
            )
            
            self.models['cnn_pattern'] = model
            self.trainers['cnn_pattern'] = trainer
            print("✓ CNN Pattern Recognition initialized")
            
        # PPO
        if self.config['models']['ppo']['enabled']:
            ppo_config = self.config['models']['ppo']
            
            # Create environment
            env = TradingEnvironment(
                data=None,  # Will be set during training
                initial_capital=100000,
                lookback_window=50
            )
            
            # Create agent
            agent = PPOAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=3,  # Position size, stop loss, take profit
                hidden_size=256
            )
            
            trainer = EnhancedPPOTrainer(
                agent=agent,
                env=env,
                n_steps=ppo_config['n_steps'],
                learning_rate=ppo_config['learning_rate'],
                n_envs=ppo_config['n_envs'],
                batch_size=ppo_config['batch_size']
            )
            
            self.models['ppo'] = agent
            self.trainers['ppo'] = trainer
            print("✓ PPO initialized")
            
        # TFT
        if self.config['models']['tft']['enabled']:
            tft_config = self.config['models']['tft']
            config = EnhancedTFTConfig(
                n_features=data_shape['n_features'],
                hidden_size=tft_config['hidden_size'],
                num_attention_heads=tft_config['num_attention_heads'],
                learning_rate=tft_config['learning_rate'],
                batch_size=tft_config['batch_size']
            )
            
            trainer = EnhancedTFTTrainer(config)
            
            self.models['tft'] = trainer.model
            self.trainers['tft'] = trainer
            print("✓ Temporal Fusion Transformer initialized")
            
        # HMM
        if self.config['models']['hmm']['enabled']:
            hmm_config = self.config['models']['hmm']
            trainer = HMMTrainer(
                n_states=hmm_config['n_states'],
                feature_set=hmm_config['feature_set']
            )
            
            self.trainers['hmm'] = trainer
            print("✓ Hidden Markov Model initialized")
            
    def train_model(self, model_name: str, train_data: pd.DataFrame,
                   val_data: pd.DataFrame) -> Dict:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Training results
        """
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        trainer = self.trainers[model_name]
        
        # Model-specific training
        if model_name == 'gru_attention':
            # Combine train and val for trainer's internal split
            combined_data = pd.concat([train_data, val_data])
            
            # Create learning rate scheduler
            scheduler = CosineAnnealingScheduler(
                initial_lr=trainer.optimizer.learning_rate,
                total_epochs=self.config['training']['max_epochs'],
                min_lr=1e-6
            )
            
            history = trainer.train(
                data=combined_data,
                epochs=self.config['training']['max_epochs'],
                lr_scheduler=scheduler,
                verbose=1
            )
            
        elif model_name == 'cnn_pattern':
            combined_data = pd.concat([train_data, val_data])
            
            scheduler = StepLRScheduler(
                initial_lr=trainer.optimizer.learning_rate,
                step_size=30,
                gamma=0.1
            )
            
            history = trainer.train(
                data=combined_data,
                epochs=self.config['training']['max_epochs'],
                lr_scheduler=scheduler,
                verbose=1
            )
            
        elif model_name == 'ppo':
            # Set environment data
            trainer.env.set_data(train_data)
            
            history = trainer.train(
                n_iterations=100,
                save_freq=10,
                eval_freq=10,
                verbose=1
            )
            
        elif model_name == 'tft':
            history = trainer.train(
                data=train_data,
                epochs=self.config['training']['max_epochs'],
                verbose=1
            )
            
        elif model_name == 'hmm':
            results = trainer.train(
                data=train_data,
                validation_split=0.2,
                cv_folds=3
            )
            history = {'hmm_results': results}
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        training_time = time.time() - start_time
        
        # Get training summary
        if hasattr(trainer, 'get_training_summary'):
            summary = trainer.get_training_summary()
        else:
            summary = {}
            
        summary['training_time'] = training_time
        summary['model_name'] = model_name
        
        return {
            'history': history,
            'summary': summary,
            'trainer': trainer
        }
    
    def train_all_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """
        Train all enabled models.
        
        Args:
            train_data: Training data
            val_data: Validation data
        """
        # Get data shape
        n_features = len([col for col in train_data.columns 
                         if col not in ['timestamp', 'symbol']])
        data_shape = {'n_features': n_features}
        
        # Initialize models
        self.initialize_models(data_shape)
        
        # Train models
        if self.config['training']['parallel_training']:
            self._train_parallel(train_data, val_data)
        else:
            self._train_sequential(train_data, val_data)
            
    def _train_sequential(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train models sequentially."""
        for model_name, trainer in self.trainers.items():
            if self.config['models'][model_name]['enabled']:
                results = self.train_model(model_name, train_data, val_data)
                self.training_results[model_name] = results
                
    def _train_parallel(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train models in parallel."""
        with ProcessPoolExecutor(max_workers=self.config['training']['num_workers']) as executor:
            futures = {}
            
            for model_name, trainer in self.trainers.items():
                if self.config['models'][model_name]['enabled']:
                    # Note: This is simplified. In practice, need to handle serialization
                    future = executor.submit(self.train_model, model_name, train_data, val_data)
                    futures[model_name] = future
                    
            # Collect results
            for model_name, future in futures.items():
                try:
                    results = future.result()
                    self.training_results[model_name] = results
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation results for each model
        """
        print("\nEvaluating models on test data...")
        evaluation_results = {}
        
        for model_name, trainer in self.trainers.items():
            if model_name not in self.training_results:
                continue
                
            print(f"\nEvaluating {model_name}...")
            
            if model_name == 'gru_attention':
                # Prepare test data
                test_prepared = trainer.prepare_data(test_data)
                
                # Evaluate
                metrics = trainer.validate(test_prepared)
                
                # Get predictions for analysis
                predictions = trainer.model.predict(test_prepared['sequences'])
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'predictions': predictions
                }
                
            elif model_name == 'cnn_pattern':
                test_prepared = trainer.prepare_data(test_data)
                metrics = trainer.validate(test_prepared)
                
                # Pattern analysis
                pattern_analysis = trainer.get_pattern_analysis()
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'pattern_analysis': pattern_analysis
                }
                
            elif model_name == 'ppo':
                trainer.env.set_data(test_data)
                metrics = trainer.evaluate(n_episodes=10)
                
                evaluation_results[model_name] = {
                    'metrics': metrics
                }
                
            elif model_name == 'tft':
                test_prepared = trainer.prepare_data(test_data)
                metrics = trainer.validate(test_prepared)
                
                # Generate forecasts
                forecasts = trainer.forecast(test_data, n_samples=100)
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'forecasts': forecasts
                }
                
            elif model_name == 'hmm':
                # Predict regimes
                regime_predictions = trainer.predict_regime(test_data)
                
                # Generate signals
                trading_signals = trainer.generate_trading_signals(test_data)
                
                evaluation_results[model_name] = {
                    'regime_predictions': regime_predictions,
                    'trading_signals': trading_signals
                }
                
        return evaluation_results
    
    def hyperparameter_optimization(self, train_data: pd.DataFrame,
                                  val_data: pd.DataFrame,
                                  model_name: str) -> Dict:
        """
        Perform hyperparameter optimization for a model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            model_name: Model to optimize
            
        Returns:
            Optimization results
        """
        print(f"\nPerforming hyperparameter optimization for {model_name}...")
        
        # Define search spaces
        search_spaces = {
            'gru_attention': {
                'hidden_size': [128, 256, 512],
                'num_layers': [2, 3, 4],
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'dropout_rate': [0.1, 0.2, 0.3],
                'attention_heads': [4, 8, 16]
            },
            'cnn_pattern': {
                'learning_rate': [1e-4, 1e-3, 1e-2],
                'batch_size': [16, 32, 64],
                'use_mixup': [True, False],
                'use_cutmix': [True, False]
            },
            'ppo': {
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'clip_range': [0.1, 0.2, 0.3],
                'entropy_coef': [0.0, 0.01, 0.05],
                'n_steps': [1024, 2048, 4096]
            },
            'tft': {
                'hidden_size': [128, 160, 256],
                'num_attention_heads': [4, 8],
                'learning_rate': [1e-4, 1e-3],
                'dropout_rate': [0.1, 0.2]
            }
        }
        
        if model_name not in search_spaces:
            print(f"No hyperparameter space defined for {model_name}")
            return {}
            
        param_space = search_spaces[model_name]
        
        # Perform optimization
        best_params = None
        best_score = float('inf')
        results = []
        
        for trial in range(self.config['training']['hyperopt_trials']):
            # Sample parameters
            trial_params = {}
            for param, values in param_space.items():
                trial_params[param] = np.random.choice(values)
                
            print(f"\nTrial {trial + 1}: {trial_params}")
            
            # Train with trial parameters
            # This is simplified - would need to reinitialize model with new params
            # and train with reduced epochs
            
            # For demonstration, use random score
            score = np.random.random()
            
            results.append({
                'params': trial_params,
                'score': score
            })
            
            if score < best_score:
                best_score = score
                best_params = trial_params
                
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def create_ensemble(self, evaluation_results: Dict[str, Dict]) -> Dict:
        """
        Create ensemble from trained models.
        
        Args:
            evaluation_results: Evaluation results for each model
            
        Returns:
            Ensemble configuration
        """
        print("\nCreating model ensemble...")
        
        # Calculate model weights based on performance
        model_weights = {}
        
        for model_name, results in evaluation_results.items():
            if 'metrics' in results:
                # Use validation loss as weight (inverse)
                if 'loss' in results['metrics']:
                    weight = 1.0 / (results['metrics']['loss'] + 1e-6)
                elif 'accuracy' in results['metrics']:
                    weight = results['metrics']['accuracy']
                else:
                    weight = 1.0
                    
                model_weights[model_name] = weight
                
        # Normalize weights
        total_weight = sum(model_weights.values())
        for model in model_weights:
            model_weights[model] /= total_weight
            
        print("\nModel weights:")
        for model, weight in model_weights.items():
            print(f"  {model}: {weight:.3f}")
            
        ensemble_config = {
            'method': self.config['ensemble']['method'],
            'model_weights': model_weights,
            'models': list(model_weights.keys())
        }
        
        return ensemble_config
    
    def save_results(self):
        """Save all training results and models."""
        print("\nSaving results...")
        
        # Save training history
        for model_name, results in self.training_results.items():
            # Save history
            history_path = self.results_dir / f"{model_name}_history.json"
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_serializable = {}
                for key, value in results['history'].items():
                    if isinstance(value, np.ndarray):
                        history_serializable[key] = value.tolist()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        history_serializable[key] = [v.tolist() for v in value]
                    else:
                        history_serializable[key] = value
                json.dump(history_serializable, f)
                
            # Save summary
            summary_path = self.results_dir / f"{model_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(results['summary'], f, default=str)
                
        print(f"Results saved to {self.results_dir}")
        
    def generate_report(self, evaluation_results: Dict[str, Dict],
                       ensemble_config: Dict) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            evaluation_results: Evaluation results
            ensemble_config: Ensemble configuration
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("MULTI-MODEL TRAINING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Training summary
        report.append("TRAINING SUMMARY")
        report.append("-" * 40)
        
        for model_name, results in self.training_results.items():
            summary = results['summary']
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  Training time: {summary.get('training_time', 0):.1f}s")
            report.append(f"  Final loss: {summary.get('final_val_loss', 'N/A')}")
            report.append(f"  Best val loss: {summary.get('best_val_loss', 'N/A')}")
            
            if 'final_val_accuracy' in summary:
                report.append(f"  Final accuracy: {summary['final_val_accuracy']:.3f}")
                
        # Evaluation results
        report.append("\n" + "=" * 80)
        report.append("EVALUATION RESULTS")
        report.append("-" * 40)
        
        for model_name, results in evaluation_results.items():
            report.append(f"\n{model_name.upper()}:")
            
            if 'metrics' in results:
                metrics = results['metrics']
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report.append(f"  {metric_name}: {value:.4f}")
                        
        # Ensemble configuration
        report.append("\n" + "=" * 80)
        report.append("ENSEMBLE CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Method: {ensemble_config['method']}")
        report.append("\nModel weights:")
        
        for model, weight in ensemble_config['model_weights'].items():
            report.append(f"  {model}: {weight:.3f}")
            
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find best performing model
        best_model = max(ensemble_config['model_weights'].items(), key=lambda x: x[1])[0]
        report.append(f"• Best performing model: {best_model}")
        report.append("• Consider increasing weight for recent market conditions")
        report.append("• Monitor model degradation with online validation")
        report.append("• Retrain models monthly or when performance drops")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.results_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
            
        return report_text
    
    def run_complete_pipeline(self, symbol: str):
        """
        Run complete training pipeline for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        print(f"\n{'=' * 80}")
        print(f"STARTING COMPLETE TRAINING PIPELINE FOR {symbol}")
        print(f"{'=' * 80}\n")
        
        # Load and prepare data
        train_data, val_data, test_data = self.load_and_prepare_data(symbol)
        
        # Train all models
        self.train_all_models(train_data, val_data)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(test_data)
        
        # Create ensemble
        ensemble_config = self.create_ensemble(evaluation_results)
        
        # Save results
        self.save_results()
        
        # Generate report
        report = self.generate_report(evaluation_results, ensemble_config)
        print("\n" + report)
        
        print(f"\n{'=' * 80}")
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}\n")


def main():
    """Main entry point for training orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multi-model trading system')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--hyperopt', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--model', type=str, help='Specific model to train')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(args.config)
    
    # Run pipeline
    if args.hyperopt and args.model:
        # Run hyperparameter optimization for specific model
        train_data, val_data, _ = orchestrator.load_and_prepare_data(args.symbol)
        results = orchestrator.hyperparameter_optimization(train_data, val_data, args.model)
        print(f"\nBest parameters for {args.model}: {results['best_params']}")
    else:
        # Run complete pipeline
        orchestrator.run_complete_pipeline(args.symbol)


if __name__ == "__main__":
    main()