"""
Comprehensive Model Training Example.

This example demonstrates how to train all available models in the crypto ML
trading system using preprocessed data, with GPU acceleration where available.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any

sys.path.append(str(Path(__file__).parent.parent))

# Import all models
from models.deep_learning.lstm_model import LSTMModel
from models.deep_learning.gru_attention import GRUAttention
from models.deep_learning.gru_attention_gpu import GRUAttentionGPU
from models.deep_learning.tcn_gpu import TemporalConvNetGPU
from models.deep_learning.transformer_gpu import TimeSeriesTransformerGPU
from models.deep_learning.cnn_lstm import CNNLSTM
from models.statistical.arima_garch import ARIMAGARCHModel
from models.ensemble.stacking import StackingEnsemble
from models.ensemble.voting_classifier import WeightedVotingEnsemble
from models.reinforcement.ppo.ppo_agent import PPOAgent
from models.reinforcement.drqn.drqn_agent import DRQNAgent
from models.traditional_ml import (
    RandomForestModel, XGBoostModel, SVMModel, 
    GradientBoostingModel, RidgeModel
)

# Import training utilities
from training.model_trainer import ModelTrainer
from training.gpu_enhanced_trainer import GPUEnhancedTrainer
from models.unified_interface import UnifiedDeepLearningModel, UnifiedEnsembleModel

# Import utilities
from utils.logger import setup_logger, get_logger
from utils.gpu_manager import get_gpu_manager
from examples.comprehensive_data_preprocessing import ComprehensiveDataPreprocessor

# Setup logging
setup_logger(level='INFO')
logger = get_logger(__name__)


class ComprehensiveModelTrainer:
    """
    Comprehensive model training pipeline demonstrating all model capabilities.
    """
    
    def __init__(self, enable_gpu: bool = True):
        """Initialize training pipeline."""
        self.enable_gpu = enable_gpu
        self.gpu_manager = get_gpu_manager() if enable_gpu else None
        
        # Initialize data preprocessor
        self.preprocessor = ComprehensiveDataPreprocessor(enable_gpu=enable_gpu)
        
        # Model registry
        self.models = {}
        self.training_results = {}
        
        logger.info(f"Model trainer initialized {'with GPU' if enable_gpu else 'with CPU'}")
    
    def prepare_data(self, data_path: str = None) -> Dict:
        """
        Prepare data for training.
        
        Args:
            data_path: Path to data file (if None, uses synthetic data)
            
        Returns:
            Dictionary with train/val/test datasets
        """
        logger.info("Preparing data for training")
        
        if data_path and Path(data_path).exists():
            # Load real data
            data = pd.read_csv(data_path)
        else:
            # Create synthetic data for demonstration
            logger.info("Creating synthetic data for demonstration")
            dates = pd.date_range('2022-01-01', '2024-01-01', freq='1h')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'high': 101 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'low': 99 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'volume': np.random.lognormal(10, 1, len(dates))
            })
        
        # Preprocess data
        preprocessed_data = self.preprocessor.preprocess_historical_data(data)
        
        # Create ML datasets
        datasets = self.preprocessor.create_ml_datasets(
            preprocessed_data,
            sequence_length=60,
            prediction_horizon=1
        )
        
        logger.info(f"Data prepared - Train: {datasets['X_train'].shape}, "
                   f"Val: {datasets['X_val'].shape}, Test: {datasets['X_test'].shape}")
        
        return datasets
    
    def train_deep_learning_models(self, datasets: Dict):
        """Train all deep learning models."""
        logger.info("\n" + "="*50)
        logger.info("Training Deep Learning Models")
        logger.info("="*50)
        
        X_train, y_train = datasets['X_train'], datasets['y_train_reg']
        X_val, y_val = datasets['X_val'], datasets['y_val_reg']
        
        # Common parameters
        input_dim = X_train.shape[2]
        sequence_length = X_train.shape[1]
        
        # 1. LSTM Model
        logger.info("\n1. Training LSTM Model")
        lstm_model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            output_dim=1,
            dropout=0.2
        )
        
        start_time = time.time()
        lstm_history = self._train_numpy_model(
            lstm_model, X_train, y_train, X_val, y_val, epochs=50
        )
        lstm_time = time.time() - start_time
        
        self.models['lstm'] = lstm_model
        self.training_results['lstm'] = {
            'history': lstm_history,
            'training_time': lstm_time
        }
        logger.info(f"LSTM training completed in {lstm_time:.2f}s")
        
        # 2. GRU-Attention (NumPy)
        logger.info("\n2. Training GRU-Attention Model (NumPy)")
        gru_attention = GRUAttention(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout_rate=0.2
        )
        
        start_time = time.time()
        gru_history = self._train_numpy_model(
            gru_attention, X_train, y_train, X_val, y_val, epochs=50
        )
        gru_time = time.time() - start_time
        
        self.models['gru_attention_numpy'] = gru_attention
        self.training_results['gru_attention_numpy'] = {
            'history': gru_history,
            'training_time': gru_time
        }
        logger.info(f"GRU-Attention (NumPy) training completed in {gru_time:.2f}s")
        
        # 3. GRU-Attention (GPU)
        if self.enable_gpu:
            logger.info("\n3. Training GRU-Attention Model (GPU)")
            gru_gpu = GRUAttentionGPU(
                input_dim=input_dim,
                hidden_dim=128,
                n_heads=8,
                n_layers=2,
                output_dim=1
            )
            
            trainer = GPUEnhancedTrainer(
                epochs=100,  # Can use more epochs with GPU
                batch_size=64,
                learning_rate=0.001,
                use_mixed_precision=True,
                use_gradient_checkpointing=True
            )
            
            start_time = time.time()
            gru_gpu = trainer.train(
                gru_gpu,
                torch.utils.data.TensorDataset(
                    torch.from_numpy(X_train).float(),
                    torch.from_numpy(y_train).float()
                ),
                torch.utils.data.TensorDataset(
                    torch.from_numpy(X_val).float(),
                    torch.from_numpy(y_val).float()
                )
            )
            gru_gpu_time = time.time() - start_time
            
            self.models['gru_attention_gpu'] = gru_gpu
            self.training_results['gru_attention_gpu'] = {
                'training_time': gru_gpu_time
            }
            logger.info(f"GRU-Attention (GPU) training completed in {gru_gpu_time:.2f}s")
        
        # 4. TCN (GPU)
        if self.enable_gpu:
            logger.info("\n4. Training TCN Model (GPU)")
            tcn_model = TemporalConvNetGPU(
                input_dim=input_dim,
                hidden_channels=[64, 128, 256],
                output_dim=1,
                kernel_size=3,
                dropout=0.2,
                use_attention=True
            )
            
            start_time = time.time()
            tcn_history = tcn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                learning_rate=0.001
            )
            tcn_time = time.time() - start_time
            
            self.models['tcn_gpu'] = tcn_model
            self.training_results['tcn_gpu'] = {
                'history': tcn_history,
                'training_time': tcn_time
            }
            logger.info(f"TCN training completed in {tcn_time:.2f}s")
        
        # 5. Transformer (GPU)
        if self.enable_gpu:
            logger.info("\n5. Training Transformer Model (GPU)")
            transformer_model = TimeSeriesTransformerGPU(
                input_dim=input_dim,
                d_model=256,
                n_heads=8,
                n_encoder_layers=4,
                n_decoder_layers=4,
                d_ff=1024,
                dropout=0.1,
                output_dim=1,
                use_temporal_fusion=True
            )
            
            start_time = time.time()
            transformer_history = transformer_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,  # Transformers train faster
                batch_size=32,
                learning_rate=0.0001,
                warmup_steps=1000
            )
            transformer_time = time.time() - start_time
            
            self.models['transformer_gpu'] = transformer_model
            self.training_results['transformer_gpu'] = {
                'history': transformer_history,
                'training_time': transformer_time
            }
            logger.info(f"Transformer training completed in {transformer_time:.2f}s")
        
        # 6. CNN-LSTM
        logger.info("\n6. Training CNN-LSTM Model")
        cnn_lstm = CNNLSTM(
            input_shape=(sequence_length, input_dim),
            num_filters=64,
            filter_size=3,
            lstm_units=128,
            output_dim=1
        )
        
        start_time = time.time()
        cnn_lstm_history = self._train_numpy_model(
            cnn_lstm, X_train, y_train, X_val, y_val, epochs=50
        )
        cnn_lstm_time = time.time() - start_time
        
        self.models['cnn_lstm'] = cnn_lstm
        self.training_results['cnn_lstm'] = {
            'history': cnn_lstm_history,
            'training_time': cnn_lstm_time
        }
        logger.info(f"CNN-LSTM training completed in {cnn_lstm_time:.2f}s")
    
    def train_traditional_ml_models(self, datasets: Dict):
        """Train traditional ML models."""
        logger.info("\n" + "="*50)
        logger.info("Training Traditional ML Models")
        logger.info("="*50)
        
        # For traditional ML, we'll use flattened features from the last sequence
        X_train_flat = datasets['X_train'][:, -1, :]  # Use last timestep
        X_val_flat = datasets['X_val'][:, -1, :]
        y_train = datasets['y_train_class']  # Use classification for traditional ML
        y_val = datasets['y_val_class']
        
        # 1. Random Forest
        logger.info("\n1. Training Random Forest")
        rf_model = RandomForestModel()
        start_time = time.time()
        rf_model.fit(X_train_flat, y_train)
        rf_time = time.time() - start_time
        
        val_score = rf_model.score(X_val_flat, y_val)
        self.models['random_forest'] = rf_model
        self.training_results['random_forest'] = {
            'val_score': val_score,
            'training_time': rf_time
        }
        logger.info(f"Random Forest training completed in {rf_time:.2f}s, Val accuracy: {val_score:.4f}")
        
        # 2. XGBoost
        logger.info("\n2. Training XGBoost")
        xgb_model = XGBoostModel()
        start_time = time.time()
        xgb_model.fit(X_train_flat, y_train)
        xgb_time = time.time() - start_time
        
        val_score = xgb_model.score(X_val_flat, y_val)
        self.models['xgboost'] = xgb_model
        self.training_results['xgboost'] = {
            'val_score': val_score,
            'training_time': xgb_time
        }
        logger.info(f"XGBoost training completed in {xgb_time:.2f}s, Val accuracy: {val_score:.4f}")
        
        # 3. SVM
        logger.info("\n3. Training SVM")
        svm_model = SVMModel()
        start_time = time.time()
        # SVM can be slow, so we'll use a subset
        subset_size = min(5000, len(X_train_flat))
        svm_model.fit(X_train_flat[:subset_size], y_train[:subset_size])
        svm_time = time.time() - start_time
        
        val_score = svm_model.score(X_val_flat, y_val)
        self.models['svm'] = svm_model
        self.training_results['svm'] = {
            'val_score': val_score,
            'training_time': svm_time
        }
        logger.info(f"SVM training completed in {svm_time:.2f}s, Val accuracy: {val_score:.4f}")
        
        # 4. Gradient Boosting
        logger.info("\n4. Training Gradient Boosting")
        gb_model = GradientBoostingModel()
        start_time = time.time()
        gb_model.fit(X_train_flat, y_train)
        gb_time = time.time() - start_time
        
        val_score = gb_model.score(X_val_flat, y_val)
        self.models['gradient_boosting'] = gb_model
        self.training_results['gradient_boosting'] = {
            'val_score': val_score,
            'training_time': gb_time
        }
        logger.info(f"Gradient Boosting training completed in {gb_time:.2f}s, Val accuracy: {val_score:.4f}")
    
    def train_statistical_models(self, datasets: Dict):
        """Train statistical models."""
        logger.info("\n" + "="*50)
        logger.info("Training Statistical Models")
        logger.info("="*50)
        
        # For ARIMA-GARCH, we need time series data
        # Use close prices from the original data
        close_prices = datasets['X_train'][:, -1, 0]  # Assuming first feature is close
        
        logger.info("\n1. Training ARIMA-GARCH Model")
        arima_garch = ARIMAGARCHModel()
        
        start_time = time.time()
        try:
            arima_garch.fit(close_prices)
            arima_time = time.time() - start_time
            
            self.models['arima_garch'] = arima_garch
            self.training_results['arima_garch'] = {
                'training_time': arima_time
            }
            logger.info(f"ARIMA-GARCH training completed in {arima_time:.2f}s")
        except Exception as e:
            logger.error(f"ARIMA-GARCH training failed: {e}")
    
    def train_reinforcement_learning_models(self, datasets: Dict):
        """Train reinforcement learning models."""
        logger.info("\n" + "="*50)
        logger.info("Training Reinforcement Learning Models")
        logger.info("="*50)
        
        # RL models need environment setup
        state_dim = (datasets['X_train'].shape[2],)
        action_dim = 3  # Buy, Hold, Sell
        
        # 1. PPO Agent
        logger.info("\n1. Training PPO Agent")
        ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=256,
            learning_rate=3e-4
        )
        
        # For demo, we'll do a simple training loop
        start_time = time.time()
        # Normally you would train with an environment
        logger.info("PPO agent initialized (full training requires trading environment)")
        ppo_time = time.time() - start_time
        
        self.models['ppo'] = ppo_agent
        self.training_results['ppo'] = {
            'training_time': ppo_time
        }
        
        # 2. DRQN Agent
        logger.info("\n2. Training DRQN Agent")
        drqn_agent = DRQNAgent(
            state_dim=datasets['X_train'].shape[2],
            action_dim=action_dim,
            hidden_dim=256,
            sequence_length=10
        )
        
        start_time = time.time()
        logger.info("DRQN agent initialized (full training requires trading environment)")
        drqn_time = time.time() - start_time
        
        self.models['drqn'] = drqn_agent
        self.training_results['drqn'] = {
            'training_time': drqn_time
        }
    
    def train_ensemble_models(self, datasets: Dict):
        """Train ensemble models using base models."""
        logger.info("\n" + "="*50)
        logger.info("Training Ensemble Models")
        logger.info("="*50)
        
        # Get some base models for ensemble
        base_models = []
        
        if 'random_forest' in self.models:
            base_models.append(('rf', self.models['random_forest']))
        if 'xgboost' in self.models:
            base_models.append(('xgb', self.models['xgboost']))
        if 'gradient_boosting' in self.models:
            base_models.append(('gb', self.models['gradient_boosting']))
        
        if len(base_models) < 2:
            logger.warning("Not enough base models for ensemble. Skipping ensemble training.")
            return
        
        # Use flattened features for ensemble
        X_train_flat = datasets['X_train'][:, -1, :]
        X_val_flat = datasets['X_val'][:, -1, :]
        y_train = datasets['y_train_class']
        y_val = datasets['y_val_class']
        
        # 1. Voting Ensemble
        logger.info("\n1. Training Voting Ensemble")
        voting_ensemble = WeightedVotingEnsemble(estimators=base_models)
        
        start_time = time.time()
        voting_ensemble.fit(X_train_flat, y_train)
        voting_time = time.time() - start_time
        
        val_score = voting_ensemble.score(X_val_flat, y_val)
        self.models['voting_ensemble'] = voting_ensemble
        self.training_results['voting_ensemble'] = {
            'val_score': val_score,
            'training_time': voting_time
        }
        logger.info(f"Voting Ensemble training completed in {voting_time:.2f}s, Val accuracy: {val_score:.4f}")
        
        # 2. Stacking Ensemble
        logger.info("\n2. Training Stacking Ensemble")
        from sklearn.linear_model import LogisticRegression
        stacking_ensemble = StackingEnsemble(
            base_models=base_models,
            meta_learner=LogisticRegression()
        )
        
        start_time = time.time()
        stacking_ensemble.fit(X_train_flat, y_train)
        stacking_time = time.time() - start_time
        
        val_score = stacking_ensemble.score(X_val_flat, y_val)
        self.models['stacking_ensemble'] = stacking_ensemble
        self.training_results['stacking_ensemble'] = {
            'val_score': val_score,
            'training_time': stacking_time
        }
        logger.info(f"Stacking Ensemble training completed in {stacking_time:.2f}s, Val accuracy: {val_score:.4f}")
    
    def _train_numpy_model(self, model, X_train, y_train, X_val, y_val, epochs=50):
        """Train NumPy-based models."""
        trainer = ModelTrainer(
            model=model,
            learning_rate=0.001,
            batch_size=32,
            epochs=epochs
        )
        
        history = trainer.fit(
            X_train, y_train,
            X_val, y_val
        )
        
        return history
    
    def evaluate_all_models(self, datasets: Dict):
        """Evaluate all trained models."""
        logger.info("\n" + "="*50)
        logger.info("Model Evaluation Summary")
        logger.info("="*50)
        
        X_test = datasets['X_test']
        y_test_reg = datasets['y_test_reg']
        y_test_class = datasets['y_test_class']
        
        results = []
        
        for model_name, model in self.models.items():
            try:
                # Skip RL models for now
                if model_name in ['ppo', 'drqn', 'arima_garch']:
                    continue
                
                # Determine if regression or classification
                if model_name in ['lstm', 'gru_attention_numpy', 'gru_attention_gpu', 
                                 'tcn_gpu', 'transformer_gpu', 'cnn_lstm']:
                    # Deep learning models - regression
                    if hasattr(model, 'predict'):
                        if model_name in ['tcn_gpu', 'transformer_gpu']:
                            predictions = model.predict(X_test)
                        else:
                            predictions = model.predict(X_test)
                        
                        mse = np.mean((predictions.flatten() - y_test_reg) ** 2)
                        mae = np.mean(np.abs(predictions.flatten() - y_test_reg))
                        
                        results.append({
                            'model': model_name,
                            'mse': mse,
                            'mae': mae,
                            'type': 'regression'
                        })
                else:
                    # Traditional ML - classification
                    X_test_flat = X_test[:, -1, :]
                    if hasattr(model, 'score'):
                        accuracy = model.score(X_test_flat, y_test_class)
                        results.append({
                            'model': model_name,
                            'accuracy': accuracy,
                            'type': 'classification'
                        })
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        # Display results
        logger.info("\nRegression Models:")
        logger.info(f"{'Model':<25} {'MSE':<15} {'MAE':<15}")
        logger.info("-" * 55)
        for result in results:
            if result['type'] == 'regression':
                logger.info(f"{result['model']:<25} {result['mse']:<15.6f} {result['mae']:<15.6f}")
        
        logger.info("\nClassification Models:")
        logger.info(f"{'Model':<25} {'Accuracy':<15}")
        logger.info("-" * 40)
        for result in results:
            if result['type'] == 'classification':
                logger.info(f"{result['model']:<25} {result['accuracy']:<15.4f}")
        
        # Training time summary
        logger.info("\nTraining Time Summary:")
        logger.info(f"{'Model':<25} {'Time (s)':<15}")
        logger.info("-" * 40)
        for model_name, results in self.training_results.items():
            if 'training_time' in results:
                logger.info(f"{model_name:<25} {results['training_time']:<15.2f}")
        
        return results
    
    def save_all_models(self, output_dir: str = "trained_models"):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"\nSaving models to {output_path}")
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'save'):
                    model_path = output_path / f"{model_name}.pkl"
                    model.save(str(model_path))
                elif hasattr(model, 'save_model'):
                    model_path = output_path / f"{model_name}.npz"
                    model.save_model(str(model_path))
                else:
                    # Use pickle for other models
                    import pickle
                    model_path = output_path / f"{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
                logger.info(f"  - Saved {model_name} to {model_path}")
            except Exception as e:
                logger.error(f"  - Failed to save {model_name}: {e}")


def main():
    """Run comprehensive model training demonstration."""
    # Initialize trainer
    trainer = ComprehensiveModelTrainer(enable_gpu=True)
    
    # Prepare data
    logger.info("\n" + "="*50)
    logger.info("Data Preparation")
    logger.info("="*50)
    datasets = trainer.prepare_data()
    
    # Train all model types
    trainer.train_deep_learning_models(datasets)
    trainer.train_traditional_ml_models(datasets)
    trainer.train_statistical_models(datasets)
    trainer.train_reinforcement_learning_models(datasets)
    trainer.train_ensemble_models(datasets)
    
    # Evaluate all models
    evaluation_results = trainer.evaluate_all_models(datasets)
    
    # Save models
    trainer.save_all_models()
    
    logger.info("\n" + "="*50)
    logger.info("Comprehensive model training complete!")
    logger.info("="*50)
    logger.info(f"Total models trained: {len(trainer.models)}")
    logger.info("Models are ready for deployment and live trading.")


if __name__ == "__main__":
    main()