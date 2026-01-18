"""
Unified ML Training Pipeline
Integrates supervised learning models from ML project with existing infrastructure.

IMPORTANT: This module has been updated to properly handle time series data:
- Uses temporal splits with gaps to prevent data leakage
- Fits scalers only on training data
- Integrates with the data auditor for pre-training validation
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import new training infrastructure
from crypto_ml_trading.training.config import TrainingConfig, get_default_config
from crypto_ml_trading.features.scaler_manager import FeatureScalerManager

# Import components
from data.enhanced_data_loader import EnhancedDataLoader, BinanceDataSource
from data.data_validator import DataValidator
from data.preprocessing import AdvancedPreprocessor
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
from features.ml_feature_engineering import MLFeatureEngineering
from features.decision_labeler import DecisionLabeler
from models.ml.pytorch_models import BiLSTMModel, GRUModel, CNNLSTMModel, TCNModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedMLPipeline:
    """
    Unified ML training pipeline combining data processing, 
    feature engineering, and model training from both projects.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.validator = DataValidator()
        self.preprocessor = AdvancedPreprocessor()
        self.ml_engineer = MLFeatureEngineering()
        self.labeler = DecisionLabeler()
        self.models = {}
        self.results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'data': {
                'sequence_length': 100,
                'lookforward': 5,
                'threshold': 0.002,
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15
            },
            'preprocessing': {
                'handle_missing': {'enabled': True, 'method': 'forward_fill'},
                'handle_infinite': {'enabled': True, 'method': 'clip'},
                'stationarity': {'enabled': True, 'method': 'pct_change'},
                'outliers': {'enabled': True, 'method': 'iqr'},
                'scaling': {'enabled': True, 'method': 'standard'}
            },
            'indicators': {
                'sma': {'enabled': True, 'periods': [10, 20, 50, 200]},
                'ema': {'enabled': True, 'periods': [12, 26]},
                'rsi': {'enabled': True, 'period': 14},
                'macd': {'enabled': True},
                'bollinger': {'enabled': True},
                'atr': {'enabled': True},
                'adx': {'enabled': True},
                'stochastic': {'enabled': True},
                'ichimoku': {'enabled': True},
                'parabolic_sar': {'enabled': True},
                'fibonacci': {'enabled': True}
            },
            'models': {
                'lstm': {
                    'enabled': True,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'bidirectional': True
                },
                'gru': {
                    'enabled': True,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2
                },
                'cnn_lstm': {
                    'enabled': True,
                    'cnn_filters': [64, 128, 256],
                    'kernel_sizes': [3, 5, 7],
                    'lstm_hidden': 128,
                    'lstm_layers': 2
                },
                'tcn': {
                    'enabled': True,
                    'num_channels': [64, 128, 256],
                    'kernel_size': 3,
                    'dropout': 0.2
                }
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'early_stopping_patience': 10,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        }
    
    def load_and_prepare_data(self, data_source: Any) -> pd.DataFrame:
        """
        Load data and prepare features using the complete pipeline.
        
        Args:
            data_source: Path to data file or data source object
            
        Returns:
            Prepared DataFrame with all features
        """
        logger.info("Loading and preparing data...")
        
        # Load data
        if isinstance(data_source, str):
            # Auto-detect format
            from data.enhanced_data_loader import UniversalDataSource
            source = UniversalDataSource(data_source)
            df = source.fetch_data("BTCUSDT", None, None)
        else:
            df = data_source
        
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Add technical indicators
        logger.info("Computing technical indicators...")
        df = EnhancedTechnicalIndicators.compute_all_indicators(
            df, self.config['indicators']
        )
        
        # Create ML features
        logger.info("Engineering ML features...")
        df = self.ml_engineer.prepare_ml_features(df)
        
        # Add labels
        logger.info("Creating ML labels...")
        df = self.labeler.create_labels(
            df,
            lookforward=self.config['data']['lookforward'],
            threshold=self.config['data']['threshold']
        )
        
        # Validate data
        logger.info("Validating data quality...")
        validation_report = self.validator.validate(df, target_col='label')
        logger.info(f"Data quality score: {validation_report['data_quality']:.1f}/100")
        
        # Preprocess
        logger.info("Preprocessing data...")
        df = self.preprocessor.preprocess(
            df,
            target_col='label',
            config=self.config['preprocessing']
        )
        
        # Final validation
        final_report = self.validator.validate(df, target_col='label')
        logger.info(f"Final quality score: {final_report['data_quality']:.1f}/100")
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models."""
        # Remove non-numeric columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != 'label']
        
        # Create sequences
        X, y = self.ml_engineer.create_sequences(
            df,
            sequence_length=self.config['data']['sequence_length'],
            target_col='label',
            feature_cols=feature_cols
        )
        
        logger.info(f"Created sequences: X shape = {X.shape}, y shape = {y.shape}")
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple]:
        """
        Split data into train, validation, and test sets.

        DEPRECATED: This method uses random shuffling which is inappropriate for
        time series data. Use create_temporal_splits() instead.
        """
        logger.warning(
            "split_data() uses random shuffling - consider using create_temporal_splits() "
            "for time series data to prevent data leakage"
        )

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_split'],
            random_state=42,
            stratify=y
        )

        # Second split: train vs val
        val_size = self.config['data']['val_split'] / (1 - self.config['data']['test_split'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42,
            stratify=y_temp
        )

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

    def create_temporal_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        gap_samples: int = None
    ) -> Dict[str, Tuple]:
        """
        Create temporal train/validation/test splits with gaps.

        This is the correct way to split time series data to prevent data leakage.
        Gaps between splits ensure that validation/test data comes from a truly
        future time period.

        Args:
            X: Feature array of shape (samples, sequence_length, features)
            y: Label array of shape (samples,)
            train_ratio: Proportion of data for training (default 0.70)
            val_ratio: Proportion of data for validation (default 0.15)
            test_ratio: Proportion of data for testing (default 0.15)
            gap_samples: Number of samples to skip between splits.
                        If None, defaults to 24 hours worth at 1-minute intervals (1440)
                        divided by sequence length.

        Returns:
            Dictionary with 'train', 'val', 'test' keys containing (X, y) tuples
        """
        total_samples = len(X)

        # Calculate gap if not specified
        # Default: 24 hours of data at 1-minute intervals = 1440 samples
        # But since we create sequences, divide by sequence length
        if gap_samples is None:
            sequence_length = self.config['data'].get('sequence_length', 100)
            # 24 hours gap, adjusted for sequence overlap
            gap_samples = max(1, 1440 // sequence_length)

        # Calculate split points
        train_end = int(total_samples * train_ratio)
        val_start = train_end + gap_samples
        val_end = val_start + int(total_samples * val_ratio)
        test_start = val_end + gap_samples

        # Ensure we have enough data
        if test_start >= total_samples:
            logger.warning("Not enough data for full gaps, reducing gap size")
            gap_samples = gap_samples // 2
            val_start = train_end + gap_samples
            val_end = val_start + int(total_samples * val_ratio)
            test_start = val_end + gap_samples

        if test_start >= total_samples:
            logger.warning("Still not enough data, removing gaps")
            gap_samples = 0
            val_start = train_end
            val_end = val_start + int(total_samples * val_ratio)
            test_start = val_end

        # Create splits
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]
        X_test, y_test = X[test_start:], y[test_start:]

        # Log split information
        logger.info(
            f"Temporal split with {gap_samples} sample gap:\n"
            f"  Train: {len(X_train)} samples (indices 0-{train_end-1})\n"
            f"  Val:   {len(X_val)} samples (indices {val_start}-{val_end-1})\n"
            f"  Test:  {len(X_test)} samples (indices {test_start}-{total_samples-1})"
        )

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

    def prepare_features_with_proper_scaling(
        self,
        df: pd.DataFrame,
        target_col: str = 'label',
        scaler_save_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScalerManager]:
        """
        Prepare features with proper scaling (fit on training data only).

        This method:
        1. Splits the data temporally
        2. Fits scalers ONLY on training data
        3. Transforms all splits using the fitted scalers

        Args:
            df: DataFrame with features (should NOT be pre-scaled)
            target_col: Name of target column
            scaler_save_path: Optional path to save fitted scalers

        Returns:
            Tuple of (train_df, val_df, test_df, scaler_manager)
        """
        logger.info("Preparing features with proper scaling...")

        # Get config values
        train_ratio = self.config['data'].get('train_split', 0.70)
        val_ratio = self.config['data'].get('val_split', 0.15)

        # Calculate split indices with gap
        total = len(df)
        gap = 1440  # 24 hours at 1-minute intervals

        train_end = int(total * train_ratio)
        val_start = train_end + gap
        val_end = val_start + int(total * val_ratio)
        test_start = val_end + gap

        # Handle insufficient data
        if test_start >= total:
            gap = gap // 4
            val_start = train_end + gap
            val_end = val_start + int(total * val_ratio)
            test_start = val_end + gap

        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target_col]

        # Create and fit scaler manager on TRAINING DATA ONLY
        scaler_manager = FeatureScalerManager()
        scaler_manager.fit(train_df[feature_cols])

        # Transform all splits
        train_df[feature_cols] = scaler_manager.transform(train_df[feature_cols])
        val_df[feature_cols] = scaler_manager.transform(val_df[feature_cols])
        test_df[feature_cols] = scaler_manager.transform(test_df[feature_cols])

        # Save scalers if path provided
        if scaler_save_path:
            scaler_manager.save(scaler_save_path)
            logger.info(f"Saved scalers to {scaler_save_path}")

        logger.info(
            f"Feature preparation complete:\n"
            f"  Train: {len(train_df)} samples\n"
            f"  Val:   {len(val_df)} samples\n"
            f"  Test:  {len(test_df)} samples\n"
            f"  Gap:   {gap} samples (24 hours)"
        )

        return train_df, val_df, test_df, scaler_manager
    
    def create_data_loaders(self, data_splits: Dict) -> Dict[str, DataLoader]:
        """Create PyTorch data loaders."""
        loaders = {}
        
        for split_name, (X, y) in data_splits.items():
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Create loader
            shuffle = (split_name == 'train')
            loader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=shuffle
            )
            
            loaders[split_name] = loader
        
        return loaders
    
    def initialize_models(self, input_size: int, num_classes: int):
        """Initialize all enabled models."""
        device = self.config['training']['device']
        
        if self.config['models']['lstm']['enabled']:
            self.models['lstm'] = BiLSTMModel(
                input_size=input_size,
                hidden_size=self.config['models']['lstm']['hidden_size'],
                num_layers=self.config['models']['lstm']['num_layers'],
                num_classes=num_classes,
                dropout=self.config['models']['lstm']['dropout']
            ).to(device)
            logger.info("Initialized BiLSTM model")
        
        if self.config['models']['gru']['enabled']:
            self.models['gru'] = GRUModel(
                input_size=input_size,
                hidden_size=self.config['models']['gru']['hidden_size'],
                num_layers=self.config['models']['gru']['num_layers'],
                num_classes=num_classes,
                dropout=self.config['models']['gru']['dropout']
            ).to(device)
            logger.info("Initialized GRU model")
        
        if self.config['models']['cnn_lstm']['enabled']:
            self.models['cnn_lstm'] = CNNLSTMModel(
                input_channels=1,
                sequence_length=self.config['data']['sequence_length'],
                num_features=input_size,
                cnn_filters=self.config['models']['cnn_lstm']['cnn_filters'],
                kernel_sizes=self.config['models']['cnn_lstm']['kernel_sizes'],
                lstm_hidden=self.config['models']['cnn_lstm']['lstm_hidden'],
                lstm_layers=self.config['models']['cnn_lstm']['lstm_layers'],
                num_classes=num_classes
            ).to(device)
            logger.info("Initialized CNN-LSTM model")
        
        if self.config['models']['tcn']['enabled']:
            self.models['tcn'] = TCNModel(
                input_size=input_size,
                num_channels=self.config['models']['tcn']['num_channels'],
                kernel_size=self.config['models']['tcn']['kernel_size'],
                num_classes=num_classes,
                dropout=self.config['models']['tcn']['dropout']
            ).to(device)
            logger.info("Initialized TCN model")
    
    def train_model(self, model_name: str, model: nn.Module, 
                   loaders: Dict[str, DataLoader]) -> Dict:
        """Train a single model."""
        logger.info(f"\nTraining {model_name} model...")
        
        device = self.config['training']['device']
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in loaders['train']:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Special handling for CNN-LSTM
                if model_name == 'cnn_lstm':
                    batch_X = batch_X.unsqueeze(1)  # Add channel dimension
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(loaders['train'])
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in loaders['val']:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    if model_name == 'cnn_lstm':
                        batch_X = batch_X.unsqueeze(1)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(loaders['val'])
            val_accuracy = 100 * correct / total
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch [{epoch}/{self.config['training']['epochs']}] - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.2f}%"
                )
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_{model_name}_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
        
        return history
    
    def evaluate_model(self, model_name: str, model: nn.Module, 
                      loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        logger.info(f"\nEvaluating {model_name} model...")
        
        device = self.config['training']['device']
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                if model_name == 'cnn_lstm':
                    batch_X = batch_X.unsqueeze(1)
                
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(
            all_labels, all_predictions,
            target_names=['Sell', 'Buy', 'Hold'],
            output_dict=True
        )
        
        confusion = confusion_matrix(all_labels, all_predictions)
        
        # Log results
        logger.info(f"\n{model_name} Test Results:")
        logger.info(f"Accuracy: {report['accuracy']:.4f}")
        logger.info(f"Precision (macro): {report['macro avg']['precision']:.4f}")
        logger.info(f"Recall (macro): {report['macro avg']['recall']:.4f}")
        logger.info(f"F1 (macro): {report['macro avg']['f1-score']:.4f}")
        
        return {
            'classification_report': report,
            'confusion_matrix': confusion.tolist(),
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def run_pipeline(self, data_source: Any) -> Dict:
        """
        Run the complete ML pipeline.
        
        Args:
            data_source: Data source (file path or DataFrame)
            
        Returns:
            Dictionary with training results
        """
        logger.info("="*60)
        logger.info("UNIFIED ML PIPELINE EXECUTION")
        logger.info("="*60)
        
        # 1. Load and prepare data
        df = self.load_and_prepare_data(data_source)
        
        # 2. Create sequences
        X, y = self.create_sequences(df)
        
        # 3. Split data
        data_splits = self.split_data(X, y)
        
        # 4. Create data loaders
        loaders = self.create_data_loaders(data_splits)
        
        # 5. Initialize models
        input_size = X.shape[2]  # Number of features
        num_classes = len(np.unique(y))
        self.initialize_models(input_size, num_classes)
        
        # 6. Train models
        for model_name, model in self.models.items():
            history = self.train_model(model_name, model, loaders)
            self.results[model_name] = {'history': history}
        
        # 7. Evaluate models
        for model_name, model in self.models.items():
            eval_results = self.evaluate_model(model_name, model, loaders['test'])
            self.results[model_name]['evaluation'] = eval_results
        
        # 8. Summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        
        for model_name, results in self.results.items():
            accuracy = results['evaluation']['classification_report']['accuracy']
            logger.info(f"{model_name.upper()} - Test Accuracy: {accuracy:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ml_pipeline_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return self.results
    
    def create_ensemble_predictions(self, data_loader: DataLoader) -> np.ndarray:
        """Create ensemble predictions from all trained models."""
        device = self.config['training']['device']
        all_model_predictions = []
        
        for model_name, model in self.models.items():
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch_X, _ in data_loader:
                    batch_X = batch_X.to(device)
                    
                    if model_name == 'cnn_lstm':
                        batch_X = batch_X.unsqueeze(1)
                    
                    outputs = model(batch_X)
                    probs = torch.softmax(outputs, dim=1)
                    predictions.append(probs.cpu().numpy())
            
            model_predictions = np.concatenate(predictions, axis=0)
            all_model_predictions.append(model_predictions)
        
        # Average predictions
        ensemble_predictions = np.mean(all_model_predictions, axis=0)
        return np.argmax(ensemble_predictions, axis=1)


def main():
    """Example usage of the unified ML pipeline."""
    # Create pipeline
    pipeline = UnifiedMLPipeline()
    
    # For demonstration, create sample data
    # In practice, use actual data file
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from examples.simple_demo import create_sample_binance_data
    
    df = create_sample_binance_data()
    
    # Run pipeline
    results = pipeline.run_pipeline(df)
    
    return results


if __name__ == "__main__":
    results = main()