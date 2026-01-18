"""
Training Configuration
Centralized configuration for ML model training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    GRU_ATTENTION = "gru_attention"
    LSTM = "lstm"
    BILSTM = "bilstm"
    TRANSFORMER = "transformer"
    TCN = "tcn"
    CNN_PATTERN = "cnn_pattern"
    TFT = "tft"
    ARIMA = "arima"
    GARCH = "garch"
    HMM = "hmm"
    PPO = "ppo"
    DRQN = "drqn"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    PROPHET = "prophet"
    SENTIMENT = "sentiment"


@dataclass
class TrainingConfig:
    """Training configuration for ML models."""

    # Data requirements
    min_coverage_percent: float = 95.0
    min_candles: int = 100000  # ~70 days minimum

    # Split configuration
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_gap_hours: int = 24  # Gap between splits to prevent leakage

    # Training parameters
    default_batch_size: int = 32
    default_epochs: int = 100
    default_learning_rate: float = 0.001
    early_stopping_patience: int = 10
    gradient_clip_max: float = 5.0

    # GPU settings (RTX 30xx/40xx optimized)
    use_mixed_precision: bool = True
    cudnn_benchmark: bool = True
    gradient_checkpointing: bool = True

    # Hyperparameter optimization
    optuna_trials: int = 50
    optuna_timeout_hours: int = 4

    # Validation settings
    walk_forward_folds: int = 5
    min_train_size_per_fold: int = 50000

    # Model deployment thresholds
    min_test_accuracy: float = 0.55  # Minimum accuracy to consider deployment
    significance_level: float = 0.05  # p-value for significance testing

    # Model-specific batch sizes (GPU memory optimized)
    batch_sizes: Dict[str, int] = field(default_factory=lambda: {
        'gru_attention': 64,
        'lstm': 128,
        'bilstm': 128,
        'transformer': 32,
        'tcn': 64,
        'cnn_pattern': 128,
        'tft': 32,
        'lightgbm': 4096,
        'random_forest': 0,  # Not batch-based
    })

    # Model-specific sequence lengths
    sequence_lengths: Dict[str, int] = field(default_factory=lambda: {
        'gru_attention': 60,
        'lstm': 60,
        'bilstm': 60,
        'transformer': 100,
        'tcn': 120,
        'cnn_pattern': 30,
        'tft': 168,  # 7 days for hourly
    })

    def get_batch_size(self, model_type: str) -> int:
        """Get batch size for a specific model type."""
        return self.batch_sizes.get(model_type, self.default_batch_size)

    def get_sequence_length(self, model_type: str) -> int:
        """Get sequence length for a specific model type."""
        return self.sequence_lengths.get(model_type, 60)


# Hyperparameter search spaces per model
HYPERPARAMETER_SPACES: Dict[str, Dict[str, Any]] = {
    'gru_attention': {
        'hidden_sizes': [[128, 64], [256, 128], [256, 128, 64], [512, 256, 128]],
        'num_attention_heads': [2, 4, 8],
        'dropout_rates': [0.1, 0.2, 0.3],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'sequence_length': [30, 60, 100],
        'l2_lambda': [1e-5, 1e-4, 1e-3],
        'bidirectional': [True, False],
        'batch_size': [32, 64, 128],
    },
    'lstm': {
        'hidden_sizes': [[128, 64], [256, 128], [512, 256]],
        'dropout_rates': [0.1, 0.2, 0.3],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'sequence_length': [30, 60, 100],
        'bidirectional': [False],
        'batch_size': [64, 128, 256],
    },
    'bilstm': {
        'hidden_sizes': [[128, 64], [256, 128], [512, 256]],
        'dropout_rates': [0.1, 0.2, 0.3],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'sequence_length': [30, 60, 100],
        'bidirectional': [True],
        'batch_size': [64, 128, 256],
    },
    'transformer': {
        'd_model': [128, 256, 512],
        'n_heads': [4, 8],
        'n_encoder_layers': [2, 4, 6],
        'd_ff': [512, 1024, 2048],
        'dropout': [0.1, 0.2],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'warmup_steps': [2000, 4000, 8000],
        'batch_size': [16, 32, 64],
    },
    'tcn': {
        'num_channels': [[64, 64, 64], [128, 128, 128], [64, 128, 256]],
        'kernel_sizes': [2, 3, 4],
        'dilation_base': [2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [32, 64, 128],
    },
    'cnn_pattern': {
        'conv_filters': [[32, 64], [64, 128], [32, 64, 128]],
        'kernel_sizes': [(3, 3), (5, 5)],
        'pool_sizes': [(2, 2), (3, 3)],
        'dense_units': [128, 256],
        'dropout': [0.3, 0.4, 0.5],
        'learning_rate': [1e-4, 5e-4],
        'batch_size': [64, 128, 256],
    },
    'tft': {
        'hidden_size': [128, 256, 512],
        'num_attention_heads': [4, 8],
        'dropout': [0.1, 0.2],
        'num_encoder_steps': [30, 60, 100],
        'num_decoder_steps': [1, 5, 10],
        'learning_rate': [1e-4, 5e-4],
        'batch_size': [16, 32],
    },
    'lightgbm': {
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 8, 16],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    },
    'random_forest': {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
    },
    'ppo': {
        'hidden_size': [128, 256, 512],
        'gamma': [0.95, 0.97, 0.99],
        'clip_param': [0.1, 0.2, 0.3],
        'entropy_coef': [0.001, 0.01, 0.05],
        'gae_lambda': [0.9, 0.95, 0.97],
        'n_epochs': [3, 5, 10],
        'batch_size': [64, 128, 256],
    },
}


# Epoch configuration per model type
EPOCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    'gru_attention': {
        'min_epochs': 50,
        'max_epochs': 300,
        'default_epochs': 150,
        'early_stopping_patience': 20,
        'lr_patience': 10,  # Reduce LR after N epochs no improvement
    },
    'transformer': {
        'min_epochs': 100,
        'max_epochs': 500,
        'default_epochs': 200,
        'early_stopping_patience': 30,  # More patience due to warmup
        'warmup_epochs': 10,
    },
    'lstm': {
        'min_epochs': 50,
        'max_epochs': 200,
        'default_epochs': 100,
        'early_stopping_patience': 15,
    },
    'bilstm': {
        'min_epochs': 50,
        'max_epochs': 200,
        'default_epochs': 100,
        'early_stopping_patience': 15,
    },
    'tcn': {
        'min_epochs': 50,
        'max_epochs': 150,
        'default_epochs': 100,
        'early_stopping_patience': 15,
    },
    'cnn_pattern': {
        'min_epochs': 50,
        'max_epochs': 200,
        'default_epochs': 100,
        'early_stopping_patience': 20,
    },
    'tft': {
        'min_epochs': 100,
        'max_epochs': 300,
        'default_epochs': 150,
        'early_stopping_patience': 25,
    },
    'ppo': {
        'min_episodes': 1000,
        'max_episodes': 10000,
        'default_episodes': 5000,
        'eval_interval': 100,
    },
    'lightgbm': {
        'min_estimators': 500,
        'max_estimators': 5000,
        'default_estimators': 2000,
        'early_stopping_rounds': 50,
    },
    'random_forest': {
        'min_estimators': 100,
        'max_estimators': 1000,
        'default_estimators': 500,
    },
}


# Hidden layer configurations per model (small/medium/large/deep/wide presets)
HIDDEN_LAYER_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'gru_attention': {
        'small': {'hidden_dim': 64, 'n_layers': 2, 'n_heads': 4, 'dropout': 0.2},
        'medium': {'hidden_dim': 128, 'n_layers': 3, 'n_heads': 4, 'dropout': 0.2},
        'large': {'hidden_dim': 256, 'n_layers': 3, 'n_heads': 8, 'dropout': 0.25},
        'deep': {'hidden_dim': 128, 'n_layers': 4, 'n_heads': 4, 'dropout': 0.3},
        'wide': {'hidden_dim': 512, 'n_layers': 2, 'n_heads': 8, 'dropout': 0.3},
    },
    'lstm': {
        'small': {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2},
        'medium': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
        'large': {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.25},
        'deep': {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3},
        'very_large': {'hidden_size': 512, 'num_layers': 2, 'dropout': 0.3},
    },
    'bilstm': {
        'small': {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2},
        'medium': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
        'large': {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.25},
        'deep': {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3},
    },
    'transformer': {
        'tiny': {'d_model': 64, 'n_heads': 2, 'n_layers': 2, 'd_ff': 256, 'dropout': 0.1},
        'small': {'d_model': 128, 'n_heads': 4, 'n_layers': 3, 'd_ff': 512, 'dropout': 0.1},
        'medium': {'d_model': 256, 'n_heads': 8, 'n_layers': 4, 'd_ff': 1024, 'dropout': 0.1},
        'large': {'d_model': 512, 'n_heads': 8, 'n_layers': 6, 'd_ff': 2048, 'dropout': 0.1},
        'base': {'d_model': 768, 'n_heads': 12, 'n_layers': 6, 'd_ff': 3072, 'dropout': 0.1},
    },
    'tcn': {
        'small': {'channels': [32, 64, 128], 'kernel_size': 2, 'dropout': 0.2},
        'medium': {'channels': [64, 128, 256], 'kernel_size': 3, 'dropout': 0.2},
        'large': {'channels': [64, 128, 256, 512], 'kernel_size': 3, 'dropout': 0.25},
        'wide': {'channels': [128, 256, 512], 'kernel_size': 3, 'dropout': 0.25},
        'deep': {'channels': [64, 64, 128, 128, 256, 256], 'kernel_size': 2, 'dropout': 0.3},
    },
    'cnn_pattern': {
        'small': {'base_filters': 16, 'n_blocks': 3, 'dropout': 0.3},
        'medium': {'base_filters': 32, 'n_blocks': 4, 'dropout': 0.3},
        'large': {'base_filters': 64, 'n_blocks': 4, 'dropout': 0.35},
        'very_large': {'base_filters': 64, 'n_blocks': 5, 'dropout': 0.4},
    },
    'ppo': {
        'small': {'hidden_size': 128, 'n_layers': 2},
        'medium': {'hidden_size': 256, 'n_layers': 2},
        'large': {'hidden_size': 512, 'n_layers': 3},
    },
}


def get_architecture_for_data_size(model_type: str, n_samples: int) -> Dict[str, Any]:
    """
    Select architecture based on dataset size.
    More data allows for larger model capacity.

    Args:
        model_type: Model type identifier
        n_samples: Number of samples in dataset

    Returns:
        Dictionary of architecture parameters
    """
    epoch_config = EPOCH_CONFIGS.get(model_type, {})

    if model_type == 'gru_attention':
        if n_samples < 100000:
            config = {'hidden_dim': 64, 'n_layers': 2, 'n_heads': 4}
            epochs = 100
        elif n_samples < 300000:
            config = {'hidden_dim': 128, 'n_layers': 3, 'n_heads': 4}
            epochs = 150
        elif n_samples < 500000:
            config = {'hidden_dim': 256, 'n_layers': 3, 'n_heads': 8}
            epochs = 200
        else:
            config = {'hidden_dim': 512, 'n_layers': 4, 'n_heads': 8}
            epochs = 300
        config['epochs'] = epochs
        config['dropout'] = 0.1 + (config['hidden_dim'] / 1000)
        return config

    elif model_type == 'transformer':
        if n_samples < 100000:
            config = {'d_model': 128, 'n_layers': 2, 'd_ff': 512, 'n_heads': 4}
            epochs = 100
        elif n_samples < 300000:
            config = {'d_model': 256, 'n_layers': 4, 'd_ff': 1024, 'n_heads': 8}
            epochs = 200
        else:
            config = {'d_model': 512, 'n_layers': 6, 'd_ff': 2048, 'n_heads': 8}
            epochs = 300
        config['epochs'] = epochs
        config['warmup_steps'] = min(4000, n_samples // 10)
        return config

    elif model_type in ('lstm', 'bilstm'):
        if n_samples < 100000:
            config = {'hidden_size': 64, 'num_layers': 1}
            epochs = 80
        elif n_samples < 300000:
            config = {'hidden_size': 128, 'num_layers': 2}
            epochs = 100
        else:
            config = {'hidden_size': 256, 'num_layers': 3}
            epochs = 150
        config['epochs'] = epochs
        config['bidirectional'] = model_type == 'bilstm'
        return config

    elif model_type == 'tcn':
        if n_samples < 100000:
            config = {'channels': [32, 64, 128], 'kernel_size': 2}
            epochs = 80
        elif n_samples < 300000:
            config = {'channels': [64, 128, 256], 'kernel_size': 3}
            epochs = 100
        else:
            config = {'channels': [64, 128, 256, 512], 'kernel_size': 3}
            epochs = 150
        config['epochs'] = epochs
        return config

    elif model_type == 'cnn_pattern':
        if n_samples < 100000:
            config = {'base_filters': 16, 'n_blocks': 3}
            epochs = 80
        elif n_samples < 300000:
            config = {'base_filters': 32, 'n_blocks': 4}
            epochs = 100
        else:
            config = {'base_filters': 64, 'n_blocks': 4}
            epochs = 150
        config['epochs'] = epochs
        return config

    elif model_type == 'lightgbm':
        if n_samples < 100000:
            return {'n_estimators': 1000, 'max_depth': 7, 'num_leaves': 63}
        elif n_samples < 300000:
            return {'n_estimators': 2000, 'max_depth': 10, 'num_leaves': 127}
        else:
            return {'n_estimators': 3000, 'max_depth': 15, 'num_leaves': 255}

    elif model_type == 'random_forest':
        if n_samples < 100000:
            return {'n_estimators': 200, 'max_depth': 15}
        elif n_samples < 300000:
            return {'n_estimators': 500, 'max_depth': 20}
        else:
            return {'n_estimators': 1000, 'max_depth': 30}

    # Default fallback
    return {'epochs': epoch_config.get('default_epochs', 100)}


def get_epoch_config(model_type: str) -> Dict[str, Any]:
    """
    Get epoch configuration for a specific model type.

    Args:
        model_type: Model type identifier

    Returns:
        Dictionary with epoch-related configuration
    """
    return EPOCH_CONFIGS.get(model_type, {
        'min_epochs': 50,
        'max_epochs': 200,
        'default_epochs': 100,
        'early_stopping_patience': 15,
    })


def get_hidden_config(model_type: str, size: str = 'medium') -> Dict[str, Any]:
    """
    Get hidden layer configuration for a specific model type and size.

    Args:
        model_type: Model type identifier
        size: Size preset (small, medium, large, deep, wide)

    Returns:
        Dictionary with hidden layer configuration
    """
    configs = HIDDEN_LAYER_CONFIGS.get(model_type, {})
    return configs.get(size, configs.get('medium', {}))


# Feature groups for different model types
FEATURE_GROUPS: Dict[str, List[str]] = {
    'price': [
        'open_price', 'high_price', 'low_price', 'close_price',
        'close_pct', 'log_returns', 'range_pct', 'body_pct'
    ],
    'volume': [
        'volume', 'quote_asset_volume', 'volume_pct_change',
        'relative_volume', 'obv', 'vwap'
    ],
    'moving_averages': [
        'sma_10', 'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26', 'ema_50'
    ],
    'momentum': [
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi', 'momentum'
    ],
    'volatility': [
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr_14', 'kc_upper', 'kc_middle', 'kc_lower',
        'parkinson_vol', 'gk_volatility'
    ],
    'trend': [
        'adx', 'plus_di', 'minus_di', 'parabolic_sar'
    ],
    'support_resistance': [
        'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3'
    ],
    'microstructure': [
        'bid_ask_spread', 'trade_intensity', 'order_imbalance'
    ],
}


# Model-specific feature requirements
MODEL_FEATURE_REQUIREMENTS: Dict[str, List[str]] = {
    'gru_attention': ['price', 'moving_averages', 'momentum', 'volatility'],
    'lstm': ['price', 'moving_averages', 'momentum', 'volatility'],
    'bilstm': ['price', 'moving_averages', 'momentum', 'volatility'],
    'transformer': ['price', 'moving_averages', 'momentum', 'volatility', 'volume'],
    'tcn': ['price', 'moving_averages', 'momentum'],
    'cnn_pattern': ['price', 'volume'],
    'tft': ['price', 'moving_averages', 'momentum', 'volatility', 'trend'],
    'lightgbm': ['price', 'moving_averages', 'momentum', 'volatility', 'volume', 'trend'],
    'random_forest': ['price', 'moving_averages', 'momentum', 'volatility', 'volume'],
}


def get_features_for_model(model_type: str) -> List[str]:
    """
    Get the list of feature names required for a specific model type.

    Args:
        model_type: Model type identifier

    Returns:
        List of feature column names
    """
    feature_groups = MODEL_FEATURE_REQUIREMENTS.get(
        model_type,
        ['price', 'moving_averages', 'momentum']  # Default
    )

    features = []
    for group in feature_groups:
        if group in FEATURE_GROUPS:
            features.extend(FEATURE_GROUPS[group])

    return features


def get_hyperparameter_space(model_type: str) -> Dict[str, Any]:
    """
    Get the hyperparameter search space for a specific model type.

    Args:
        model_type: Model type identifier

    Returns:
        Dictionary of hyperparameter names to possible values
    """
    return HYPERPARAMETER_SPACES.get(model_type, {})


@dataclass
class GPUConfig:
    """GPU-specific configuration."""
    device: str = "cuda"
    use_cuda: bool = True
    use_mixed_precision: bool = True
    cudnn_benchmark: bool = True
    memory_fraction: float = 0.95  # Use 95% of GPU memory
    num_workers: int = 4  # DataLoader workers

    @property
    def device_string(self) -> str:
        """Get PyTorch device string."""
        if self.use_cuda:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_interval: str = "1m"
    lookback_days: int = 365
    min_data_points: int = 100000
    max_data_points: int = 525600  # 1 year of 1-minute data

    # Feature engineering
    calculate_indicators: bool = True
    calculate_patterns: bool = True
    calculate_microstructure: bool = False

    # Label configuration
    prediction_horizon: int = 60  # minutes
    label_threshold: float = 0.001  # 0.1% for binary classification


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_database: bool = True
    log_dir: str = "logs/training"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # Save every N epochs


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_gpu_config() -> GPUConfig:
    """Get GPU configuration."""
    return GPUConfig()


def get_data_config() -> DataConfig:
    """Get data configuration."""
    return DataConfig()


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return LoggingConfig()
