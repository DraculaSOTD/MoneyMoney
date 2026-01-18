from .unified_ml_pipeline import UnifiedMLPipeline
from .ppo_trainer import PPOTrainer, TradingActorCritic
from .config import (
    TrainingConfig,
    get_default_config,
    HYPERPARAMETER_SPACES,
    EPOCH_CONFIGS,
    HIDDEN_LAYER_CONFIGS,
    get_architecture_for_data_size,
    get_epoch_config,
    get_hidden_config,
)

# Data caching and HDF5 datasets for large-scale training
from .data_cache import (
    PreprocessedDataCache,
    MODEL_CACHE_CONFIG,
    get_data_cache,
)
from .hdf5_dataset import (
    HDF5SequenceDataset,
    HDF5FlatDataset,
    HDF5DataModule,
    HDF5SubsetDataset,
)
from .data_auditor import ProfileDataAuditor, DataAuditReport, DataGap
from .gap_filler import DataGapFiller, GapFillReport
from .data_validator import DataQualityValidator, DataQualityReport
from .evaluation import ModelEvaluator, EvaluationReport
from .training_logger import TrainingLogger, create_training_logger
from .walk_forward import WalkForwardValidator, WalkForwardFold, WalkForwardResult
from .hyperopt import HyperparameterOptimizer, OptimizationResult
from .model_comparison import ModelComparator, ComparisonResult
from .orchestrator import ModelTrainingOrchestrator, TrainingReport
from .drift_detector import ConceptDriftDetector, DriftReport

# Learning rate schedulers
from .lr_schedulers import (
    create_scheduler,
    is_epoch_based_scheduler,
    requires_metric,
    NoamScheduler,
    WarmupCosineScheduler,
    LinearDecayScheduler,
    SCHEDULER_CONFIGS,
)

# Data augmentation
from .augmentation import (
    TimeSeriesAugmentation,
    TorchAugmentation,
    AugmentationConfig,
    create_augmentation,
)

# Custom loss functions
from .loss_functions import (
    FocalLoss,
    LabelSmoothingLoss,
    WeightedCrossEntropyLoss,
    TradingLoss,
    SoftmaxFocalLoss,
    create_loss_function,
)

# GPU-enhanced training
from .gpu_enhanced_trainer import (
    GPUEnhancedTrainer,
    ModelAverager,
    CryptoDataset,
)

__all__ = [
    # Existing
    'UnifiedMLPipeline',
    'PPOTrainer',
    'TradingActorCritic',
    # Configuration
    'TrainingConfig',
    'get_default_config',
    'HYPERPARAMETER_SPACES',
    'EPOCH_CONFIGS',
    'HIDDEN_LAYER_CONFIGS',
    'get_architecture_for_data_size',
    'get_epoch_config',
    'get_hidden_config',
    # Data Caching (for large-scale training)
    'PreprocessedDataCache',
    'MODEL_CACHE_CONFIG',
    'get_data_cache',
    'HDF5SequenceDataset',
    'HDF5FlatDataset',
    'HDF5DataModule',
    'HDF5SubsetDataset',
    # Data Foundation
    'ProfileDataAuditor',
    'DataAuditReport',
    'DataGap',
    'DataGapFiller',
    'GapFillReport',
    'DataQualityValidator',
    'DataQualityReport',
    # Evaluation
    'ModelEvaluator',
    'EvaluationReport',
    # Logging
    'TrainingLogger',
    'create_training_logger',
    # Validation
    'WalkForwardValidator',
    'WalkForwardFold',
    'WalkForwardResult',
    # Hyperparameter Optimization
    'HyperparameterOptimizer',
    'OptimizationResult',
    # Model Comparison
    'ModelComparator',
    'ComparisonResult',
    # Orchestration
    'ModelTrainingOrchestrator',
    'TrainingReport',
    # Drift Detection
    'ConceptDriftDetector',
    'DriftReport',
    # Learning Rate Schedulers
    'create_scheduler',
    'is_epoch_based_scheduler',
    'requires_metric',
    'NoamScheduler',
    'WarmupCosineScheduler',
    'LinearDecayScheduler',
    'SCHEDULER_CONFIGS',
    # Data Augmentation
    'TimeSeriesAugmentation',
    'TorchAugmentation',
    'AugmentationConfig',
    'create_augmentation',
    # Loss Functions
    'FocalLoss',
    'LabelSmoothingLoss',
    'WeightedCrossEntropyLoss',
    'TradingLoss',
    'SoftmaxFocalLoss',
    'create_loss_function',
    # GPU-Enhanced Training
    'GPUEnhancedTrainer',
    'ModelAverager',
    'CryptoDataset',
]