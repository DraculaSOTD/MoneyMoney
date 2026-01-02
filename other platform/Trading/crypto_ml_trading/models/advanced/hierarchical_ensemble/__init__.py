"""
Hierarchical Ensemble Architecture for Cryptocurrency Trading.

This module implements a sophisticated hierarchical ensemble system that combines
predictions from multiple specialized models across different time horizons and
market conditions with advanced meta-learning capabilities.

Key Features:
- Multi-level ensemble hierarchy with specialized sub-ensembles
- Dynamic model weighting based on market conditions
- Meta-learning for ensemble optimization
- Cross-validation and model selection
- Real-time ensemble coordination
- Performance attribution and analysis
"""

from models.advanced.hierarchical_ensemble.base_ensemble import BaseEnsemble
from models.advanced.hierarchical_ensemble.meta_learner import MetaLearner
from models.advanced.hierarchical_ensemble.ensemble_coordinator import EnsembleCoordinator
from models.advanced.hierarchical_ensemble.model_selector import ModelSelector
from models.advanced.hierarchical_ensemble.ensemble_optimizer import EnsembleOptimizer

__all__ = [
    'BaseEnsemble',
    'MetaLearner', 
    'EnsembleCoordinator',
    'ModelSelector',
    'EnsembleOptimizer'
]