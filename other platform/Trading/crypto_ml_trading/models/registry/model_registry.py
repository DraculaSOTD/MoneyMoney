"""
Model Registry System for tracking and managing all trading models.

Provides centralized registration, versioning, and metadata management
for all models in the trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import hashlib
import os
import pickle
import shutil
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    """Types of models in the system."""
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"
    SENTIMENT = "sentiment"
    ALTERNATIVE_DATA = "alternative_data"
    RISK_MANAGEMENT = "risk_management"


@dataclass
class ModelMetadata:
    """Comprehensive metadata for a registered model."""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    
    # Model specifics
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    
    # Data information
    training_data_hash: Optional[str] = None
    feature_names: List[str] = field(default_factory=list)
    target_type: str = "classification"
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    framework: str = "numpy"
    
    # Tags and categories
    tags: List[str] = field(default_factory=list)
    asset_classes: List[str] = field(default_factory=list)
    time_horizons: List[str] = field(default_factory=list)
    
    # Deployment info
    deployment_requirements: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage
    parent_model_id: Optional[str] = None
    experiment_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        # Convert enums
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value
        # Convert datetime
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        # Convert enums back
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        # Convert datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    version_id: str
    model_id: str
    version_number: str
    created_at: datetime
    commit_message: str
    
    # Changes from previous version
    changes: Dict[str, Any]
    performance_delta: Dict[str, float]
    
    # Storage paths
    model_path: str
    metadata_path: str
    
    # Validation
    is_validated: bool = False
    validation_results: Optional[Dict[str, Any]] = None
    
    # Approval
    is_approved: bool = False
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None


class ModelRegistry:
    """
    Central registry for all trading models.
    
    Features:
    - Model registration and tracking
    - Version management
    - Metadata storage
    - Model discovery
    - Lifecycle management
    - Performance tracking
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Base path for registry storage
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry components
        self.models_path = self.registry_path / "models"
        self.metadata_path = self.registry_path / "metadata"
        self.versions_path = self.registry_path / "versions"
        
        # Create directories
        self.models_path.mkdir(exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)
        self.versions_path.mkdir(exist_ok=True)
        
        # Load registry
        self.registry = self._load_registry()
        
        # Active models cache
        self.active_models = {}
        
        logger.info(f"Model registry initialized at {self.registry_path}")
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load existing registry from disk."""
        registry = {}
        registry_file = self.metadata_path / "registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
                
            for model_id, metadata_dict in registry_data.items():
                try:
                    registry[model_id] = ModelMetadata.from_dict(metadata_dict)
                except Exception as e:
                    logger.error(f"Failed to load model {model_id}: {e}")
        
        return registry
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_data = {
            model_id: metadata.to_dict() 
            for model_id, metadata in self.registry.items()
        }
        
        registry_file = self.metadata_path / "registry.json"
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def generate_model_id(self, model_name: str, model_type: ModelType) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_id = f"{model_type.value}_{model_name}_{timestamp}"
        
        # Add hash for uniqueness
        hash_input = f"{base_id}_{np.random.random()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"{base_id}_{hash_suffix}"
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: ModelType,
        author: str,
        description: str,
        architecture: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        training_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
        **kwargs
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: The model instance
            model_name: Name of the model
            model_type: Type of model
            author: Model author
            description: Model description
            architecture: Architecture details
            hyperparameters: Model hyperparameters
            training_config: Training configuration
            performance_metrics: Performance metrics
            validation_metrics: Validation metrics
            **kwargs: Additional metadata
            
        Returns:
            Model ID
        """
        # Generate model ID
        model_id = self.generate_model_id(model_name, model_type)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version="1.0.0",
            status=ModelStatus.DEVELOPMENT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            author=author,
            description=description,
            architecture=architecture,
            hyperparameters=hyperparameters,
            training_config=training_config,
            performance_metrics=performance_metrics,
            validation_metrics=validation_metrics,
            **kwargs
        )
        
        # Save model
        model_path = self.models_path / f"{model_id}.pkl"
        self._save_model(model, model_path)
        
        # Save metadata
        metadata_path = self.metadata_path / f"{model_id}_metadata.json"
        self._save_metadata(metadata, metadata_path)
        
        # Update registry
        self.registry[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model {model_name} with ID {model_id}")
        
        return model_id
    
    def _save_model(self, model: Any, path: Path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    def _save_metadata(self, metadata: ModelMetadata, path: Path):
        """Save metadata to disk."""
        with open(path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Load a model and its metadata.
        
        Args:
            model_id: Model ID
            
        Returns:
            Tuple of (model, metadata)
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.registry[model_id]
        model_path = self.models_path / f"{model_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, metadata
    
    def update_model_status(self, model_id: str, new_status: ModelStatus, 
                          approved_by: Optional[str] = None):
        """Update model status in registry."""
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.registry[model_id]
        old_status = metadata.status
        
        # Validate status transition
        valid_transitions = {
            ModelStatus.DEVELOPMENT: [ModelStatus.STAGING, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.DEVELOPMENT, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.STAGING, ModelStatus.DEPRECATED],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED],
            ModelStatus.ARCHIVED: []
        }
        
        if new_status not in valid_transitions.get(old_status, []):
            raise ValueError(f"Invalid status transition: {old_status} -> {new_status}")
        
        # Update status
        metadata.status = new_status
        metadata.updated_at = datetime.now()
        
        # Log status change
        logger.info(f"Model {model_id} status changed from {old_status} to {new_status}")
        
        # Save updates
        self._save_registry()
        metadata_path = self.metadata_path / f"{model_id}_metadata.json"
        self._save_metadata(metadata, metadata_path)
    
    def search_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
        asset_classes: Optional[List[str]] = None,
        min_performance: Optional[Dict[str, float]] = None
    ) -> List[ModelMetadata]:
        """
        Search for models based on criteria.
        
        Args:
            model_type: Filter by model type
            status: Filter by status
            tags: Filter by tags
            asset_classes: Filter by asset classes
            min_performance: Minimum performance requirements
            
        Returns:
            List of matching models
        """
        results = []
        
        for model_id, metadata in self.registry.items():
            # Type filter
            if model_type and metadata.model_type != model_type:
                continue
            
            # Status filter
            if status and metadata.status != status:
                continue
            
            # Tags filter
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            # Asset classes filter
            if asset_classes and not any(
                asset in metadata.asset_classes for asset in asset_classes
            ):
                continue
            
            # Performance filter
            if min_performance:
                meets_requirements = True
                for metric, min_value in min_performance.items():
                    if metadata.performance_metrics.get(metric, 0) < min_value:
                        meets_requirements = False
                        break
                if not meets_requirements:
                    continue
            
            results.append(metadata)
        
        # Sort by performance (example: by sharpe_ratio)
        results.sort(
            key=lambda m: m.performance_metrics.get('sharpe_ratio', 0),
            reverse=True
        )
        
        return results
    
    def get_production_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """Get all production models."""
        return self.search_models(model_type=model_type, status=ModelStatus.PRODUCTION)
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for model_id in model_ids:
            if model_id not in self.registry:
                continue
            
            metadata = self.registry[model_id]
            row = {
                'model_id': model_id,
                'model_name': metadata.model_name,
                'model_type': metadata.model_type.value,
                'status': metadata.status.value,
                'version': metadata.version,
                'created_at': metadata.created_at,
                **metadata.performance_metrics,
                **{f"val_{k}": v for k, v in metadata.validation_metrics.items()}
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_model_lineage(self, model_id: str) -> List[ModelMetadata]:
        """Get model lineage (parent models)."""
        lineage = []
        current_id = model_id
        
        while current_id and current_id in self.registry:
            metadata = self.registry[current_id]
            lineage.append(metadata)
            current_id = metadata.parent_model_id
        
        return lineage
    
    def export_registry_report(self, output_path: str):
        """Export comprehensive registry report."""
        report = {
            'registry_info': {
                'total_models': len(self.registry),
                'last_updated': datetime.now().isoformat(),
                'registry_path': str(self.registry_path)
            },
            'models_by_type': {},
            'models_by_status': {},
            'top_performers': {},
            'recent_models': []
        }
        
        # Models by type
        for model_type in ModelType:
            count = sum(1 for m in self.registry.values() if m.model_type == model_type)
            report['models_by_type'][model_type.value] = count
        
        # Models by status
        for status in ModelStatus:
            count = sum(1 for m in self.registry.values() if m.status == status)
            report['models_by_status'][status.value] = count
        
        # Top performers by metric
        metrics = ['sharpe_ratio', 'accuracy', 'roi']
        for metric in metrics:
            sorted_models = sorted(
                self.registry.values(),
                key=lambda m: m.performance_metrics.get(metric, 0),
                reverse=True
            )[:5]
            
            report['top_performers'][metric] = [
                {
                    'model_id': m.model_id,
                    'model_name': m.model_name,
                    'value': m.performance_metrics.get(metric, 0)
                }
                for m in sorted_models
            ]
        
        # Recent models
        recent_models = sorted(
            self.registry.values(),
            key=lambda m: m.created_at,
            reverse=True
        )[:10]
        
        report['recent_models'] = [
            {
                'model_id': m.model_id,
                'model_name': m.model_name,
                'created_at': m.created_at.isoformat(),
                'author': m.author
            }
            for m in recent_models
        ]
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Registry report exported to {output_path}")