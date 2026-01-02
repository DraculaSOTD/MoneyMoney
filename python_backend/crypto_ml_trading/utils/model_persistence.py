"""
Model Persistence System for Crypto ML Trading
Handles saving, loading, versioning, and management of trained models.
"""

import pickle
import json
import joblib
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timezone
import logging
import hashlib
import shutil
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for saved models."""
    model_name: str
    model_type: str
    version: str
    created_at: str
    file_size: int
    checksum: str
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    feature_names: List[str] = None
    model_parameters: Dict[str, Any] = None
    training_data_info: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


class ModelCheckpoint:
    """Model checkpoint for incremental saving."""
    
    def __init__(self, model_name: str, checkpoint_dir: Union[str, Path]):
        """
        Initialize model checkpoint.
        
        Args:
            model_name: Name of the model
            checkpoint_dir: Directory to save checkpoints
        """
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_counter = 0
        self._load_counter()
    
    def _load_counter(self):
        """Load checkpoint counter from file."""
        counter_file = self.checkpoint_dir / f"{self.model_name}_counter.txt"
        if counter_file.exists():
            try:
                with open(counter_file, 'r') as f:
                    self.checkpoint_counter = int(f.read().strip())
            except (ValueError, IOError):
                self.checkpoint_counter = 0
    
    def _save_counter(self):
        """Save checkpoint counter to file."""
        counter_file = self.checkpoint_dir / f"{self.model_name}_counter.txt"
        try:
            with open(counter_file, 'w') as f:
                f.write(str(self.checkpoint_counter))
        except IOError as e:
            logger.error(f"Failed to save checkpoint counter: {e}")
    
    def save_checkpoint(self, model: Any, epoch: int = None, 
                       metrics: Dict[str, float] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Training epoch (optional)
            metrics: Performance metrics (optional)
            
        Returns:
            Path to saved checkpoint
        """
        self.checkpoint_counter += 1
        
        if epoch is not None:
            checkpoint_name = f"{self.model_name}_epoch_{epoch:04d}_checkpoint_{self.checkpoint_counter:04d}.pkl"
        else:
            checkpoint_name = f"{self.model_name}_checkpoint_{self.checkpoint_counter:04d}.pkl"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        checkpoint_data = {
            'model': model,
            'checkpoint_counter': self.checkpoint_counter,
            'epoch': epoch,
            'metrics': metrics or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with gzip.open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self._save_counter()
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_latest_checkpoint(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Load latest checkpoint.
        
        Returns:
            Tuple of (model, checkpoint_info)
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for {self.model_name}")
        
        # Sort by modification time (latest first)
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            with gzip.open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"Loaded checkpoint: {latest_checkpoint}")
            return checkpoint_data['model'], {
                'epoch': checkpoint_data.get('epoch'),
                'metrics': checkpoint_data.get('metrics', {}),
                'timestamp': checkpoint_data.get('timestamp'),
                'path': str(latest_checkpoint)
            }
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of checkpoints to keep
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_*.pkl"))
        if len(checkpoints) <= keep_last:
            return
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-keep_last]:
            try:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint}: {e}")


class ModelPersistence:
    """
    Model persistence system with versioning, compression, and metadata management.
    
    Features:
    - Multiple serialization formats (pickle, joblib)
    - Model versioning
    - Metadata tracking
    - Compression support
    - Checksum validation
    - Model registry
    - Backup and restore
    """
    
    def __init__(self, models_dir: Union[str, Path] = "saved_models"):
        """
        Initialize model persistence system.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()
        self._lock = threading.Lock()
    
    def _load_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load model registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except IOError:
            return ""
    
    def save_model(self, 
                   model: Any,
                   model_name: str,
                   model_type: str = "unknown",
                   version: str = None,
                   training_config: Dict[str, Any] = None,
                   performance_metrics: Dict[str, float] = None,
                   feature_names: List[str] = None,
                   compression: bool = True,
                   format: str = "pickle") -> str:
        """
        Save model with metadata.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            model_type: Type of model (e.g., 'gru_attention', 'cnn')
            version: Model version (auto-generated if None)
            training_config: Training configuration used
            performance_metrics: Model performance metrics
            feature_names: List of feature names
            compression: Whether to compress the model file
            format: Serialization format ('pickle' or 'joblib')
            
        Returns:
            Path to saved model file
        """
        with self._lock:
            if version is None:
                # Auto-generate version based on timestamp
                version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            # Create model directory
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Determine file extension
            ext = f".{format}"
            if compression:
                ext += ".gz"
            
            model_file = model_dir / f"{model_name}_v{version}{ext}"
            
            # Save model
            try:
                if compression:
                    with gzip.open(model_file, 'wb') as f:
                        if format == "joblib":
                            joblib.dump(model, f)
                        else:  # pickle
                            pickle.dump(model, f)
                else:
                    with open(model_file, 'wb') as f:
                        if format == "joblib":
                            joblib.dump(model, f)
                        else:  # pickle
                            pickle.dump(model, f)
                
                # Create metadata
                file_size = model_file.stat().st_size
                checksum = self._calculate_checksum(model_file)
                
                metadata = ModelMetadata(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    file_size=file_size,
                    checksum=checksum,
                    training_config=training_config or {},
                    performance_metrics=performance_metrics or {},
                    feature_names=feature_names
                )
                
                # Save metadata
                metadata_file = model_dir / f"{model_name}_v{version}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                # Update registry
                if model_name not in self.registry:
                    self.registry[model_name] = []
                
                self.registry[model_name].append({
                    'version': version,
                    'file_path': str(model_file),
                    'metadata_path': str(metadata_file),
                    'created_at': metadata.created_at,
                    'file_size': file_size,
                    'checksum': checksum
                })
                
                # Sort by creation time (newest first)
                self.registry[model_name].sort(
                    key=lambda x: x['created_at'], reverse=True
                )
                
                self._save_registry()
                
                logger.info(f"Model saved: {model_file}")
                return str(model_file)
                
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
                raise
    
    def load_model(self, 
                   model_name: str, 
                   version: str = None,
                   verify_checksum: bool = True) -> Tuple[Any, ModelMetadata]:
        """
        Load model with metadata.
        
        Args:
            model_name: Name of the model to load
            version: Specific version to load (latest if None)
            verify_checksum: Whether to verify file integrity
            
        Returns:
            Tuple of (model, metadata)
        """
        with self._lock:
            if model_name not in self.registry:
                raise FileNotFoundError(f"Model '{model_name}' not found in registry")
            
            model_versions = self.registry[model_name]
            
            if version is None:
                # Load latest version
                if not model_versions:
                    raise FileNotFoundError(f"No versions found for model '{model_name}'")
                model_info = model_versions[0]  # Already sorted by creation time
            else:
                # Load specific version
                model_info = None
                for info in model_versions:
                    if info['version'] == version:
                        model_info = info
                        break
                
                if model_info is None:
                    raise FileNotFoundError(
                        f"Version '{version}' not found for model '{model_name}'"
                    )
            
            model_file = Path(model_info['file_path'])
            metadata_file = Path(model_info['metadata_path'])
            
            # Verify files exist
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
            # Verify checksum
            if verify_checksum:
                current_checksum = self._calculate_checksum(model_file)
                if current_checksum != model_info['checksum']:
                    raise ValueError(f"Checksum mismatch for {model_file}")
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
            
            # Load model
            try:
                # Determine format from file extension
                is_compressed = str(model_file).endswith('.gz')
                is_joblib = '.joblib' in str(model_file)
                
                if is_compressed:
                    with gzip.open(model_file, 'rb') as f:
                        if is_joblib:
                            model = joblib.load(f)
                        else:
                            model = pickle.load(f)
                else:
                    with open(model_file, 'rb') as f:
                        if is_joblib:
                            model = joblib.load(f)
                        else:
                            model = pickle.load(f)
                
                logger.info(f"Model loaded: {model_file}")
                return model, metadata
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary mapping model names to version info
        """
        return self.registry.copy()
    
    def delete_model(self, model_name: str, version: str = None):
        """
        Delete model and its metadata.
        
        Args:
            model_name: Name of the model
            version: Specific version to delete (all versions if None)
        """
        with self._lock:
            if model_name not in self.registry:
                raise FileNotFoundError(f"Model '{model_name}' not found")
            
            if version is None:
                # Delete all versions
                for model_info in self.registry[model_name]:
                    self._delete_model_files(model_info)
                
                # Remove from registry
                del self.registry[model_name]
                
                # Remove model directory if empty
                model_dir = self.models_dir / model_name
                try:
                    model_dir.rmdir()
                except OSError:
                    pass  # Directory not empty
                
            else:
                # Delete specific version
                model_versions = self.registry[model_name]
                model_info = None
                
                for i, info in enumerate(model_versions):
                    if info['version'] == version:
                        model_info = info
                        del model_versions[i]
                        break
                
                if model_info is None:
                    raise FileNotFoundError(
                        f"Version '{version}' not found for model '{model_name}'"
                    )
                
                self._delete_model_files(model_info)
                
                # Remove model from registry if no versions left
                if not model_versions:
                    del self.registry[model_name]
            
            self._save_registry()
            logger.info(f"Deleted model: {model_name} (version: {version or 'all'})")
    
    def _delete_model_files(self, model_info: Dict[str, Any]):
        """Delete model and metadata files."""
        try:
            Path(model_info['file_path']).unlink(missing_ok=True)
            Path(model_info['metadata_path']).unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to delete model files: {e}")
    
    def backup_models(self, backup_dir: Union[str, Path]):
        """
        Create backup of all models and registry.
        
        Args:
            backup_dir: Directory to store backup
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"model_backup_{timestamp}"
        
        try:
            # Copy entire models directory
            shutil.copytree(self.models_dir, backup_path)
            logger.info(f"Models backed up to: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup models: {e}")
            raise
    
    def restore_models(self, backup_path: Union[str, Path]):
        """
        Restore models from backup.
        
        Args:
            backup_path: Path to backup directory
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup directory not found: {backup_path}")
        
        try:
            # Remove current models directory
            if self.models_dir.exists():
                shutil.rmtree(self.models_dir)
            
            # Restore from backup
            shutil.copytree(backup_path, self.models_dir)
            
            # Reload registry
            self.registry = self._load_registry()
            
            logger.info(f"Models restored from: {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to restore models: {e}")
            raise
    
    @contextmanager
    def model_checkpoint(self, model_name: str):
        """
        Context manager for model checkpointing during training.
        
        Args:
            model_name: Name of the model
            
        Yields:
            ModelCheckpoint instance
        """
        checkpoint_dir = self.models_dir / model_name / "checkpoints"
        checkpoint = ModelCheckpoint(model_name, checkpoint_dir)
        
        try:
            yield checkpoint
        finally:
            # Cleanup old checkpoints
            checkpoint.cleanup_old_checkpoints(keep_last=5)


# Example usage and testing
if __name__ == "__main__":
    # Test model persistence
    import numpy as np
    
    # Create test model (simple numpy array)
    test_model = {
        'weights': np.random.randn(100, 50),
        'biases': np.random.randn(50),
        'config': {'layers': [100, 50, 10]}
    }
    
    # Initialize persistence system
    persistence = ModelPersistence("test_models")
    
    # Save model
    model_path = persistence.save_model(
        model=test_model,
        model_name="test_neural_network",
        model_type="feedforward",
        training_config={'epochs': 100, 'lr': 0.001},
        performance_metrics={'accuracy': 0.95, 'loss': 0.05},
        feature_names=['feature_' + str(i) for i in range(100)]
    )
    
    print(f"Model saved to: {model_path}")
    
    # Load model
    loaded_model, metadata = persistence.load_model("test_neural_network")
    
    print(f"Loaded model type: {metadata.model_type}")
    print(f"Performance metrics: {metadata.performance_metrics}")
    
    # Test checkpointing
    with persistence.model_checkpoint("test_neural_network") as checkpoint:
        for epoch in range(5):
            # Simulate training
            checkpoint.save_checkpoint(
                test_model, 
                epoch=epoch, 
                metrics={'loss': 0.1 - epoch * 0.01}
            )
    
    # List models
    models = persistence.list_models()
    print("Available models:", list(models.keys()))
    
    # Create backup
    backup_path = persistence.backup_models("test_backup")
    print(f"Backup created at: {backup_path}")
    
    print("Model persistence test completed!")