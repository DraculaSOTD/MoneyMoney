"""
Pipeline Builder for Model Orchestration.

Provides a fluent interface for building complex trading pipelines
with proper dependency management and validation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

from model_orchestrator import (
    ModelNode, PipelineConfig, ModelOrchestrator, 
    PipelineStage, ModelStatus
)


@dataclass
class ModelSpec:
    """Specification for a model in the pipeline."""
    name: str
    model_type: str
    model_class: type
    config: Dict[str, Any]
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


class PipelineBuilder:
    """
    Fluent interface for building trading pipelines.
    
    Example:
        pipeline = (PipelineBuilder("portfolio_prediction")
                   .add_data_source("market_data", MarketDataFetcher())
                   .add_feature_extractor("features", FeatureExtractor())
                   .add_model("lstm", LSTMPredictor(), inputs={"data": "features"})
                   .add_model("tcn", TCNPredictor(), inputs={"data": "features"})
                   .add_ensemble("ensemble", ["lstm", "tcn"])
                   .add_risk_manager("risk", RiskManager(), inputs={"predictions": "ensemble"})
                   .add_executor("trader", TradeExecutor(), inputs={"signals": "risk"})
                   .with_parallel_models(5)
                   .with_caching(ttl=300)
                   .build())
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize pipeline builder."""
        self.name = name
        self.description = description or f"Pipeline: {name}"
        self.pipeline_id = f"pipeline_{name}_{uuid.uuid4().hex[:8]}"
        
        # Pipeline components
        self.models: Dict[str, ModelSpec] = {}
        self.stages: List[PipelineStage] = []
        self.connections: List[Tuple[str, str, str]] = []  # (source, target, input_name)
        
        # Configuration
        self.config = {
            'max_parallel_models': 5,
            'global_timeout': 3600,
            'error_handling': 'continue',
            'monitoring_interval': 30,
            'performance_tracking': True,
            'enable_caching': True,
            'cache_ttl': 300
        }
        
        # Validation state
        self.validated = False
    
    def add_data_source(self, name: str, source: Any, 
                       outputs: Optional[List[str]] = None) -> 'PipelineBuilder':
        """Add a data source to the pipeline."""
        outputs = outputs or ['data']
        
        spec = ModelSpec(
            name=name,
            model_type='data_source',
            model_class=type(source),
            config={},
            outputs=outputs
        )
        
        self.models[name] = spec
        
        # Add data ingestion stage if not present
        if PipelineStage.DATA_INGESTION not in self.stages:
            self.stages.append(PipelineStage.DATA_INGESTION)
        
        return self
    
    def add_feature_extractor(self, name: str, extractor: Any,
                            inputs: Optional[Dict[str, str]] = None,
                            outputs: Optional[List[str]] = None) -> 'PipelineBuilder':
        """Add a feature extraction model."""
        inputs = inputs or {'data': 'input'}
        outputs = outputs or ['features']
        
        spec = ModelSpec(
            name=name,
            model_type='feature_extraction',
            model_class=type(extractor),
            config={},
            inputs=inputs,
            outputs=outputs
        )
        
        self.models[name] = spec
        self._update_dependencies(name, inputs)
        
        # Add feature engineering stage
        if PipelineStage.FEATURE_ENGINEERING not in self.stages:
            self.stages.append(PipelineStage.FEATURE_ENGINEERING)
        
        return self
    
    def add_model(self, name: str, model: Any,
                 model_type: str = 'prediction',
                 inputs: Optional[Dict[str, str]] = None,
                 outputs: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None) -> 'PipelineBuilder':
        """Add a prediction model to the pipeline."""
        inputs = inputs or {'data': 'input'}
        outputs = outputs or ['predictions']
        config = config or {}
        
        spec = ModelSpec(
            name=name,
            model_type=model_type,
            model_class=type(model),
            config=config,
            inputs=inputs,
            outputs=outputs
        )
        
        self.models[name] = spec
        self._update_dependencies(name, inputs)
        
        # Add model prediction stage
        if PipelineStage.MODEL_PREDICTION not in self.stages:
            self.stages.append(PipelineStage.MODEL_PREDICTION)
        
        return self
    
    def add_ensemble(self, name: str, model_names: List[str],
                    ensemble_method: str = 'weighted_average',
                    weights: Optional[List[float]] = None) -> 'PipelineBuilder':
        """Add an ensemble combining multiple models."""
        # Create inputs from all component models
        inputs = {f'model_{i}': model_name 
                 for i, model_name in enumerate(model_names)}
        
        # Mock ensemble model
        class EnsembleModel:
            def __init__(self, method, weights):
                self.method = method
                self.weights = weights or [1.0/len(model_names)] * len(model_names)
            
            def predict(self, inputs_dict):
                predictions = [inputs_dict[key] for key in sorted(inputs_dict.keys())]
                if self.method == 'weighted_average':
                    return np.average(predictions, weights=self.weights, axis=0)
                elif self.method == 'voting':
                    return np.median(predictions, axis=0)
                else:
                    return np.mean(predictions, axis=0)
        
        ensemble = EnsembleModel(ensemble_method, weights)
        
        spec = ModelSpec(
            name=name,
            model_type='ensemble',
            model_class=EnsembleModel,
            config={'method': ensemble_method, 'weights': weights},
            inputs=inputs,
            outputs=['ensemble_predictions']
        )
        
        self.models[name] = spec
        self._update_dependencies(name, inputs)
        
        return self
    
    def add_risk_manager(self, name: str, risk_manager: Any,
                        inputs: Optional[Dict[str, str]] = None,
                        config: Optional[Dict[str, Any]] = None) -> 'PipelineBuilder':
        """Add a risk management component."""
        inputs = inputs or {'predictions': 'model_predictions'}
        config = config or {}
        
        spec = ModelSpec(
            name=name,
            model_type='risk_assessment',
            model_class=type(risk_manager),
            config=config,
            inputs=inputs,
            outputs=['risk_adjusted_signals']
        )
        
        self.models[name] = spec
        self._update_dependencies(name, inputs)
        
        # Add risk management stage
        if PipelineStage.RISK_MANAGEMENT not in self.stages:
            self.stages.append(PipelineStage.RISK_MANAGEMENT)
        
        return self
    
    def add_executor(self, name: str, executor: Any,
                    inputs: Optional[Dict[str, str]] = None) -> 'PipelineBuilder':
        """Add a trade execution component."""
        inputs = inputs or {'signals': 'risk_adjusted_signals'}
        
        spec = ModelSpec(
            name=name,
            model_type='execution',
            model_class=type(executor),
            config={},
            inputs=inputs,
            outputs=['execution_results']
        )
        
        self.models[name] = spec
        self._update_dependencies(name, inputs)
        
        # Add execution stage
        if PipelineStage.EXECUTION not in self.stages:
            self.stages.append(PipelineStage.EXECUTION)
        
        return self
    
    def add_custom(self, name: str, component: Any,
                  model_type: str,
                  inputs: Optional[Dict[str, str]] = None,
                  outputs: Optional[List[str]] = None,
                  stage: Optional[PipelineStage] = None) -> 'PipelineBuilder':
        """Add a custom component to the pipeline."""
        inputs = inputs or {}
        outputs = outputs or ['output']
        
        spec = ModelSpec(
            name=name,
            model_type=model_type,
            model_class=type(component),
            config={},
            inputs=inputs,
            outputs=outputs
        )
        
        self.models[name] = spec
        self._update_dependencies(name, inputs)
        
        # Add stage if specified
        if stage and stage not in self.stages:
            self.stages.append(stage)
        
        return self
    
    def connect(self, source: str, target: str, 
               input_name: str = 'data') -> 'PipelineBuilder':
        """Manually connect two models."""
        if source not in self.models:
            raise ValueError(f"Source model {source} not found")
        if target not in self.models:
            raise ValueError(f"Target model {target} not found")
        
        self.connections.append((source, target, input_name))
        self._update_dependencies(target, {input_name: source})
        
        return self
    
    def with_parallel_models(self, max_parallel: int) -> 'PipelineBuilder':
        """Set maximum parallel model execution."""
        self.config['max_parallel_models'] = max_parallel
        return self
    
    def with_timeout(self, timeout_seconds: int) -> 'PipelineBuilder':
        """Set global pipeline timeout."""
        self.config['global_timeout'] = timeout_seconds
        return self
    
    def with_error_handling(self, strategy: str) -> 'PipelineBuilder':
        """Set error handling strategy (continue, stop, retry)."""
        if strategy not in ['continue', 'stop', 'retry']:
            raise ValueError(f"Invalid error handling strategy: {strategy}")
        self.config['error_handling'] = strategy
        return self
    
    def with_caching(self, enabled: bool = True, ttl: int = 300) -> 'PipelineBuilder':
        """Configure result caching."""
        self.config['enable_caching'] = enabled
        self.config['cache_ttl'] = ttl
        return self
    
    def with_monitoring(self, interval: int = 30) -> 'PipelineBuilder':
        """Configure performance monitoring."""
        self.config['performance_tracking'] = True
        self.config['monitoring_interval'] = interval
        return self
    
    def _update_dependencies(self, model_name: str, inputs: Dict[str, str]) -> None:
        """Update model dependencies based on inputs."""
        dependencies = []
        
        for input_name, source in inputs.items():
            if source != 'input' and source in self.models:
                dependencies.append(source)
        
        if model_name in self.models:
            self.models[model_name].dependencies = dependencies
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the pipeline configuration."""
        errors = []
        
        # Check for cycles
        if self._has_cycles():
            errors.append("Pipeline contains circular dependencies")
        
        # Check all inputs are satisfied
        for model_name, spec in self.models.items():
            for input_name, source in spec.inputs.items():
                if source != 'input' and source not in self.models:
                    errors.append(f"Model {model_name} input '{input_name}' "
                                f"references unknown source '{source}'")
        
        # Check stages are in correct order
        stage_order = [
            PipelineStage.DATA_INGESTION,
            PipelineStage.PREPROCESSING,
            PipelineStage.FEATURE_ENGINEERING,
            PipelineStage.MODEL_PREDICTION,
            PipelineStage.POST_PROCESSING,
            PipelineStage.RISK_MANAGEMENT,
            PipelineStage.EXECUTION
        ]
        
        current_stages = [s for s in stage_order if s in self.stages]
        if current_stages != sorted(self.stages, key=lambda s: stage_order.index(s)):
            errors.append("Pipeline stages are not in correct order")
        
        self.validated = len(errors) == 0
        return self.validated, errors
    
    def _has_cycles(self) -> bool:
        """Check if the dependency graph has cycles."""
        # Build adjacency list
        graph = {name: spec.dependencies for name, spec in self.models.items()}
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle_util(node):
                    return True
        
        return False
    
    def build(self, orchestrator: Optional[ModelOrchestrator] = None) -> Tuple[str, ModelOrchestrator]:
        """
        Build the pipeline and register with orchestrator.
        
        Returns:
            Tuple of (pipeline_id, orchestrator)
        """
        # Validate first
        valid, errors = self.validate()
        if not valid:
            raise ValueError(f"Pipeline validation failed: {errors}")
        
        # Create orchestrator if not provided
        if orchestrator is None:
            orchestrator = ModelOrchestrator()
        
        # Create pipeline config
        pipeline_config = PipelineConfig(
            pipeline_id=self.pipeline_id,
            pipeline_name=self.name,
            description=self.description,
            stages=self.stages,
            **self.config
        )
        
        # Register pipeline
        orchestrator.register_pipeline(pipeline_config)
        
        # Create and register model nodes
        for model_name, spec in self.models.items():
            # Create model instance
            if hasattr(spec.model_class, '__init__'):
                model_instance = spec.model_class(**spec.config)
            else:
                model_instance = spec.model_class
            
            # Create model node
            node = ModelNode(
                node_id=f"{self.pipeline_id}_{model_name}",
                model_name=spec.name,
                model_type=spec.model_type,
                model_instance=model_instance,
                dependencies=[f"{self.pipeline_id}_{dep}" for dep in spec.dependencies],
                inputs={k: f"{self.pipeline_id}_{v}" if v != 'input' else v 
                       for k, v in spec.inputs.items()},
                outputs=spec.outputs,
                resource_requirements=spec.resource_requirements
            )
            
            orchestrator.register_model(node)
        
        return self.pipeline_id, orchestrator
    
    def visualize(self) -> str:
        """Generate a text visualization of the pipeline."""
        lines = []
        lines.append(f"Pipeline: {self.name}")
        lines.append(f"ID: {self.pipeline_id}")
        lines.append(f"Description: {self.description}")
        lines.append("\nStages:")
        for stage in self.stages:
            lines.append(f"  - {stage.value}")
        
        lines.append("\nModels:")
        
        # Topological sort for display
        sorted_models = self._topological_sort()
        
        for model_name in sorted_models:
            spec = self.models[model_name]
            lines.append(f"\n  {model_name} ({spec.model_type}):")
            
            if spec.inputs:
                lines.append("    Inputs:")
                for input_name, source in spec.inputs.items():
                    lines.append(f"      - {input_name} <- {source}")
            
            if spec.outputs:
                lines.append("    Outputs:")
                for output in spec.outputs:
                    lines.append(f"      - {output}")
            
            if spec.dependencies:
                lines.append("    Dependencies:")
                for dep in spec.dependencies:
                    lines.append(f"      - {dep}")
        
        lines.append("\nConfiguration:")
        for key, value in self.config.items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort on models."""
        graph = {name: spec.dependencies for name, spec in self.models.items()}
        in_degree = {name: 0 for name in self.models}
        
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [node for node, degree in in_degree.items() if degree == 0]
        sorted_nodes = []
        
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            
            for dep in graph.get(node, []):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        
        return sorted_nodes
    
    def export_config(self, filepath: str) -> None:
        """Export pipeline configuration to JSON."""
        config = {
            'name': self.name,
            'description': self.description,
            'pipeline_id': self.pipeline_id,
            'stages': [s.value for s in self.stages],
            'models': {
                name: {
                    'type': spec.model_type,
                    'class': spec.model_class.__name__,
                    'config': spec.config,
                    'inputs': spec.inputs,
                    'outputs': spec.outputs,
                    'dependencies': spec.dependencies
                }
                for name, spec in self.models.items()
            },
            'connections': self.connections,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_config(cls, filepath: str) -> 'PipelineBuilder':
        """Load pipeline configuration from JSON."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        builder = cls(config['name'], config['description'])
        builder.pipeline_id = config['pipeline_id']
        builder.config = config['config']
        
        # Note: This is simplified - in practice, you'd need a registry
        # to reconstruct the actual model instances from class names
        
        return builder