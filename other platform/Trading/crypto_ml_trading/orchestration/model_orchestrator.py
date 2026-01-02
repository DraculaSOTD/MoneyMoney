"""
Advanced Model Orchestration System.

Coordinates multiple models, manages execution pipelines, and handles
complex trading strategies with failover and monitoring capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import queue
import time
import json
import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model execution status."""
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DISABLED = "disabled"


class PipelineStage(Enum):
    """Pipeline execution stages."""
    DATA_INGESTION = "data_ingestion"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_PREDICTION = "model_prediction"
    POST_PROCESSING = "post_processing"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION = "execution"


@dataclass
class ModelNode:
    """Represents a model in the orchestration graph."""
    node_id: str
    model_name: str
    model_type: str
    model_instance: Any
    dependencies: List[str] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)  # input_name -> source_node_id
    outputs: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3
    priority: int = 1
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    status: ModelStatus = ModelStatus.READY
    last_execution: Optional[datetime] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for orchestration pipeline."""
    pipeline_id: str
    pipeline_name: str
    description: str
    stages: List[PipelineStage]
    max_parallel_models: int = 5
    global_timeout: int = 3600  # 1 hour
    error_handling: str = "continue"  # continue, stop, retry
    monitoring_interval: int = 30  # seconds
    performance_tracking: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes


@dataclass
class OrchestrationResult:
    """Result from orchestration execution."""
    pipeline_id: str
    execution_id: str
    start_time: datetime
    end_time: datetime
    status: str
    model_results: Dict[str, Any]
    stage_timings: Dict[str, float]
    errors: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    cache_hits: int
    total_models_executed: int


class ModelOrchestrator:
    """
    Advanced model orchestration system for complex trading pipelines.
    
    Features:
    - DAG-based model execution
    - Parallel processing with dependency management
    - Dynamic pipeline construction
    - Failure handling and recovery
    - Performance monitoring
    - Resource management
    - Result caching
    """
    
    def __init__(self, 
                 max_workers: int = 10,
                 result_cache_size: int = 1000,
                 monitoring_enabled: bool = True):
        """
        Initialize model orchestrator.
        
        Args:
            max_workers: Maximum parallel execution threads
            result_cache_size: Size of result cache
            monitoring_enabled: Enable performance monitoring
        """
        self.max_workers = max_workers
        self.monitoring_enabled = monitoring_enabled
        
        # Model registry
        self.models: Dict[str, ModelNode] = {}
        self.pipelines: Dict[str, PipelineConfig] = {}
        
        # Execution management
        self.execution_queue = queue.PriorityQueue()
        self.worker_pool: List[threading.Thread] = []
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # Result caching
        self.result_cache = deque(maxlen=result_cache_size)
        self.cache_index: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.model_metrics = defaultdict(lambda: {
            'executions': 0,
            'failures': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        })
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        logger.info("Model orchestrator initialized")
    
    def register_model(self, model_node: ModelNode) -> None:
        """Register a model in the orchestration system."""
        if model_node.node_id in self.models:
            logger.warning(f"Model {model_node.node_id} already registered, updating...")
        
        self.models[model_node.node_id] = model_node
        logger.info(f"Registered model: {model_node.model_name} (ID: {model_node.node_id})")
    
    def register_pipeline(self, pipeline_config: PipelineConfig) -> None:
        """Register a pipeline configuration."""
        self.pipelines[pipeline_config.pipeline_id] = pipeline_config
        logger.info(f"Registered pipeline: {pipeline_config.pipeline_name}")
    
    def build_execution_graph(self, pipeline_id: str, 
                            input_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Build execution graph based on model dependencies.
        
        Returns:
            Adjacency list representation of execution graph
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Add all models to graph
        for model_id, model in self.models.items():
            if model.status == ModelStatus.DISABLED:
                continue
                
            for dep in model.dependencies:
                if dep in self.models and self.models[dep].status != ModelStatus.DISABLED:
                    graph[dep].append(model_id)
                    in_degree[model_id] += 1
        
        # Topological sort to determine execution order
        execution_order = []
        queue = [node for node in self.models if in_degree[node] == 0 and 
                self.models[node].status != ModelStatus.DISABLED]
        
        while queue:
            node = queue.pop(0)
            execution_order.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(execution_order) != len([m for m in self.models.values() 
                                       if m.status != ModelStatus.DISABLED]):
            raise ValueError("Circular dependencies detected in model graph")
        
        return graph, execution_order
    
    def execute_pipeline(self, pipeline_id: str, 
                        input_data: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """
        Execute a complete pipeline.
        
        Args:
            pipeline_id: Pipeline to execute
            input_data: Input data for models
            context: Additional execution context
            
        Returns:
            Orchestration result with all outputs
        """
        start_time = datetime.now()
        execution_id = f"{pipeline_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting pipeline execution: {execution_id}")
        
        # Initialize execution tracking
        execution_state = {
            'pipeline_id': pipeline_id,
            'execution_id': execution_id,
            'start_time': start_time,
            'input_data': input_data,
            'context': context or {},
            'model_results': {},
            'stage_timings': {},
            'errors': [],
            'cache_hits': 0
        }
        
        self.active_executions[execution_id] = execution_state
        
        try:
            # Build execution graph
            graph, execution_order = self.build_execution_graph(pipeline_id, input_data)
            
            # Execute pipeline stages
            pipeline_config = self.pipelines[pipeline_id]
            
            for stage in pipeline_config.stages:
                stage_start = time.time()
                
                try:
                    self._execute_stage(stage, execution_state, execution_order)
                except Exception as e:
                    logger.error(f"Stage {stage.value} failed: {e}")
                    execution_state['errors'].append({
                        'stage': stage.value,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
                    if pipeline_config.error_handling == "stop":
                        break
                
                execution_state['stage_timings'][stage.value] = time.time() - stage_start
            
            # Compile results
            end_time = datetime.now()
            
            result = OrchestrationResult(
                pipeline_id=pipeline_id,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                status="completed" if not execution_state['errors'] else "completed_with_errors",
                model_results=execution_state['model_results'],
                stage_timings=execution_state['stage_timings'],
                errors=execution_state['errors'],
                performance_metrics=self._calculate_performance_metrics(execution_state),
                cache_hits=execution_state['cache_hits'],
                total_models_executed=len(execution_state['model_results'])
            )
            
            # Store in performance history
            if self.monitoring_enabled:
                self.performance_history.append({
                    'execution_id': execution_id,
                    'timestamp': end_time,
                    'duration': (end_time - start_time).total_seconds(),
                    'models_executed': len(execution_state['model_results']),
                    'errors': len(execution_state['errors']),
                    'cache_hits': execution_state['cache_hits']
                })
            
            return result
            
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def _execute_stage(self, stage: PipelineStage, 
                      execution_state: Dict[str, Any],
                      execution_order: List[str]) -> None:
        """Execute a specific pipeline stage."""
        stage_models = self._get_models_for_stage(stage, execution_order)
        
        if stage == PipelineStage.MODEL_PREDICTION:
            # Execute models in parallel respecting dependencies
            self._execute_models_parallel(stage_models, execution_state)
        else:
            # Execute stage-specific logic
            self._execute_stage_logic(stage, execution_state, stage_models)
    
    def _get_models_for_stage(self, stage: PipelineStage, 
                            execution_order: List[str]) -> List[str]:
        """Get models that belong to a specific stage."""
        # For demonstration, assign models to stages based on type
        stage_models = []
        
        for model_id in execution_order:
            model = self.models[model_id]
            
            if stage == PipelineStage.MODEL_PREDICTION:
                if model.model_type in ['prediction', 'forecasting', 'classification']:
                    stage_models.append(model_id)
            elif stage == PipelineStage.FEATURE_ENGINEERING:
                if model.model_type in ['feature_extraction', 'transformation']:
                    stage_models.append(model_id)
            elif stage == PipelineStage.RISK_MANAGEMENT:
                if model.model_type in ['risk_assessment', 'position_sizing']:
                    stage_models.append(model_id)
        
        return stage_models
    
    def _execute_models_parallel(self, model_ids: List[str], 
                               execution_state: Dict[str, Any]) -> None:
        """Execute models in parallel with dependency resolution."""
        completed = set()
        execution_queue = queue.Queue()
        results_queue = queue.Queue()
        
        # Worker function
        def model_worker():
            while True:
                try:
                    model_id = execution_queue.get(timeout=1)
                    if model_id is None:
                        break
                    
                    # Execute model
                    result = self._execute_single_model(model_id, execution_state)
                    results_queue.put((model_id, result))
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker error executing {model_id}: {e}")
                    results_queue.put((model_id, {'error': str(e)}))
        
        # Start workers
        workers = []
        for _ in range(min(self.max_workers, len(model_ids))):
            worker = threading.Thread(target=model_worker)
            worker.start()
            workers.append(worker)
        
        # Schedule models based on dependencies
        remaining = set(model_ids)
        
        while remaining or not results_queue.empty():
            # Check for completed models
            try:
                model_id, result = results_queue.get(timeout=0.1)
                execution_state['model_results'][model_id] = result
                completed.add(model_id)
                remaining.discard(model_id)
            except queue.Empty:
                pass
            
            # Schedule ready models
            for model_id in list(remaining):
                model = self.models[model_id]
                
                # Check if dependencies are satisfied
                if all(dep in completed or dep not in model_ids 
                      for dep in model.dependencies):
                    execution_queue.put(model_id)
                    remaining.remove(model_id)
        
        # Stop workers
        for _ in workers:
            execution_queue.put(None)
        
        for worker in workers:
            worker.join()
    
    def _execute_single_model(self, model_id: str, 
                            execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single model."""
        model = self.models[model_id]
        
        # Check cache
        cache_key = self._generate_cache_key(model_id, execution_state)
        if cache_key in self.cache_index:
            execution_state['cache_hits'] += 1
            logger.info(f"Cache hit for model {model_id}")
            return self.cache_index[cache_key]
        
        # Prepare inputs
        model_inputs = {}
        for input_name, source_node in model.inputs.items():
            if source_node in execution_state['model_results']:
                model_inputs[input_name] = execution_state['model_results'][source_node]
            elif source_node == "input":
                model_inputs[input_name] = execution_state['input_data']
        
        # Execute model
        start_time = time.time()
        model.status = ModelStatus.RUNNING
        model.last_execution = datetime.now()
        
        try:
            # Call model's predict method
            if hasattr(model.model_instance, 'predict'):
                result = model.model_instance.predict(model_inputs)
            elif callable(model.model_instance):
                result = model.model_instance(model_inputs)
            else:
                raise ValueError(f"Model {model_id} has no callable interface")
            
            # Update status
            model.status = ModelStatus.COMPLETED
            model.execution_time = time.time() - start_time
            
            # Update metrics
            self._update_model_metrics(model_id, model.execution_time, success=True)
            
            # Cache result
            if self.pipelines[execution_state['pipeline_id']].enable_caching:
                self.cache_index[cache_key] = result
                self.result_cache.append((cache_key, result, datetime.now()))
            
            return {
                'output': result,
                'execution_time': model.execution_time,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            model.status = ModelStatus.FAILED
            model.error_message = str(e)
            model.execution_time = time.time() - start_time
            
            # Update metrics
            self._update_model_metrics(model_id, model.execution_time, success=False)
            
            # Handle retry
            if model.retry_count > 0:
                logger.warning(f"Model {model_id} failed, retrying...")
                model.retry_count -= 1
                return self._execute_single_model(model_id, execution_state)
            
            raise
    
    def _execute_stage_logic(self, stage: PipelineStage, 
                           execution_state: Dict[str, Any],
                           stage_models: List[str]) -> None:
        """Execute stage-specific logic."""
        if stage == PipelineStage.DATA_INGESTION:
            # Data ingestion logic
            logger.info("Executing data ingestion stage")
            # Would implement data fetching, validation, etc.
            
        elif stage == PipelineStage.PREPROCESSING:
            # Preprocessing logic
            logger.info("Executing preprocessing stage")
            # Would implement data cleaning, normalization, etc.
            
        elif stage == PipelineStage.POST_PROCESSING:
            # Post-processing logic
            logger.info("Executing post-processing stage")
            # Would implement result aggregation, formatting, etc.
            
        elif stage == PipelineStage.RISK_MANAGEMENT:
            # Risk management logic
            logger.info("Executing risk management stage")
            # Would implement position sizing, risk limits, etc.
            
        elif stage == PipelineStage.EXECUTION:
            # Trade execution logic
            logger.info("Executing trade execution stage")
            # Would implement order placement, monitoring, etc.
    
    def _generate_cache_key(self, model_id: str, 
                          execution_state: Dict[str, Any]) -> str:
        """Generate cache key for model result."""
        # Create deterministic key based on model and inputs
        key_parts = [model_id]
        
        model = self.models[model_id]
        for input_name, source_node in sorted(model.inputs.items()):
            if source_node in execution_state['model_results']:
                # Use hash of result
                result_hash = hash(str(execution_state['model_results'][source_node]))
                key_parts.append(f"{input_name}:{result_hash}")
            elif source_node == "input":
                # Use hash of input data
                input_hash = hash(str(execution_state['input_data']))
                key_parts.append(f"{input_name}:{input_hash}")
        
        return "_".join(key_parts)
    
    def _update_model_metrics(self, model_id: str, 
                            execution_time: float, 
                            success: bool) -> None:
        """Update model performance metrics."""
        metrics = self.model_metrics[model_id]
        
        metrics['executions'] += 1
        if not success:
            metrics['failures'] += 1
        
        metrics['total_time'] += execution_time
        metrics['avg_time'] = metrics['total_time'] / metrics['executions']
    
    def _calculate_performance_metrics(self, 
                                     execution_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for execution."""
        total_time = sum(execution_state['stage_timings'].values())
        model_times = [
            r.get('execution_time', 0) 
            for r in execution_state['model_results'].values()
            if isinstance(r, dict) and 'execution_time' in r
        ]
        
        return {
            'total_execution_time': total_time,
            'avg_model_time': np.mean(model_times) if model_times else 0,
            'max_model_time': max(model_times) if model_times else 0,
            'cache_hit_rate': execution_state['cache_hits'] / 
                            max(len(execution_state['model_results']), 1),
            'error_rate': len(execution_state['errors']) / 
                         max(len(self.models), 1)
        }
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Orchestrator monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Orchestrator monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Clean expired cache entries
                self._clean_cache()
                
                # Log performance statistics
                if self.performance_history:
                    recent_perf = list(self.performance_history)[-10:]
                    avg_duration = np.mean([p['duration'] for p in recent_perf])
                    avg_models = np.mean([p['models_executed'] for p in recent_perf])
                    
                    logger.info(f"Recent performance - Avg duration: {avg_duration:.2f}s, "
                              f"Avg models: {avg_models:.1f}")
                
                # Check for stuck executions
                for exec_id, exec_state in self.active_executions.items():
                    duration = (datetime.now() - exec_state['start_time']).total_seconds()
                    pipeline = self.pipelines.get(exec_state['pipeline_id'])
                    
                    if pipeline and duration > pipeline.global_timeout:
                        logger.warning(f"Execution {exec_id} exceeded timeout")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _clean_cache(self) -> None:
        """Clean expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, result, timestamp in list(self.result_cache):
            age_seconds = (current_time - timestamp).total_seconds()
            
            # Get cache TTL from pipeline config (default 5 minutes)
            if age_seconds > 300:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.cache_index:
                del self.cache_index[key]
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get current status of a model."""
        if model_id not in self.models:
            return {'error': f'Model {model_id} not found'}
        
        model = self.models[model_id]
        metrics = self.model_metrics[model_id]
        
        return {
            'model_id': model_id,
            'model_name': model.model_name,
            'status': model.status.value,
            'last_execution': model.last_execution,
            'execution_time': model.execution_time,
            'error_message': model.error_message,
            'metrics': dict(metrics)
        }
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get current status of a pipeline."""
        if pipeline_id not in self.pipelines:
            return {'error': f'Pipeline {pipeline_id} not found'}
        
        pipeline = self.pipelines[pipeline_id]
        
        # Get recent executions
        recent_executions = [
            perf for perf in self.performance_history
            if perf.get('execution_id', '').startswith(pipeline_id)
        ][-5:]
        
        return {
            'pipeline_id': pipeline_id,
            'pipeline_name': pipeline.pipeline_name,
            'stages': [s.value for s in pipeline.stages],
            'recent_executions': recent_executions,
            'active_executions': [
                exec_id for exec_id in self.active_executions
                if exec_id.startswith(pipeline_id)
            ]
        }
    
    def create_dynamic_pipeline(self, 
                              name: str,
                              model_ids: List[str],
                              stages: Optional[List[PipelineStage]] = None) -> str:
        """
        Create a pipeline dynamically from a set of models.
        
        Args:
            name: Pipeline name
            model_ids: Models to include
            stages: Pipeline stages (auto-detected if None)
            
        Returns:
            Pipeline ID
        """
        pipeline_id = f"dynamic_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Auto-detect stages if not provided
        if stages is None:
            stages = self._detect_pipeline_stages(model_ids)
        
        pipeline_config = PipelineConfig(
            pipeline_id=pipeline_id,
            pipeline_name=name,
            description=f"Dynamically created pipeline with {len(model_ids)} models",
            stages=stages,
            max_parallel_models=min(5, len(model_ids)),
            error_handling="continue"
        )
        
        self.register_pipeline(pipeline_config)
        
        logger.info(f"Created dynamic pipeline: {pipeline_id}")
        return pipeline_id
    
    def _detect_pipeline_stages(self, model_ids: List[str]) -> List[PipelineStage]:
        """Auto-detect required stages based on models."""
        stages = [PipelineStage.DATA_INGESTION]
        
        model_types = {self.models[mid].model_type for mid in model_ids if mid in self.models}
        
        if any(t in ['feature_extraction', 'transformation'] for t in model_types):
            stages.append(PipelineStage.FEATURE_ENGINEERING)
        
        stages.append(PipelineStage.MODEL_PREDICTION)
        
        if any(t in ['risk_assessment', 'position_sizing'] for t in model_types):
            stages.append(PipelineStage.RISK_MANAGEMENT)
        
        stages.extend([PipelineStage.POST_PROCESSING, PipelineStage.EXECUTION])
        
        return stages
    
    def export_orchestration_graph(self, pipeline_id: str, 
                                 output_path: str) -> None:
        """Export orchestration graph for visualization."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        graph, execution_order = self.build_execution_graph(pipeline_id, {})
        
        # Create graph description
        graph_data = {
            'pipeline': self.pipelines[pipeline_id].__dict__,
            'nodes': {},
            'edges': []
        }
        
        # Add nodes
        for model_id in execution_order:
            model = self.models[model_id]
            graph_data['nodes'][model_id] = {
                'id': model_id,
                'name': model.model_name,
                'type': model.model_type,
                'status': model.status.value,
                'dependencies': model.dependencies
            }
        
        # Add edges
        for source, targets in graph.items():
            for target in targets:
                graph_data['edges'].append({
                    'source': source,
                    'target': target
                })
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        logger.info(f"Exported orchestration graph to {output_path}")