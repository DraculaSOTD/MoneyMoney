"""
Advanced Model Orchestration System.

Provides comprehensive pipeline management for complex trading strategies
with parallel execution, monitoring, and error handling capabilities.
"""

from .model_orchestrator import (
    ModelOrchestrator,
    ModelNode,
    PipelineConfig,
    ModelStatus,
    PipelineStage,
    OrchestrationResult
)

from .pipeline_builder import (
    PipelineBuilder,
    ModelSpec
)

from .pipeline_templates import (
    PipelineTemplates,
    MockModel
)

__all__ = [
    # Orchestrator
    'ModelOrchestrator',
    'ModelNode',
    'PipelineConfig',
    'ModelStatus',
    'PipelineStage',
    'OrchestrationResult',
    
    # Builder
    'PipelineBuilder',
    'ModelSpec',
    
    # Templates
    'PipelineTemplates',
    'MockModel'
]

__version__ = '1.0.0'