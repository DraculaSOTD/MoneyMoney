"""
Demonstration of Advanced Model Orchestration System.

Shows complete pipeline construction, execution, and monitoring capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from model_orchestrator import ModelOrchestrator, ModelNode, PipelineConfig, PipelineStage
from pipeline_builder import PipelineBuilder
from pipeline_templates import PipelineTemplates, MockModel


def demonstrate_basic_orchestration():
    """Demonstrate basic orchestration capabilities."""
    print("\n" + "="*80)
    print("BASIC ORCHESTRATION DEMONSTRATION")
    print("="*80)
    
    # Create orchestrator
    orchestrator = ModelOrchestrator(max_workers=5)
    
    # Create simple models
    class DataFetcher:
        def predict(self, inputs):
            print("Fetching market data...")
            return {
                'prices': np.random.randn(100) * 0.02 + 1.0,
                'volumes': np.random.lognormal(10, 1, 100)
            }
    
    class FeatureExtractor:
        def predict(self, inputs):
            print("Extracting features...")
            data = inputs['data']['output']
            return {
                'features': {
                    'returns': np.diff(data['prices']),
                    'volatility': np.std(data['prices'][-20:]),
                    'volume_trend': np.polyfit(range(20), data['volumes'][-20:], 1)[0]
                }
            }
    
    class Predictor:
        def predict(self, inputs):
            print("Making predictions...")
            features = inputs['features']['output']['features']
            # Mock prediction
            prediction = np.mean(features['returns']) * 100 + np.random.randn() * 0.01
            return {'prediction': prediction, 'confidence': 0.75}
    
    # Register models
    models = [
        ModelNode(
            node_id="data_fetcher",
            model_name="Market Data Fetcher",
            model_type="data_source",
            model_instance=DataFetcher(),
            outputs=["market_data"]
        ),
        ModelNode(
            node_id="feature_extractor",
            model_name="Feature Extractor",
            model_type="feature_extraction",
            model_instance=FeatureExtractor(),
            dependencies=["data_fetcher"],
            inputs={"data": "data_fetcher"},
            outputs=["features"]
        ),
        ModelNode(
            node_id="predictor",
            model_name="Price Predictor",
            model_type="prediction",
            model_instance=Predictor(),
            dependencies=["feature_extractor"],
            inputs={"features": "feature_extractor"},
            outputs=["prediction"]
        )
    ]
    
    for model in models:
        orchestrator.register_model(model)
    
    # Create pipeline
    pipeline_config = PipelineConfig(
        pipeline_id="simple_pipeline",
        pipeline_name="Simple Prediction Pipeline",
        description="Basic demonstration pipeline",
        stages=[
            PipelineStage.DATA_INGESTION,
            PipelineStage.FEATURE_ENGINEERING,
            PipelineStage.MODEL_PREDICTION
        ]
    )
    
    orchestrator.register_pipeline(pipeline_config)
    
    # Execute pipeline
    print("\n--- Executing Pipeline ---")
    result = orchestrator.execute_pipeline(
        pipeline_id="simple_pipeline",
        input_data={"timestamp": datetime.now()},
        context={"mode": "demo"}
    )
    
    print(f"\nPipeline Status: {result.status}")
    print(f"Execution Time: {(result.end_time - result.start_time).total_seconds():.2f}s")
    print(f"Models Executed: {result.total_models_executed}")
    
    # Display results
    print("\n--- Model Results ---")
    for model_id, model_result in result.model_results.items():
        if isinstance(model_result, dict) and 'output' in model_result:
            print(f"\n{model_id}:")
            output = model_result['output']
            if isinstance(output, dict):
                for key, value in output.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: array of shape {value.shape}")
                    else:
                        print(f"  {key}: {value}")
    
    return orchestrator


def demonstrate_pipeline_builder():
    """Demonstrate pipeline builder functionality."""
    print("\n" + "="*80)
    print("PIPELINE BUILDER DEMONSTRATION")
    print("="*80)
    
    # Build a complex pipeline using fluent interface
    builder = (PipelineBuilder("advanced_trading", "Advanced multi-model trading pipeline")
              .add_data_source("market_data", MockModel("DataFetcher"))
              .add_feature_extractor("technical_features", 
                                   MockModel("TechnicalIndicators"),
                                   inputs={"data": "market_data"})
              .add_feature_extractor("sentiment_features",
                                   MockModel("SentimentAnalyzer"),
                                   inputs={"data": "market_data"})
              # Add multiple prediction models
              .add_model("lstm_model", MockModel("LSTM"), 
                        inputs={"features": "technical_features"})
              .add_model("transformer_model", MockModel("Transformer"),
                        inputs={"features": "technical_features"})
              .add_model("sentiment_model", MockModel("SentimentPredictor"),
                        inputs={"features": "sentiment_features"})
              # Create ensemble
              .add_ensemble("ensemble", ["lstm_model", "transformer_model", "sentiment_model"])
              # Risk management
              .add_risk_manager("risk_manager", MockModel("RiskManager"),
                              inputs={"predictions": "ensemble"})
              # Execution
              .add_executor("trader", MockModel("TradeExecutor"),
                          inputs={"signals": "risk_manager"})
              # Configuration
              .with_parallel_models(3)
              .with_caching(enabled=True, ttl=300)
              .with_monitoring(interval=30)
              .with_error_handling("continue"))
    
    # Validate pipeline
    print("\n--- Pipeline Validation ---")
    valid, errors = builder.validate()
    print(f"Valid: {valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Visualize pipeline
    print("\n--- Pipeline Structure ---")
    print(builder.visualize())
    
    # Build and register
    pipeline_id, orchestrator = builder.build()
    print(f"\nBuilt pipeline: {pipeline_id}")
    
    return builder, orchestrator


def demonstrate_pipeline_templates():
    """Demonstrate pre-built pipeline templates."""
    print("\n" + "="*80)
    print("PIPELINE TEMPLATES DEMONSTRATION")
    print("="*80)
    
    # 1. Momentum Trading Pipeline
    print("\n--- Momentum Trading Pipeline ---")
    momentum_builder = PipelineTemplates.momentum_trading_pipeline(
        assets=["BTC", "ETH", "SOL"],
        lookback_period=20
    )
    print(momentum_builder.visualize())
    
    # 2. ML Ensemble Pipeline
    print("\n--- ML Ensemble Pipeline ---")
    ml_builder = PipelineTemplates.ml_ensemble_pipeline(
        n_models=3,
        prediction_horizon=24
    )
    pipeline_id, orchestrator = ml_builder.build()
    
    # Execute ML pipeline
    print("\nExecuting ML Ensemble Pipeline...")
    result = orchestrator.execute_pipeline(
        pipeline_id=pipeline_id,
        input_data={"timestamp": datetime.now(), "asset": "BTC"}
    )
    print(f"Status: {result.status}")
    print(f"Models executed: {result.total_models_executed}")
    
    # 3. Arbitrage Pipeline
    print("\n--- Arbitrage Pipeline ---")
    arb_builder = PipelineTemplates.arbitrage_pipeline(
        exchanges=["binance", "coinbase", "kraken"],
        min_spread=0.001
    )
    
    # 4. Portfolio Optimization Pipeline
    print("\n--- Portfolio Optimization Pipeline ---")
    portfolio_builder = PipelineTemplates.portfolio_optimization_pipeline(
        assets=["BTC", "ETH", "BNB", "SOL", "ADA"],
        rebalance_frequency="daily"
    )
    
    return orchestrator


def demonstrate_parallel_execution():
    """Demonstrate parallel model execution."""
    print("\n" + "="*80)
    print("PARALLEL EXECUTION DEMONSTRATION")
    print("="*80)
    
    orchestrator = ModelOrchestrator(max_workers=5)
    
    # Create models with different execution times
    class SlowModel:
        def __init__(self, name: str, sleep_time: float):
            self.name = name
            self.sleep_time = sleep_time
        
        def predict(self, inputs):
            print(f"{self.name} starting (will take {self.sleep_time}s)...")
            time.sleep(self.sleep_time)
            print(f"{self.name} completed!")
            return {"result": f"{self.name}_output"}
    
    # Create dependency graph:
    #     A
    #    / \
    #   B   C
    #   |\ /|
    #   | X |
    #   |/ \|
    #   D   E
    #    \ /
    #     F
    
    models = [
        ModelNode("A", "Model A", "source", SlowModel("A", 1.0)),
        ModelNode("B", "Model B", "processing", SlowModel("B", 2.0), 
                 dependencies=["A"], inputs={"data": "A"}),
        ModelNode("C", "Model C", "processing", SlowModel("C", 1.5),
                 dependencies=["A"], inputs={"data": "A"}),
        ModelNode("D", "Model D", "processing", SlowModel("D", 1.0),
                 dependencies=["B", "C"], inputs={"b_data": "B", "c_data": "C"}),
        ModelNode("E", "Model E", "processing", SlowModel("E", 1.0),
                 dependencies=["B", "C"], inputs={"b_data": "B", "c_data": "C"}),
        ModelNode("F", "Model F", "output", SlowModel("F", 0.5),
                 dependencies=["D", "E"], inputs={"d_data": "D", "e_data": "E"})
    ]
    
    for model in models:
        orchestrator.register_model(model)
    
    # Create pipeline
    pipeline = PipelineConfig(
        pipeline_id="parallel_demo",
        pipeline_name="Parallel Execution Demo",
        description="Demonstrates parallel model execution",
        stages=[PipelineStage.MODEL_PREDICTION],
        max_parallel_models=4
    )
    
    orchestrator.register_pipeline(pipeline)
    
    # Execute with timing
    print("\nExecuting pipeline with parallel models...")
    print("Expected execution order: A -> (B||C) -> (D||E) -> F")
    print("With parallelism, total time should be ~5.5s instead of 7s")
    
    start_time = time.time()
    result = orchestrator.execute_pipeline("parallel_demo", {})
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f}s")
    print(f"Stage timings: {result.stage_timings}")
    
    return orchestrator


def demonstrate_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n" + "="*80)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*80)
    
    orchestrator = ModelOrchestrator()
    
    # Create models with potential failures
    class UnreliableModel:
        def __init__(self, name: str, failure_rate: float = 0.3):
            self.name = name
            self.failure_rate = failure_rate
            self.attempt = 0
        
        def predict(self, inputs):
            self.attempt += 1
            if np.random.random() < self.failure_rate and self.attempt < 3:
                raise Exception(f"{self.name} failed (attempt {self.attempt})")
            return {"result": f"{self.name}_success"}
    
    # Build pipeline with error handling
    builder = (PipelineBuilder("error_handling_demo", "Demonstrates error handling")
              .add_model("reliable_model", MockModel("ReliableModel"))
              .add_model("unreliable_model", UnreliableModel("UnreliableModel", 0.5))
              .add_model("dependent_model", MockModel("DependentModel"),
                        inputs={"data": "unreliable_model"})
              .with_error_handling("continue"))  # Continue on error
    
    pipeline_id, _ = builder.build(orchestrator)
    
    print("\nExecuting pipeline with unreliable models...")
    result = orchestrator.execute_pipeline(pipeline_id, {})
    
    print(f"\nStatus: {result.status}")
    print(f"Errors encountered: {len(result.errors)}")
    for error in result.errors:
        print(f"  - {error['error']}")
    
    # Test with stop-on-error
    builder.with_error_handling("stop")
    pipeline_id2, _ = builder.build(orchestrator)
    
    print("\nExecuting with stop-on-error policy...")
    result2 = orchestrator.execute_pipeline(pipeline_id2, {})
    print(f"Status: {result2.status}")
    print(f"Models executed: {result2.total_models_executed}")
    
    return orchestrator


def demonstrate_monitoring_and_performance():
    """Demonstrate monitoring and performance tracking."""
    print("\n" + "="*80)
    print("MONITORING AND PERFORMANCE DEMONSTRATION")
    print("="*80)
    
    orchestrator = ModelOrchestrator(monitoring_enabled=True)
    
    # Start monitoring
    orchestrator.start_monitoring()
    
    # Create a pipeline
    builder = PipelineTemplates.ml_ensemble_pipeline(n_models=3)
    pipeline_id, _ = builder.build(orchestrator)
    
    # Execute multiple times to gather statistics
    print("\nExecuting pipeline multiple times for performance analysis...")
    execution_times = []
    
    for i in range(5):
        print(f"\nExecution {i+1}/5")
        start = time.time()
        result = orchestrator.execute_pipeline(pipeline_id, {"iteration": i})
        execution_times.append(time.time() - start)
        
        # Add some variation
        time.sleep(0.5)
    
    # Get model statistics
    print("\n--- Model Performance Statistics ---")
    for model_id in ["ml_ensemble_lstm_model_0", "ml_ensemble_tcn_model_1", "ml_ensemble_transformer_model_2"]:
        full_id = f"{pipeline_id}_{model_id}"
        status = orchestrator.get_model_status(full_id)
        if 'metrics' in status:
            metrics = status['metrics']
            print(f"\n{model_id}:")
            print(f"  Executions: {metrics['executions']}")
            print(f"  Failures: {metrics['failures']}")
            print(f"  Avg Time: {metrics['avg_time']:.3f}s")
    
    # Get pipeline statistics
    print("\n--- Pipeline Performance ---")
    pipeline_status = orchestrator.get_pipeline_status(pipeline_id)
    print(f"Pipeline: {pipeline_status['pipeline_name']}")
    print(f"Recent executions: {len(pipeline_status['recent_executions'])}")
    
    # Visualize performance
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(execution_times, 'b-o')
    plt.xlabel('Execution')
    plt.ylabel('Time (seconds)')
    plt.title('Pipeline Execution Times')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(execution_times)), execution_times)
    plt.xlabel('Execution')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Stop monitoring
    orchestrator.stop_monitoring()
    
    return orchestrator, execution_times


def demonstrate_dynamic_pipelines():
    """Demonstrate dynamic pipeline creation."""
    print("\n" + "="*80)
    print("DYNAMIC PIPELINE CREATION DEMONSTRATION")
    print("="*80)
    
    orchestrator = ModelOrchestrator()
    
    # Register various models
    models = [
        ModelNode("data_1", "Data Source 1", "data_source", MockModel("DataSource1")),
        ModelNode("data_2", "Data Source 2", "data_source", MockModel("DataSource2")),
        ModelNode("feature_1", "Feature Extractor 1", "feature_extraction", MockModel("Features1")),
        ModelNode("model_1", "Prediction Model 1", "prediction", MockModel("Model1")),
        ModelNode("model_2", "Prediction Model 2", "prediction", MockModel("Model2")),
        ModelNode("risk_1", "Risk Manager", "risk_assessment", MockModel("Risk1"))
    ]
    
    for model in models:
        orchestrator.register_model(model)
    
    # Dynamically create pipeline based on requirements
    print("\nCreating dynamic pipeline based on available models...")
    
    # Scenario 1: Use all prediction models
    pipeline_id1 = orchestrator.create_dynamic_pipeline(
        name="all_predictions",
        model_ids=["data_1", "feature_1", "model_1", "model_2"]
    )
    
    print(f"Created pipeline: {pipeline_id1}")
    
    # Scenario 2: Risk-aware pipeline
    pipeline_id2 = orchestrator.create_dynamic_pipeline(
        name="risk_aware",
        model_ids=["data_2", "model_1", "risk_1"],
        stages=[PipelineStage.DATA_INGESTION, 
               PipelineStage.MODEL_PREDICTION,
               PipelineStage.RISK_MANAGEMENT]
    )
    
    print(f"Created pipeline: {pipeline_id2}")
    
    # Export orchestration graph
    orchestrator.export_orchestration_graph(
        pipeline_id1,
        "orchestration_graph.json"
    )
    print("\nExported orchestration graph to orchestration_graph.json")
    
    return orchestrator


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("ADVANCED MODEL ORCHESTRATION SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Basic orchestration
    orchestrator1 = demonstrate_basic_orchestration()
    
    # Pipeline builder
    builder, orchestrator2 = demonstrate_pipeline_builder()
    
    # Pipeline templates
    orchestrator3 = demonstrate_pipeline_templates()
    
    # Parallel execution
    orchestrator4 = demonstrate_parallel_execution()
    
    # Error handling
    orchestrator5 = demonstrate_error_handling()
    
    # Monitoring and performance
    orchestrator6, perf_data = demonstrate_monitoring_and_performance()
    
    # Dynamic pipelines
    orchestrator7 = demonstrate_dynamic_pipelines()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("✓ DAG-based model orchestration")
    print("✓ Fluent pipeline builder interface")
    print("✓ Pre-built pipeline templates")
    print("✓ Parallel model execution")
    print("✓ Error handling and recovery")
    print("✓ Performance monitoring")
    print("✓ Dynamic pipeline creation")
    print("✓ Result caching and optimization")


if __name__ == "__main__":
    main()