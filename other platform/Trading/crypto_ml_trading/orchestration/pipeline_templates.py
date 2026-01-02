"""
Pre-built Pipeline Templates for Common Trading Strategies.

Provides ready-to-use pipeline configurations for various trading approaches.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from pipeline_builder import PipelineBuilder
from model_orchestrator import ModelOrchestrator


class MockModel:
    """Mock model for demonstration."""
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
    
    def predict(self, inputs: Dict[str, Any]) -> Any:
        # Mock prediction
        return np.random.randn(24)  # 24-hour prediction


class PipelineTemplates:
    """Collection of pre-built pipeline templates."""
    
    @staticmethod
    def momentum_trading_pipeline(assets: List[str], 
                                lookback_period: int = 20) -> PipelineBuilder:
        """
        Create a momentum-based trading pipeline.
        
        Features:
        - Technical indicator calculation
        - Momentum signal generation
        - Risk-adjusted position sizing
        - Multi-asset support
        """
        builder = (PipelineBuilder("momentum_trading", 
                                 "Multi-asset momentum trading strategy")
                  .add_data_source("market_data", 
                                 MockModel("MarketDataFetcher", assets=assets))
                  .add_feature_extractor("technical_indicators",
                                       MockModel("TechnicalIndicators", 
                                               indicators=["rsi", "macd", "bb"]),
                                       inputs={"data": "market_data"})
                  .add_model("momentum_signals",
                           MockModel("MomentumSignalGenerator", 
                                   lookback=lookback_period),
                           model_type="signal_generation",
                           inputs={"indicators": "technical_indicators"})
                  .add_risk_manager("position_sizer",
                                  MockModel("PositionSizer", 
                                          max_position=0.1,
                                          volatility_scaling=True),
                                  inputs={"signals": "momentum_signals"})
                  .add_executor("trade_executor",
                              MockModel("TradeExecutor", 
                                      order_type="market"),
                              inputs={"positions": "position_sizer"})
                  .with_parallel_models(3)
                  .with_caching(enabled=True, ttl=60)
                  .with_error_handling("continue"))
        
        return builder
    
    @staticmethod
    def ml_ensemble_pipeline(n_models: int = 5,
                           prediction_horizon: int = 24) -> PipelineBuilder:
        """
        Create an ML ensemble prediction pipeline.
        
        Features:
        - Multiple ML models (LSTM, TCN, Transformer)
        - Feature engineering pipeline
        - Model ensemble with voting
        - Confidence-based position sizing
        """
        builder = (PipelineBuilder("ml_ensemble",
                                 "Machine learning ensemble for price prediction")
                  .add_data_source("price_data",
                                 MockModel("PriceDataFetcher", 
                                         timeframe="1h"))
                  .add_feature_extractor("feature_engineer",
                                       MockModel("FeatureEngineer",
                                               features=["price", "volume", "volatility"]),
                                       inputs={"raw_data": "price_data"}))
        
        # Add multiple ML models
        model_names = []
        for i in range(n_models):
            model_type = ["lstm", "tcn", "transformer", "gru", "cnn"][i % 5]
            model_name = f"{model_type}_model_{i}"
            
            builder.add_model(model_name,
                            MockModel(f"{model_type.upper()}Model",
                                    horizon=prediction_horizon),
                            inputs={"features": "feature_engineer"})
            model_names.append(model_name)
        
        # Add ensemble
        builder = (builder
                  .add_ensemble("ensemble_predictions",
                              model_names,
                              ensemble_method="weighted_average")
                  .add_model("confidence_estimator",
                           MockModel("ConfidenceEstimator"),
                           model_type="uncertainty",
                           inputs={"predictions": "ensemble_predictions"})
                  .add_risk_manager("confidence_sizer",
                                  MockModel("ConfidenceBasedSizer"),
                                  inputs={"predictions": "ensemble_predictions",
                                        "confidence": "confidence_estimator"})
                  .add_executor("ml_trader",
                              MockModel("MLTradeExecutor"),
                              inputs={"signals": "confidence_sizer"})
                  .with_parallel_models(n_models)
                  .with_timeout(600))
        
        return builder
    
    @staticmethod
    def arbitrage_pipeline(exchanges: List[str],
                         min_spread: float = 0.001) -> PipelineBuilder:
        """
        Create a cross-exchange arbitrage pipeline.
        
        Features:
        - Multi-exchange data collection
        - Spread calculation and monitoring
        - Latency-aware execution
        - Risk limits
        """
        builder = PipelineBuilder("arbitrage",
                                "Cross-exchange arbitrage strategy")
        
        # Add data sources for each exchange
        for exchange in exchanges:
            builder.add_data_source(f"{exchange}_data",
                                  MockModel(f"{exchange}DataFetcher",
                                          exchange=exchange))
        
        # Aggregate data
        builder.add_custom("data_aggregator",
                         MockModel("ExchangeDataAggregator"),
                         model_type="aggregation",
                         inputs={f"ex_{i}": f"{ex}_data" 
                                for i, ex in enumerate(exchanges)},
                         outputs=["aggregated_data"])
        
        # Add arbitrage components
        builder = (builder
                  .add_model("spread_calculator",
                           MockModel("SpreadCalculator",
                                   min_spread=min_spread),
                           model_type="analysis",
                           inputs={"data": "data_aggregator"})
                  .add_model("opportunity_detector",
                           MockModel("ArbitrageOpportunityDetector",
                                   latency_threshold=100),
                           model_type="signal_generation",
                           inputs={"spreads": "spread_calculator"})
                  .add_risk_manager("arb_risk_manager",
                                  MockModel("ArbitrageRiskManager",
                                          max_exposure=10000),
                                  inputs={"opportunities": "opportunity_detector"})
                  .add_custom("multi_exchange_executor",
                            MockModel("MultiExchangeExecutor",
                                    exchanges=exchanges),
                            model_type="execution",
                            inputs={"orders": "arb_risk_manager"})
                  .with_parallel_models(len(exchanges))
                  .with_caching(enabled=False)  # Real-time data
                  .with_timeout(30))  # Fast execution required
        
        return builder
    
    @staticmethod
    def sentiment_driven_pipeline(news_sources: List[str],
                                social_platforms: List[str]) -> PipelineBuilder:
        """
        Create a sentiment-driven trading pipeline.
        
        Features:
        - Multi-source sentiment analysis
        - News and social media integration
        - Sentiment aggregation
        - Contrarian/momentum strategies
        """
        builder = PipelineBuilder("sentiment_trading",
                                "Sentiment-driven trading strategy")
        
        # Add news sources
        for source in news_sources:
            builder.add_data_source(f"{source}_news",
                                  MockModel(f"{source}NewsFetcher",
                                          source=source))
        
        # Add social media sources
        for platform in social_platforms:
            builder.add_data_source(f"{platform}_social",
                                  MockModel(f"{platform}SocialFetcher",
                                          platform=platform))
        
        # Sentiment analysis
        all_sources = [f"{s}_news" for s in news_sources] + \
                     [f"{p}_social" for p in social_platforms]
        
        builder.add_custom("text_aggregator",
                         MockModel("TextDataAggregator"),
                         model_type="aggregation",
                         inputs={f"source_{i}": source 
                                for i, source in enumerate(all_sources)},
                         outputs=["aggregated_text"])
        
        builder = (builder
                  .add_model("sentiment_analyzer",
                           MockModel("SentimentAnalyzer",
                                   model="finbert"),
                           model_type="nlp",
                           inputs={"text": "text_aggregator"})
                  .add_model("sentiment_aggregator",
                           MockModel("SentimentAggregator",
                                   aggregation="weighted"),
                           model_type="aggregation",
                           inputs={"sentiments": "sentiment_analyzer"})
                  .add_data_source("price_data",
                                 MockModel("PriceDataFetcher"))
                  .add_model("sentiment_signals",
                           MockModel("SentimentSignalGenerator",
                                   strategy="contrarian"),
                           model_type="signal_generation",
                           inputs={"sentiment": "sentiment_aggregator",
                                 "prices": "price_data"})
                  .add_risk_manager("sentiment_risk",
                                  MockModel("SentimentRiskManager",
                                          sentiment_threshold=0.7),
                                  inputs={"signals": "sentiment_signals"})
                  .add_executor("sentiment_trader",
                              MockModel("SentimentTradeExecutor"),
                              inputs={"positions": "sentiment_risk"})
                  .with_parallel_models(len(all_sources))
                  .with_monitoring(interval=60))
        
        return builder
    
    @staticmethod
    def portfolio_optimization_pipeline(assets: List[str],
                                      rebalance_frequency: str = "daily") -> PipelineBuilder:
        """
        Create a portfolio optimization pipeline.
        
        Features:
        - Multi-asset universe
        - Return prediction
        - Risk modeling
        - Portfolio optimization (Markowitz, Risk Parity, etc.)
        - Rebalancing logic
        """
        builder = (PipelineBuilder("portfolio_optimization",
                                 "Dynamic portfolio optimization strategy")
                  .add_data_source("universe_data",
                                 MockModel("UniverseDataFetcher",
                                         assets=assets))
                  .add_feature_extractor("return_features",
                                       MockModel("ReturnFeatureExtractor",
                                               lookback_days=252),
                                       inputs={"data": "universe_data"}))
        
        # Add return prediction models
        builder = (builder
                  .add_model("return_predictor",
                           MockModel("ReturnPredictor",
                                   method="factor_model"),
                           inputs={"features": "return_features"})
                  .add_model("risk_model",
                           MockModel("RiskModel",
                                   method="covariance_shrinkage"),
                           model_type="risk_modeling",
                           inputs={"returns": "universe_data"})
                  .add_model("correlation_model",
                           MockModel("CorrelationModel",
                                   method="dcc_garch"),
                           model_type="risk_modeling",
                           inputs={"returns": "universe_data"}))
        
        # Portfolio optimization
        builder = (builder
                  .add_custom("optimizer",
                            MockModel("PortfolioOptimizer",
                                    method="mean_variance",
                                    constraints={"max_weight": 0.2}),
                            model_type="optimization",
                            inputs={"expected_returns": "return_predictor",
                                  "risk_matrix": "risk_model",
                                  "correlations": "correlation_model"},
                            outputs=["optimal_weights"])
                  .add_model("rebalance_calculator",
                           MockModel("RebalanceCalculator",
                                   frequency=rebalance_frequency,
                                   threshold=0.05),
                           model_type="rebalancing",
                           inputs={"target_weights": "optimizer",
                                 "current_weights": "universe_data"})
                  .add_risk_manager("portfolio_risk",
                                  MockModel("PortfolioRiskManager",
                                          var_limit=0.02),
                                  inputs={"rebalance": "rebalance_calculator"})
                  .add_executor("portfolio_executor",
                              MockModel("PortfolioExecutor",
                                      execution_algo="twap"),
                              inputs={"orders": "portfolio_risk"})
                  .with_caching(enabled=True, ttl=3600))
        
        return builder
    
    @staticmethod
    def high_frequency_pipeline(symbol: str,
                              tick_data: bool = True) -> PipelineBuilder:
        """
        Create a high-frequency trading pipeline.
        
        Features:
        - Tick data processing
        - Microstructure analysis
        - Low-latency prediction
        - Market making or stat arb
        """
        data_type = "tick" if tick_data else "millisecond"
        
        builder = (PipelineBuilder("high_frequency",
                                 f"High-frequency trading strategy for {symbol}")
                  .add_data_source("hf_data",
                                 MockModel("HighFreqDataFetcher",
                                         symbol=symbol,
                                         data_type=data_type))
                  .add_feature_extractor("microstructure",
                                       MockModel("MicrostructureFeatures",
                                               features=["spread", "depth", "imbalance"]),
                                       inputs={"data": "hf_data"})
                  .add_model("hf_predictor",
                           MockModel("HighFreqPredictor",
                                   horizon_ms=100,
                                   model="lightgbm"),
                           inputs={"features": "microstructure"})
                  .add_model("market_maker",
                           MockModel("MarketMakingStrategy",
                                   spread_target=0.0001),
                           model_type="strategy",
                           inputs={"predictions": "hf_predictor",
                                 "market_data": "hf_data"})
                  .add_risk_manager("hf_risk",
                                  MockModel("HighFreqRiskManager",
                                          position_limit=1000,
                                          max_drawdown=0.005),
                                  inputs={"signals": "market_maker"})
                  .add_executor("hf_executor",
                              MockModel("HighFreqExecutor",
                                      latency_target_us=50),
                              inputs={"orders": "hf_risk"})
                  .with_parallel_models(1)  # Sequential for low latency
                  .with_caching(enabled=False)  # No caching for HFT
                  .with_timeout(1)  # 1 second timeout
                  .with_error_handling("stop"))  # Stop on any error
        
        return builder
    
    @staticmethod
    def regime_adaptive_pipeline(regimes: List[str] = None) -> PipelineBuilder:
        """
        Create a regime-adaptive trading pipeline.
        
        Features:
        - Market regime detection
        - Strategy switching based on regime
        - Adaptive risk management
        - Performance attribution
        """
        if regimes is None:
            regimes = ["trending", "ranging", "volatile", "crisis"]
        
        builder = (PipelineBuilder("regime_adaptive",
                                 "Adaptive strategy based on market regimes")
                  .add_data_source("market_data",
                                 MockModel("MarketDataFetcher"))
                  .add_model("regime_detector",
                           MockModel("MarketRegimeDetector",
                                   regimes=regimes,
                                   method="hidden_markov"),
                           model_type="regime_detection",
                           inputs={"data": "market_data"}))
        
        # Add strategy for each regime
        for regime in regimes:
            builder.add_model(f"{regime}_strategy",
                            MockModel(f"{regime.capitalize()}Strategy"),
                            model_type="strategy",
                            inputs={"data": "market_data"})
        
        # Strategy selector
        strategy_inputs = {"regime": "regime_detector",
                          "market_data": "market_data"}
        for regime in regimes:
            strategy_inputs[f"{regime}_signal"] = f"{regime}_strategy"
        
        builder = (builder
                  .add_custom("strategy_selector",
                            MockModel("StrategySelector"),
                            model_type="selection",
                            inputs=strategy_inputs,
                            outputs=["selected_strategy"])
                  .add_model("regime_confidence",
                           MockModel("RegimeConfidenceEstimator"),
                           model_type="confidence",
                           inputs={"regime": "regime_detector"})
                  .add_risk_manager("adaptive_risk",
                                  MockModel("AdaptiveRiskManager"),
                                  inputs={"strategy": "strategy_selector",
                                        "confidence": "regime_confidence"})
                  .add_executor("adaptive_executor",
                              MockModel("AdaptiveExecutor"),
                              inputs={"orders": "adaptive_risk"})
                  .with_monitoring(interval=15))
        
        return builder
    
    @staticmethod
    def create_custom_pipeline(config: Dict[str, Any]) -> PipelineBuilder:
        """
        Create a custom pipeline from configuration.
        
        Args:
            config: Pipeline configuration dictionary
            
        Returns:
            Configured pipeline builder
        """
        builder = PipelineBuilder(
            config.get('name', 'custom_pipeline'),
            config.get('description', 'Custom trading pipeline')
        )
        
        # Add components from config
        for component in config.get('components', []):
            comp_type = component['type']
            
            if comp_type == 'data_source':
                builder.add_data_source(
                    component['name'],
                    MockModel(component['model'], **component.get('config', {}))
                )
            elif comp_type == 'model':
                builder.add_model(
                    component['name'],
                    MockModel(component['model'], **component.get('config', {})),
                    inputs=component.get('inputs', {})
                )
            elif comp_type == 'risk_manager':
                builder.add_risk_manager(
                    component['name'],
                    MockModel(component['model'], **component.get('config', {})),
                    inputs=component.get('inputs', {})
                )
            elif comp_type == 'executor':
                builder.add_executor(
                    component['name'],
                    MockModel(component['model'], **component.get('config', {})),
                    inputs=component.get('inputs', {})
                )
        
        # Apply configuration
        if 'parallel_models' in config:
            builder.with_parallel_models(config['parallel_models'])
        
        if 'caching' in config:
            builder.with_caching(**config['caching'])
        
        if 'error_handling' in config:
            builder.with_error_handling(config['error_handling'])
        
        return builder