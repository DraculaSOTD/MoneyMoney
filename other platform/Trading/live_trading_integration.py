import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import pandas as pd
import numpy as np
import yaml
import os

from exchanges.binance_connector import OrderSide, OrderType
from data_feeds.real_time_manager import RealTimeDataManager, DataType
from trading.execution_engine import ExecutionEngine, RiskLimits
from security.key_manager import SecureKeyManager, KeyProvider, APIKey
from database.repository import TradingRepository
from api.main import app, state

# Import ML system components
from main import CryptoMLTradingSystem
from integrated_trading_system import IntegratedCryptoMLSystem
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
from features.market_microstructure import MarketMicrostructure
from decision_engine import DecisionEngine

# Import monitoring components
from monitoring.alerting import AlertManager, AlertBuilder, AlertType, AlertSeverity
from monitoring.metrics_collector import MetricsCollector

# Import alternative data components
from alternative_data.sentiment_analyzer import CryptoSentimentAnalyzer

logger = logging.getLogger(__name__)

class LiveTradingIntegration:
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.key_manager = SecureKeyManager(
            provider=KeyProvider(self.config.get('security', {}).get('key_provider', 'environment'))
        )
        
        self.data_manager: Optional[RealTimeDataManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.ml_system: Optional[IntegratedCryptoMLSystem] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.repository: Optional[TradingRepository] = None
        self.alert_manager: Optional[AlertManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.sentiment_analyzer: Optional[CryptoSentimentAnalyzer] = None
        
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    async def initialize(self):
        logger.info("Initializing live trading integration...")
        
        # Initialize database
        db_config = self.config.get('database', {})
        if db_config.get('enabled', False):
            db_url = db_config.get('url', 'postgresql://user:pass@localhost/trading')
            self.repository = TradingRepository(db_url)
            
        # Initialize data manager
        redis_url = self.config.get('redis', {}).get('url') if self.config.get('redis', {}).get('enabled', False) else None
        self.data_manager = RealTimeDataManager(redis_url)
        await self.data_manager.start()
        
        # Initialize exchanges
        await self._initialize_exchanges()
        
        # Initialize execution engine
        risk_config = self.config.get('risk_management', {})
        risk_limits = RiskLimits(
            max_position_size=Decimal(str(risk_config.get('max_position_size', 10000))),
            max_positions=risk_config.get('max_positions', 10),
            max_daily_loss=Decimal(str(risk_config.get('max_daily_loss', 1000))),
            max_drawdown=Decimal(str(risk_config.get('max_drawdown', 0.1))),
            max_exposure=Decimal(str(risk_config.get('max_total_exposure', 50000))),
            stop_loss_pct=Decimal(str(risk_config.get('stop_loss_percent', 0.02))),
            take_profit_pct=Decimal(str(risk_config.get('take_profit_percent', 0.05)))
        )
        
        self.execution_engine = ExecutionEngine(risk_limits)
        await self.execution_engine.initialize(self.data_manager)
        
        # Initialize ML system
        await self._initialize_ml_system()
        
        # Initialize decision engine
        self.decision_engine = DecisionEngine(self.config)
        
        # Initialize monitoring and alerting
        self.alert_manager = AlertManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        await self.metrics_collector.start()
        
        # Initialize sentiment analyzer
        if self.config.get('alternative_data', {}).get('enabled', False):
            self.sentiment_analyzer = CryptoSentimentAnalyzer(self.config, self.key_manager)
            logger.info("Sentiment analyzer initialized")
            
            # Start sentiment monitoring
            symbols = self.config.get('trading', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
            crypto_symbols = [s.replace('USDT', '') for s in symbols]  # Convert to base symbols
            sentiment_task = asyncio.create_task(
                self.sentiment_analyzer.start_monitoring(crypto_symbols, interval_minutes=15)
            )
            self.tasks.append(sentiment_task)
        
        # Send startup alert
        await self.alert_manager.send_alert(AlertBuilder.system_started())
        
        # Start data streams
        await self._start_data_streams()
        
        # Register callbacks
        self._register_callbacks()
        
        self.running = True
        logger.info("Live trading integration initialized successfully")
        
    async def _initialize_exchanges(self):
        # Binance
        binance_key = self.key_manager.get_exchange_keys("binance")
        if binance_key:
            testnet = self.config.get('exchanges', {}).get('binance', {}).get('testnet', True)
            await self.data_manager.add_binance_exchange(
                api_key=binance_key.key,
                api_secret=binance_key.secret,
                testnet=testnet
            )
            logger.info(f"Initialized Binance exchange (testnet={testnet})")
        else:
            logger.warning("No Binance API keys found")
            
        # Add more exchanges as implemented
        
    async def _initialize_ml_system(self):
        try:
            # Use the integrated system that combines all models
            self.ml_system = IntegratedCryptoMLSystem(self.config_path)
            
            # Load pre-trained models if available
            models_path = self.config.get('model_storage', {}).get('path', 'models')
            if os.path.exists(models_path):
                self.ml_system.load_models(models_path)
                logger.info("Loaded pre-trained ML models")
            else:
                logger.warning("No pre-trained models found, will train on live data")
                
        except Exception as e:
            logger.error(f"Failed to initialize ML system: {e}")
            
    async def _start_data_streams(self):
        symbols = self.config.get('trading', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
        interval = self.config.get('data', {}).get('kline_interval', '1m')
        
        # Subscribe to market data
        await self.data_manager.start_multiple_streams(
            "binance",
            symbols,
            [DataType.TICKER, DataType.KLINE],
            interval
        )
        
        logger.info(f"Started data streams for {symbols}")
        
    def _register_callbacks(self):
        # Register data processing callbacks
        self.data_manager.register_processing_callback(
            DataType.KLINE,
            self._process_kline_for_ml
        )
        
        # Register order/position callbacks
        self.execution_engine.register_order_callback(self._on_order_update)
        self.execution_engine.register_position_callback(self._on_position_update)
        
    def _process_kline_for_ml(self, market_data) -> Optional[Any]:
        try:
            # Get recent klines for ML features
            df = self.data_manager.get_latest_klines(market_data.symbol, n=200)
            
            if len(df) < 100:  # Need minimum data for indicators
                return None
                
            # Calculate technical indicators
            enhanced_indicators = EnhancedTechnicalIndicators()
            df_with_indicators = enhanced_indicators.calculate_all(df)
            
            # Calculate market microstructure if we have order book data
            # This would require orderbook stream implementation
            
            # Get ML predictions
            if self.ml_system and self.ml_system.is_ready():
                predictions = self.ml_system.predict(df_with_indicators)
                
                # Store predictions in database
                if self.repository:
                    for model_name, prediction in predictions.items():
                        self.repository.save_model_prediction({
                            'model_name': model_name,
                            'symbol': market_data.symbol,
                            'prediction': prediction['prediction'],
                            'confidence': prediction.get('confidence', 0),
                            'features': prediction.get('features', {}),
                            'actual_price': float(market_data.data.close)
                        })
                        
                # Generate trading signals
                asyncio.create_task(self._generate_trading_signals(
                    market_data.symbol, predictions, df_with_indicators
                ))
                
        except Exception as e:
            logger.error(f"Error processing kline for ML: {e}")
            
        return None
        
    async def _generate_trading_signals(self, symbol: str, predictions: Dict[str, Any], 
                                      features_df: pd.DataFrame):
        try:
            # Get sentiment data if available
            if self.sentiment_analyzer:
                crypto_symbol = symbol.replace('USDT', '')  # Convert BTCUSDT -> BTC
                sentiment = await self.sentiment_analyzer.get_combined_sentiment(crypto_symbol)
                
                if sentiment:
                    # Add sentiment to predictions
                    sentiment_trend = self.sentiment_analyzer.get_sentiment_trend(crypto_symbol)
                    predictions['sentiment'] = {
                        'score': sentiment.sentiment_score,
                        'confidence': sentiment.confidence,
                        'volume': sentiment.volume,
                        'trend': sentiment_trend['trend'],
                        'volatility': sentiment_trend['volatility']
                    }
                    
                    # Record sentiment for metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_model_prediction(
                            'sentiment', sentiment.sentiment_score
                        )
            
            # Use decision engine to generate signals
            signal = self.decision_engine.generate_signal(
                symbol=symbol,
                predictions=predictions,
                features=features_df.iloc[-1].to_dict(),
                current_positions=self.execution_engine.get_positions()
            )
            
            if signal and signal['action'] != 'hold':
                await self._execute_signal(symbol, signal)
                
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            
    async def _execute_signal(self, symbol: str, signal: Dict[str, Any]):
        try:
            action = signal['action']
            confidence = signal.get('confidence', 0)
            strategy = signal.get('strategy', 'ml_ensemble')
            
            # Check confidence threshold
            min_confidence = self.config.get('trading', {}).get('min_confidence', 0.6)
            if confidence < min_confidence:
                logger.info(f"Signal confidence {confidence} below threshold {min_confidence}")
                return
                
            # Get current position
            position = self.execution_engine.get_position(
                symbol, 
                OrderSide.BUY if action == 'sell' else OrderSide.SELL
            )
            
            # Calculate position size
            ticker = self.data_manager.get_latest_ticker(symbol)
            if not ticker:
                logger.warning(f"No ticker data for {symbol}")
                return
                
            position_size = self._calculate_position_size(
                symbol, ticker.last_price, confidence
            )
            
            if action == 'buy' and not position:
                # Open long position
                order = await self.execution_engine.place_order(
                    "binance", symbol, OrderSide.BUY, OrderType.MARKET,
                    position_size, strategy=strategy
                )
                if order:
                    await self.alert_manager.send_alert(
                        AlertBuilder.position_opened(symbol, "BUY", float(position_size), float(ticker.last_price))
                    )
                logger.info(f"Opened long position: {symbol} qty={position_size}")
                
            elif action == 'sell' and not position:
                # Open short position (if supported)
                if self.config.get('trading', {}).get('allow_short', False):
                    order = await self.execution_engine.place_order(
                        "binance", symbol, OrderSide.SELL, OrderType.MARKET,
                        position_size, strategy=strategy
                    )
                    logger.info(f"Opened short position: {symbol} qty={position_size}")
                    
            elif action == 'sell' and position and position.side == OrderSide.BUY:
                # Close long position
                order = await self.execution_engine.close_position(
                    symbol, OrderSide.BUY
                )
                if order and position:
                    pnl = float(position.realized_pnl)
                    return_pct = pnl / float(position.entry_price * position.quantity)
                    await self.alert_manager.send_alert(
                        AlertBuilder.position_closed(symbol, pnl, return_pct)
                    )
                logger.info(f"Closed long position: {symbol}")
                
            elif action == 'buy' and position and position.side == OrderSide.SELL:
                # Close short position
                order = await self.execution_engine.close_position(
                    symbol, OrderSide.SELL
                )
                logger.info(f"Closed short position: {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            
    def _calculate_position_size(self, symbol: str, price: Decimal, 
                               confidence: float) -> Decimal:
        # Kelly Criterion or fixed fractional position sizing
        base_position_value = Decimal(str(
            self.config.get('trading', {}).get('base_position_size', 1000)
        ))
        
        # Adjust by confidence
        adjusted_value = base_position_value * Decimal(str(confidence))
        
        # Calculate quantity
        quantity = adjusted_value / price
        
        # Round to exchange precision (simplified, should use exchange info)
        precision = 6  # Most crypto pairs
        quantity = Decimal(str(round(float(quantity), precision)))
        
        return quantity
        
    def _on_order_update(self, order):
        logger.info(f"Order update: {order.order_id} - {order.status.value}")
        
        # Record metrics
        if self.metrics_collector and order.status.value == 'FILLED':
            self.metrics_collector.record_trade({
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': float(order.executed_qty),
                'price': float(order.price),
                'exchange': 'binance'
            })
            
            # Send trade alert
            asyncio.create_task(self.alert_manager.send_alert(
                AlertBuilder.trade_executed(
                    order.symbol, order.side.value, 
                    float(order.executed_qty), float(order.price)
                )
            ))
        
        if self.repository:
            self.repository.save_order({
                'order_id': order.order_id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'price': order.price,
                'quantity': order.quantity,
                'executed_qty': order.executed_qty,
                'timestamp': order.timestamp,
                'update_time': order.update_time
            })
            
    def _on_position_update(self, position):
        logger.info(f"Position update: {position.symbol} - PnL: {position.unrealized_pnl}")
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.update_position({
                'symbol': position.symbol,
                'side': position.side.value,
                'status': position.status.value,
                'quantity': float(position.quantity),
                'entry_price': float(position.entry_price),
                'current_price': float(position.current_price),
                'unrealized_pnl': float(position.unrealized_pnl)
            })
        
        if self.repository:
            self.repository.save_position({
                'symbol': position.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'current_price': position.current_price,
                'entry_time': position.entry_time,
                'status': position.status,
                'strategy': position.strategy
            })
            
    async def start_trading_loop(self):
        while self.running:
            try:
                # Update risk metrics
                await self._update_risk_metrics()
                
                # Check for ML model retraining
                await self._check_model_retraining()
                
                # Clean up old data
                if self.repository:
                    self.repository.cleanup_old_data(days_to_keep=30)
                    
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
                
    async def _update_risk_metrics(self):
        if not self.repository:
            return
            
        metrics = self.execution_engine.get_metrics()
        
        # Calculate additional risk metrics
        positions = self.execution_engine.get_positions()
        total_exposure = sum(p.quantity * p.current_price for p in positions)
        
        # Update metrics collector
        if self.metrics_collector:
            self.metrics_collector.update_portfolio_value(metrics.get('portfolio_value', 0))
            self.metrics_collector.update_pnl(metrics['daily_pnl'], metrics['total_pnl'])
            
            # Check for risk alerts
            drawdown = self.metrics_collector._calculate_current_drawdown()
            if drawdown > self.config.get('risk_management', {}).get('max_drawdown', 0.1):
                await self.alert_manager.send_alert(
                    AlertBuilder.large_drawdown(
                        drawdown, 
                        self.config.get('risk_management', {}).get('max_drawdown', 0.1)
                    )
                )
                
            # Check daily loss limit
            if metrics['daily_pnl'] < -self.config.get('risk_management', {}).get('max_daily_loss', 1000):
                await self.alert_manager.send_alert(
                    AlertBuilder.risk_limit_reached(
                        'daily_loss',
                        abs(metrics['daily_pnl']),
                        self.config.get('risk_management', {}).get('max_daily_loss', 1000)
                    )
                )
        
        self.repository.save_risk_metrics({
            'portfolio_value': metrics.get('portfolio_value', 0),
            'daily_pnl': metrics['daily_pnl'],
            'total_pnl': metrics['total_pnl'],
            'exposure': float(total_exposure),
            'open_positions': len(positions)
        })
        
    async def _check_model_retraining(self):
        # Check if models need retraining based on performance
        if not self.ml_system or not self.repository:
            return
            
        # Get recent prediction accuracy
        recent_predictions = []
        for symbol in self.config.get('trading', {}).get('symbols', []):
            predictions = self.repository.get_recent_predictions(
                'ensemble', symbol, hours=24
            )
            recent_predictions.extend(predictions)
            
        if len(recent_predictions) > 100:
            # Calculate accuracy metrics
            errors = [p.error for p in recent_predictions if p.error is not None]
            if errors:
                avg_error = np.mean(np.abs(errors))
                
                # Retrain if error exceeds threshold
                error_threshold = self.config.get('ml', {}).get('retrain_error_threshold', 0.02)
                if avg_error > error_threshold:
                    logger.info(f"Average prediction error {avg_error} exceeds threshold, retraining...")
                    asyncio.create_task(self._retrain_models())
                    
    async def _retrain_models(self):
        try:
            # Get recent market data for training
            training_data = {}
            for symbol in self.config.get('trading', {}).get('symbols', []):
                df = self.data_manager.get_latest_klines(symbol, n=1000)
                if len(df) > 500:
                    training_data[symbol] = df
                    
            if training_data:
                # Retrain models with recent data
                self.ml_system.retrain(training_data)
                
                # Save updated models
                models_path = self.config.get('model_storage', {}).get('path', 'models')
                self.ml_system.save_models(models_path)
                
                logger.info("Models retrained successfully")
                
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            
    async def shutdown(self):
        logger.info("Shutting down live trading integration...")
        
        self.running = False
        
        # Send shutdown alert
        if self.alert_manager:
            await self.alert_manager.send_alert(AlertBuilder.system_stopped())
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown components
        if self.metrics_collector:
            await self.metrics_collector.stop()
            
        if self.alert_manager:
            await self.alert_manager.close()
            
        if self.data_manager:
            await self.data_manager.stop()
            
        logger.info("Live trading integration shutdown complete")
        
    def get_status(self) -> Dict[str, Any]:
        return {
            'running': self.running,
            'data_manager': self.data_manager is not None,
            'execution_engine': self.execution_engine is not None,
            'ml_system': self.ml_system is not None,
            'ml_ready': self.ml_system.is_ready() if self.ml_system else False,
            'active_streams': self.data_manager.get_market_snapshot() if self.data_manager else {},
            'execution_metrics': self.execution_engine.get_metrics() if self.execution_engine else {},
            'key_manager': self.key_manager.export_public_config()
        }

async def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create integration
    integration = LiveTradingIntegration()
    
    try:
        # Initialize
        await integration.initialize()
        
        # Start trading loop
        trading_task = asyncio.create_task(integration.start_trading_loop())
        
        # Keep running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())