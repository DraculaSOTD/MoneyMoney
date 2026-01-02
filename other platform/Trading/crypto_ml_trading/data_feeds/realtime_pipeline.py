"""
Real-time Data Pipeline.

Unified pipeline for processing market data, alternative data,
and feeding it to ML models with GPU acceleration.
"""

import asyncio
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import logging
import threading
import queue
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_feeds.exchange_connector import ExchangeDataManager, MarketData, OrderBookUpdate, Trade
from data_feeds.alternative_data import AlternativeDataAggregator
from feature_engineering.gpu_indicators import GPUTechnicalIndicators
from feature_engineering.ml_features import MLFeatureEngineer
from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedData:
    """Processed data ready for ML models."""
    symbol: str
    timestamp: datetime
    features: np.ndarray
    market_data: MarketData
    sentiment_score: float
    volume_profile: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeDataPipeline:
    """
    Real-time data pipeline with GPU acceleration.
    
    Features:
    - Multi-source data integration
    - GPU-accelerated feature engineering
    - Real-time sentiment analysis
    - Streaming data processing
    - Model-ready output
    """
    
    def __init__(self,
                 symbols: List[str],
                 exchanges: List[str],
                 lookback_periods: int = 100,
                 update_interval: float = 1.0,
                 enable_gpu: bool = True):
        """
        Initialize real-time data pipeline.
        
        Args:
            symbols: List of symbols to track
            exchanges: List of exchanges to connect to
            lookback_periods: Number of periods for technical indicators
            update_interval: Update interval in seconds
            enable_gpu: Enable GPU acceleration
        """
        self.symbols = symbols
        self.exchanges = exchanges
        self.lookback_periods = lookback_periods
        self.update_interval = update_interval
        self.enable_gpu = enable_gpu
        
        # Initialize components
        self.gpu_manager = get_gpu_manager() if enable_gpu else None
        self.exchange_manager = ExchangeDataManager(gpu_accelerated=enable_gpu)
        self.alt_data_aggregator = AlternativeDataAggregator()
        self.gpu_indicators = GPUTechnicalIndicators(use_gpu=enable_gpu)
        self.feature_engineer = MLFeatureEngineer()
        
        # Data storage
        self.ohlcv_buffers = defaultdict(lambda: {
            'open': deque(maxlen=lookback_periods * 2),
            'high': deque(maxlen=lookback_periods * 2),
            'low': deque(maxlen=lookback_periods * 2),
            'close': deque(maxlen=lookback_periods * 2),
            'volume': deque(maxlen=lookback_periods * 2),
            'timestamp': deque(maxlen=lookback_periods * 2)
        })
        
        self.processed_data_queue = queue.Queue(maxsize=1000)
        self.feature_cache = {}
        
        # State
        self.is_running = False
        self.processing_thread = None
        self.alt_data_thread = None
        
        # Callbacks
        self.data_callbacks = []
        self.model_callbacks = []
        
        # Performance tracking
        self.metrics = {
            'messages_processed': 0,
            'features_computed': 0,
            'processing_time': deque(maxlen=100),
            'queue_depth': 0
        }
        
        # Initialize exchanges
        self._setup_exchanges()
        
        logger.info(f"Real-time pipeline initialized for {symbols} on {exchanges}")
    
    def _setup_exchanges(self):
        """Setup exchange connections."""
        for exchange in self.exchanges:
            self.exchange_manager.add_exchange(exchange, self.symbols)
        
        # Register callbacks
        self.exchange_manager.register_callback(
            'market_data',
            lambda data: self._handle_market_data(data)
        )
        self.exchange_manager.register_callback(
            'order_book',
            lambda data: self._handle_order_book(data)
        )
        self.exchange_manager.register_callback(
            'trade',
            lambda data: self._handle_trade(data)
        )
    
    def start(self):
        """Start the real-time pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        
        # Start exchange connections
        self.exchange_manager.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Start alternative data thread
        self.alt_data_thread = threading.Thread(
            target=self._alt_data_loop,
            daemon=True
        )
        self.alt_data_thread.start()
        
        logger.info("Real-time pipeline started")
    
    def stop(self):
        """Stop the real-time pipeline."""
        self.is_running = False
        
        # Stop exchange connections
        self.exchange_manager.stop()
        
        # Wait for threads
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        if self.alt_data_thread:
            self.alt_data_thread.join(timeout=5)
        
        # Clear GPU memory
        if self.enable_gpu:
            self.gpu_indicators.clear_cache()
            if self.gpu_manager:
                self.gpu_manager.clear_cache()
        
        logger.info("Real-time pipeline stopped")
    
    def _handle_market_data(self, data: MarketData):
        """Handle incoming market data."""
        # Update OHLCV buffer
        symbol_key = f"{data.exchange}:{data.symbol}"
        buffer = self.ohlcv_buffers[symbol_key]
        
        buffer['open'].append(data.open_24h)
        buffer['high'].append(data.high_24h)
        buffer['low'].append(data.low_24h)
        buffer['close'].append(data.last)
        buffer['volume'].append(data.volume_24h)
        buffer['timestamp'].append(data.timestamp)
        
        self.metrics['messages_processed'] += 1
    
    def _handle_order_book(self, data: OrderBookUpdate):
        """Handle order book updates."""
        # Calculate order book features
        if data.bids and data.asks:
            bid_ask_spread = data.asks[0][0] - data.bids[0][0]
            mid_price = (data.asks[0][0] + data.bids[0][0]) / 2
            
            # Store in feature cache
            symbol_key = f"{data.exchange}:{data.symbol}"
            if symbol_key not in self.feature_cache:
                self.feature_cache[symbol_key] = {}
            
            self.feature_cache[symbol_key].update({
                'bid_ask_spread': bid_ask_spread,
                'mid_price': mid_price,
                'order_book_imbalance': self._calculate_order_book_imbalance(data),
                'order_book_timestamp': data.timestamp
            })
    
    def _handle_trade(self, data: Trade):
        """Handle trade updates."""
        # Update volume profile
        symbol_key = f"{data.exchange}:{data.symbol}"
        if symbol_key not in self.feature_cache:
            self.feature_cache[symbol_key] = {}
        
        # Track buy/sell volume
        if 'buy_volume' not in self.feature_cache[symbol_key]:
            self.feature_cache[symbol_key]['buy_volume'] = 0
            self.feature_cache[symbol_key]['sell_volume'] = 0
        
        if data.side == 'buy':
            self.feature_cache[symbol_key]['buy_volume'] += data.size
        else:
            self.feature_cache[symbol_key]['sell_volume'] += data.size
    
    def _calculate_order_book_imbalance(self, order_book: OrderBookUpdate) -> float:
        """Calculate order book imbalance."""
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        # Calculate total bid/ask volumes (top 5 levels)
        bid_volume = sum(size for _, size in order_book.bids[:5])
        ask_volume = sum(size for _, size in order_book.asks[:5])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    for exchange in self.exchanges:
                        symbol_key = f"{exchange}:{symbol}"
                        
                        # Check if we have enough data
                        if len(self.ohlcv_buffers[symbol_key]['close']) >= self.lookback_periods:
                            processed = self._process_symbol_data(symbol, exchange)
                            if processed:
                                # Add to queue
                                try:
                                    self.processed_data_queue.put_nowait(processed)
                                except queue.Full:
                                    # Remove oldest
                                    try:
                                        self.processed_data_queue.get_nowait()
                                        self.processed_data_queue.put_nowait(processed)
                                    except:
                                        pass
                                
                                # Emit to callbacks
                                self._emit_processed_data(processed)
                
                # Update metrics
                self.metrics['queue_depth'] = self.processed_data_queue.qsize()
                
                # Sleep
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(1)
    
    def _process_symbol_data(self, symbol: str, exchange: str) -> Optional[ProcessedData]:
        """Process data for a single symbol."""
        start_time = time.time()
        symbol_key = f"{exchange}:{symbol}"
        
        try:
            # Get OHLCV data
            buffer = self.ohlcv_buffers[symbol_key]
            ohlcv_data = {
                'open': np.array(list(buffer['open'])),
                'high': np.array(list(buffer['high'])),
                'low': np.array(list(buffer['low'])),
                'close': np.array(list(buffer['close'])),
                'volume': np.array(list(buffer['volume']))
            }
            
            # Calculate technical indicators on GPU
            indicators = self.gpu_indicators.calculate_all_indicators(ohlcv_data)
            
            # Get latest market data
            latest_price = self.exchange_manager.get_latest_price(symbol, exchange)
            if latest_price is None:
                return None
            
            # Create feature vector
            features = self._create_feature_vector(
                symbol_key, ohlcv_data, indicators
            )
            
            # Get sentiment score
            sentiment_summary = self.alt_data_aggregator.get_sentiment_summary(symbol, hours=1)
            sentiment_score = sentiment_summary.get('overall_sentiment', 0.0)
            
            # Get volume profile
            volume_profile = {
                'buy_volume': self.feature_cache.get(symbol_key, {}).get('buy_volume', 0),
                'sell_volume': self.feature_cache.get(symbol_key, {}).get('sell_volume', 0),
                'total_volume': ohlcv_data['volume'][-1]
            }
            
            # Create processed data
            processed = ProcessedData(
                symbol=symbol,
                timestamp=datetime.now(tz=timezone.utc),
                features=features,
                market_data=MarketData(
                    exchange=exchange,
                    symbol=symbol,
                    timestamp=datetime.now(tz=timezone.utc),
                    bid=latest_price * 0.999,  # Approximate
                    ask=latest_price * 1.001,  # Approximate
                    last=latest_price,
                    volume_24h=ohlcv_data['volume'][-1],
                    high_24h=np.max(ohlcv_data['high'][-24:]) if len(ohlcv_data['high']) >= 24 else ohlcv_data['high'][-1],
                    low_24h=np.min(ohlcv_data['low'][-24:]) if len(ohlcv_data['low']) >= 24 else ohlcv_data['low'][-1],
                    open_24h=ohlcv_data['open'][-24] if len(ohlcv_data['open']) >= 24 else ohlcv_data['open'][0]
                ),
                sentiment_score=sentiment_score,
                volume_profile=volume_profile,
                metadata={
                    'exchange': exchange,
                    'indicators': list(indicators.keys()),
                    'feature_count': len(features)
                }
            )
            
            # Track performance
            processing_time = time.time() - start_time
            self.metrics['processing_time'].append(processing_time)
            self.metrics['features_computed'] += 1
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing {symbol_key}: {e}")
            return None
    
    def _create_feature_vector(self, symbol_key: str, 
                              ohlcv_data: Dict[str, np.ndarray],
                              indicators: Dict[str, np.ndarray]) -> np.ndarray:
        """Create feature vector for ML models."""
        features = []
        
        # Price features
        close_prices = ohlcv_data['close']
        features.extend([
            close_prices[-1],  # Current price
            (close_prices[-1] - close_prices[-2]) / close_prices[-2],  # Return
            np.std(np.diff(np.log(close_prices[-20:]))),  # Volatility
        ])
        
        # Technical indicators (latest values)
        for indicator_name in ['rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr_14']:
            if indicator_name in indicators:
                features.append(indicators[indicator_name][-1])
        
        # Order book features
        cache = self.feature_cache.get(symbol_key, {})
        features.extend([
            cache.get('bid_ask_spread', 0),
            cache.get('order_book_imbalance', 0),
        ])
        
        # Volume features
        volumes = ohlcv_data['volume']
        features.extend([
            volumes[-1] / np.mean(volumes) if len(volumes) > 0 else 1.0,  # Volume ratio
            cache.get('buy_volume', 0) / (cache.get('buy_volume', 0) + cache.get('sell_volume', 1)),  # Buy ratio
        ])
        
        # Convert to numpy array
        feature_array = np.array(features, dtype=np.float32)
        
        # Handle NaN/Inf values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize if GPU available
        if self.enable_gpu and self.gpu_manager:
            feature_tensor = torch.from_numpy(feature_array)
            feature_tensor = self.gpu_manager.to_device(feature_tensor)
            # Simple normalization
            feature_tensor = (feature_tensor - feature_tensor.mean()) / (feature_tensor.std() + 1e-8)
            feature_array = self.gpu_manager.to_numpy(feature_tensor)
        
        return feature_array
    
    def _alt_data_loop(self):
        """Alternative data fetching loop."""
        while self.is_running:
            try:
                # Fetch alternative data periodically
                asyncio.run(self._fetch_alt_data())
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Alternative data loop error: {e}")
                time.sleep(60)
    
    async def _fetch_alt_data(self):
        """Fetch alternative data asynchronously."""
        try:
            data = await self.alt_data_aggregator.fetch_all_data(
                self.symbols,
                lookback_hours=1
            )
            
            # Analyze sentiment if analyzer available
            if hasattr(self.alt_data_aggregator, 'sentiment_analyzer') and \
               self.alt_data_aggregator.sentiment_analyzer:
                await self.alt_data_aggregator.analyze_sentiment(data)
            
            logger.info(f"Fetched alternative data: {len(data.get('news', []))} news, "
                       f"{len(data.get('twitter', []))} tweets, "
                       f"{len(data.get('reddit', []))} reddit posts")
            
        except Exception as e:
            logger.error(f"Error fetching alternative data: {e}")
    
    def _emit_processed_data(self, data: ProcessedData):
        """Emit processed data to callbacks."""
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Data callback error: {e}")
    
    def register_data_callback(self, callback: Callable[[ProcessedData], None]):
        """Register callback for processed data."""
        self.data_callbacks.append(callback)
    
    def register_model_callback(self, model_fn: Callable[[np.ndarray], np.ndarray],
                               output_callback: Callable[[Dict[str, Any]], None]):
        """
        Register ML model for real-time predictions.
        
        Args:
            model_fn: Function that takes features and returns predictions
            output_callback: Function to handle model outputs
        """
        def model_wrapper(data: ProcessedData):
            try:
                # Make prediction
                prediction = model_fn(data.features.reshape(1, -1))
                
                # Create output
                output = {
                    'symbol': data.symbol,
                    'timestamp': data.timestamp,
                    'prediction': prediction[0] if len(prediction.shape) > 1 else prediction,
                    'features': data.features,
                    'market_data': data.market_data,
                    'sentiment': data.sentiment_score,
                    'metadata': data.metadata
                }
                
                # Call output callback
                output_callback(output)
                
            except Exception as e:
                logger.error(f"Model callback error: {e}")
        
        self.data_callbacks.append(model_wrapper)
    
    def get_latest_data(self, symbol: str, exchange: Optional[str] = None) -> Optional[ProcessedData]:
        """Get latest processed data for a symbol."""
        # Search in queue (most recent first)
        recent_data = []
        
        # Temporarily get all items
        temp_items = []
        try:
            while True:
                item = self.processed_data_queue.get_nowait()
                temp_items.append(item)
                if item.symbol == symbol and (not exchange or item.metadata.get('exchange') == exchange):
                    recent_data.append(item)
        except queue.Empty:
            pass
        
        # Put items back
        for item in temp_items:
            try:
                self.processed_data_queue.put_nowait(item)
            except:
                pass
        
        # Return most recent
        if recent_data:
            return max(recent_data, key=lambda x: x.timestamp)
        return None
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'pipeline': {
                'running': self.is_running,
                'symbols': self.symbols,
                'exchanges': self.exchanges,
                'messages_processed': self.metrics['messages_processed'],
                'features_computed': self.metrics['features_computed'],
                'queue_depth': self.metrics['queue_depth'],
                'avg_processing_time': np.mean(self.metrics['processing_time']) if self.metrics['processing_time'] else 0,
                'gpu_enabled': self.enable_gpu
            },
            'exchanges': self.exchange_manager.get_stats(),
            'alternative_data': self.alt_data_aggregator.get_stats()
        }
        
        if self.enable_gpu and self.gpu_manager:
            stats['gpu'] = {
                'device': str(self.gpu_manager.device),
                'memory_used': self.gpu_manager.get_memory_info()
            }
        
        return stats