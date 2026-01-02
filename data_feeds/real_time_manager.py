import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import json
import aioredis
from decimal import Decimal

from python_backend.exchanges.binance_connector import BinanceConnector, Ticker, Kline, OrderSide

logger = logging.getLogger(__name__)

class DataType(Enum):
    TICKER = "ticker"
    KLINE = "kline"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    AGG_TRADES = "agg_trades"

@dataclass
class MarketData:
    symbol: str
    data_type: DataType
    data: Any
    timestamp: datetime
    exchange: str = "binance"

@dataclass
class DataBuffer:
    max_size: int = 1000
    data: deque = field(default_factory=deque)
    
    def add(self, item: Any):
        if len(self.data) >= self.max_size:
            self.data.popleft()
        self.data.append(item)
        
    def get_latest(self, n: int = 1) -> List[Any]:
        return list(self.data)[-n:]
        
    def get_all(self) -> List[Any]:
        return list(self.data)
        
    def to_dataframe(self) -> pd.DataFrame:
        if not self.data:
            return pd.DataFrame()
        return pd.DataFrame(self.data)

class RealTimeDataManager:
    def __init__(self, redis_url: Optional[str] = None):
        self.exchanges: Dict[str, Any] = {}
        self.subscriptions: Dict[str, Dict[str, List[Callable]]] = defaultdict(lambda: defaultdict(list))
        self.buffers: Dict[str, Dict[str, DataBuffer]] = defaultdict(dict)
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.processing_callbacks: Dict[str, Callable] = {}
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        
    async def start(self):
        self.running = True
        if self.redis_url:
            self.redis = await aioredis.from_url(self.redis_url)
        logger.info("Real-time data manager started")
        
    async def stop(self):
        self.running = False
        
        for task in self.tasks:
            task.cancel()
            
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'disconnect'):
                await exchange.disconnect()
                
        if self.redis:
            await self.redis.close()
            
        logger.info("Real-time data manager stopped")
        
    def add_exchange(self, name: str, exchange: Any):
        self.exchanges[name] = exchange
        logger.info(f"Added exchange: {name}")
        
    async def add_binance_exchange(self, api_key: Optional[str] = None, 
                                  api_secret: Optional[str] = None, 
                                  testnet: bool = True):
        exchange = BinanceConnector(api_key, api_secret, testnet)
        await exchange.connect()
        self.add_exchange("binance", exchange)
        return exchange
        
    def subscribe(self, symbol: str, data_type: DataType, callback: Callable, 
                 buffer_size: int = 1000):
        key = f"{symbol}:{data_type.value}"
        self.subscriptions[symbol][data_type.value].append(callback)
        
        if key not in self.buffers:
            self.buffers[key] = DataBuffer(max_size=buffer_size)
            
        logger.info(f"Subscribed to {key}")
        
    def unsubscribe(self, symbol: str, data_type: DataType, callback: Callable):
        if symbol in self.subscriptions and data_type.value in self.subscriptions[symbol]:
            if callback in self.subscriptions[symbol][data_type.value]:
                self.subscriptions[symbol][data_type.value].remove(callback)
                logger.info(f"Unsubscribed from {symbol}:{data_type.value}")
                
    async def start_ticker_stream(self, exchange_name: str, symbol: str):
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found")
            
        while self.running:
            try:
                ticker = await exchange.get_ticker(symbol)
                market_data = MarketData(
                    symbol=symbol,
                    data_type=DataType.TICKER,
                    data=ticker,
                    timestamp=ticker.timestamp,
                    exchange=exchange_name
                )
                
                await self._process_data(market_data)
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Ticker stream error for {symbol}: {e}")
                await asyncio.sleep(5)
                
    async def start_kline_stream(self, exchange_name: str, symbol: str, interval: str = '1m'):
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            raise ValueError(f"Exchange {exchange_name} not found")
            
        if isinstance(exchange, BinanceConnector):
            exchange.subscribe('kline', lambda kline: asyncio.create_task(
                self._process_kline(symbol, kline, exchange_name)
            ))
            
            task = asyncio.create_task(exchange.start_kline_stream(symbol, interval))
            self.tasks.append(task)
            
    async def _process_kline(self, symbol: str, kline: Kline, exchange_name: str):
        market_data = MarketData(
            symbol=symbol,
            data_type=DataType.KLINE,
            data=kline,
            timestamp=kline.close_time,
            exchange=exchange_name
        )
        await self._process_data(market_data)
        
    async def _process_data(self, market_data: MarketData):
        key = f"{market_data.symbol}:{market_data.data_type.value}"
        
        if key in self.buffers:
            self.buffers[key].add(market_data.data)
            
        if self.redis:
            await self._store_in_redis(market_data)
            
        if market_data.symbol in self.subscriptions:
            callbacks = self.subscriptions[market_data.symbol].get(market_data.data_type.value, [])
            for callback in callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        if market_data.data_type in self.processing_callbacks:
            try:
                processed = self.processing_callbacks[market_data.data_type](market_data)
                if processed:
                    await self._process_data(processed)
            except Exception as e:
                logger.error(f"Processing callback error: {e}")
                
    async def _store_in_redis(self, market_data: MarketData):
        try:
            key = f"market_data:{market_data.exchange}:{market_data.symbol}:{market_data.data_type.value}"
            
            if market_data.data_type == DataType.TICKER:
                value = {
                    'bid': str(market_data.data.bid_price),
                    'ask': str(market_data.data.ask_price),
                    'last': str(market_data.data.last_price),
                    'volume': str(market_data.data.volume),
                    'timestamp': market_data.timestamp.isoformat()
                }
            elif market_data.data_type == DataType.KLINE:
                value = {
                    'open': str(market_data.data.open),
                    'high': str(market_data.data.high),
                    'low': str(market_data.data.low),
                    'close': str(market_data.data.close),
                    'volume': str(market_data.data.volume),
                    'timestamp': market_data.timestamp.isoformat()
                }
            else:
                value = {'timestamp': market_data.timestamp.isoformat()}
                
            await self.redis.hset(key, mapping=value)
            await self.redis.expire(key, 3600)  # Expire after 1 hour
            
            ts_key = f"{key}:ts"
            await self.redis.zadd(ts_key, {json.dumps(value): market_data.timestamp.timestamp()})
            await self.redis.expire(ts_key, 86400)  # Expire after 24 hours
            
        except Exception as e:
            logger.error(f"Redis storage error: {e}")
            
    def get_buffer_data(self, symbol: str, data_type: DataType, 
                       as_dataframe: bool = False) -> Any:
        key = f"{symbol}:{data_type.value}"
        buffer = self.buffers.get(key)
        
        if not buffer:
            return pd.DataFrame() if as_dataframe else []
            
        if as_dataframe:
            return buffer.to_dataframe()
        return buffer.get_all()
        
    def get_latest_ticker(self, symbol: str) -> Optional[Ticker]:
        key = f"{symbol}:{DataType.TICKER.value}"
        buffer = self.buffers.get(key)
        
        if buffer and buffer.data:
            return buffer.get_latest(1)[0]
        return None
        
    def get_latest_klines(self, symbol: str, n: int = 100) -> pd.DataFrame:
        key = f"{symbol}:{DataType.KLINE.value}"
        buffer = self.buffers.get(key)
        
        if not buffer or not buffer.data:
            return pd.DataFrame()
            
        klines = buffer.get_latest(n)
        df = pd.DataFrame([{
            'timestamp': k.close_time,
            'open': float(k.open),
            'high': float(k.high),
            'low': float(k.low),
            'close': float(k.close),
            'volume': float(k.volume)
        } for k in klines])
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
        
    def register_processing_callback(self, data_type: DataType, callback: Callable):
        self.processing_callbacks[data_type] = callback
        
    async def get_aggregated_data(self, symbols: List[str], 
                                 data_type: DataType) -> Dict[str, Any]:
        result = {}
        for symbol in symbols:
            if data_type == DataType.TICKER:
                ticker = self.get_latest_ticker(symbol)
                if ticker:
                    result[symbol] = {
                        'bid': float(ticker.bid_price),
                        'ask': float(ticker.ask_price),
                        'last': float(ticker.last_price),
                        'volume': float(ticker.volume)
                    }
            elif data_type == DataType.KLINE:
                klines = self.get_latest_klines(symbol, n=1)
                if not klines.empty:
                    latest = klines.iloc[-1]
                    result[symbol] = {
                        'open': latest['open'],
                        'high': latest['high'],
                        'low': latest['low'],
                        'close': latest['close'],
                        'volume': latest['volume']
                    }
        return result
        
    async def start_multiple_streams(self, exchange_name: str, symbols: List[str], 
                                   data_types: List[DataType], interval: str = '1m'):
        tasks = []
        for symbol in symbols:
            for data_type in data_types:
                if data_type == DataType.TICKER:
                    task = asyncio.create_task(
                        self.start_ticker_stream(exchange_name, symbol)
                    )
                elif data_type == DataType.KLINE:
                    task = asyncio.create_task(
                        self.start_kline_stream(exchange_name, symbol, interval)
                    )
                tasks.append(task)
                
        self.tasks.extend(tasks)
        logger.info(f"Started {len(tasks)} data streams")
        
    def get_market_snapshot(self) -> Dict[str, Any]:
        snapshot = {
            'timestamp': datetime.now(),
            'exchanges': list(self.exchanges.keys()),
            'symbols': {},
            'buffer_sizes': {}
        }
        
        for key, buffer in self.buffers.items():
            symbol, data_type = key.split(':')
            if symbol not in snapshot['symbols']:
                snapshot['symbols'][symbol] = []
            snapshot['symbols'][symbol].append(data_type)
            snapshot['buffer_sizes'][key] = len(buffer.data)
            
        return snapshot