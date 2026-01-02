"""
Real-time Exchange Data Connector.

Provides WebSocket connections to major cryptocurrency exchanges
for live market data streaming with automatic reconnection and error handling.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict, deque
import logging
import websockets
import aiohttp
import numpy as np
from abc import ABC, abstractmethod
import threading
from queue import Queue, Empty
import ssl
import certifi

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MarketData:
    """Standardized market data across exchanges."""
    exchange: str
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume_24h: float
    high_24h: float
    low_24h: float
    open_24h: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookUpdate:
    """Order book update data."""
    exchange: str
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Individual trade data."""
    exchange: str
    symbol: str
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    trade_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExchangeConnector(ABC):
    """Abstract base class for exchange connectors."""
    
    def __init__(self, name: str, symbols: List[str], 
                 callbacks: Optional[Dict[str, Callable]] = None):
        """
        Initialize exchange connector.
        
        Args:
            name: Exchange name
            symbols: List of symbols to subscribe to
            callbacks: Dictionary of callbacks for different data types
        """
        self.name = name
        self.symbols = symbols
        self.callbacks = callbacks or {}
        self.is_connected = False
        self.ws = None
        self.running = False
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        self.data_queue = Queue(maxsize=10000)
        
        # Performance tracking
        self.message_count = 0
        self.error_count = 0
        self.last_message_time = None
        
    @abstractmethod
    async def _get_ws_url(self) -> str:
        """Get WebSocket URL for the exchange."""
        pass
    
    @abstractmethod
    async def _subscribe(self, ws: websockets.WebSocketClientProtocol):
        """Subscribe to market data streams."""
        pass
    
    @abstractmethod
    async def _process_message(self, message: Dict[str, Any]):
        """Process incoming message from exchange."""
        pass
    
    async def connect(self):
        """Connect to exchange WebSocket."""
        self.running = True
        reconnect_delay = self.reconnect_delay
        
        while self.running:
            try:
                url = await self._get_ws_url()
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                
                async with websockets.connect(url, ssl=ssl_context) as ws:
                    self.ws = ws
                    self.is_connected = True
                    reconnect_delay = self.reconnect_delay  # Reset delay
                    
                    logger.info(f"Connected to {self.name} WebSocket")
                    
                    # Subscribe to streams
                    await self._subscribe(ws)
                    
                    # Message handling loop
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._process_message(data)
                            
                            self.message_count += 1
                            self.last_message_time = time.time()
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error from {self.name}: {e}")
                            self.error_count += 1
                        except Exception as e:
                            logger.error(f"Error processing message from {self.name}: {e}")
                            self.error_count += 1
                            
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed for {self.name}: {e}")
                self.is_connected = False
                
            except Exception as e:
                logger.error(f"WebSocket error for {self.name}: {e}")
                self.is_connected = False
                self.error_count += 1
            
            if self.running:
                logger.info(f"Reconnecting to {self.name} in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self.max_reconnect_delay)
    
    async def disconnect(self):
        """Disconnect from exchange WebSocket."""
        self.running = False
        if self.ws:
            await self.ws.close()
        self.is_connected = False
        logger.info(f"Disconnected from {self.name}")
    
    def _emit_market_data(self, data: MarketData):
        """Emit market data to callbacks."""
        if 'market_data' in self.callbacks:
            try:
                self.callbacks['market_data'](data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Add to queue
        try:
            self.data_queue.put_nowait(data)
        except:
            pass  # Queue full, drop oldest
    
    def _emit_order_book(self, data: OrderBookUpdate):
        """Emit order book update to callbacks."""
        if 'order_book' in self.callbacks:
            try:
                self.callbacks['order_book'](data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _emit_trade(self, data: Trade):
        """Emit trade data to callbacks."""
        if 'trade' in self.callbacks:
            try:
                self.callbacks['trade'](data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            'exchange': self.name,
            'connected': self.is_connected,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.message_count, 1),
            'last_message': self.last_message_time,
            'queue_size': self.data_queue.qsize()
        }


class BinanceConnector(ExchangeConnector):
    """Binance WebSocket connector."""
    
    def __init__(self, symbols: List[str], callbacks: Optional[Dict[str, Callable]] = None):
        super().__init__('Binance', symbols, callbacks)
        self.base_url = 'wss://stream.binance.com:9443/ws'
        
    async def _get_ws_url(self) -> str:
        """Get Binance WebSocket URL."""
        # Convert symbols to streams
        streams = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower().replace('/', '')
            streams.extend([
                f"{symbol_lower}@ticker",
                f"{symbol_lower}@depth20@100ms",
                f"{symbol_lower}@trade"
            ])
        
        return f"{self.base_url}/{'/'.join(streams)}"
    
    async def _subscribe(self, ws: websockets.WebSocketClientProtocol):
        """Binance subscription is done via URL."""
        pass  # Subscription handled in URL
    
    async def _process_message(self, message: Dict[str, Any]):
        """Process Binance message."""
        if 'e' not in message:
            return
        
        event_type = message['e']
        
        if event_type == '24hrTicker':
            # 24hr ticker update
            market_data = MarketData(
                exchange=self.name,
                symbol=self._normalize_symbol(message['s']),
                timestamp=datetime.fromtimestamp(message['E'] / 1000, tz=timezone.utc),
                bid=float(message['b']),
                ask=float(message['a']),
                last=float(message['c']),
                volume_24h=float(message['v']),
                high_24h=float(message['h']),
                low_24h=float(message['l']),
                open_24h=float(message['o']),
                metadata={'count': message['n']}
            )
            self._emit_market_data(market_data)
            
        elif event_type == 'depthUpdate':
            # Order book update
            order_book = OrderBookUpdate(
                exchange=self.name,
                symbol=self._normalize_symbol(message['s']),
                timestamp=datetime.fromtimestamp(message['E'] / 1000, tz=timezone.utc),
                bids=[(float(p), float(q)) for p, q in message['b']],
                asks=[(float(p), float(q)) for p, q in message['a']],
                metadata={'update_id': message['u']}
            )
            self._emit_order_book(order_book)
            
        elif event_type == 'trade':
            # Trade update
            trade = Trade(
                exchange=self.name,
                symbol=self._normalize_symbol(message['s']),
                timestamp=datetime.fromtimestamp(message['T'] / 1000, tz=timezone.utc),
                price=float(message['p']),
                size=float(message['q']),
                side='sell' if message['m'] else 'buy',  # m = true means seller is maker
                trade_id=str(message['t']),
                metadata={'buyer_maker': message['m']}
            )
            self._emit_trade(trade)
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize Binance symbol format."""
        # Binance uses BTCUSDT format
        if symbol.endswith('USDT'):
            return f"{symbol[:-4]}/USDT"
        elif symbol.endswith('BTC'):
            return f"{symbol[:-3]}/BTC"
        elif symbol.endswith('ETH'):
            return f"{symbol[:-3]}/ETH"
        return symbol


class CoinbaseConnector(ExchangeConnector):
    """Coinbase Pro WebSocket connector."""
    
    def __init__(self, symbols: List[str], callbacks: Optional[Dict[str, Callable]] = None):
        super().__init__('Coinbase', symbols, callbacks)
        self.base_url = 'wss://ws-feed.pro.coinbase.com'
        
    async def _get_ws_url(self) -> str:
        """Get Coinbase WebSocket URL."""
        return self.base_url
    
    async def _subscribe(self, ws: websockets.WebSocketClientProtocol):
        """Subscribe to Coinbase channels."""
        # Convert symbols to Coinbase format
        product_ids = [symbol.replace('/', '-') for symbol in self.symbols]
        
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": ["ticker", "level2", "matches"]
        }
        
        await ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to Coinbase channels for {product_ids}")
    
    async def _process_message(self, message: Dict[str, Any]):
        """Process Coinbase message."""
        msg_type = message.get('type')
        
        if msg_type == 'ticker':
            # Ticker update
            market_data = MarketData(
                exchange=self.name,
                symbol=message['product_id'].replace('-', '/'),
                timestamp=datetime.fromisoformat(message['time'].replace('Z', '+00:00')),
                bid=float(message['best_bid']),
                ask=float(message['best_ask']),
                last=float(message['price']),
                volume_24h=float(message['volume_24h']),
                high_24h=float(message['high_24h']),
                low_24h=float(message['low_24h']),
                open_24h=float(message['open_24h']),
                metadata={'side': message.get('side'), 'trade_id': message.get('trade_id')}
            )
            self._emit_market_data(market_data)
            
        elif msg_type == 'l2update':
            # Level 2 order book update
            changes = message['changes']
            bids = []
            asks = []
            
            for side, price, size in changes:
                price = float(price)
                size = float(size)
                if side == 'buy':
                    bids.append((price, size))
                else:
                    asks.append((price, size))
            
            order_book = OrderBookUpdate(
                exchange=self.name,
                symbol=message['product_id'].replace('-', '/'),
                timestamp=datetime.fromisoformat(message['time'].replace('Z', '+00:00')),
                bids=bids,
                asks=asks,
                metadata={'type': 'update'}
            )
            self._emit_order_book(order_book)
            
        elif msg_type == 'match':
            # Trade match
            trade = Trade(
                exchange=self.name,
                symbol=message['product_id'].replace('-', '/'),
                timestamp=datetime.fromisoformat(message['time'].replace('Z', '+00:00')),
                price=float(message['price']),
                size=float(message['size']),
                side=message['side'],
                trade_id=str(message['trade_id']),
                metadata={
                    'maker_order_id': message['maker_order_id'],
                    'taker_order_id': message['taker_order_id']
                }
            )
            self._emit_trade(trade)


class KrakenConnector(ExchangeConnector):
    """Kraken WebSocket connector."""
    
    def __init__(self, symbols: List[str], callbacks: Optional[Dict[str, Callable]] = None):
        super().__init__('Kraken', symbols, callbacks)
        self.base_url = 'wss://ws.kraken.com'
        self.channel_map = {}
        
    async def _get_ws_url(self) -> str:
        """Get Kraken WebSocket URL."""
        return self.base_url
    
    async def _subscribe(self, ws: websockets.WebSocketClientProtocol):
        """Subscribe to Kraken channels."""
        # Convert symbols to Kraken format
        for symbol in self.symbols:
            kraken_symbol = self._to_kraken_symbol(symbol)
            
            # Subscribe to ticker
            await ws.send(json.dumps({
                "event": "subscribe",
                "pair": [kraken_symbol],
                "subscription": {"name": "ticker"}
            }))
            
            # Subscribe to order book
            await ws.send(json.dumps({
                "event": "subscribe",
                "pair": [kraken_symbol],
                "subscription": {"name": "book", "depth": 25}
            }))
            
            # Subscribe to trades
            await ws.send(json.dumps({
                "event": "subscribe",
                "pair": [kraken_symbol],
                "subscription": {"name": "trade"}
            }))
            
        logger.info(f"Subscribed to Kraken channels for {self.symbols}")
    
    async def _process_message(self, message: Union[Dict, List]):
        """Process Kraken message."""
        # Kraken sends arrays for channel data
        if isinstance(message, list) and len(message) >= 4:
            channel_id = message[-2]
            channel_name = message[-1]
            pair = message[3] if len(message) > 4 else None
            
            if channel_name == 'ticker':
                # Ticker data
                ticker_data = message[1]
                market_data = MarketData(
                    exchange=self.name,
                    symbol=self._from_kraken_symbol(pair),
                    timestamp=datetime.now(tz=timezone.utc),
                    bid=float(ticker_data['b'][0]),
                    ask=float(ticker_data['a'][0]),
                    last=float(ticker_data['c'][0]),
                    volume_24h=float(ticker_data['v'][1]),  # 24h volume
                    high_24h=float(ticker_data['h'][1]),   # 24h high
                    low_24h=float(ticker_data['l'][1]),    # 24h low
                    open_24h=float(ticker_data['o'][1]),   # 24h open
                    metadata={'vwap': float(ticker_data['p'][1])}  # 24h VWAP
                )
                self._emit_market_data(market_data)
                
            elif channel_name.startswith('book'):
                # Order book data
                if 'as' in message[1] or 'bs' in message[1]:
                    # Snapshot
                    bids = [(float(p), float(v)) for p, v in message[1].get('bs', [])]
                    asks = [(float(p), float(v)) for p, v in message[1].get('as', [])]
                else:
                    # Update
                    bids = [(float(p), float(v)) for p, v in message[1].get('b', [])]
                    asks = [(float(p), float(v)) for p, v in message[1].get('a', [])]
                
                order_book = OrderBookUpdate(
                    exchange=self.name,
                    symbol=self._from_kraken_symbol(pair),
                    timestamp=datetime.now(tz=timezone.utc),
                    bids=bids,
                    asks=asks,
                    metadata={'channel_id': channel_id}
                )
                self._emit_order_book(order_book)
                
            elif channel_name == 'trade':
                # Trade data
                for trade_data in message[1]:
                    trade = Trade(
                        exchange=self.name,
                        symbol=self._from_kraken_symbol(pair),
                        timestamp=datetime.fromtimestamp(float(trade_data[2]), tz=timezone.utc),
                        price=float(trade_data[0]),
                        size=float(trade_data[1]),
                        side='buy' if trade_data[3] == 'b' else 'sell',
                        trade_id=str(int(float(trade_data[2]) * 1000)),  # Use timestamp as ID
                        metadata={'type': trade_data[4]}
                    )
                    self._emit_trade(trade)
        
        elif isinstance(message, dict) and message.get('event') == 'subscriptionStatus':
            # Store channel mapping
            if message.get('status') == 'subscribed':
                self.channel_map[message.get('channelID')] = {
                    'name': message.get('channelName'),
                    'pair': message.get('pair')
                }
    
    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Kraken format."""
        # BTC/USD -> XBT/USD for Kraken
        base, quote = symbol.split('/')
        if base == 'BTC':
            base = 'XBT'
        return f"{base}/{quote}"
    
    def _from_kraken_symbol(self, symbol: str) -> str:
        """Convert Kraken symbol to standard format."""
        if not symbol:
            return ""
        # XBT/USD -> BTC/USD
        base, quote = symbol.split('/')
        if base == 'XBT':
            base = 'BTC'
        return f"{base}/{quote}"


class ExchangeDataManager:
    """
    Manages multiple exchange connections and provides unified data access.
    
    Features:
    - Multi-exchange connectivity
    - Data aggregation
    - Automatic reconnection
    - Performance monitoring
    """
    
    def __init__(self, gpu_accelerated: bool = True):
        """
        Initialize exchange data manager.
        
        Args:
            gpu_accelerated: Whether to use GPU for data processing
        """
        self.gpu_accelerated = gpu_accelerated
        self.connectors = {}
        self.running = False
        self.event_loop = None
        self.thread = None
        
        # Data storage
        self.market_data_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.order_book_cache = {}
        self.trade_buffer = defaultdict(lambda: deque(maxlen=1000))
        
        # Aggregated data
        self.aggregated_prices = {}
        self.aggregated_volumes = {}
        
        # Callbacks
        self.callbacks = defaultdict(list)
        
        logger.info("Exchange data manager initialized")
    
    def add_exchange(self, exchange: str, symbols: List[str]):
        """
        Add exchange connector.
        
        Args:
            exchange: Exchange name ('binance', 'coinbase', 'kraken')
            symbols: List of symbols to subscribe to
        """
        # Create callbacks for this exchange
        callbacks = {
            'market_data': lambda data: self._handle_market_data(data),
            'order_book': lambda data: self._handle_order_book(data),
            'trade': lambda data: self._handle_trade(data)
        }
        
        # Create connector
        if exchange.lower() == 'binance':
            connector = BinanceConnector(symbols, callbacks)
        elif exchange.lower() == 'coinbase':
            connector = CoinbaseConnector(symbols, callbacks)
        elif exchange.lower() == 'kraken':
            connector = KrakenConnector(symbols, callbacks)
        else:
            raise ValueError(f"Unknown exchange: {exchange}")
        
        self.connectors[exchange] = connector
        logger.info(f"Added {exchange} connector for symbols: {symbols}")
    
    def start(self):
        """Start all exchange connections."""
        if self.running:
            logger.warning("Exchange data manager already running")
            return
        
        self.running = True
        
        # Start event loop in separate thread
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        logger.info("Exchange data manager started")
    
    def stop(self):
        """Stop all exchange connections."""
        self.running = False
        
        if self.event_loop:
            # Schedule disconnect tasks
            for connector in self.connectors.values():
                asyncio.run_coroutine_threadsafe(
                    connector.disconnect(), self.event_loop
                )
            
            # Stop event loop
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Exchange data manager stopped")
    
    def _run_event_loop(self):
        """Run event loop in thread."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        # Create tasks for all connectors
        tasks = []
        for connector in self.connectors.values():
            task = self.event_loop.create_task(connector.connect())
            tasks.append(task)
        
        # Run until stopped
        try:
            self.event_loop.run_until_complete(asyncio.gather(*tasks))
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self.event_loop.close()
    
    def _handle_market_data(self, data: MarketData):
        """Handle market data update."""
        # Store in buffer
        key = f"{data.exchange}:{data.symbol}"
        self.market_data_buffer[key].append(data)
        
        # Update aggregated data
        self._update_aggregated_data(data)
        
        # Emit to callbacks
        for callback in self.callbacks['market_data']:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Market data callback error: {e}")
    
    def _handle_order_book(self, data: OrderBookUpdate):
        """Handle order book update."""
        # Update cache
        key = f"{data.exchange}:{data.symbol}"
        self.order_book_cache[key] = data
        
        # Emit to callbacks
        for callback in self.callbacks['order_book']:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Order book callback error: {e}")
    
    def _handle_trade(self, data: Trade):
        """Handle trade update."""
        # Store in buffer
        key = f"{data.exchange}:{data.symbol}"
        self.trade_buffer[key].append(data)
        
        # Emit to callbacks
        for callback in self.callbacks['trade']:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    def _update_aggregated_data(self, data: MarketData):
        """Update aggregated market data."""
        symbol = data.symbol
        
        if symbol not in self.aggregated_prices:
            self.aggregated_prices[symbol] = {}
            self.aggregated_volumes[symbol] = {}
        
        # Update price and volume
        self.aggregated_prices[symbol][data.exchange] = {
            'bid': data.bid,
            'ask': data.ask,
            'last': data.last,
            'timestamp': data.timestamp
        }
        
        self.aggregated_volumes[symbol][data.exchange] = data.volume_24h
    
    def register_callback(self, data_type: str, callback: Callable):
        """Register callback for data updates."""
        self.callbacks[data_type].append(callback)
    
    def get_latest_price(self, symbol: str, exchange: Optional[str] = None) -> Optional[float]:
        """Get latest price for symbol."""
        if exchange:
            key = f"{exchange}:{symbol}"
            if key in self.market_data_buffer and self.market_data_buffer[key]:
                return self.market_data_buffer[key][-1].last
        else:
            # Get best price across exchanges
            if symbol in self.aggregated_prices:
                prices = []
                for exc_data in self.aggregated_prices[symbol].values():
                    prices.append(exc_data['last'])
                return np.mean(prices) if prices else None
        
        return None
    
    def get_order_book(self, symbol: str, exchange: str) -> Optional[OrderBookUpdate]:
        """Get latest order book."""
        key = f"{exchange}:{symbol}"
        return self.order_book_cache.get(key)
    
    def get_recent_trades(self, symbol: str, exchange: str, limit: int = 100) -> List[Trade]:
        """Get recent trades."""
        key = f"{exchange}:{symbol}"
        if key in self.trade_buffer:
            trades = list(self.trade_buffer[key])
            return trades[-limit:]
        return []
    
    def get_aggregated_volume(self, symbol: str) -> float:
        """Get total volume across all exchanges."""
        if symbol in self.aggregated_volumes:
            return sum(self.aggregated_volumes[symbol].values())
        return 0.0
    
    def get_best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask across all exchanges."""
        best_bid = None
        best_ask = None
        
        for exchange, connector in self.connectors.items():
            order_book = self.get_order_book(symbol, exchange)
            if order_book and order_book.bids and order_book.asks:
                # Get top of book
                exchange_bid = order_book.bids[0][0]
                exchange_ask = order_book.asks[0][0]
                
                # Update best prices
                if best_bid is None or exchange_bid > best_bid:
                    best_bid = exchange_bid
                if best_ask is None or exchange_ask < best_ask:
                    best_ask = exchange_ask
        
        return best_bid, best_ask
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data manager statistics."""
        stats = {
            'exchanges': {},
            'total_market_data': sum(len(buffer) for buffer in self.market_data_buffer.values()),
            'total_trades': sum(len(buffer) for buffer in self.trade_buffer.values()),
            'symbols_tracked': len(set(key.split(':')[1] for key in self.market_data_buffer.keys()))
        }
        
        for exchange, connector in self.connectors.items():
            stats['exchanges'][exchange] = connector.get_stats()
        
        return stats