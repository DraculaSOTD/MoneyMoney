import asyncio
import json
import time
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import aiohttp
import websockets
import logging
from decimal import Decimal
import pandas as pd

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class Ticker:
    symbol: str
    bid_price: Decimal
    bid_qty: Decimal
    ask_price: Decimal
    ask_qty: Decimal
    last_price: Decimal
    volume: Decimal
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime

@dataclass
class Order:
    symbol: str
    order_id: str
    client_order_id: str
    side: OrderSide
    type: OrderType
    status: OrderStatus
    price: Optional[Decimal]
    quantity: Decimal
    executed_qty: Decimal
    timestamp: datetime
    update_time: datetime

@dataclass
class Trade:
    symbol: str
    trade_id: str
    order_id: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    commission: Decimal
    commission_asset: str
    timestamp: datetime

@dataclass
class Kline:
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime
    quote_volume: Decimal
    trades: int

class BinanceConnector:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if testnet:
            self.rest_base = "https://testnet.binance.vision"
            self.ws_base = "wss://testnet.binance.vision"
        else:
            self.rest_base = "https://api.binance.com"
            self.ws_base = "wss://stream.binance.com:9443"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.listen_key: Optional[str] = None
        self.callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        
    async def connect(self):
        self.session = aiohttp.ClientSession()
        self.running = True
        
    async def disconnect(self):
        self.running = False
        if self.ws_connection:
            await self.ws_connection.close()
        if self.session:
            await self.session.close()
            
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    def _add_timestamp_and_signature(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._generate_signature(params)
        return params
        
    async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                      signed: bool = False) -> Dict:
        if params is None:
            params = {}
            
        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret required for signed endpoints")
            params = self._add_timestamp_and_signature(params)
            
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
            
        url = f"{self.rest_base}{endpoint}"
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                data = await response.json()
                if response.status != 200:
                    logger.error(f"API error: {response.status} - {data}")
                    raise Exception(f"API error: {data}")
                return data
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
            
    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/api/v3/exchangeInfo', params)
        
    async def get_ticker(self, symbol: str) -> Ticker:
        data = await self._request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
        return Ticker(
            symbol=data['symbol'],
            bid_price=Decimal(data['bidPrice']),
            bid_qty=Decimal(data['bidQty']),
            ask_price=Decimal(data['askPrice']),
            ask_qty=Decimal(data['askQty']),
            last_price=Decimal(data['lastPrice']),
            volume=Decimal(data['volume']),
            high_24h=Decimal(data['highPrice']),
            low_24h=Decimal(data['lowPrice']),
            timestamp=datetime.fromtimestamp(data['closeTime'] / 1000)
        )
        
    async def get_klines(self, symbol: str, interval: str = '1m',
                        limit: int = 500, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Kline]:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        # Add optional time range parameters
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)

        data = await self._request('GET', '/api/v3/klines', params)
        
        klines = []
        for kline_data in data:
            klines.append(Kline(
                open_time=datetime.fromtimestamp(kline_data[0] / 1000),
                open=Decimal(kline_data[1]),
                high=Decimal(kline_data[2]),
                low=Decimal(kline_data[3]),
                close=Decimal(kline_data[4]),
                volume=Decimal(kline_data[5]),
                close_time=datetime.fromtimestamp(kline_data[6] / 1000),
                quote_volume=Decimal(kline_data[7]),
                trades=int(kline_data[8])
            ))
        return klines
        
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return await self._request('GET', '/api/v3/depth', params)
        
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: Decimal, price: Optional[Decimal] = None,
                         time_in_force: str = 'GTC') -> Order:
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': str(quantity),
            'newOrderRespType': 'FULL'
        }
        
        if order_type == OrderType.LIMIT:
            if not price:
                raise ValueError("Price required for limit orders")
            params['price'] = str(price)
            params['timeInForce'] = time_in_force
            
        data = await self._request('POST', '/api/v3/order', params, signed=True)
        
        return Order(
            symbol=data['symbol'],
            order_id=str(data['orderId']),
            client_order_id=data['clientOrderId'],
            side=OrderSide(data['side']),
            type=OrderType(data['type']),
            status=OrderStatus(data['status']),
            price=Decimal(data.get('price', '0')) if data.get('price') else None,
            quantity=Decimal(data['origQty']),
            executed_qty=Decimal(data['executedQty']),
            timestamp=datetime.fromtimestamp(data['transactTime'] / 1000),
            update_time=datetime.fromtimestamp(data['transactTime'] / 1000)
        )
        
    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        data = await self._request('DELETE', '/api/v3/order', params, signed=True)
        
        return Order(
            symbol=data['symbol'],
            order_id=str(data['orderId']),
            client_order_id=data['clientOrderId'],
            side=OrderSide(data['side']),
            type=OrderType(data['type']),
            status=OrderStatus(data['status']),
            price=Decimal(data.get('price', '0')) if data.get('price') else None,
            quantity=Decimal(data['origQty']),
            executed_qty=Decimal(data['executedQty']),
            timestamp=datetime.fromtimestamp(data['transactTime'] / 1000),
            update_time=datetime.fromtimestamp(data['transactTime'] / 1000)
        )
        
    async def get_order(self, symbol: str, order_id: str) -> Order:
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        data = await self._request('GET', '/api/v3/order', params, signed=True)
        
        return Order(
            symbol=data['symbol'],
            order_id=str(data['orderId']),
            client_order_id=data['clientOrderId'],
            side=OrderSide(data['side']),
            type=OrderType(data['type']),
            status=OrderStatus(data['status']),
            price=Decimal(data.get('price', '0')) if data.get('price') else None,
            quantity=Decimal(data['origQty']),
            executed_qty=Decimal(data['executedQty']),
            timestamp=datetime.fromtimestamp(data['time'] / 1000),
            update_time=datetime.fromtimestamp(data['updateTime'] / 1000)
        )
        
    async def get_account(self) -> Dict:
        return await self._request('GET', '/api/v3/account', signed=True)
        
    async def get_listen_key(self) -> str:
        data = await self._request('POST', '/api/v3/userDataStream')
        return data['listenKey']
        
    async def keepalive_listen_key(self, listen_key: str):
        await self._request('PUT', '/api/v3/userDataStream', {'listenKey': listen_key})
        
    async def delete_listen_key(self, listen_key: str):
        await self._request('DELETE', '/api/v3/userDataStream', {'listenKey': listen_key})
        
    def subscribe(self, channel: str, callback: Callable):
        if channel not in self.callbacks:
            self.callbacks[channel] = []
        self.callbacks[channel].append(callback)
        
    def _handle_message(self, channel: str, data: Any):
        if channel in self.callbacks:
            for callback in self.callbacks[channel]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
    async def start_kline_stream(self, symbol: str, interval: str = '1m'):
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"{self.ws_base}/ws/{stream_name}"
        
        async with websockets.connect(url) as websocket:
            while self.running:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data['e'] == 'kline':
                        kline_data = data['k']
                        kline = Kline(
                            open_time=datetime.fromtimestamp(kline_data['t'] / 1000),
                            open=Decimal(kline_data['o']),
                            high=Decimal(kline_data['h']),
                            low=Decimal(kline_data['l']),
                            close=Decimal(kline_data['c']),
                            volume=Decimal(kline_data['v']),
                            close_time=datetime.fromtimestamp(kline_data['T'] / 1000),
                            quote_volume=Decimal(kline_data['q']),
                            trades=int(kline_data['n'])
                        )
                        self._handle_message('kline', kline)
                        
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await asyncio.sleep(5)
                    
    async def start_user_stream(self):
        if not self.api_key:
            raise ValueError("API key required for user stream")
            
        self.listen_key = await self.get_listen_key()
        url = f"{self.ws_base}/ws/{self.listen_key}"
        
        keepalive_task = asyncio.create_task(self._keepalive_loop())
        
        try:
            async with websockets.connect(url) as websocket:
                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data['e'] == 'executionReport':
                            order = Order(
                                symbol=data['s'],
                                order_id=str(data['i']),
                                client_order_id=data['c'],
                                side=OrderSide(data['S']),
                                type=OrderType(data['o']),
                                status=OrderStatus(data['X']),
                                price=Decimal(data['p']) if data['p'] != '0' else None,
                                quantity=Decimal(data['q']),
                                executed_qty=Decimal(data['z']),
                                timestamp=datetime.fromtimestamp(data['O'] / 1000),
                                update_time=datetime.fromtimestamp(data['T'] / 1000)
                            )
                            self._handle_message('order_update', order)
                            
                        elif data['e'] == 'outboundAccountPosition':
                            self._handle_message('account_update', data)
                            
                    except Exception as e:
                        logger.error(f"User stream error: {e}")
                        await asyncio.sleep(5)
                        
        finally:
            keepalive_task.cancel()
            if self.listen_key:
                await self.delete_listen_key(self.listen_key)
                
    async def _keepalive_loop(self):
        while self.running and self.listen_key:
            try:
                await asyncio.sleep(30 * 60)  # 30 minutes
                await self.keepalive_listen_key(self.listen_key)
            except Exception as e:
                logger.error(f"Keepalive error: {e}")
                
    async def get_historical_data(self, symbol: str, interval: str = '1m',
                                 days_back: int = 30) -> pd.DataFrame:
        """
        Fetch historical kline data. Default interval is 1m (1 minute).
        IMPORTANT: For the trading system, ALWAYS use 1m intervals.
        Other timeframes should be aggregated from 1m data.
        """
        klines = await self.get_klines(symbol, interval, limit=1000)
        
        df = pd.DataFrame([{
            'timestamp': k.open_time,
            'open': float(k.open),
            'high': float(k.high),
            'low': float(k.low),
            'close': float(k.close),
            'volume': float(k.volume),
            'quote_volume': float(k.quote_volume),
            'trades': k.trades
        } for k in klines])
        
        df.set_index('timestamp', inplace=True)
        return df