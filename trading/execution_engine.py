import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import pandas as pd
import numpy as np
from collections import defaultdict

from python_backend.exchanges.binance_connector import (
    BinanceConnector, Order, OrderSide, OrderType, OrderStatus, Trade
)
from data_feeds.real_time_manager import RealTimeDataManager, DataType

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"

class RiskLimitType(Enum):
    POSITION_SIZE = "position_size"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    EXPOSURE = "exposure"

@dataclass
class Position:
    symbol: str
    side: OrderSide
    entry_price: Decimal
    quantity: Decimal
    current_price: Decimal
    entry_time: datetime
    status: PositionStatus = PositionStatus.OPEN
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')
    exchange: str = "binance"
    strategy: Optional[str] = None
    
    def update_price(self, current_price: Decimal):
        self.current_price = current_price
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            
    def close(self, exit_price: Decimal, exit_quantity: Decimal):
        if self.side == OrderSide.BUY:
            pnl = (exit_price - self.entry_price) * exit_quantity
        else:
            pnl = (self.entry_price - exit_price) * exit_quantity
            
        self.realized_pnl += pnl
        self.quantity -= exit_quantity
        
        if self.quantity == 0:
            self.status = PositionStatus.CLOSED
        else:
            self.status = PositionStatus.PARTIAL

@dataclass
class RiskLimits:
    max_position_size: Decimal = Decimal('10000')  # USD value
    max_positions: int = 10
    max_daily_loss: Decimal = Decimal('1000')  # USD
    max_drawdown: Decimal = Decimal('0.1')  # 10%
    max_exposure: Decimal = Decimal('50000')  # Total USD exposure
    stop_loss_pct: Decimal = Decimal('0.02')  # 2%
    take_profit_pct: Decimal = Decimal('0.05')  # 5%

@dataclass
class ExecutionMetrics:
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_volume: Decimal = Decimal('0')
    total_fees: Decimal = Decimal('0')
    avg_slippage: Decimal = Decimal('0')
    daily_pnl: Decimal = Decimal('0')
    open_positions: int = 0

class ExecutionEngine:
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.exchanges: Dict[str, Any] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.risk_limits = risk_limits or RiskLimits()
        self.metrics = ExecutionMetrics()
        self.data_manager: Optional[RealTimeDataManager] = None
        self.balance: Dict[str, Decimal] = defaultdict(Decimal)
        self.running = False
        self.order_callbacks: List[Any] = []
        self.position_callbacks: List[Any] = []
        self.daily_pnl: Decimal = Decimal('0')
        self.peak_balance: Decimal = Decimal('0')
        self._last_reset = datetime.now()
        
    async def initialize(self, data_manager: RealTimeDataManager):
        self.data_manager = data_manager
        self.running = True
        
        for name, exchange in data_manager.exchanges.items():
            self.exchanges[name] = exchange
            
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._daily_reset())
        
        logger.info("Execution engine initialized")
        
    async def place_order(self, exchange_name: str, symbol: str, side: OrderSide,
                         order_type: OrderType, quantity: Decimal,
                         price: Optional[Decimal] = None, strategy: Optional[str] = None) -> Optional[Order]:
        try:
            if not await self._check_risk_limits(symbol, side, quantity, price):
                logger.warning(f"Order rejected due to risk limits: {symbol} {side} {quantity}")
                return None
                
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")
                
            order = await exchange.place_order(symbol, side, order_type, quantity, price)
            
            self.orders[order.order_id] = order
            self.metrics.total_orders += 1
            
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                self.metrics.successful_orders += 1
                await self._update_position(order, strategy)
            elif order.status == OrderStatus.REJECTED:
                self.metrics.failed_orders += 1
                
            for callback in self.order_callbacks:
                callback(order)
                
            logger.info(f"Order placed: {order.order_id} - {symbol} {side} {quantity}")
            return order
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            self.metrics.failed_orders += 1
            return None
            
    async def cancel_order(self, exchange_name: str, symbol: str, order_id: str) -> Optional[Order]:
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")
                
            order = await exchange.cancel_order(symbol, order_id)
            
            if order.order_id in self.orders:
                self.orders[order.order_id] = order
                
            logger.info(f"Order cancelled: {order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return None
            
    async def _check_risk_limits(self, symbol: str, side: OrderSide, 
                                quantity: Decimal, price: Optional[Decimal]) -> bool:
        current_price = price
        if not current_price and self.data_manager:
            ticker = self.data_manager.get_latest_ticker(symbol)
            if ticker:
                current_price = ticker.last_price
                
        if not current_price:
            logger.warning(f"Cannot determine price for {symbol}")
            return False
            
        position_value = quantity * current_price
        
        if position_value > self.risk_limits.max_position_size:
            logger.warning(f"Position size {position_value} exceeds limit {self.risk_limits.max_position_size}")
            return False
            
        open_positions = sum(1 for p in self.positions.values() if p.status == PositionStatus.OPEN)
        if open_positions >= self.risk_limits.max_positions:
            logger.warning(f"Max positions limit reached: {open_positions}")
            return False
            
        if self.daily_pnl < -self.risk_limits.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False
            
        total_exposure = sum(p.quantity * p.current_price for p in self.positions.values() 
                           if p.status == PositionStatus.OPEN)
        if total_exposure + position_value > self.risk_limits.max_exposure:
            logger.warning(f"Exposure limit would be exceeded: {total_exposure + position_value}")
            return False
            
        current_balance = sum(self.balance.values())
        if current_balance > 0 and self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if drawdown > self.risk_limits.max_drawdown:
                logger.warning(f"Drawdown limit reached: {drawdown}")
                return False
                
        return True
        
    async def _update_position(self, order: Order, strategy: Optional[str] = None):
        position_key = f"{order.symbol}:{order.side.value}"
        
        if position_key in self.positions and self.positions[position_key].status == PositionStatus.OPEN:
            position = self.positions[position_key]
            
            new_quantity = position.quantity + order.executed_qty
            new_avg_price = ((position.entry_price * position.quantity) + 
                           (order.price * order.executed_qty)) / new_quantity
                           
            position.quantity = new_quantity
            position.entry_price = new_avg_price
        else:
            position = Position(
                symbol=order.symbol,
                side=order.side,
                entry_price=order.price,
                quantity=order.executed_qty,
                current_price=order.price,
                entry_time=order.timestamp,
                strategy=strategy
            )
            self.positions[position_key] = position
            
        self.metrics.total_volume += order.executed_qty * order.price
        self.metrics.open_positions = sum(1 for p in self.positions.values() 
                                        if p.status == PositionStatus.OPEN)
                                        
        for callback in self.position_callbacks:
            callback(position)
            
    async def close_position(self, symbol: str, side: OrderSide, 
                           quantity: Optional[Decimal] = None) -> Optional[Order]:
        position_key = f"{symbol}:{side.value}"
        position = self.positions.get(position_key)
        
        if not position or position.status != PositionStatus.OPEN:
            logger.warning(f"No open position found for {position_key}")
            return None
            
        close_quantity = quantity or position.quantity
        close_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        
        order = await self.place_order(
            position.exchange, symbol, close_side, 
            OrderType.MARKET, close_quantity
        )
        
        if order and order.status == OrderStatus.FILLED:
            position.close(order.price, order.executed_qty)
            self.daily_pnl += position.realized_pnl
            
        return order
        
    async def _monitor_positions(self):
        while self.running:
            try:
                for position in list(self.positions.values()):
                    if position.status != PositionStatus.OPEN:
                        continue
                        
                    ticker = self.data_manager.get_latest_ticker(position.symbol)
                    if ticker:
                        position.update_price(ticker.last_price)
                        
                        if position.side == OrderSide.BUY:
                            pnl_pct = (position.current_price - position.entry_price) / position.entry_price
                        else:
                            pnl_pct = (position.entry_price - position.current_price) / position.entry_price
                            
                        if pnl_pct <= -self.risk_limits.stop_loss_pct:
                            logger.warning(f"Stop loss triggered for {position.symbol}")
                            await self.close_position(position.symbol, position.side)
                            
                        elif pnl_pct >= self.risk_limits.take_profit_pct:
                            logger.info(f"Take profit triggered for {position.symbol}")
                            await self.close_position(position.symbol, position.side)
                            
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def _daily_reset(self):
        while self.running:
            now = datetime.now()
            if now.date() > self._last_reset.date():
                self.daily_pnl = Decimal('0')
                self._last_reset = now
                logger.info("Daily P&L reset")
                
            await asyncio.sleep(3600)  # Check every hour
            
    def get_positions(self, status: Optional[PositionStatus] = None) -> List[Position]:
        if status:
            return [p for p in self.positions.values() if p.status == status]
        return list(self.positions.values())
        
    def get_position(self, symbol: str, side: OrderSide) -> Optional[Position]:
        position_key = f"{symbol}:{side.value}"
        return self.positions.get(position_key)
        
    def get_total_pnl(self) -> Decimal:
        realized = sum(p.realized_pnl for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values() 
                        if p.status == PositionStatus.OPEN)
        return realized + unrealized
        
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'total_orders': self.metrics.total_orders,
            'successful_orders': self.metrics.successful_orders,
            'failed_orders': self.metrics.failed_orders,
            'success_rate': (self.metrics.successful_orders / self.metrics.total_orders 
                           if self.metrics.total_orders > 0 else 0),
            'total_volume': float(self.metrics.total_volume),
            'total_fees': float(self.metrics.total_fees),
            'daily_pnl': float(self.daily_pnl),
            'total_pnl': float(self.get_total_pnl()),
            'open_positions': self.metrics.open_positions,
            'positions': [
                {
                    'symbol': p.symbol,
                    'side': p.side.value,
                    'quantity': float(p.quantity),
                    'entry_price': float(p.entry_price),
                    'current_price': float(p.current_price),
                    'unrealized_pnl': float(p.unrealized_pnl),
                    'status': p.status.value
                }
                for p in self.get_positions(PositionStatus.OPEN)
            ]
        }
        
    def register_order_callback(self, callback: Any):
        self.order_callbacks.append(callback)
        
    def register_position_callback(self, callback: Any):
        self.position_callbacks.append(callback)
        
    async def update_balances(self, exchange_name: str):
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return
                
            account = await exchange.get_account()
            for balance in account.get('balances', []):
                asset = balance['asset']
                free = Decimal(balance['free'])
                locked = Decimal(balance['locked'])
                self.balance[asset] = free + locked
                
            total_balance = sum(self.balance.values())
            if total_balance > self.peak_balance:
                self.peak_balance = total_balance
                
        except Exception as e:
            logger.error(f"Balance update failed: {e}")