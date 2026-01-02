"""
Live Trading System with GPU Acceleration.

Provides automated trading capabilities with real-time signal generation,
order management, position tracking, and risk management.
"""

import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from decimal import Decimal
from enum import Enum
import logging
import json
import threading
import queue
import time
import uuid
from abc import ABC, abstractmethod

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_feeds.realtime_pipeline import RealTimeDataPipeline, ProcessedData
from data_feeds.exchange_connector import MarketData, OrderBookUpdate, Trade
from utils.gpu_manager import GPUManager, get_gpu_manager
from utils.model_converters import HybridModel, ModelFormat
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    exchange: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Position representation."""
    symbol: str
    exchange: str
    side: OrderSide
    quantity: float
    average_entry_price: float
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_pnl(self, current_price: float):
        """Update P&L calculations."""
        self.current_price = current_price
        self.updated_at = datetime.now(tz=timezone.utc)
        
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (current_price - self.average_entry_price) * self.quantity
        else:  # SELL
            self.unrealized_pnl = (self.average_entry_price - current_price) * self.quantity


@dataclass
class TradingSignal:
    """Trading signal from ML models."""
    symbol: str
    exchange: str
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    predicted_price: Optional[float] = None
    predicted_return: Optional[float] = None
    features: Optional[np.ndarray] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """
    Risk management system for trading.
    
    Features:
    - Position sizing
    - Stop-loss management
    - Portfolio exposure limits
    - Drawdown protection
    """
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_portfolio_exposure: float = 0.8,
                 max_drawdown: float = 0.2,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.05):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_exposure: Maximum total exposure
            max_drawdown: Maximum allowed drawdown
            stop_loss_pct: Default stop-loss percentage
            take_profit_pct: Default take-profit percentage
        """
        self.max_position_size = max_position_size
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Portfolio tracking
        self.portfolio_value = 100000.0  # Starting value
        self.peak_value = self.portfolio_value
        self.current_exposure = 0.0
        self.positions_by_symbol = {}
        
        logger.info(f"Risk manager initialized with max position: {max_position_size}, "
                   f"max exposure: {max_portfolio_exposure}")
    
    def check_signal(self, signal: TradingSignal, current_price: float) -> Dict[str, Any]:
        """
        Check if signal passes risk checks.
        
        Returns:
            Dictionary with 'approved', 'quantity', 'stop_loss', 'take_profit'
        """
        # Check drawdown
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            logger.warning(f"Max drawdown exceeded: {current_drawdown:.2%}")
            return {'approved': False, 'reason': 'max_drawdown_exceeded'}
        
        # Check exposure
        if self.current_exposure >= self.max_portfolio_exposure:
            logger.warning(f"Max exposure reached: {self.current_exposure:.2%}")
            return {'approved': False, 'reason': 'max_exposure_reached'}
        
        # Calculate position size
        position_value = self.portfolio_value * self.max_position_size
        quantity = position_value / current_price
        
        # Adjust for remaining exposure
        remaining_exposure = self.max_portfolio_exposure - self.current_exposure
        if remaining_exposure < self.max_position_size:
            quantity *= (remaining_exposure / self.max_position_size)
        
        # Calculate stop-loss and take-profit
        if signal.action == 'buy':
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:  # sell
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        
        return {
            'approved': True,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_value': position_value
        }
    
    def update_position(self, symbol: str, position: Optional[Position]):
        """Update position tracking."""
        if position:
            self.positions_by_symbol[symbol] = position
        elif symbol in self.positions_by_symbol:
            del self.positions_by_symbol[symbol]
        
        # Recalculate exposure
        self.current_exposure = sum(
            abs(pos.quantity * pos.current_price) / self.portfolio_value
            for pos in self.positions_by_symbol.values()
        )
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value and peak."""
        self.portfolio_value = value
        if value > self.peak_value:
            self.peak_value = value


class ExchangeAPIConnector(ABC):
    """Abstract base class for exchange API connections."""
    
    @abstractmethod
    async def place_order(self, order: Order) -> bool:
        """Place an order on the exchange."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balances."""
        pass


class MockExchangeAPI(ExchangeAPIConnector):
    """Mock exchange API for testing."""
    
    def __init__(self, initial_balance: Dict[str, float] = None):
        self.balances = initial_balance or {'USDT': 100000.0}
        self.orders = {}
        self.filled_orders = []
        self.order_fills = defaultdict(list)
        
    async def place_order(self, order: Order) -> bool:
        """Simulate order placement."""
        order.status = OrderStatus.OPEN
        order.updated_at = datetime.now(tz=timezone.utc)
        self.orders[order.order_id] = order
        
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            await self._fill_order(order, order.price or 0)
        
        return True
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Simulate order cancellation."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now(tz=timezone.utc)
                return True
        return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get simulated order status."""
        return self.orders.get(order_id)
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get simulated account balances."""
        return self.balances.copy()
    
    async def _fill_order(self, order: Order, fill_price: float):
        """Simulate order fill."""
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now(tz=timezone.utc)
        
        # Update balances
        base, quote = order.symbol.split('/')
        
        if order.side == OrderSide.BUY:
            # Deduct quote currency
            cost = order.quantity * fill_price
            self.balances[quote] = self.balances.get(quote, 0) - cost
            # Add base currency
            self.balances[base] = self.balances.get(base, 0) + order.quantity
        else:  # SELL
            # Deduct base currency
            self.balances[base] = self.balances.get(base, 0) - order.quantity
            # Add quote currency
            revenue = order.quantity * fill_price
            self.balances[quote] = self.balances.get(quote, 0) + revenue
        
        self.filled_orders.append(order)
        self.order_fills[order.symbol].append({
            'order_id': order.order_id,
            'side': order.side,
            'quantity': order.quantity,
            'price': fill_price,
            'timestamp': order.updated_at
        })


class SignalGenerator:
    """
    GPU-accelerated signal generation from ML models.
    
    Features:
    - Multi-model ensemble
    - GPU batch processing
    - Confidence scoring
    - Signal filtering
    """
    
    def __init__(self, models: List[Tuple[str, Any]], 
                 confidence_threshold: float = 0.6,
                 enable_gpu: bool = True):
        """
        Initialize signal generator.
        
        Args:
            models: List of (name, model) tuples
            confidence_threshold: Minimum confidence for signals
            enable_gpu: Enable GPU acceleration
        """
        self.models = models
        self.confidence_threshold = confidence_threshold
        self.gpu_manager = get_gpu_manager() if enable_gpu else None
        
        # Convert models to hybrid format for GPU execution
        self.hybrid_models = []
        for name, model in models:
            if hasattr(model, 'predict'):
                hybrid = HybridModel(
                    {ModelFormat.NUMPY: model},
                    preferred_format=ModelFormat.PYTORCH if enable_gpu else ModelFormat.NUMPY
                )
                self.hybrid_models.append((name, hybrid))
            else:
                self.hybrid_models.append((name, model))
        
        # Signal history for analysis
        self.signal_history = deque(maxlen=1000)
        
        logger.info(f"Signal generator initialized with {len(models)} models")
    
    def generate_signals(self, data: ProcessedData) -> List[TradingSignal]:
        """
        Generate trading signals from processed data.
        
        Args:
            data: Processed market data with features
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Prepare features
        features = data.features.reshape(1, -1)  # Add batch dimension
        
        # Get predictions from all models
        predictions = []
        for name, model in self.hybrid_models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(features)
                else:
                    # Assume it's a callable model
                    pred = model(features)
                
                predictions.append((name, pred))
            except Exception as e:
                logger.error(f"Error in model {name}: {e}")
                continue
        
        if not predictions:
            return signals
        
        # Ensemble predictions
        all_preds = np.array([p[1].flatten()[0] for _, p in predictions])
        mean_pred = np.mean(all_preds)
        std_pred = np.std(all_preds)
        
        # Determine action based on predictions
        if mean_pred > 0.5:
            action = 'buy'
            confidence = mean_pred
        elif mean_pred < -0.5:
            action = 'sell'
            confidence = abs(mean_pred)
        else:
            action = 'hold'
            confidence = 1 - abs(mean_pred)
        
        # Adjust confidence based on model agreement (lower std = higher agreement)
        if std_pred > 0:
            confidence *= (1 - min(std_pred, 1))
        
        # Create signal if confidence meets threshold
        if confidence >= self.confidence_threshold or action == 'hold':
            signal = TradingSignal(
                symbol=data.symbol,
                exchange=data.metadata.get('exchange', 'unknown'),
                timestamp=data.timestamp,
                action=action,
                confidence=confidence,
                predicted_return=mean_pred,
                features=data.features,
                model_name='ensemble',
                metadata={
                    'model_predictions': {name: float(pred.flatten()[0]) for name, pred in predictions},
                    'sentiment_score': data.sentiment_score,
                    'volume_ratio': data.volume_profile.get('buy_volume', 0) / 
                                   max(data.volume_profile.get('sell_volume', 1), 1)
                }
            )
            
            signals.append(signal)
            self.signal_history.append(signal)
        
        return signals
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        if not self.signal_history:
            return {}
        
        recent_signals = list(self.signal_history)
        
        # Count by action
        action_counts = defaultdict(int)
        confidence_by_action = defaultdict(list)
        
        for signal in recent_signals:
            action_counts[signal.action] += 1
            confidence_by_action[signal.action].append(signal.confidence)
        
        return {
            'total_signals': len(recent_signals),
            'action_counts': dict(action_counts),
            'avg_confidence': {
                action: np.mean(confidences) 
                for action, confidences in confidence_by_action.items()
            },
            'signal_rate': len(recent_signals) / max(len(self.signal_history), 1)
        }


class LiveTradingSystem:
    """
    Complete live trading system with GPU acceleration.
    
    Features:
    - Real-time data integration
    - GPU-accelerated signal generation
    - Order management
    - Position tracking
    - Risk management
    - Performance monitoring
    """
    
    def __init__(self,
                 symbols: List[str],
                 exchanges: List[str],
                 models: List[Tuple[str, Any]],
                 risk_params: Optional[Dict[str, float]] = None,
                 enable_gpu: bool = True,
                 paper_trading: bool = True):
        """
        Initialize live trading system.
        
        Args:
            symbols: List of symbols to trade
            exchanges: List of exchanges to connect to
            models: List of (name, model) tuples for signal generation
            risk_params: Risk management parameters
            enable_gpu: Enable GPU acceleration
            paper_trading: Use mock exchange for paper trading
        """
        self.symbols = symbols
        self.exchanges = exchanges
        self.enable_gpu = enable_gpu
        self.paper_trading = paper_trading
        
        # Initialize components
        self.gpu_manager = get_gpu_manager() if enable_gpu else None
        
        # Data pipeline
        self.data_pipeline = RealTimeDataPipeline(
            symbols=symbols,
            exchanges=exchanges,
            enable_gpu=enable_gpu
        )
        
        # Signal generator
        self.signal_generator = SignalGenerator(
            models=models,
            enable_gpu=enable_gpu
        )
        
        # Risk manager
        risk_params = risk_params or {}
        self.risk_manager = RiskManager(**risk_params)
        
        # Exchange API
        if paper_trading:
            self.exchange_api = MockExchangeAPI()
        else:
            # Would initialize real exchange APIs here
            raise NotImplementedError("Real exchange APIs not implemented in this demo")
        
        # Order and position tracking
        self.active_orders = {}
        self.positions = {}
        self.order_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(tz=timezone.utc),
            'peak_balance': 100000.0,
            'current_balance': 100000.0
        }
        
        # Control
        self.is_running = False
        self.trading_thread = None
        
        # Callbacks
        self.trade_callbacks = []
        self.signal_callbacks = []
        
        logger.info(f"Live trading system initialized for {symbols} on {exchanges}")
    
    def start(self):
        """Start the live trading system."""
        if self.is_running:
            logger.warning("Trading system already running")
            return
        
        self.is_running = True
        
        # Start data pipeline
        self.data_pipeline.start()
        
        # Register data callback
        self.data_pipeline.register_data_callback(self._handle_processed_data)
        
        # Start trading loop
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            daemon=True
        )
        self.trading_thread.start()
        
        logger.info("Live trading system started")
    
    def stop(self):
        """Stop the live trading system."""
        self.is_running = False
        
        # Stop data pipeline
        self.data_pipeline.stop()
        
        # Cancel all active orders
        asyncio.run(self._cancel_all_orders())
        
        # Wait for thread
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        
        logger.info("Live trading system stopped")
    
    def _handle_processed_data(self, data: ProcessedData):
        """Handle incoming processed data."""
        try:
            # Generate signals
            signals = self.signal_generator.generate_signals(data)
            
            # Process each signal
            for signal in signals:
                self._process_signal(signal, data)
            
            # Update positions with current prices
            self._update_positions(data)
            
        except Exception as e:
            logger.error(f"Error handling data: {e}")
    
    def _process_signal(self, signal: TradingSignal, data: ProcessedData):
        """Process a trading signal."""
        # Emit signal to callbacks
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
        
        # Skip if not actionable
        if signal.action == 'hold':
            return
        
        # Check with risk manager
        risk_check = self.risk_manager.check_signal(signal, data.market_data.last)
        
        if not risk_check['approved']:
            logger.info(f"Signal rejected by risk manager: {risk_check.get('reason')}")
            return
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            exchange=signal.exchange,
            side=OrderSide.BUY if signal.action == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=risk_check['quantity'],
            price=data.market_data.last,
            metadata={
                'signal': signal,
                'stop_loss': risk_check['stop_loss'],
                'take_profit': risk_check['take_profit']
            }
        )
        
        # Place order asynchronously
        asyncio.run_coroutine_threadsafe(
            self._place_order(order),
            asyncio.get_event_loop()
        )
    
    async def _place_order(self, order: Order):
        """Place an order on the exchange."""
        try:
            # Place order
            success = await self.exchange_api.place_order(order)
            
            if success:
                self.active_orders[order.order_id] = order
                logger.info(f"Order placed: {order.order_id} - {order.side.value} "
                          f"{order.quantity:.4f} {order.symbol} @ {order.price:.2f}")
                
                # Update order status
                await self._check_order_status(order.order_id)
            else:
                logger.error(f"Failed to place order: {order.order_id}")
        
        except Exception as e:
            logger.error(f"Error placing order: {e}")
    
    async def _check_order_status(self, order_id: str):
        """Check and update order status."""
        if order_id not in self.active_orders:
            return
        
        order = self.active_orders[order_id]
        
        try:
            # Get updated order from exchange
            updated_order = await self.exchange_api.get_order_status(
                order_id, order.symbol
            )
            
            if updated_order:
                # Update local order
                self.active_orders[order_id] = updated_order
                
                # Handle filled orders
                if updated_order.status == OrderStatus.FILLED:
                    await self._handle_filled_order(updated_order)
                
                # Handle cancelled/rejected orders
                elif updated_order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    del self.active_orders[order_id]
                    self.order_history.append(updated_order)
        
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
    
    async def _handle_filled_order(self, order: Order):
        """Handle a filled order."""
        # Remove from active orders
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        
        # Add to history
        self.order_history.append(order)
        
        # Update or create position
        position_key = f"{order.exchange}:{order.symbol}"
        
        if position_key in self.positions:
            # Update existing position
            position = self.positions[position_key]
            
            if order.side == position.side:
                # Adding to position
                total_quantity = position.quantity + order.filled_quantity
                position.average_entry_price = (
                    (position.average_entry_price * position.quantity + 
                     order.average_fill_price * order.filled_quantity) / 
                    total_quantity
                )
                position.quantity = total_quantity
            else:
                # Reducing or closing position
                if order.filled_quantity >= position.quantity:
                    # Position closed
                    realized_pnl = self._calculate_realized_pnl(position, order)
                    position.realized_pnl += realized_pnl
                    
                    # Update performance metrics
                    self.performance_metrics['total_trades'] += 1
                    self.performance_metrics['total_pnl'] += realized_pnl
                    if realized_pnl > 0:
                        self.performance_metrics['winning_trades'] += 1
                    
                    # Remove position
                    del self.positions[position_key]
                    
                    logger.info(f"Position closed: {order.symbol} - P&L: ${realized_pnl:.2f}")
                else:
                    # Partial close
                    realized_pnl = self._calculate_realized_pnl(
                        position, order, order.filled_quantity
                    )
                    position.realized_pnl += realized_pnl
                    position.quantity -= order.filled_quantity
        else:
            # New position
            position = Position(
                symbol=order.symbol,
                exchange=order.exchange,
                side=order.side,
                quantity=order.filled_quantity,
                average_entry_price=order.average_fill_price,
                current_price=order.average_fill_price
            )
            self.positions[position_key] = position
            
            logger.info(f"Position opened: {order.symbol} - "
                       f"{order.side.value} {order.filled_quantity:.4f} @ {order.average_fill_price:.2f}")
        
        # Update risk manager
        self.risk_manager.update_position(
            position_key, 
            self.positions.get(position_key)
        )
        
        # Emit to callbacks
        for callback in self.trade_callbacks:
            try:
                callback(order, self.positions.get(position_key))
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
        
        # Update balances
        await self._update_balances()
    
    def _calculate_realized_pnl(self, position: Position, order: Order, 
                               quantity: Optional[float] = None) -> float:
        """Calculate realized P&L from closing trade."""
        close_quantity = quantity or position.quantity
        
        if position.side == OrderSide.BUY:
            # Closing long position
            pnl = (order.average_fill_price - position.average_entry_price) * close_quantity
        else:
            # Closing short position
            pnl = (position.average_entry_price - order.average_fill_price) * close_quantity
        
        return pnl
    
    def _update_positions(self, data: ProcessedData):
        """Update position P&L with current prices."""
        position_key = f"{data.metadata.get('exchange', 'unknown')}:{data.symbol}"
        
        if position_key in self.positions:
            position = self.positions[position_key]
            position.update_pnl(data.market_data.last)
            
            # Check stop-loss and take-profit
            order_meta = next(
                (o.metadata for o in self.active_orders.values() 
                 if o.symbol == data.symbol and o.status == OrderStatus.OPEN),
                None
            )
            
            if order_meta:
                stop_loss = order_meta.get('stop_loss')
                take_profit = order_meta.get('take_profit')
                
                should_close = False
                close_reason = ""
                
                if position.side == OrderSide.BUY:
                    if stop_loss and data.market_data.last <= stop_loss:
                        should_close = True
                        close_reason = "stop_loss"
                    elif take_profit and data.market_data.last >= take_profit:
                        should_close = True
                        close_reason = "take_profit"
                else:  # SELL
                    if stop_loss and data.market_data.last >= stop_loss:
                        should_close = True
                        close_reason = "stop_loss"
                    elif take_profit and data.market_data.last <= take_profit:
                        should_close = True
                        close_reason = "take_profit"
                
                if should_close:
                    self._close_position(position, close_reason)
    
    def _close_position(self, position: Position, reason: str):
        """Close a position."""
        # Create closing order
        close_order = Order(
            order_id=str(uuid.uuid4()),
            symbol=position.symbol,
            exchange=position.exchange,
            side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            metadata={'close_reason': reason}
        )
        
        # Place order asynchronously
        asyncio.run_coroutine_threadsafe(
            self._place_order(close_order),
            asyncio.get_event_loop()
        )
        
        logger.info(f"Closing position {position.symbol} due to {reason}")
    
    async def _cancel_all_orders(self):
        """Cancel all active orders."""
        tasks = []
        for order_id, order in self.active_orders.items():
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                tasks.append(
                    self.exchange_api.cancel_order(order_id, order.symbol)
                )
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            cancelled = sum(1 for r in results if r and not isinstance(r, Exception))
            logger.info(f"Cancelled {cancelled} orders")
    
    async def _update_balances(self):
        """Update account balances."""
        try:
            balances = await self.exchange_api.get_account_balance()
            
            # Calculate total value in USDT
            total_value = balances.get('USDT', 0)
            
            # Add value of other holdings
            for symbol, quantity in balances.items():
                if symbol != 'USDT' and quantity > 0:
                    # Get current price
                    price = self.data_pipeline.get_latest_price(f"{symbol}/USDT")
                    if price:
                        total_value += quantity * price
            
            # Update performance metrics
            self.performance_metrics['current_balance'] = total_value
            if total_value > self.performance_metrics['peak_balance']:
                self.performance_metrics['peak_balance'] = total_value
            
            # Update risk manager
            self.risk_manager.update_portfolio_value(total_value)
            
        except Exception as e:
            logger.error(f"Error updating balances: {e}")
    
    def _trading_loop(self):
        """Main trading loop for periodic tasks."""
        while self.is_running:
            try:
                # Check order statuses
                for order_id in list(self.active_orders.keys()):
                    asyncio.run(self._check_order_status(order_id))
                
                # Update balances
                asyncio.run(self._update_balances())
                
                # Log performance
                self._log_performance()
                
                # Sleep
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(5)
    
    def _log_performance(self):
        """Log trading performance."""
        metrics = self.performance_metrics
        runtime = (datetime.now(tz=timezone.utc) - metrics['start_time']).total_seconds() / 3600
        
        win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        total_return = ((metrics['current_balance'] - 100000) / 100000 * 100)
        
        logger.info(f"Performance - Trades: {metrics['total_trades']}, "
                   f"Win Rate: {win_rate:.1f}%, "
                   f"Total P&L: ${metrics['total_pnl']:.2f}, "
                   f"Return: {total_return:.2f}%, "
                   f"Runtime: {runtime:.1f}h")
    
    def register_trade_callback(self, callback: Callable[[Order, Optional[Position]], None]):
        """Register callback for trade events."""
        self.trade_callbacks.append(callback)
    
    def register_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """Register callback for signal events."""
        self.signal_callbacks.append(callback)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        metrics = self.performance_metrics.copy()
        
        # Add current positions
        metrics['open_positions'] = []
        for position in self.positions.values():
            metrics['open_positions'].append({
                'symbol': position.symbol,
                'side': position.side.value,
                'quantity': position.quantity,
                'entry_price': position.average_entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl
            })
        
        # Add risk metrics
        metrics['risk_metrics'] = {
            'current_exposure': self.risk_manager.current_exposure,
            'max_drawdown': (self.risk_manager.peak_value - self.risk_manager.portfolio_value) / 
                           self.risk_manager.peak_value if self.risk_manager.peak_value > 0 else 0
        }
        
        # Add signal stats
        metrics['signal_stats'] = self.signal_generator.get_signal_stats()
        
        # Add data pipeline stats
        metrics['pipeline_stats'] = self.data_pipeline.get_pipeline_stats()
        
        return metrics
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        trades = []
        
        for order in list(self.order_history)[-limit:]:
            if order.status == OrderStatus.FILLED:
                trades.append({
                    'timestamp': order.updated_at.isoformat(),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.filled_quantity,
                    'price': order.average_fill_price,
                    'order_id': order.order_id
                })
        
        return trades