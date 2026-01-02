"""
High-Performance Order Execution Engine with GPU Optimization.

Provides advanced order execution strategies, smart order routing,
and execution analysis with GPU acceleration.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from decimal import Decimal
import logging
import heapq
from enum import Enum
import threading
import queue

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from trading.live_trading_system import Order, OrderType, OrderSide, OrderStatus, ExchangeAPIConnector
from data_feeds.exchange_connector import OrderBookUpdate
from utils.gpu_manager import get_gpu_manager
from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"    # Percentage of Volume
    ICEBERG = "iceberg"  # Iceberg orders
    ADAPTIVE = "adaptive"  # Adaptive execution


@dataclass
class ExecutionPlan:
    """Execution plan for an order."""
    order_id: str
    algorithm: ExecutionAlgorithm
    parent_order: Order
    child_orders: List[Order] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    end_time: Optional[datetime] = None
    target_price: Optional[float] = None
    urgency: float = 0.5  # 0 = patient, 1 = urgent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketMicrostructure:
    """Market microstructure data."""
    symbol: str
    exchange: str
    timestamp: datetime
    bid_ask_spread: float
    order_book_imbalance: float
    average_trade_size: float
    trades_per_minute: float
    price_volatility: float
    volume_profile: Dict[float, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderBookAnalyzer:
    """
    GPU-accelerated order book analysis.
    
    Features:
    - Market impact estimation
    - Optimal execution price calculation
    - Liquidity detection
    - Price level clustering
    """
    
    def __init__(self, enable_gpu: bool = True):
        self.gpu_manager = get_gpu_manager() if enable_gpu else None
        self.enable_gpu = enable_gpu
        
        # Historical data
        self.spread_history = defaultdict(lambda: deque(maxlen=1000))
        self.imbalance_history = defaultdict(lambda: deque(maxlen=1000))
        self.trade_size_history = defaultdict(lambda: deque(maxlen=1000))
    
    def analyze_order_book(self, order_book: OrderBookUpdate) -> Dict[str, Any]:
        """
        Analyze order book for execution insights.
        
        Returns:
            Dictionary with analysis results
        """
        symbol_key = f"{order_book.exchange}:{order_book.symbol}"
        
        # Calculate basic metrics
        if order_book.bids and order_book.asks:
            best_bid = order_book.bids[0][0]
            best_ask = order_book.asks[0][0]
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            # Store historical data
            self.spread_history[symbol_key].append(spread)
            
            # Calculate order book imbalance
            imbalance = self._calculate_imbalance(order_book)
            self.imbalance_history[symbol_key].append(imbalance)
            
            # Analyze liquidity
            liquidity_profile = self._analyze_liquidity(order_book)
            
            # Estimate market impact
            impact_bps = self._estimate_market_impact(order_book, spread)
            
            # Detect support/resistance levels
            levels = self._detect_price_levels(order_book)
            
            return {
                'mid_price': mid_price,
                'spread': spread,
                'spread_bps': (spread / mid_price) * 10000,
                'imbalance': imbalance,
                'liquidity_profile': liquidity_profile,
                'market_impact_bps': impact_bps,
                'support_resistance': levels,
                'avg_spread': np.mean(self.spread_history[symbol_key]) if self.spread_history[symbol_key] else spread,
                'spread_volatility': np.std(self.spread_history[symbol_key]) if len(self.spread_history[symbol_key]) > 1 else 0
            }
        
        return {}
    
    def _calculate_imbalance(self, order_book: OrderBookUpdate) -> float:
        """Calculate order book imbalance using GPU."""
        # Get top N levels
        n_levels = 10
        bid_volumes = [size for _, size in order_book.bids[:n_levels]]
        ask_volumes = [size for _, size in order_book.asks[:n_levels]]
        
        if self.enable_gpu and self.gpu_manager:
            # Convert to GPU tensors
            bid_tensor = torch.tensor(bid_volumes, device=self.gpu_manager.device)
            ask_tensor = torch.tensor(ask_volumes, device=self.gpu_manager.device)
            
            # Weighted imbalance (closer levels have more weight)
            weights = torch.exp(-torch.arange(len(bid_volumes), device=self.gpu_manager.device) * 0.1)
            
            weighted_bid_volume = torch.sum(bid_tensor * weights[:len(bid_tensor)])
            weighted_ask_volume = torch.sum(ask_tensor * weights[:len(ask_tensor)])
            
            total_volume = weighted_bid_volume + weighted_ask_volume
            if total_volume > 0:
                imbalance = (weighted_bid_volume - weighted_ask_volume) / total_volume
                return float(imbalance.cpu())
        else:
            # CPU fallback
            weights = np.exp(-np.arange(len(bid_volumes)) * 0.1)
            weighted_bid = np.sum(np.array(bid_volumes) * weights[:len(bid_volumes)])
            weighted_ask = np.sum(np.array(ask_volumes) * weights[:len(ask_volumes)])
            
            total = weighted_bid + weighted_ask
            if total > 0:
                return (weighted_bid - weighted_ask) / total
        
        return 0.0
    
    def _analyze_liquidity(self, order_book: OrderBookUpdate) -> Dict[str, float]:
        """Analyze liquidity at different price levels."""
        liquidity = {
            'bid_depth_1bps': 0.0,
            'ask_depth_1bps': 0.0,
            'bid_depth_5bps': 0.0,
            'ask_depth_5bps': 0.0,
            'bid_depth_10bps': 0.0,
            'ask_depth_10bps': 0.0
        }
        
        if not order_book.bids or not order_book.asks:
            return liquidity
        
        mid_price = (order_book.bids[0][0] + order_book.asks[0][0]) / 2
        
        # Accumulate liquidity at different price levels
        for price, size in order_book.bids:
            distance_bps = ((mid_price - price) / mid_price) * 10000
            if distance_bps <= 1:
                liquidity['bid_depth_1bps'] += size
            if distance_bps <= 5:
                liquidity['bid_depth_5bps'] += size
            if distance_bps <= 10:
                liquidity['bid_depth_10bps'] += size
        
        for price, size in order_book.asks:
            distance_bps = ((price - mid_price) / mid_price) * 10000
            if distance_bps <= 1:
                liquidity['ask_depth_1bps'] += size
            if distance_bps <= 5:
                liquidity['ask_depth_5bps'] += size
            if distance_bps <= 10:
                liquidity['ask_depth_10bps'] += size
        
        return liquidity
    
    def _estimate_market_impact(self, order_book: OrderBookUpdate, spread: float) -> float:
        """Estimate market impact in basis points."""
        if not order_book.bids or not order_book.asks:
            return 0.0
        
        # Simple linear impact model
        # Impact = α * (order_size / avg_liquidity) + β * spread
        
        avg_bid_size = np.mean([size for _, size in order_book.bids[:5]])
        avg_ask_size = np.mean([size for _, size in order_book.asks[:5]])
        avg_liquidity = (avg_bid_size + avg_ask_size) / 2
        
        mid_price = (order_book.bids[0][0] + order_book.asks[0][0]) / 2
        
        # Coefficients (would be calibrated from historical data)
        alpha = 0.1  # Size impact coefficient
        beta = 0.5   # Spread impact coefficient
        
        # Assume typical order size is 10% of average liquidity
        size_ratio = 0.1
        spread_bps = (spread / mid_price) * 10000
        
        impact_bps = alpha * size_ratio * 10000 + beta * spread_bps
        
        return impact_bps
    
    def _detect_price_levels(self, order_book: OrderBookUpdate) -> Dict[str, List[float]]:
        """Detect support and resistance levels from order book."""
        levels = {
            'support': [],
            'resistance': []
        }
        
        # Cluster large orders
        size_threshold = np.percentile([size for _, size in order_book.bids + order_book.asks], 90)
        
        # Find support levels (large bid orders)
        for price, size in order_book.bids:
            if size > size_threshold:
                levels['support'].append(price)
        
        # Find resistance levels (large ask orders)
        for price, size in order_book.asks:
            if size > size_threshold:
                levels['resistance'].append(price)
        
        return levels


class SmartOrderRouter:
    """
    Smart order routing across multiple exchanges.
    
    Features:
    - Best execution routing
    - Multi-exchange arbitrage
    - Latency optimization
    - Fee minimization
    """
    
    def __init__(self, exchanges: Dict[str, ExchangeAPIConnector]):
        self.exchanges = exchanges
        self.order_book_analyzer = OrderBookAnalyzer()
        
        # Exchange characteristics
        self.exchange_fees = {
            'binance': {'maker': 0.0010, 'taker': 0.0010},
            'coinbase': {'maker': 0.0050, 'taker': 0.0050},
            'kraken': {'maker': 0.0016, 'taker': 0.0026}
        }
        
        # Latency estimates (ms)
        self.exchange_latency = {
            'binance': 50,
            'coinbase': 100,
            'kraken': 150
        }
    
    async def route_order(self, order: Order, order_books: Dict[str, OrderBookUpdate]) -> List[Order]:
        """
        Route order to best exchange(s).
        
        Returns:
            List of child orders for each exchange
        """
        # Analyze all available order books
        exchange_scores = {}
        
        for exchange, order_book in order_books.items():
            if exchange not in self.exchanges:
                continue
            
            analysis = self.order_book_analyzer.analyze_order_book(order_book)
            
            # Calculate execution score
            score = self._calculate_exchange_score(
                order, order_book, analysis, exchange
            )
            
            exchange_scores[exchange] = score
        
        # Sort exchanges by score
        sorted_exchanges = sorted(
            exchange_scores.items(), 
            key=lambda x: x[1]['total_score'], 
            reverse=True
        )
        
        # Split order across exchanges if beneficial
        child_orders = self._split_order(order, sorted_exchanges)
        
        return child_orders
    
    def _calculate_exchange_score(self, order: Order, order_book: OrderBookUpdate, 
                                 analysis: Dict[str, Any], exchange: str) -> Dict[str, float]:
        """Calculate exchange routing score."""
        scores = {
            'price_score': 0.0,
            'liquidity_score': 0.0,
            'impact_score': 0.0,
            'fee_score': 0.0,
            'latency_score': 0.0,
            'total_score': 0.0
        }
        
        if not analysis:
            return scores
        
        # Price score (better price = higher score)
        mid_price = analysis.get('mid_price', 0)
        if order.side == OrderSide.BUY:
            # Lower ask is better
            best_ask = order_book.asks[0][0] if order_book.asks else float('inf')
            scores['price_score'] = 1 / (1 + best_ask - mid_price) if mid_price > 0 else 0
        else:
            # Higher bid is better
            best_bid = order_book.bids[0][0] if order_book.bids else 0
            scores['price_score'] = 1 / (1 + mid_price - best_bid) if mid_price > 0 else 0
        
        # Liquidity score
        liquidity_key = 'ask_depth_5bps' if order.side == OrderSide.BUY else 'bid_depth_5bps'
        available_liquidity = analysis.get('liquidity_profile', {}).get(liquidity_key, 0)
        scores['liquidity_score'] = min(available_liquidity / order.quantity, 1.0) if order.quantity > 0 else 0
        
        # Impact score (lower impact is better)
        impact_bps = analysis.get('market_impact_bps', 0)
        scores['impact_score'] = 1 / (1 + impact_bps / 100)
        
        # Fee score (lower fees are better)
        fees = self.exchange_fees.get(exchange.lower(), {'taker': 0.005})
        scores['fee_score'] = 1 / (1 + fees['taker'] * 100)
        
        # Latency score (lower latency is better)
        latency = self.exchange_latency.get(exchange.lower(), 200)
        scores['latency_score'] = 1 / (1 + latency / 100)
        
        # Calculate weighted total score
        weights = {
            'price_score': 0.3,
            'liquidity_score': 0.3,
            'impact_score': 0.2,
            'fee_score': 0.15,
            'latency_score': 0.05
        }
        
        scores['total_score'] = sum(
            scores[key] * weights[key] 
            for key in weights
        )
        
        return scores
    
    def _split_order(self, parent_order: Order, 
                    sorted_exchanges: List[Tuple[str, Dict[str, float]]]) -> List[Order]:
        """Split order across multiple exchanges if beneficial."""
        child_orders = []
        remaining_quantity = parent_order.quantity
        
        # Use top exchanges that meet minimum score threshold
        min_score_threshold = 0.5
        
        for exchange, scores in sorted_exchanges:
            if scores['total_score'] < min_score_threshold or remaining_quantity <= 0:
                break
            
            # Allocate quantity based on liquidity score
            allocation_ratio = min(scores['liquidity_score'], 0.5)  # Max 50% per exchange
            allocated_quantity = min(
                parent_order.quantity * allocation_ratio,
                remaining_quantity
            )
            
            if allocated_quantity > 0:
                child_order = Order(
                    order_id=f"{parent_order.order_id}_{exchange}",
                    symbol=parent_order.symbol,
                    exchange=exchange,
                    side=parent_order.side,
                    order_type=parent_order.order_type,
                    quantity=allocated_quantity,
                    price=parent_order.price,
                    metadata={
                        'parent_order_id': parent_order.order_id,
                        'routing_scores': scores
                    }
                )
                
                child_orders.append(child_order)
                remaining_quantity -= allocated_quantity
        
        # If no suitable exchanges, use the best one for full order
        if not child_orders and sorted_exchanges:
            best_exchange = sorted_exchanges[0][0]
            child_order = Order(
                order_id=f"{parent_order.order_id}_{best_exchange}",
                symbol=parent_order.symbol,
                exchange=best_exchange,
                side=parent_order.side,
                order_type=parent_order.order_type,
                quantity=parent_order.quantity,
                price=parent_order.price,
                metadata={
                    'parent_order_id': parent_order.order_id,
                    'routing_scores': sorted_exchanges[0][1]
                }
            )
            child_orders.append(child_order)
        
        return child_orders


class ExecutionEngine:
    """
    Advanced execution engine with GPU acceleration.
    
    Features:
    - Multiple execution algorithms
    - Smart order routing
    - Real-time adaptation
    - Performance analytics
    """
    
    def __init__(self, 
                 exchanges: Dict[str, ExchangeAPIConnector],
                 enable_gpu: bool = True):
        self.exchanges = exchanges
        self.smart_router = SmartOrderRouter(exchanges)
        self.order_book_analyzer = OrderBookAnalyzer(enable_gpu)
        self.gpu_manager = get_gpu_manager() if enable_gpu else None
        
        # Execution plans
        self.active_plans = {}
        self.completed_plans = deque(maxlen=1000)
        
        # Order books cache
        self.order_books = {}
        
        # Performance tracking
        self.execution_metrics = defaultdict(lambda: {
            'total_orders': 0,
            'filled_orders': 0,
            'avg_slippage_bps': 0,
            'avg_fill_time_ms': 0
        })
        
        # Control
        self.is_running = False
        self.execution_thread = None
        
        logger.info("Execution engine initialized")
    
    async def execute_order(self, order: Order, algorithm: ExecutionAlgorithm = ExecutionAlgorithm.ADAPTIVE) -> ExecutionPlan:
        """
        Execute an order using specified algorithm.
        
        Args:
            order: Order to execute
            algorithm: Execution algorithm to use
            
        Returns:
            ExecutionPlan with child orders
        """
        # Create execution plan
        plan = ExecutionPlan(
            order_id=f"exec_{order.order_id}",
            algorithm=algorithm,
            parent_order=order
        )
        
        # Generate child orders based on algorithm
        if algorithm == ExecutionAlgorithm.TWAP:
            child_orders = await self._execute_twap(order, plan)
        elif algorithm == ExecutionAlgorithm.VWAP:
            child_orders = await self._execute_vwap(order, plan)
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            child_orders = await self._execute_iceberg(order, plan)
        else:  # ADAPTIVE
            child_orders = await self._execute_adaptive(order, plan)
        
        plan.child_orders = child_orders
        self.active_plans[plan.order_id] = plan
        
        # Execute child orders
        for child_order in child_orders:
            await self._place_child_order(child_order)
        
        return plan
    
    async def _execute_twap(self, order: Order, plan: ExecutionPlan) -> List[Order]:
        """Execute order using Time-Weighted Average Price algorithm."""
        # Split order into time slices
        num_slices = 10
        slice_quantity = order.quantity / num_slices
        
        child_orders = []
        
        for i in range(num_slices):
            # Route each slice
            slice_order = Order(
                order_id=f"{order.order_id}_twap_{i}",
                symbol=order.symbol,
                exchange=order.exchange,
                side=order.side,
                order_type=order.order_type,
                quantity=slice_quantity,
                price=order.price,
                metadata={
                    'parent_plan': plan.order_id,
                    'slice': i,
                    'algorithm': 'twap'
                }
            )
            
            # Get routing for this slice
            routed_orders = await self.smart_router.route_order(
                slice_order, self.order_books
            )
            
            child_orders.extend(routed_orders)
        
        return child_orders
    
    async def _execute_vwap(self, order: Order, plan: ExecutionPlan) -> List[Order]:
        """Execute order using Volume-Weighted Average Price algorithm."""
        # Analyze historical volume profile
        symbol_key = f"{order.exchange}:{order.symbol}"
        
        # For demo, use simple volume distribution
        volume_curve = self._get_volume_curve()
        
        child_orders = []
        
        for i, volume_pct in enumerate(volume_curve):
            slice_quantity = order.quantity * volume_pct
            
            if slice_quantity > 0:
                slice_order = Order(
                    order_id=f"{order.order_id}_vwap_{i}",
                    symbol=order.symbol,
                    exchange=order.exchange,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=slice_quantity,
                    price=order.price,
                    metadata={
                        'parent_plan': plan.order_id,
                        'time_slot': i,
                        'volume_pct': volume_pct,
                        'algorithm': 'vwap'
                    }
                )
                
                routed_orders = await self.smart_router.route_order(
                    slice_order, self.order_books
                )
                
                child_orders.extend(routed_orders)
        
        return child_orders
    
    async def _execute_iceberg(self, order: Order, plan: ExecutionPlan) -> List[Order]:
        """Execute order using Iceberg algorithm."""
        # Show only a small portion of the order at a time
        visible_ratio = 0.1  # Show 10% at a time
        visible_quantity = order.quantity * visible_ratio
        
        child_orders = []
        remaining_quantity = order.quantity
        slice_num = 0
        
        while remaining_quantity > 0:
            slice_quantity = min(visible_quantity, remaining_quantity)
            
            slice_order = Order(
                order_id=f"{order.order_id}_iceberg_{slice_num}",
                symbol=order.symbol,
                exchange=order.exchange,
                side=order.side,
                order_type=OrderType.LIMIT,  # Iceberg uses limit orders
                quantity=slice_quantity,
                price=order.price,
                metadata={
                    'parent_plan': plan.order_id,
                    'slice': slice_num,
                    'hidden_quantity': remaining_quantity - slice_quantity,
                    'algorithm': 'iceberg'
                }
            )
            
            routed_orders = await self.smart_router.route_order(
                slice_order, self.order_books
            )
            
            child_orders.extend(routed_orders)
            remaining_quantity -= slice_quantity
            slice_num += 1
        
        return child_orders
    
    async def _execute_adaptive(self, order: Order, plan: ExecutionPlan) -> List[Order]:
        """Execute order using Adaptive algorithm."""
        # Analyze current market conditions
        market_analysis = await self._analyze_market_conditions(order)
        
        # Choose strategy based on conditions
        if market_analysis.get('high_volatility', False):
            # Use smaller slices in volatile markets
            return await self._execute_iceberg(order, plan)
        elif market_analysis.get('low_liquidity', False):
            # Use TWAP for low liquidity
            return await self._execute_twap(order, plan)
        else:
            # Use VWAP for normal conditions
            return await self._execute_vwap(order, plan)
    
    def _get_volume_curve(self) -> List[float]:
        """Get intraday volume distribution curve."""
        # Simplified U-shaped volume curve (high at open/close)
        hours = 24
        curve = []
        
        for hour in range(hours):
            if hour < 2 or hour > 22:
                volume = 0.08  # 8% during first/last hours
            elif hour < 4 or hour > 20:
                volume = 0.06  # 6% during early/late hours
            else:
                volume = 0.034  # ~3.4% during regular hours
            
            curve.append(volume)
        
        # Normalize
        total = sum(curve)
        return [v/total for v in curve]
    
    async def _analyze_market_conditions(self, order: Order) -> Dict[str, Any]:
        """Analyze current market conditions using GPU."""
        symbol_key = f"{order.exchange}:{order.symbol}"
        
        conditions = {
            'high_volatility': False,
            'low_liquidity': False,
            'trending': False,
            'large_spread': False
        }
        
        # Get recent order book analyses
        if symbol_key in self.order_book_analyzer.spread_history:
            spreads = list(self.order_book_analyzer.spread_history[symbol_key])
            
            if len(spreads) > 10:
                # Calculate volatility
                spread_volatility = np.std(spreads) / np.mean(spreads)
                conditions['high_volatility'] = spread_volatility > 0.5
                
                # Check for large spreads
                current_spread = spreads[-1] if spreads else 0
                avg_spread = np.mean(spreads)
                conditions['large_spread'] = current_spread > 2 * avg_spread
        
        # Check order book for liquidity
        if symbol_key in self.order_books:
            order_book = self.order_books[symbol_key]
            analysis = self.order_book_analyzer.analyze_order_book(order_book)
            
            # Check liquidity
            liquidity_key = 'ask_depth_5bps' if order.side == OrderSide.BUY else 'bid_depth_5bps'
            available_liquidity = analysis.get('liquidity_profile', {}).get(liquidity_key, 0)
            conditions['low_liquidity'] = available_liquidity < order.quantity * 2
            
            # Check for trending (persistent imbalance)
            if symbol_key in self.order_book_analyzer.imbalance_history:
                imbalances = list(self.order_book_analyzer.imbalance_history[symbol_key])[-20:]
                if len(imbalances) > 10:
                    avg_imbalance = np.mean(imbalances)
                    conditions['trending'] = abs(avg_imbalance) > 0.3
        
        return conditions
    
    async def _place_child_order(self, order: Order):
        """Place a child order on the exchange."""
        exchange_name = order.exchange.lower()
        if exchange_name in self.exchanges:
            try:
                success = await self.exchanges[exchange_name].place_order(order)
                
                if success:
                    logger.info(f"Child order placed: {order.order_id} - "
                              f"{order.quantity:.4f} {order.symbol} on {order.exchange}")
                else:
                    logger.error(f"Failed to place child order: {order.order_id}")
                    
            except Exception as e:
                logger.error(f"Error placing child order: {e}")
    
    def update_order_book(self, order_book: OrderBookUpdate):
        """Update cached order book."""
        symbol_key = f"{order_book.exchange}:{order_book.symbol}"
        self.order_books[symbol_key] = order_book
    
    def get_execution_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics."""
        if symbol:
            return self.execution_metrics.get(symbol, {})
        
        # Aggregate stats
        total_stats = {
            'total_orders': sum(m['total_orders'] for m in self.execution_metrics.values()),
            'filled_orders': sum(m['filled_orders'] for m in self.execution_metrics.values()),
            'active_plans': len(self.active_plans),
            'completed_plans': len(self.completed_plans)
        }
        
        # Average metrics
        if self.execution_metrics:
            avg_slippage = np.mean([m['avg_slippage_bps'] for m in self.execution_metrics.values()])
            avg_fill_time = np.mean([m['avg_fill_time_ms'] for m in self.execution_metrics.values()])
            
            total_stats['avg_slippage_bps'] = avg_slippage
            total_stats['avg_fill_time_ms'] = avg_fill_time
        
        return total_stats