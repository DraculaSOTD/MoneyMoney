"""
Sentiment-Based Trading Strategies.

Implements various trading strategies that utilize sentiment analysis
for signal generation and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging

from enhanced_sentiment_analyzer import SentimentSignal
from stream_processor import SentimentStreamProcessor

logger = logging.getLogger(__name__)


@dataclass
class TradingPosition:
    """Represents a trading position."""
    asset: str
    side: str  # long or short
    entry_price: float
    size: float
    entry_time: datetime
    entry_sentiment: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """Enhanced trading signal with sentiment context."""
    timestamp: datetime
    asset: str
    action: str  # buy, sell, hold, close
    confidence: float
    sentiment_score: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: float = 1.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SentimentStrategy(ABC):
    """Abstract base class for sentiment-based strategies."""
    
    @abstractmethod
    def generate_signal(self, sentiment_data: Dict[str, Any], 
                       market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal from sentiment and market data."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class MomentumSentimentStrategy(SentimentStrategy):
    """
    Momentum-based sentiment strategy.
    
    Trades based on sentiment momentum and acceleration.
    """
    
    def __init__(self,
                 momentum_threshold: float = 0.3,
                 confidence_threshold: float = 0.6,
                 position_sizing: str = 'fixed'):
        """
        Initialize momentum strategy.
        
        Args:
            momentum_threshold: Minimum momentum for signal
            confidence_threshold: Minimum confidence
            position_sizing: Position sizing method
        """
        self.momentum_threshold = momentum_threshold
        self.confidence_threshold = confidence_threshold
        self.position_sizing = position_sizing
        self.recent_signals = deque(maxlen=10)
        
    def generate_signal(self, sentiment_data: Dict[str, Any], 
                       market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate signal based on sentiment momentum."""
        # Extract sentiment metrics
        current_sentiment = sentiment_data.get('sentiment_score', 0)
        sentiment_velocity = sentiment_data.get('velocity', 0)
        sentiment_acceleration = sentiment_data.get('acceleration', 0)
        confidence = sentiment_data.get('confidence', 0)
        
        if confidence < self.confidence_threshold:
            return None
        
        # Check momentum conditions
        if abs(sentiment_velocity) < self.momentum_threshold:
            return None
        
        # Determine action
        if sentiment_velocity > self.momentum_threshold and sentiment_acceleration > 0:
            action = 'buy'
            reason = f"Positive sentiment momentum: velocity={sentiment_velocity:.3f}"
        elif sentiment_velocity < -self.momentum_threshold and sentiment_acceleration < 0:
            action = 'sell'
            reason = f"Negative sentiment momentum: velocity={sentiment_velocity:.3f}"
        else:
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(
            sentiment_velocity, confidence, market_data
        )
        
        # Create signal
        signal = TradingSignal(
            timestamp=datetime.now(),
            asset=sentiment_data.get('asset', 'UNKNOWN'),
            action=action,
            confidence=confidence,
            sentiment_score=current_sentiment,
            position_size=position_size,
            reason=reason,
            metadata={
                'strategy': self.get_strategy_name(),
                'velocity': sentiment_velocity,
                'acceleration': sentiment_acceleration
            }
        )
        
        self.recent_signals.append(signal)
        return signal
    
    def _calculate_position_size(self, velocity: float, confidence: float,
                               market_data: Dict[str, Any]) -> float:
        """Calculate position size based on sentiment strength."""
        if self.position_sizing == 'fixed':
            return 1.0
        elif self.position_sizing == 'proportional':
            # Size proportional to momentum and confidence
            base_size = min(abs(velocity) / 0.5, 1.0)  # Cap at 1.0
            return base_size * confidence
        elif self.position_sizing == 'kelly':
            # Simplified Kelly criterion
            win_rate = 0.5 + confidence * 0.3  # Estimated from confidence
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0, min(kelly_fraction * 0.25, 1.0))  # Conservative Kelly
        else:
            return 1.0
    
    def get_strategy_name(self) -> str:
        return "MomentumSentiment"


class ContrarianSentimentStrategy(SentimentStrategy):
    """
    Contrarian sentiment strategy.
    
    Trades against extreme sentiment levels.
    """
    
    def __init__(self,
                 extreme_threshold: float = 0.8,
                 reversion_periods: int = 20):
        """
        Initialize contrarian strategy.
        
        Args:
            extreme_threshold: Threshold for extreme sentiment
            reversion_periods: Periods for mean reversion calculation
        """
        self.extreme_threshold = extreme_threshold
        self.reversion_periods = reversion_periods
        self.sentiment_history = deque(maxlen=reversion_periods * 2)
        
    def generate_signal(self, sentiment_data: Dict[str, Any],
                       market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate contrarian signal."""
        current_sentiment = sentiment_data.get('sentiment_score', 0)
        confidence = sentiment_data.get('confidence', 0)
        
        # Update history
        self.sentiment_history.append(current_sentiment)
        
        if len(self.sentiment_history) < self.reversion_periods:
            return None
        
        # Calculate mean and extremity
        sentiment_mean = np.mean(list(self.sentiment_history)[-self.reversion_periods:])
        sentiment_std = np.std(list(self.sentiment_history)[-self.reversion_periods:])
        
        if sentiment_std == 0:
            return None
        
        z_score = (current_sentiment - sentiment_mean) / sentiment_std
        
        # Check for extreme sentiment
        if abs(z_score) < 2:  # Not extreme enough
            return None
        
        # Contrarian signals
        if z_score > 2 and current_sentiment > self.extreme_threshold:
            action = 'sell'
            reason = f"Extreme positive sentiment: z-score={z_score:.2f}"
        elif z_score < -2 and current_sentiment < -self.extreme_threshold:
            action = 'buy'
            reason = f"Extreme negative sentiment: z-score={z_score:.2f}"
        else:
            return None
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            asset=sentiment_data.get('asset', 'UNKNOWN'),
            action=action,
            confidence=confidence * 0.8,  # Reduce confidence for contrarian
            sentiment_score=current_sentiment,
            position_size=min(abs(z_score) / 3, 1.0),
            reason=reason,
            metadata={
                'strategy': self.get_strategy_name(),
                'z_score': z_score,
                'sentiment_mean': sentiment_mean,
                'sentiment_std': sentiment_std
            }
        )
        
        return signal
    
    def get_strategy_name(self) -> str:
        return "ContrarianSentiment"


class EntityFocusedStrategy(SentimentStrategy):
    """
    Strategy focused on specific entity sentiment.
    
    Trades based on sentiment around specific entities (coins, people, etc).
    """
    
    def __init__(self,
                 target_entities: List[str],
                 entity_threshold: float = 0.5,
                 correlation_threshold: float = 0.7):
        """
        Initialize entity-focused strategy.
        
        Args:
            target_entities: Entities to track
            entity_threshold: Sentiment threshold for entities
            correlation_threshold: Correlation threshold for confirmation
        """
        self.target_entities = target_entities
        self.entity_threshold = entity_threshold
        self.correlation_threshold = correlation_threshold
        self.entity_histories = defaultdict(lambda: deque(maxlen=50))
        
    def generate_signal(self, sentiment_data: Dict[str, Any],
                       market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate signal based on entity sentiment."""
        entities = sentiment_data.get('entities', {})
        
        # Check each target entity
        for entity in self.target_entities:
            if entity not in entities:
                continue
            
            entity_data = entities[entity]
            entity_sentiment = entity_data.get('sentiment', 0)
            mentions = entity_data.get('mentions', 0)
            
            # Update history
            self.entity_histories[entity].append({
                'sentiment': entity_sentiment,
                'mentions': mentions,
                'timestamp': datetime.now()
            })
            
            # Need sufficient history
            if len(self.entity_histories[entity]) < 10:
                continue
            
            # Check sentiment threshold
            if abs(entity_sentiment) < self.entity_threshold:
                continue
            
            # Check mention volume (importance)
            recent_mentions = sum(h['mentions'] for h in list(self.entity_histories[entity])[-5:])
            if recent_mentions < 10:  # Not enough discussion
                continue
            
            # Generate signal
            if entity_sentiment > self.entity_threshold:
                action = 'buy'
                reason = f"Positive {entity} sentiment: {entity_sentiment:.3f}"
            else:
                action = 'sell'
                reason = f"Negative {entity} sentiment: {entity_sentiment:.3f}"
            
            # Calculate confidence based on mention volume and consistency
            recent_sentiments = [h['sentiment'] for h in list(self.entity_histories[entity])[-10:]]
            consistency = 1 - np.std(recent_sentiments)
            confidence = min(0.9, consistency * (recent_mentions / 50))
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                asset=sentiment_data.get('asset', entity),
                action=action,
                confidence=confidence,
                sentiment_score=sentiment_data.get('sentiment_score', entity_sentiment),
                position_size=confidence,
                reason=reason,
                metadata={
                    'strategy': self.get_strategy_name(),
                    'entity': entity,
                    'entity_sentiment': entity_sentiment,
                    'mentions': mentions,
                    'consistency': consistency
                }
            )
            
            return signal  # Return first valid signal
        
        return None
    
    def get_strategy_name(self) -> str:
        return "EntityFocused"


class SentimentDivergenceStrategy(SentimentStrategy):
    """
    Strategy based on sentiment-price divergence.
    
    Identifies when sentiment and price move in opposite directions.
    """
    
    def __init__(self,
                 divergence_threshold: float = 0.3,
                 lookback_periods: int = 20):
        """
        Initialize divergence strategy.
        
        Args:
            divergence_threshold: Minimum divergence for signal
            lookback_periods: Periods for divergence calculation
        """
        self.divergence_threshold = divergence_threshold
        self.lookback_periods = lookback_periods
        self.history = deque(maxlen=lookback_periods * 2)
        
    def generate_signal(self, sentiment_data: Dict[str, Any],
                       market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate signal based on sentiment-price divergence."""
        current_sentiment = sentiment_data.get('sentiment_score', 0)
        current_price = market_data.get('price', 0)
        
        if current_price == 0:
            return None
        
        # Update history
        self.history.append({
            'sentiment': current_sentiment,
            'price': current_price,
            'timestamp': datetime.now()
        })
        
        if len(self.history) < self.lookback_periods:
            return None
        
        # Calculate divergence
        recent_data = list(self.history)[-self.lookback_periods:]
        
        # Price trend
        prices = [d['price'] for d in recent_data]
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        # Sentiment trend
        sentiments = [d['sentiment'] for d in recent_data]
        sentiment_start = np.mean(sentiments[:5])
        sentiment_end = np.mean(sentiments[-5:])
        sentiment_change = sentiment_end - sentiment_start
        
        # Check for divergence
        divergence_score = 0
        
        if price_change > 0.05 and sentiment_change < -self.divergence_threshold:
            # Price up, sentiment down - bearish divergence
            divergence_score = -abs(sentiment_change)
            action = 'sell'
            reason = f"Bearish divergence: price up {price_change:.2%}, sentiment down {sentiment_change:.3f}"
        elif price_change < -0.05 and sentiment_change > self.divergence_threshold:
            # Price down, sentiment up - bullish divergence
            divergence_score = abs(sentiment_change)
            action = 'buy'
            reason = f"Bullish divergence: price down {price_change:.2%}, sentiment up {sentiment_change:.3f}"
        else:
            return None
        
        # Calculate confidence
        confidence = min(0.9, abs(divergence_score))
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            asset=sentiment_data.get('asset', 'UNKNOWN'),
            action=action,
            confidence=confidence,
            sentiment_score=current_sentiment,
            position_size=confidence * 0.8,
            reason=reason,
            metadata={
                'strategy': self.get_strategy_name(),
                'price_change': price_change,
                'sentiment_change': sentiment_change,
                'divergence_score': divergence_score
            }
        )
        
        return signal
    
    def get_strategy_name(self) -> str:
        return "SentimentDivergence"


class SentimentStrategyManager:
    """
    Manages multiple sentiment strategies and position tracking.
    
    Features:
    - Multi-strategy execution
    - Position management
    - Risk controls
    - Performance tracking
    """
    
    def __init__(self,
                 strategies: List[SentimentStrategy],
                 max_positions: int = 10,
                 max_position_size: float = 0.1,
                 risk_per_trade: float = 0.02):
        """
        Initialize strategy manager.
        
        Args:
            strategies: List of strategies to manage
            max_positions: Maximum concurrent positions
            max_position_size: Maximum position size (fraction of capital)
            risk_per_trade: Maximum risk per trade
        """
        self.strategies = strategies
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        
        # Position tracking
        self.positions = {}
        self.position_history = []
        
        # Performance tracking
        self.signal_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(lambda: {
            'total_signals': 0,
            'executed_signals': 0,
            'winning_trades': 0,
            'total_pnl': 0.0
        })
        
        # Risk management
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()
        
        logger.info(f"Strategy manager initialized with {len(strategies)} strategies")
    
    def process_sentiment_update(self, sentiment_data: Dict[str, Any],
                               market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Process sentiment update through all strategies."""
        # Reset daily PnL if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = current_date
        
        # Check risk limits
        if self.daily_pnl < -self.daily_loss_limit:
            logger.warning("Daily loss limit reached, no new positions")
            return []
        
        # Generate signals from all strategies
        all_signals = []
        
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(sentiment_data, market_data)
                if signal:
                    all_signals.append(signal)
                    self.performance_metrics[strategy.get_strategy_name()]['total_signals'] += 1
            except Exception as e:
                logger.error(f"Strategy {strategy.get_strategy_name()} error: {e}")
        
        # Filter and prioritize signals
        filtered_signals = self._filter_signals(all_signals, market_data)
        
        # Update signal history
        self.signal_history.extend(filtered_signals)
        
        return filtered_signals
    
    def _filter_signals(self, signals: List[TradingSignal],
                       market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Filter and prioritize trading signals."""
        if not signals:
            return []
        
        # Check position limits
        if len(self.positions) >= self.max_positions:
            # Only allow closing signals
            signals = [s for s in signals if s.action in ['sell', 'close']]
        
        # Remove conflicting signals for same asset
        asset_signals = defaultdict(list)
        for signal in signals:
            asset_signals[signal.asset].append(signal)
        
        filtered = []
        for asset, asset_sigs in asset_signals.items():
            if len(asset_sigs) == 1:
                filtered.append(asset_sigs[0])
            else:
                # Choose highest confidence signal
                best_signal = max(asset_sigs, key=lambda s: s.confidence)
                filtered.append(best_signal)
        
        # Apply position sizing limits
        for signal in filtered:
            signal.position_size = min(signal.position_size, self.max_position_size)
        
        return filtered
    
    def execute_signal(self, signal: TradingSignal, 
                      execution_price: float) -> Optional[TradingPosition]:
        """Execute a trading signal."""
        asset = signal.asset
        
        # Check if we have an existing position
        if asset in self.positions:
            existing = self.positions[asset]
            
            if signal.action in ['sell', 'close']:
                # Close position
                pnl = self._calculate_pnl(existing, execution_price)
                self.daily_pnl += pnl
                
                # Update performance metrics
                strategy_name = signal.metadata.get('strategy', 'unknown')
                self.performance_metrics[strategy_name]['executed_signals'] += 1
                if pnl > 0:
                    self.performance_metrics[strategy_name]['winning_trades'] += 1
                self.performance_metrics[strategy_name]['total_pnl'] += pnl
                
                # Remove position
                del self.positions[asset]
                
                logger.info(f"Closed {asset} position: PnL={pnl:.4f}")
                return None
            else:
                logger.warning(f"Already have position in {asset}, ignoring buy signal")
                return None
        
        # Open new position
        if signal.action == 'buy':
            position = TradingPosition(
                asset=asset,
                side='long',
                entry_price=execution_price,
                size=signal.position_size,
                entry_time=signal.timestamp,
                entry_sentiment=signal.sentiment_score,
                stop_loss=signal.stop_loss,
                take_profit=signal.price_target,
                metadata=signal.metadata
            )
            
            self.positions[asset] = position
            self.position_history.append(position)
            
            # Update metrics
            strategy_name = signal.metadata.get('strategy', 'unknown')
            self.performance_metrics[strategy_name]['executed_signals'] += 1
            
            logger.info(f"Opened {asset} position: size={position.size:.3f}")
            return position
        
        return None
    
    def _calculate_pnl(self, position: TradingPosition, exit_price: float) -> float:
        """Calculate position PnL."""
        if position.side == 'long':
            price_change = (exit_price - position.entry_price) / position.entry_price
        else:
            price_change = (position.entry_price - exit_price) / position.entry_price
        
        return price_change * position.size
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all strategies."""
        report = {
            'timestamp': datetime.now(),
            'active_positions': len(self.positions),
            'total_positions_opened': len(self.position_history),
            'daily_pnl': self.daily_pnl,
            'strategy_performance': {}
        }
        
        for strategy_name, metrics in self.performance_metrics.items():
            win_rate = 0
            if metrics['executed_signals'] > 0:
                win_rate = metrics['winning_trades'] / metrics['executed_signals']
            
            report['strategy_performance'][strategy_name] = {
                'total_signals': metrics['total_signals'],
                'executed_signals': metrics['executed_signals'],
                'execution_rate': metrics['executed_signals'] / max(metrics['total_signals'], 1),
                'win_rate': win_rate,
                'total_pnl': metrics['total_pnl']
            }
        
        return report