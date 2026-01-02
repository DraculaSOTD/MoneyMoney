import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class TradingState:
    """Current state of the trading environment."""
    position: float  # Current position (-1 to 1)
    cash: float  # Available cash
    portfolio_value: float  # Total portfolio value
    current_price: float
    timestamp: pd.Timestamp
    step: int
    done: bool = False


class TradingEnvironment:
    """
    Cryptocurrency trading environment for reinforcement learning.
    
    Features:
    - Continuous action space for position sizing
    - Transaction costs and slippage
    - Risk-adjusted rewards
    - Market impact modeling
    - Multi-timeframe observations
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 max_position: float = 1.0,
                 reward_scaling: float = 1.0,
                 lookback_window: int = 60):
        """
        Initialize trading environment.
        
        Args:
            data: Historical OHLCV data
            initial_capital: Starting capital
            commission_rate: Commission per trade
            slippage_rate: Slippage percentage
            max_position: Maximum position size
            reward_scaling: Reward scaling factor
            lookback_window: Number of periods for observation
        """
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.lookback_window = lookback_window
        
        # Environment state
        self.current_step = 0
        self.state = None
        self.episode_trades = []
        self.episode_returns = []
        
        # Observation and action spaces
        self.observation_shape = self._get_observation_shape()
        self.action_dim = 3  # [position_size, stop_loss, take_profit]
        
        # Performance tracking
        self.metrics = {
            'total_reward': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
    def _get_observation_shape(self) -> Tuple[int, ...]:
        """Calculate observation space shape."""
        # Features: OHLCV + technical indicators + position info
        n_price_features = 5  # OHLCV
        n_technical_features = 20  # Technical indicators
        n_position_features = 5  # Position, PnL, etc.
        n_market_features = 10  # Market microstructure
        
        total_features = (n_price_features + n_technical_features + 
                         n_position_features + n_market_features)
        
        return (self.lookback_window, total_features)
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = self.lookback_window
        
        self.state = TradingState(
            position=0.0,
            cash=self.initial_capital,
            portfolio_value=self.initial_capital,
            current_price=self.data.iloc[self.current_step]['close'],
            timestamp=self.data.iloc[self.current_step]['timestamp'],
            step=0
        )
        
        self.episode_trades = []
        self.episode_returns = []
        self.metrics = {
            'total_reward': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Trading action [position_size, stop_loss, take_profit]
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Clip and process action
        position_target = np.clip(action[0], -self.max_position, self.max_position)
        stop_loss_pct = np.clip(action[1], 0.01, 0.05)  # 1-5% stop loss
        take_profit_pct = np.clip(action[2], 0.02, 0.10)  # 2-10% take profit
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        prev_price = self.state.current_price
        current_price = current_data['close']
        
        # Execute trade if position changes
        position_change = position_target - self.state.position
        if abs(position_change) > 0.01:  # Minimum position change
            self._execute_trade(position_change, current_price)
            
        # Update position value
        self._update_portfolio(current_price)
        
        # Calculate reward
        reward = self._calculate_reward(prev_price, current_price)
        
        # Update state
        self.state.current_price = current_price
        self.state.timestamp = current_data['timestamp']
        self.state.step += 1
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                self.state.portfolio_value < self.initial_capital * 0.5)  # 50% loss stops episode
        
        self.state.done = done
        
        # Get next observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self.state.portfolio_value,
            'position': self.state.position,
            'trades': len(self.episode_trades),
            'current_price': current_price,
            'timestamp': self.state.timestamp
        }
        
        return observation, reward, done, info
    
    def _execute_trade(self, position_change: float, current_price: float):
        """Execute a trade with costs."""
        # Calculate trade value
        trade_value = abs(position_change) * self.state.portfolio_value
        
        # Apply slippage
        if position_change > 0:  # Buying
            execution_price = current_price * (1 + self.slippage_rate)
        else:  # Selling
            execution_price = current_price * (1 - self.slippage_rate)
            
        # Calculate commission
        commission = trade_value * self.commission_rate
        
        # Update cash and position
        cash_change = -position_change * self.state.portfolio_value - commission
        self.state.cash += cash_change
        self.state.position += position_change
        
        # Record trade
        self.episode_trades.append({
            'timestamp': self.state.timestamp,
            'action': 'buy' if position_change > 0 else 'sell',
            'size': abs(position_change),
            'price': execution_price,
            'commission': commission
        })
        
        self.metrics['total_trades'] += 1
        
    def _update_portfolio(self, current_price: float):
        """Update portfolio value based on current price."""
        # Position value
        position_value = self.state.position * self.state.portfolio_value
        
        # Price change effect on position
        if self.state.position != 0 and self.state.current_price > 0:
            price_change = (current_price - self.state.current_price) / self.state.current_price
            position_pnl = position_value * price_change
            self.state.cash += position_pnl
            
        # Update total portfolio value
        self.state.portfolio_value = self.state.cash
        
        # Track returns
        returns = (self.state.portfolio_value - self.initial_capital) / self.initial_capital
        self.episode_returns.append(returns)
        
        # Update metrics
        if len(self.episode_returns) > 1:
            self.metrics['max_drawdown'] = self._calculate_max_drawdown()
            
    def _calculate_reward(self, prev_price: float, current_price: float) -> float:
        """
        Calculate step reward.
        
        Uses a combination of:
        - Realized PnL
        - Risk-adjusted returns
        - Transaction cost penalty
        """
        # Base reward: portfolio return
        current_return = (self.state.portfolio_value - self.initial_capital) / self.initial_capital
        prev_return = self.episode_returns[-2] if len(self.episode_returns) > 1 else 0
        step_return = current_return - prev_return
        
        # Risk adjustment (Sharpe-like)
        if len(self.episode_returns) > 20:
            recent_returns = np.diff(self.episode_returns[-20:])
            volatility = np.std(recent_returns) + 1e-6
            risk_adjusted_return = step_return / volatility
        else:
            risk_adjusted_return = step_return
            
        # Transaction penalty
        n_recent_trades = sum(1 for t in self.episode_trades[-10:] 
                            if t['timestamp'] > self.state.timestamp - pd.Timedelta(minutes=10))
        transaction_penalty = -0.001 * n_recent_trades  # Penalize overtrading
        
        # Position penalty (encourage position sizing)
        position_penalty = -0.0001 * (1 - abs(self.state.position)) if abs(step_return) > 0.001 else 0
        
        # Combine rewards
        reward = (risk_adjusted_return * 100 +  # Scale returns
                 transaction_penalty +
                 position_penalty)
        
        return reward * self.reward_scaling
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Observation array
        """
        # Get historical data window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step + 1
        window_data = self.data.iloc[start_idx:end_idx]
        
        observations = []
        
        for _, row in window_data.iterrows():
            obs = []
            
            # Price features (normalized)
            obs.extend([
                row['open'] / row['close'],
                row['high'] / row['close'],
                row['low'] / row['close'],
                1.0,  # close/close = 1
                np.log(row['volume'] + 1)
            ])
            
            # Technical indicators (if available)
            tech_features = []
            for col in ['rsi', 'macd', 'bb_upper', 'bb_lower', 'atr']:
                if col in row:
                    tech_features.append(row[col])
                else:
                    tech_features.append(0.0)
                    
            # Pad technical features
            tech_features.extend([0.0] * (20 - len(tech_features)))
            obs.extend(tech_features)
            
            # Position features
            obs.extend([
                self.state.position,
                self.state.portfolio_value / self.initial_capital,
                self.state.cash / self.initial_capital,
                float(len(self.episode_trades)),
                self.metrics.get('max_drawdown', 0)
            ])
            
            # Market microstructure features
            market_features = self._calculate_market_features(row)
            obs.extend(market_features)
            
            observations.append(obs)
            
        # Pad if necessary
        while len(observations) < self.lookback_window:
            observations.insert(0, observations[0] if observations else [0] * self.observation_shape[1])
            
        return np.array(observations)
    
    def _calculate_market_features(self, row: pd.Series) -> List[float]:
        """Calculate market microstructure features."""
        features = []
        
        # Price momentum
        if self.current_step > 10:
            recent_prices = self.data.iloc[self.current_step-10:self.current_step+1]['close']
            momentum = (row['close'] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            volatility = recent_prices.pct_change().std()
        else:
            momentum = 0.0
            volatility = 0.02
            
        features.extend([momentum, volatility])
        
        # Volume profile
        if self.current_step > 20:
            recent_volumes = self.data.iloc[self.current_step-20:self.current_step+1]['volume']
            volume_ratio = row['volume'] / (recent_volumes.mean() + 1e-6)
            volume_trend = (recent_volumes.iloc[-5:].mean() - recent_volumes.iloc[:5].mean()) / (recent_volumes.mean() + 1e-6)
        else:
            volume_ratio = 1.0
            volume_trend = 0.0
            
        features.extend([volume_ratio, volume_trend])
        
        # Spread and liquidity
        spread = (row['high'] - row['low']) / row['close']
        features.append(spread)
        
        # Pad to 10 features
        while len(features) < 10:
            features.append(0.0)
            
        return features[:10]
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.episode_returns:
            return 0.0
            
        peak = self.initial_capital
        max_dd = 0.0
        
        for ret in self.episode_returns:
            value = self.initial_capital * (1 + ret)
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
            
        return max_dd
    
    def get_episode_summary(self) -> Dict:
        """Get summary statistics for the episode."""
        if not self.episode_returns:
            return {}
            
        returns = np.array(self.episode_returns)
        
        # Calculate metrics
        total_return = returns[-1] if len(returns) > 0 else 0
        
        if len(returns) > 1:
            daily_returns = np.diff(returns)
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        winning_trades = sum(1 for t in self.episode_trades 
                           if self._calculate_trade_return(t) > 0)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.metrics['max_drawdown'],
            'num_trades': len(self.episode_trades),
            'win_rate': winning_trades / max(1, len(self.episode_trades)),
            'final_portfolio_value': self.state.portfolio_value
        }
    
    def _calculate_trade_return(self, trade: Dict) -> float:
        """Calculate return for a single trade (simplified)."""
        # This is a placeholder - actual implementation would track entry/exit
        return np.random.randn() * 0.02  # Random for now
    
    def render(self, mode: str = 'human'):
        """Render the environment (text-based)."""
        if mode == 'human':
            print(f"\nStep: {self.state.step}")
            print(f"Price: ${self.state.current_price:.2f}")
            print(f"Position: {self.state.position:.2%}")
            print(f"Portfolio Value: ${self.state.portfolio_value:.2f}")
            print(f"Return: {(self.state.portfolio_value/self.initial_capital - 1):.2%}")
            print(f"Trades: {len(self.episode_trades)}")