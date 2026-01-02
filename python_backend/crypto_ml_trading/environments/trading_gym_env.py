"""
OpenAI Gym-compatible Trading Environment
Integrates with existing system for RL training
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any, Union
from dataclasses import dataclass
import logging

# Import system components
from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
from features.ml_feature_engineering import MLFeatureEngineering
from data.preprocessing import AdvancedPreprocessor
from models.risk_management.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Current state of the trading environment."""
    timestamp: pd.Timestamp
    position: float  # Current position size (-1 to 1)
    entry_price: float  # Entry price if in position, 0 otherwise
    cash: float  # Available cash
    portfolio_value: float  # Total portfolio value
    unrealized_pnl: float  # Unrealized P&L
    realized_pnl: float  # Realized P&L
    current_price: float  # Current market price
    steps_in_position: int  # Number of steps in current position


class TradingGymEnvironment(gym.Env):
    """
    OpenAI Gym-compatible trading environment for RL training.
    
    Action Space:
        - Continuous: [position_size, stop_loss_pct, take_profit_pct]
        - Discrete: 0=Sell, 1=Hold, 2=Buy
    
    Observation Space:
        - Market features (OHLCV, indicators)
        - Portfolio state (position, P&L, etc.)
        - Risk metrics
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 10000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 reward_type: str = 'sharpe',
                 lookback_window: int = 50,
                 discrete_actions: bool = False,
                 feature_config: Optional[Dict] = None,
                 risk_config: Optional[Dict] = None,
                 max_steps: Optional[int] = None):
        """
        Initialize trading environment.
        
        Args:
            data: Historical market data
            initial_balance: Starting capital
            commission: Trading commission (percentage)
            slippage: Slippage factor
            reward_type: Type of reward function
            lookback_window: Number of periods for observation
            discrete_actions: Use discrete action space
            feature_config: Feature engineering configuration
            risk_config: Risk management configuration
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        # Market data
        self.data = data
        self.lookback_window = lookback_window
        self.max_steps = max_steps or len(data) - lookback_window - 1
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        
        # Feature engineering
        self.ml_engineer = MLFeatureEngineering()
        self.preprocessor = AdvancedPreprocessor()
        self.feature_config = feature_config or self._default_feature_config()
        
        # Risk management
        self.risk_manager = RiskManager(risk_config or self._default_risk_config())
        
        # Prepare features
        self._prepare_features()
        
        # Action and observation spaces
        self.discrete_actions = discrete_actions
        self._setup_spaces()
        
        # Episode state
        self.current_step = 0
        self.state = None
        self.history = []
        
    def _default_feature_config(self) -> Dict:
        """Default feature configuration."""
        return {
            'indicators': {
                'sma': {'enabled': True, 'periods': [10, 20, 50]},
                'ema': {'enabled': True, 'periods': [12, 26]},
                'rsi': {'enabled': True, 'period': 14},
                'macd': {'enabled': True},
                'bollinger': {'enabled': True},
                'atr': {'enabled': True},
                'adx': {'enabled': True}
            },
            'percentage_features': True,
            'lagged_features': True,
            'rolling_features': True
        }
    
    def _default_risk_config(self) -> Dict:
        """Default risk configuration."""
        return {
            'max_position_size': 1.0,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'max_drawdown': 0.2
        }
    
    def _prepare_features(self):
        """Prepare all features for the entire dataset."""
        logger.info("Preparing features for RL environment...")
        
        # Add technical indicators
        self.data = EnhancedTechnicalIndicators.compute_all_indicators(
            self.data, self.feature_config['indicators']
        )
        
        # Create ML features
        self.data = self.ml_engineer.prepare_ml_features(
            self.data, self.feature_config
        )
        
        # Preprocess
        self.data = self.preprocessor.preprocess(self.data)
        
        # Get feature columns
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        # Normalize features to [-1, 1] for RL
        from sklearn.preprocessing import MinMaxScaler
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data[self.feature_columns] = self.feature_scaler.fit_transform(
            self.data[self.feature_columns].fillna(0)
        )
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Observation space: market features + portfolio state
        n_features = len(self.feature_columns)
        n_portfolio_features = 8  # position, cash_ratio, pnl, etc.
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, n_features + n_portfolio_features),
            dtype=np.float32
        )
        
        # Action space
        if self.discrete_actions:
            self.action_space = spaces.Discrete(3)  # Sell, Hold, Buy
        else:
            # [position_size, stop_loss_pct, take_profit_pct]
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),
                high=np.array([1.0, 0.1, 0.2]),
                dtype=np.float32
            )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Random starting point
        self.current_step = np.random.randint(
            self.lookback_window,
            len(self.data) - self.max_steps
        )
        
        # Initialize state
        self.state = TradingState(
            timestamp=self.data.iloc[self.current_step]['timestamp'],
            position=0.0,
            entry_price=0.0,
            cash=self.initial_balance,
            portfolio_value=self.initial_balance,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            current_price=self.data.iloc[self.current_step]['close'],
            steps_in_position=0
        )
        
        # Clear history
        self.history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Market features
        start_idx = self.current_step - self.lookback_window + 1
        end_idx = self.current_step + 1
        
        market_features = self.data[self.feature_columns].iloc[start_idx:end_idx].values
        
        # Portfolio features (repeated for each timestep)
        portfolio_features = np.array([
            self.state.position,
            self.state.cash / self.initial_balance,  # Cash ratio
            self.state.unrealized_pnl / self.initial_balance,  # Unrealized P&L ratio
            self.state.realized_pnl / self.initial_balance,  # Realized P&L ratio
            self.state.portfolio_value / self.initial_balance,  # Portfolio value ratio
            self.state.steps_in_position / 100,  # Normalized steps in position
            1.0 if self.state.position > 0 else 0.0,  # Long position indicator
            1.0 if self.state.position < 0 else 0.0,  # Short position indicator
        ])
        
        # Repeat portfolio features for each timestep
        portfolio_features = np.tile(portfolio_features, (self.lookback_window, 1))
        
        # Combine features
        observation = np.concatenate([market_features, portfolio_features], axis=1)
        
        return observation.astype(np.float32)
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Trading action (discrete or continuous)
            
        Returns:
            observation: Next observation
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        # Store previous state
        prev_portfolio_value = self.state.portfolio_value
        prev_position = self.state.position
        
        # Parse action
        if self.discrete_actions:
            position_size = self._discrete_to_continuous(action)
            stop_loss_pct = 0.02  # Default 2%
            take_profit_pct = 0.04  # Default 4%
        else:
            position_size = float(action[0])
            stop_loss_pct = float(action[1])
            take_profit_pct = float(action[2])
        
        # Execute trade
        trade_info = self._execute_trade(position_size, stop_loss_pct, take_profit_pct)
        
        # Update market
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False
            self.state.current_price = self.data.iloc[self.current_step]['close']
            self.state.timestamp = self.data.iloc[self.current_step]['timestamp']
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value)
        
        # Check if done (stop conditions)
        if self.state.portfolio_value <= self.initial_balance * 0.5:  # 50% loss
            done = True
            reward -= 1.0  # Penalty for large loss
        
        if self.current_step - self.lookback_window >= self.max_steps:
            done = True
        
        # Store history
        self.history.append({
            'step': self.current_step,
            'action': action,
            'position': self.state.position,
            'portfolio_value': self.state.portfolio_value,
            'reward': reward,
            'price': self.state.current_price
        })
        
        # Info dictionary
        info = {
            'trade_info': trade_info,
            'portfolio_value': self.state.portfolio_value,
            'position': self.state.position,
            'realized_pnl': self.state.realized_pnl,
            'unrealized_pnl': self.state.unrealized_pnl
        }
        
        return self._get_observation(), reward, done, info
    
    def _discrete_to_continuous(self, action: int) -> float:
        """Convert discrete action to continuous position size."""
        if action == 0:  # Sell
            return -1.0
        elif action == 1:  # Hold
            return self.state.position  # Maintain current position
        else:  # Buy
            return 1.0
    
    def _execute_trade(self, target_position: float, 
                      stop_loss_pct: float, take_profit_pct: float) -> Dict:
        """Execute trade based on target position."""
        current_position = self.state.position
        position_change = target_position - current_position
        
        trade_info = {
            'executed': False,
            'position_change': 0.0,
            'cost': 0.0,
            'commission': 0.0
        }
        
        if abs(position_change) < 0.01:  # No significant change
            if self.state.position != 0:
                self.state.steps_in_position += 1
            return trade_info
        
        # Calculate trade cost
        trade_value = abs(position_change) * self.state.current_price * self.initial_balance
        commission = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        total_cost = commission + slippage_cost
        
        # Check if we have enough cash
        if position_change > 0:  # Buying
            required_cash = trade_value + total_cost
            if required_cash > self.state.cash:
                # Adjust position change to available cash
                available_for_position = self.state.cash - total_cost
                position_change = available_for_position / (self.state.current_price * self.initial_balance)
                trade_value = position_change * self.state.current_price * self.initial_balance
                total_cost = trade_value * (self.commission + self.slippage)
        
        # Close existing position if changing sides
        if current_position != 0 and np.sign(target_position) != np.sign(current_position):
            # Close current position first
            close_value = abs(current_position) * self.state.current_price * self.initial_balance
            close_cost = close_value * (self.commission + self.slippage)
            
            # Calculate P&L
            pnl = (self.state.current_price - self.state.entry_price) * current_position * self.initial_balance
            self.state.realized_pnl += pnl - close_cost
            self.state.cash += close_value - close_cost
            
            # Reset position
            self.state.position = 0
            self.state.entry_price = 0
            self.state.unrealized_pnl = 0
            self.state.steps_in_position = 0
        
        # Open new position
        if position_change != 0:
            self.state.position = target_position
            self.state.entry_price = self.state.current_price * (1 + self.slippage * np.sign(position_change))
            self.state.cash -= trade_value + total_cost
            self.state.steps_in_position = 1
            
            trade_info['executed'] = True
            trade_info['position_change'] = position_change
            trade_info['cost'] = trade_value
            trade_info['commission'] = total_cost
        
        return trade_info
    
    def _update_portfolio_value(self):
        """Update portfolio value and unrealized P&L."""
        if self.state.position != 0:
            # Update unrealized P&L
            self.state.unrealized_pnl = (
                (self.state.current_price - self.state.entry_price) * 
                self.state.position * self.initial_balance
            )
            
            # Portfolio value = cash + position value
            position_value = abs(self.state.position) * self.state.current_price * self.initial_balance
            self.state.portfolio_value = self.state.cash + position_value + self.state.unrealized_pnl
        else:
            self.state.unrealized_pnl = 0
            self.state.portfolio_value = self.state.cash
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """Calculate reward based on reward type."""
        # Portfolio return
        portfolio_return = (self.state.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        if self.reward_type == 'simple':
            # Simple return-based reward
            reward = portfolio_return * 100
            
        elif self.reward_type == 'sharpe':
            # Sharpe ratio-based reward
            if len(self.history) > 10:
                recent_returns = []
                for i in range(1, min(11, len(self.history))):
                    prev_val = self.history[-i]['portfolio_value']
                    curr_val = self.history[-i+1]['portfolio_value'] if i > 1 else self.state.portfolio_value
                    ret = (curr_val - prev_val) / prev_val
                    recent_returns.append(ret)
                
                if len(recent_returns) > 1:
                    mean_return = np.mean(recent_returns)
                    std_return = np.std(recent_returns)
                    sharpe = mean_return / (std_return + 1e-8) * np.sqrt(252)  # Annualized
                    reward = sharpe / 10  # Scale down
                else:
                    reward = portfolio_return * 100
            else:
                reward = portfolio_return * 100
                
        elif self.reward_type == 'risk_adjusted':
            # Risk-adjusted reward
            reward = portfolio_return * 100
            
            # Penalty for holding position too long
            if self.state.steps_in_position > 50:
                reward -= 0.01 * (self.state.steps_in_position - 50)
            
            # Penalty for large drawdown
            max_value = max([h['portfolio_value'] for h in self.history[-100:]], default=self.state.portfolio_value)
            current_drawdown = (max_value - self.state.portfolio_value) / max_value
            if current_drawdown > 0.1:  # 10% drawdown
                reward -= current_drawdown * 10
                
        else:
            reward = portfolio_return * 100
        
        return float(reward)
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the environment."""
        if mode == 'ansi':
            output = f"\n[Step {self.current_step}] "
            output += f"Price: ${self.state.current_price:.2f} | "
            output += f"Position: {self.state.position:.2f} | "
            output += f"Portfolio: ${self.state.portfolio_value:.2f} | "
            output += f"P&L: ${self.state.realized_pnl + self.state.unrealized_pnl:.2f}"
            return output
        
        elif mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}")
            print(f"Timestamp: {self.state.timestamp}")
            print(f"Current Price: ${self.state.current_price:.2f}")
            print(f"Position: {self.state.position:.2f}")
            print(f"Entry Price: ${self.state.entry_price:.2f}" if self.state.position != 0 else "Entry Price: N/A")
            print(f"Cash: ${self.state.cash:.2f}")
            print(f"Portfolio Value: ${self.state.portfolio_value:.2f}")
            print(f"Unrealized P&L: ${self.state.unrealized_pnl:.2f}")
            print(f"Realized P&L: ${self.state.realized_pnl:.2f}")
            print(f"Total P&L: ${self.state.realized_pnl + self.state.unrealized_pnl:.2f}")
            print(f"Return: {(self.state.portfolio_value / self.initial_balance - 1) * 100:.2f}%")
            print(f"{'='*60}")
    
    def get_episode_statistics(self) -> Dict:
        """Get statistics for the completed episode."""
        if not self.history:
            return {}
        
        portfolio_values = [h['portfolio_value'] for h in self.history]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate max drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (cummax - portfolio_values) / cummax
        max_drawdown = np.max(drawdowns)
        
        # Win rate
        trades = [h for h in self.history if h.get('trade_info', {}).get('executed', False)]
        winning_trades = sum(1 for t in trades if t['reward'] > 0)
        win_rate = winning_trades / len(trades) if trades else 0
        
        return {
            'total_return': (self.state.portfolio_value / self.initial_balance - 1) * 100,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': max_drawdown * 100,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'final_portfolio_value': self.state.portfolio_value,
            'total_pnl': self.state.realized_pnl + self.state.unrealized_pnl
        }


class MultiAssetTradingEnv(TradingGymEnvironment):
    """
    Multi-asset trading environment for portfolio management.
    """
    
    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame],
                 initial_balance: float = 10000,
                 **kwargs):
        """
        Initialize multi-asset environment.
        
        Args:
            data_dict: Dictionary of asset_name -> price data
            initial_balance: Starting capital
            **kwargs: Additional arguments for parent class
        """
        self.assets = list(data_dict.keys())
        self.data_dict = data_dict
        
        # Align all dataframes by timestamp
        self._align_data()
        
        # Initialize with first asset (for compatibility)
        super().__init__(self.aligned_data[self.assets[0]], initial_balance, **kwargs)
        
        # Override action space for multiple assets
        self._setup_multi_asset_spaces()
    
    def _align_data(self):
        """Align all asset data by timestamp."""
        # Find common timestamps
        timestamps = None
        for asset, df in self.data_dict.items():
            if timestamps is None:
                timestamps = set(df.index)
            else:
                timestamps = timestamps.intersection(set(df.index))
        
        # Create aligned dataframes
        self.aligned_data = {}
        for asset, df in self.data_dict.items():
            self.aligned_data[asset] = df.loc[sorted(timestamps)]
    
    def _setup_multi_asset_spaces(self):
        """Setup action and observation spaces for multiple assets."""
        n_assets = len(self.assets)
        
        # Action space: position size for each asset
        if self.discrete_actions:
            # Multi-discrete: 3 actions per asset
            self.action_space = spaces.MultiDiscrete([3] * n_assets)
        else:
            # Continuous: position for each asset + cash weight
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n_assets + 1,),  # +1 for cash weight
                dtype=np.float32
            )
        
        # Observation space: features for all assets
        n_features = len(self.feature_columns)
        n_portfolio_features = 8 * n_assets + 2  # Portfolio features per asset + global
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, n_features * n_assets + n_portfolio_features),
            dtype=np.float32
        )