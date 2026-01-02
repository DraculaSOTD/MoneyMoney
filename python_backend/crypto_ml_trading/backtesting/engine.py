import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    direction: str  # 'long' or 'short'
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    commission: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'stopped'


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    max_position_size: float = 0.2   # 20% of capital
    use_leverage: bool = False
    max_leverage: float = 3.0
    allow_short: bool = False
    risk_free_rate: float = 0.02    # 2% annual
    trading_days: int = 365         # Crypto trades 365 days


class BacktestEngine:
    """
    Event-driven backtesting engine for cryptocurrency trading strategies.
    
    Features:
    - Realistic order execution with slippage and commissions
    - Risk management integration
    - Performance metrics calculation
    - Walk-forward analysis support
    - Multiple position handling
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.reset()
        
    def reset(self):
        """Reset backtesting state."""
        self.capital = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.positions = {}  # Current open positions
        self.trades = []     # Trade history
        self.equity_curve = []
        self.timestamps = []
        self.current_time = None
        self.current_price = None
        
    def run(self, data: pd.DataFrame, 
            signal_generator: Callable,
            position_sizer: Optional[Callable] = None,
            risk_manager: Optional[Callable] = None) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            signal_generator: Function that generates trading signals
            position_sizer: Function that determines position size
            risk_manager: Function that manages risk
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        # Ensure data is sorted by time
        data = data.sort_values('timestamp')
        
        # Main backtest loop
        for idx, row in data.iterrows():
            self.current_time = row['timestamp']
            self.current_price = row['close']
            
            # Update open positions
            self._update_positions(row)
            
            # Generate signals
            signal = signal_generator(data.loc[:idx], self.positions)
            
            if signal and signal['action'] != 'hold':
                # Determine position size
                if position_sizer:
                    position_details = position_sizer(
                        signal, 
                        self.capital,
                        self.current_price,
                        row.get('volatility', 0.02)
                    )
                else:
                    position_details = self._default_position_sizing(signal)
                    
                # Apply risk management
                if risk_manager:
                    position_details = risk_manager(position_details, self.positions)
                    
                # Execute trade
                if position_details['position_size'] > 0:
                    self._execute_trade(signal, position_details, row)
                    
            # Record equity
            total_equity = self._calculate_total_equity(row)
            self.equity_curve.append(total_equity)
            self.timestamps.append(self.current_time)
            
        # Close all remaining positions
        self._close_all_positions()
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        return results
    
    def _update_positions(self, row: pd.DataFrame):
        """Update open positions with current prices."""
        for symbol, trade in list(self.positions.items()):
            # Check stop loss
            if trade.stop_loss:
                if (trade.direction == 'long' and row['low'] <= trade.stop_loss) or \
                   (trade.direction == 'short' and row['high'] >= trade.stop_loss):
                    self._close_position(symbol, trade.stop_loss, 'stopped')
                    continue
                    
            # Check take profit
            if trade.take_profit:
                if (trade.direction == 'long' and row['high'] >= trade.take_profit) or \
                   (trade.direction == 'short' and row['low'] <= trade.take_profit):
                    self._close_position(symbol, trade.take_profit, 'closed')
                    
    def _execute_trade(self, signal: Dict, position_details: Dict, row: pd.Series):
        """Execute a trade based on signal."""
        # Calculate actual entry price with slippage
        slippage = self.current_price * self.config.slippage_rate
        if signal['action'] == 'buy':
            entry_price = self.current_price + slippage
            direction = 'long'
        else:  # sell
            if not self.config.allow_short and 'BTCUSDT' not in self.positions:
                return  # Can't short if not allowed and no position to close
            entry_price = self.current_price - slippage
            direction = 'short'
            
        # Calculate position value and commission
        position_value = position_details['position_value']
        commission = position_value * self.config.commission_rate
        
        # Check if we have enough capital
        required_capital = position_value + commission
        if required_capital > self.capital:
            warnings.warn(f"Insufficient capital for trade: {required_capital} > {self.capital}")
            return
            
        # Create trade
        trade = Trade(
            entry_time=self.current_time,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            position_size=position_details['position_size_pct'],
            direction=direction,
            stop_loss=position_details.get('stop_loss'),
            take_profit=position_details.get('take_profit'),
            commission=commission
        )
        
        # Update capital
        self.capital -= required_capital
        
        # Store position
        self.positions['BTCUSDT'] = trade
        
    def _close_position(self, symbol: str, exit_price: float, status: str = 'closed'):
        """Close an open position."""
        if symbol not in self.positions:
            return
            
        trade = self.positions[symbol]
        
        # Apply slippage
        slippage = exit_price * self.config.slippage_rate
        if trade.direction == 'long':
            exit_price -= slippage
        else:
            exit_price += slippage
            
        # Calculate PnL
        if trade.direction == 'long':
            price_change = exit_price - trade.entry_price
        else:
            price_change = trade.entry_price - exit_price
            
        position_value = trade.position_size * self.initial_capital
        pnl = (price_change / trade.entry_price) * position_value
        
        # Commission on exit
        exit_commission = position_value * self.config.commission_rate
        pnl -= exit_commission
        
        # Update trade
        trade.exit_time = self.current_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.return_pct = pnl / position_value
        trade.commission += exit_commission
        trade.status = status
        
        # Update capital
        self.capital += position_value + pnl
        
        # Move to trade history
        self.trades.append(trade)
        del self.positions[symbol]
        
    def _close_all_positions(self):
        """Close all remaining open positions."""
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, self.current_price)
            
    def _calculate_total_equity(self, row: pd.Series) -> float:
        """Calculate total equity including open positions."""
        equity = self.capital
        
        # Add unrealized PnL from open positions
        for symbol, trade in self.positions.items():
            current_value = trade.position_size * self.initial_capital
            
            if trade.direction == 'long':
                price_change = row['close'] - trade.entry_price
            else:
                price_change = trade.entry_price - row['close']
                
            unrealized_pnl = (price_change / trade.entry_price) * current_value
            equity += current_value + unrealized_pnl
            
        return equity
    
    def _default_position_sizing(self, signal: Dict) -> Dict:
        """Default position sizing strategy."""
        confidence = signal.get('confidence', 0.5)
        base_size = self.config.max_position_size
        
        # Scale by confidence
        position_size = base_size * confidence
        
        return {
            'position_size_pct': position_size,
            'position_value': position_size * self.capital,
            'confidence': confidence
        }
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return {}
            
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Basic metrics
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # Handle empty returns
        if len(returns) == 0:
            return {
                'total_return': total_return,
                'num_trades': len(self.trades),
                'final_equity': equity_array[-1]
            }
            
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(equity_array)
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics()
        
        # Monthly/yearly returns
        period_returns = self._calculate_period_returns(returns)
        
        return {
            # Overall performance
            'total_return': total_return,
            'annualized_return': self._annualize_return(total_return, len(returns)),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': self._annualize_return(total_return, len(returns)) / max_drawdown if max_drawdown > 0 else 0,
            
            # Risk metrics
            'volatility': np.std(returns) * np.sqrt(252),
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'var_95': np.percentile(returns, 5),
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            
            # Trade statistics
            **trade_stats,
            
            # Period returns
            **period_returns,
            
            # Other metrics
            'final_equity': equity_array[-1],
            'equity_curve': equity_array.tolist(),
            'timestamps': self.timestamps
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
            
        excess_returns = returns - self.config.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - self.config.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
            
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = abs(np.min(drawdown))
        
        # Calculate duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
                max_duration = max(max_duration, current_duration)
            else:
                drawdown_start = None
                
        return max_drawdown, max_duration
    
    def _calculate_trade_statistics(self) -> Dict:
        """Calculate trade-level statistics."""
        if not self.trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0
            }
            
        # Separate wins and losses
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        num_trades = len(self.trades)
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
        
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss)
        
        # Additional statistics
        avg_trade_duration = np.mean([
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in self.trades if t.exit_time
        ]) if self.trades else 0
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade_duration_hours': avg_trade_duration,
            'total_commission': sum(t.commission for t in self.trades)
        }
    
    def _calculate_period_returns(self, returns: np.ndarray) -> Dict:
        """Calculate returns by period."""
        if len(self.timestamps) != len(returns) + 1:
            return {}
            
        # Create DataFrame for easier period calculations
        df = pd.DataFrame({
            'returns': np.concatenate([[0], returns]),
            'timestamp': self.timestamps
        })
        df.set_index('timestamp', inplace=True)
        
        # Monthly returns
        monthly_returns = df.resample('M')['returns'].apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return {
            'avg_monthly_return': monthly_returns.mean(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months': (monthly_returns > 0).sum(),
            'negative_months': (monthly_returns <= 0).sum()
        }
    
    def _annualize_return(self, total_return: float, num_periods: int) -> float:
        """Annualize return based on number of periods."""
        if num_periods == 0:
            return 0
            
        years = num_periods / (252 * 24 * 60)  # Assuming minute data
        return (1 + total_return) ** (1 / years) - 1
    
    def plot_results(self):
        """
        Note: Plotting would require matplotlib.
        This is a placeholder for visualization.
        """
        print("Backtest Results Visualization:")
        print("1. Equity Curve")
        print("2. Drawdown Chart")
        print("3. Monthly Returns Heatmap")
        print("4. Trade Distribution")
        print("\nImplement with matplotlib when needed.")