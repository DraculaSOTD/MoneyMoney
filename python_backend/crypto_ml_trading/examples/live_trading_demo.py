"""
Live Trading System Demo.

Demonstrates the complete live trading system with GPU acceleration,
including real-time data, signal generation, and order execution.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from trading.live_trading_system import (
    LiveTradingSystem, TradingSignal, Order, Position, OrderSide
)
from trading.execution_engine import ExecutionEngine, ExecutionAlgorithm
from trading.portfolio_optimizer import PortfolioOptimizer, Asset
from models.deep_learning.gru_attention_gpu import GRUAttentionGPU
from utils.logger import get_logger, setup_logger

# Setup logging
setup_logger(level='INFO')
logger = get_logger(__name__)


def create_demo_models():
    """Create demo ML models for signal generation."""
    models = []
    
    # GRU-Attention model
    gru_model = GRUAttentionGPU(
        input_dim=20,  # Simplified feature set
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        output_dim=1
    )
    
    # Wrap in prediction function
    def gru_predictor(features):
        import torch
        
        # Convert to tensor and add batch dimension
        x = torch.tensor(features, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = gru_model(x)
        
        # Convert to trading signal (-1 to 1)
        return torch.tanh(output).numpy()
    
    models.append(('GRU_Attention', gru_predictor))
    
    # Simple momentum model
    def momentum_predictor(features):
        # Extract price change feature (assumed to be first feature)
        price_change = features[:, 0] if features.ndim > 1 else features[0]
        
        # Simple momentum signal
        if price_change > 0.01:  # 1% up
            return np.array([[0.7]])  # Buy signal
        elif price_change < -0.01:  # 1% down
            return np.array([[-0.7]])  # Sell signal
        else:
            return np.array([[0.0]])  # Hold
    
    models.append(('Momentum', momentum_predictor))
    
    # Mean reversion model
    def mean_reversion_predictor(features):
        # Extract RSI feature (assumed to be second feature)
        rsi = features[:, 1] if features.ndim > 1 else features[1] if len(features) > 1 else 50
        
        # Mean reversion signal
        if rsi > 70:  # Overbought
            return np.array([[-0.6]])  # Sell signal
        elif rsi < 30:  # Oversold
            return np.array([[0.6]])  # Buy signal
        else:
            return np.array([[0.0]])  # Hold
    
    models.append(('MeanReversion', mean_reversion_predictor))
    
    return models


class TradingSystemDemo:
    """Demo application for live trading system."""
    
    def __init__(self):
        # Trading parameters
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.exchanges = ['binance', 'coinbase']
        
        # Create ML models
        self.models = create_demo_models()
        
        # Risk parameters
        self.risk_params = {
            'max_position_size': 0.1,      # 10% per position
            'max_portfolio_exposure': 0.6,  # 60% total exposure
            'max_drawdown': 0.15,          # 15% max drawdown
            'stop_loss_pct': 0.02,         # 2% stop loss
            'take_profit_pct': 0.04        # 4% take profit
        }
        
        # Initialize systems
        self.trading_system = None
        self.execution_engine = None
        self.portfolio_optimizer = None
        
        # Performance tracking
        self.performance_log = []
    
    def initialize(self):
        """Initialize all trading components."""
        logger.info("Initializing trading system demo...")
        
        # Create live trading system
        self.trading_system = LiveTradingSystem(
            symbols=self.symbols,
            exchanges=self.exchanges,
            models=self.models,
            risk_params=self.risk_params,
            enable_gpu=True,
            paper_trading=True  # Demo mode
        )
        
        # Create execution engine
        self.execution_engine = ExecutionEngine(
            exchanges={'mock': self.trading_system.exchange_api},
            enable_gpu=True
        )
        
        # Create portfolio optimizer
        self.portfolio_optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            enable_gpu=True,
            rebalance_threshold=0.05
        )
        
        # Register callbacks
        self.trading_system.register_signal_callback(self._on_signal)
        self.trading_system.register_trade_callback(self._on_trade)
        
        logger.info("Trading system initialized successfully")
    
    def _on_signal(self, signal: TradingSignal):
        """Handle trading signals."""
        logger.info(f"Signal received - {signal.symbol}: {signal.action} "
                   f"(confidence: {signal.confidence:.2f})")
        
        # Log signal details
        if signal.metadata:
            logger.debug(f"Signal metadata: {signal.metadata}")
    
    def _on_trade(self, order: Order, position: Optional[Position]):
        """Handle trade executions."""
        logger.info(f"Trade executed - Order: {order.order_id}, "
                   f"Symbol: {order.symbol}, Side: {order.side.value}, "
                   f"Quantity: {order.filled_quantity:.4f}, "
                   f"Price: {order.average_fill_price:.2f}")
        
        if position:
            logger.info(f"Position updated - {position.symbol}: "
                       f"{position.quantity:.4f} @ {position.average_entry_price:.2f}, "
                       f"Unrealized P&L: ${position.unrealized_pnl:.2f}")
    
    def run_demo(self, duration_minutes: int = 5):
        """Run the trading demo."""
        logger.info(f"Starting {duration_minutes}-minute trading demo...")
        
        # Start trading system
        self.trading_system.start()
        
        # Run for specified duration
        start_time = time.time()
        update_interval = 30  # Log updates every 30 seconds
        last_update = start_time
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                current_time = time.time()
                
                # Periodic updates
                if current_time - last_update >= update_interval:
                    self._log_performance()
                    self._check_portfolio_rebalancing()
                    last_update = current_time
                
                time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            # Stop trading system
            self.trading_system.stop()
            
            # Final performance report
            self._generate_final_report()
    
    def _log_performance(self):
        """Log current performance metrics."""
        stats = self.trading_system.get_performance_stats()
        
        logger.info("=== Performance Update ===")
        logger.info(f"Total trades: {stats['total_trades']}")
        logger.info(f"Win rate: {stats.get('winning_trades', 0) / max(stats['total_trades'], 1) * 100:.1f}%")
        logger.info(f"Total P&L: ${stats['total_pnl']:.2f}")
        logger.info(f"Current balance: ${stats['current_balance']:.2f}")
        
        # Log positions
        if stats['open_positions']:
            logger.info("Open positions:")
            for pos in stats['open_positions']:
                logger.info(f"  {pos['symbol']}: {pos['side']} {pos['quantity']:.4f} @ "
                           f"{pos['entry_price']:.2f}, P&L: ${pos['unrealized_pnl']:.2f}")
        
        # Store for analysis
        self.performance_log.append({
            'timestamp': datetime.now(tz=timezone.utc),
            'stats': stats.copy()
        })
    
    def _check_portfolio_rebalancing(self):
        """Check if portfolio needs rebalancing."""
        positions = self.trading_system.positions
        
        if not positions:
            return
        
        # Calculate current weights
        total_value = sum(pos.quantity * pos.current_price for pos in positions.values())
        
        if total_value <= 0:
            return
        
        current_weights = {}
        symbols = []
        
        for key, pos in positions.items():
            symbol = pos.symbol
            weight = (pos.quantity * pos.current_price) / total_value
            current_weights[symbol] = weight
            symbols.append(symbol)
        
        # Get historical returns (mock data for demo)
        returns_data = self._get_mock_returns_data(symbols)
        
        # Optimize portfolio
        assets = [Asset(symbol=s, current_price=100, weight=current_weights.get(s, 0)) 
                 for s in symbols]
        
        optimization_result = self.portfolio_optimizer.optimize_portfolio(
            assets=assets,
            returns_data=returns_data,
            optimization_method='max_sharpe'
        )
        
        optimal_weights = optimization_result['weights']
        
        # Check if rebalancing needed
        current_array = np.array([current_weights.get(s, 0) for s in symbols])
        if self.portfolio_optimizer.check_rebalancing_needed(current_array, optimal_weights):
            logger.info("Portfolio rebalancing recommended:")
            for i, symbol in enumerate(symbols):
                logger.info(f"  {symbol}: {current_array[i]:.1%} -> {optimal_weights[i]:.1%}")
    
    def _get_mock_returns_data(self, symbols: List[str]) -> pd.DataFrame:
        """Generate mock returns data for demo."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Generate correlated returns
        n_assets = len(symbols)
        mean_returns = np.random.uniform(-0.001, 0.002, n_assets)
        volatilities = np.random.uniform(0.01, 0.03, n_assets)
        
        # Correlation matrix
        correlation = 0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)
        
        # Generate returns
        returns = np.random.multivariate_normal(
            mean_returns,
            np.outer(volatilities, volatilities) * correlation,
            size=len(dates)
        )
        
        return pd.DataFrame(returns, index=dates, columns=symbols)
    
    def _generate_final_report(self):
        """Generate final performance report."""
        logger.info("\n" + "="*50)
        logger.info("FINAL TRADING REPORT")
        logger.info("="*50)
        
        stats = self.trading_system.get_performance_stats()
        
        # Trading metrics
        logger.info("\nTrading Metrics:")
        logger.info(f"  Total trades: {stats['total_trades']}")
        logger.info(f"  Winning trades: {stats.get('winning_trades', 0)}")
        logger.info(f"  Win rate: {stats.get('winning_trades', 0) / max(stats['total_trades'], 1) * 100:.1f}%")
        logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
        
        # Return metrics
        initial_balance = 100000
        final_balance = stats['current_balance']
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        logger.info("\nReturn Metrics:")
        logger.info(f"  Initial balance: ${initial_balance:,.2f}")
        logger.info(f"  Final balance: ${final_balance:,.2f}")
        logger.info(f"  Total return: {total_return:.2f}%")
        logger.info(f"  Peak balance: ${stats['peak_balance']:,.2f}")
        
        # Risk metrics
        if 'risk_metrics' in stats:
            logger.info("\nRisk Metrics:")
            logger.info(f"  Current exposure: {stats['risk_metrics']['current_exposure']:.1%}")
            logger.info(f"  Max drawdown: {stats['risk_metrics']['max_drawdown']:.1%}")
        
        # Signal statistics
        if 'signal_stats' in stats:
            logger.info("\nSignal Statistics:")
            logger.info(f"  Total signals: {stats['signal_stats'].get('total_signals', 0)}")
            
            action_counts = stats['signal_stats'].get('action_counts', {})
            for action, count in action_counts.items():
                logger.info(f"  {action.capitalize()} signals: {count}")
        
        # System performance
        if 'pipeline_stats' in stats:
            pipeline = stats['pipeline_stats']['pipeline']
            logger.info("\nSystem Performance:")
            logger.info(f"  Messages processed: {pipeline['messages_processed']}")
            logger.info(f"  Features computed: {pipeline['features_computed']}")
            logger.info(f"  Avg processing time: {pipeline['avg_processing_time']*1000:.1f}ms")
            
            if 'gpu' in stats['pipeline_stats']:
                gpu_info = stats['pipeline_stats']['gpu']
                logger.info(f"  GPU device: {gpu_info['device']}")
                logger.info(f"  GPU memory used: {gpu_info['memory_used']}")
        
        # Trade history
        trades = self.trading_system.get_trade_history(limit=10)
        if trades:
            logger.info("\nRecent Trades:")
            for trade in trades[-5:]:  # Show last 5 trades
                logger.info(f"  {trade['timestamp']}: {trade['side']} {trade['quantity']:.4f} "
                           f"{trade['symbol']} @ {trade['price']:.2f}")
        
        logger.info("\n" + "="*50)
        logger.info("Demo completed successfully!")
        
        # Save performance log
        self._save_performance_log()
    
    def _save_performance_log(self):
        """Save performance log to file."""
        import json
        
        log_file = Path(__file__).parent / 'trading_performance_log.json'
        
        # Convert to serializable format
        serializable_log = []
        for entry in self.performance_log:
            serializable_entry = {
                'timestamp': entry['timestamp'].isoformat(),
                'total_trades': entry['stats']['total_trades'],
                'total_pnl': entry['stats']['total_pnl'],
                'current_balance': entry['stats']['current_balance'],
                'open_positions': len(entry['stats'].get('open_positions', []))
            }
            serializable_log.append(serializable_entry)
        
        with open(log_file, 'w') as f:
            json.dump(serializable_log, f, indent=2)
        
        logger.info(f"Performance log saved to {log_file}")


def main():
    """Run the live trading demo."""
    demo = TradingSystemDemo()
    
    # Initialize systems
    demo.initialize()
    
    # Run demo
    demo.run_demo(duration_minutes=2)  # 2-minute demo


if __name__ == "__main__":
    main()