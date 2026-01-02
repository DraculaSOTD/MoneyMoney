#!/usr/bin/env python3
"""
ML Trading with Backtesting Demo
Shows complete pipeline: training -> prediction -> backtesting -> evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Import components
from training.unified_ml_pipeline import UnifiedMLPipeline
from strategies.ml_strategy import create_ml_signal_generator
from backtesting.engine import BacktestingEngine
from backtesting.metrics import PerformanceMetrics
from models.risk_management.risk_manager import RiskManager

# For data generation
from examples.simple_demo import create_sample_binance_data
from data.enhanced_data_loader import BinanceDataSource


def prepare_data_for_demo(n_samples: int = 5000):
    """Prepare realistic crypto data for demonstration."""
    print("1. Preparing demo data...")
    
    # Generate Binance format data
    df_binance = create_sample_binance_data()
    
    # Add more samples if needed
    if len(df_binance) < n_samples:
        # Generate more data
        n_additional = n_samples - len(df_binance)
        base_price = df_binance['Close'].iloc[-1]
        
        # Continue the time series
        last_time = df_binance['Open time'].iloc[-1]
        new_timestamps = pd.date_range(
            start=pd.to_datetime(last_time, unit='ms') + pd.Timedelta(hours=1),
            periods=n_additional,
            freq='1h'
        )
        
        # Generate continuous prices
        returns = np.random.normal(0.0001, 0.01, n_additional)
        new_prices = base_price * (1 + returns).cumprod()
        
        # Create additional data
        additional_data = pd.DataFrame({
            'Open time': new_timestamps.astype(np.int64) // 10**9 * 1000,
            'Open': new_prices * (1 + np.random.normal(0, 0.001, n_additional)),
            'High': new_prices * (1 + abs(np.random.normal(0, 0.005, n_additional))),
            'Low': new_prices * (1 - abs(np.random.normal(0, 0.005, n_additional))),
            'Close': new_prices,
            'Volume': np.random.lognormal(10, 1, n_additional),
            'Close time': (new_timestamps + pd.Timedelta(hours=1)).astype(np.int64) // 10**9 * 1000,
            'Quote asset volume': np.random.lognormal(15, 1, n_additional),
            'Number of trades': np.random.poisson(100, n_additional),
            'Taker buy base asset volume': np.random.lognormal(9.5, 1, n_additional),
            'Taker buy quote asset volume': np.random.lognormal(14.5, 1, n_additional),
            'Ignore': 0
        })
        
        df_binance = pd.concat([df_binance, additional_data], ignore_index=True)
    
    # Convert to standard format
    data_source = BinanceDataSource('')
    df_standard = data_source._convert_binance_to_standard(df_binance)
    
    print(f"✓ Generated {len(df_standard)} hours of data")
    print(f"✓ Date range: {df_standard['timestamp'].min()} to {df_standard['timestamp'].max()}")
    
    return df_standard


def train_ml_models(df_train: pd.DataFrame, config: dict):
    """Train ML models on historical data."""
    print("\n2. Training ML models...")
    
    # Create pipeline
    pipeline = UnifiedMLPipeline()
    pipeline.config = config
    
    # Run training
    results = pipeline.run_pipeline(df_train)
    
    # Print summary
    print("\nTraining Summary:")
    for model_name, model_results in results.items():
        acc = model_results['evaluation']['classification_report']['accuracy']
        f1 = model_results['evaluation']['classification_report']['macro avg']['f1-score']
        print(f"  {model_name}: Accuracy={acc:.3f}, F1={f1:.3f}")
    
    return pipeline, results


def run_backtest_with_ml(df_test: pd.DataFrame, 
                        trained_models: dict,
                        feature_config: dict,
                        preprocessor):
    """Run backtest using ML predictions."""
    print("\n3. Running backtest with ML strategy...")
    
    # Create ML signal generator
    signal_generator = create_ml_signal_generator(
        trained_models,
        feature_config,
        preprocessor=preprocessor,
        sequence_length=50,
        risk_config={
            'max_position_size': 0.2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
    )
    
    # Position sizing function
    def position_sizer(signal, capital, current_positions):
        """Simple position sizing based on signal confidence."""
        if signal['action'] == 'hold':
            return 0
        
        # Use confidence-based sizing
        base_size = capital * 0.1  # 10% base position
        confidence_multiplier = signal['confidence']
        
        position_size = base_size * confidence_multiplier
        
        # Limit total exposure
        total_exposure = sum(pos['value'] for pos in current_positions.values())
        if total_exposure + position_size > capital * 0.5:  # Max 50% exposure
            position_size = max(0, capital * 0.5 - total_exposure)
        
        return position_size
    
    # Create backtesting engine
    engine = BacktestingEngine(
        initial_capital=10000,
        commission=0.001,  # 0.1% commission
        slippage=0.0005    # 0.05% slippage
    )
    
    # Run backtest
    results = engine.run_backtest(
        data=df_test,
        signal_generator=signal_generator,
        position_sizer=position_sizer,
        stop_loss=0.02,  # 2% stop loss
        take_profit=0.04  # 4% take profit
    )
    
    print(f"✓ Backtest complete: {len(results['trades'])} trades executed")
    
    return results, signal_generator.strategy


def run_benchmark_strategies(df_test: pd.DataFrame):
    """Run benchmark strategies for comparison."""
    print("\n4. Running benchmark strategies...")
    
    benchmarks = {}
    
    # Buy and Hold
    def buy_and_hold_signal(data, positions, **kwargs):
        if not positions:  # No position yet
            return {'action': 'buy', 'confidence': 1.0}
        return {'action': 'hold', 'confidence': 1.0}
    
    # Simple SMA crossover
    def sma_crossover_signal(data, positions, **kwargs):
        if len(data) < 50:
            return {'action': 'hold', 'confidence': 0.0}
        
        sma_short = data['close'].rolling(10).mean().iloc[-1]
        sma_long = data['close'].rolling(50).mean().iloc[-1]
        
        if sma_short > sma_long:
            return {'action': 'buy', 'confidence': 0.7}
        elif sma_short < sma_long:
            return {'action': 'sell', 'confidence': 0.7}
        return {'action': 'hold', 'confidence': 0.5}
    
    # Position sizer for benchmarks
    def simple_position_sizer(signal, capital, positions):
        if signal['action'] == 'buy' and not positions:
            return capital * 0.95  # Use 95% of capital
        return 0
    
    # Run benchmarks
    engine = BacktestingEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
    
    # Buy and Hold
    benchmarks['buy_hold'] = engine.run_backtest(
        data=df_test,
        signal_generator=buy_and_hold_signal,
        position_sizer=simple_position_sizer
    )
    
    # SMA Crossover
    benchmarks['sma'] = engine.run_backtest(
        data=df_test,
        signal_generator=sma_crossover_signal,
        position_sizer=simple_position_sizer
    )
    
    return benchmarks


def visualize_results(ml_results: dict, benchmarks: dict, ml_strategy):
    """Create comprehensive visualization of results."""
    print("\n5. Creating visualizations...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('ML Trading Strategy Performance', fontsize=16)
    
    # 1. Equity curves
    ax1 = axes[0, 0]
    
    # ML strategy
    ml_equity = pd.Series(ml_results['equity_curve'])
    ml_returns = ml_equity.pct_change().fillna(0)
    ax1.plot(ml_equity.index, ml_equity.values, label='ML Strategy', linewidth=2)
    
    # Benchmarks
    for name, benchmark in benchmarks.items():
        equity = pd.Series(benchmark['equity_curve'])
        ax1.plot(equity.index, equity.values, label=name.upper(), alpha=0.7)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Equity Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdowns
    ax2 = axes[0, 1]
    
    # Calculate drawdowns
    ml_cummax = ml_equity.cummax()
    ml_drawdown = (ml_equity - ml_cummax) / ml_cummax * 100
    
    ax2.fill_between(ml_drawdown.index, ml_drawdown.values, 0, 
                     color='red', alpha=0.3, label='ML Drawdown')
    ax2.plot(ml_drawdown.index, ml_drawdown.values, color='red', linewidth=1)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns distribution
    ax3 = axes[1, 0]
    
    # ML returns histogram
    ax3.hist(ml_returns * 100, bins=50, alpha=0.7, label='ML Returns', density=True)
    ax3.axvline(ml_returns.mean() * 100, color='red', linestyle='--', 
                label=f'Mean: {ml_returns.mean()*100:.2f}%')
    
    ax3.set_xlabel('Daily Returns (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Returns Distribution')
    ax3.legend()
    
    # 4. Performance metrics comparison
    ax4 = axes[1, 1]
    
    # Calculate metrics
    metrics_calc = PerformanceMetrics()
    ml_metrics = metrics_calc.calculate_metrics(ml_results)
    
    metrics_data = {
        'ML Strategy': {
            'Sharpe': ml_metrics['sharpe_ratio'],
            'Max DD': ml_metrics['max_drawdown'],
            'Win Rate': ml_metrics['win_rate']
        }
    }
    
    for name, benchmark in benchmarks.items():
        bench_metrics = metrics_calc.calculate_metrics(benchmark)
        metrics_data[name.upper()] = {
            'Sharpe': bench_metrics['sharpe_ratio'],
            'Max DD': bench_metrics['max_drawdown'],
            'Win Rate': bench_metrics['win_rate']
        }
    
    # Plot metrics
    strategies = list(metrics_data.keys())
    sharpe_values = [metrics_data[s]['Sharpe'] for s in strategies]
    
    x = np.arange(len(strategies))
    ax4.bar(x, sharpe_values)
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_title('Sharpe Ratio Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=45)
    
    # Add value labels
    for i, v in enumerate(sharpe_values):
        ax4.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    # 5. ML prediction accuracy over time
    ax5 = axes[2, 0]
    
    if hasattr(ml_strategy, 'prediction_history') and ml_strategy.prediction_history:
        # Rolling accuracy
        df_pred = pd.DataFrame(ml_strategy.prediction_history)
        df_pred['rolling_accuracy'] = df_pred['correct'].rolling(50).mean()
        
        ax5.plot(range(len(df_pred)), df_pred['rolling_accuracy'], label='50-trade MA')
        ax5.axhline(df_pred['correct'].mean(), color='red', linestyle='--', 
                   label=f'Overall: {df_pred["correct"].mean():.2%}')
        
        ax5.set_xlabel('Trade Number')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('ML Prediction Accuracy Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Trade analysis
    ax6 = axes[2, 1]
    
    if ml_results['trades']:
        # Profit/Loss by trade
        trades_df = pd.DataFrame(ml_results['trades'])
        if 'pnl' in trades_df.columns:
            profits = trades_df[trades_df['pnl'] > 0]['pnl']
            losses = trades_df[trades_df['pnl'] <= 0]['pnl']
            
            ax6.hist([profits, losses], bins=30, label=['Profits', 'Losses'], 
                    color=['green', 'red'], alpha=0.7)
            ax6.set_xlabel('P&L ($)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Trade P&L Distribution')
            ax6.legend()
    
    plt.tight_layout()
    plt.savefig('ml_backtest_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to ml_backtest_results.png")
    
    return fig


def print_detailed_report(ml_results: dict, benchmarks: dict, ml_strategy):
    """Print detailed performance report."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE REPORT")
    print("="*80)
    
    # Calculate metrics
    metrics_calc = PerformanceMetrics()
    ml_metrics = metrics_calc.calculate_metrics(ml_results)
    
    print("\n1. ML STRATEGY PERFORMANCE")
    print("-"*40)
    print(f"Total Return: {ml_metrics['total_return']:.2%}")
    print(f"Annual Return: {ml_metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {ml_metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {ml_metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {ml_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {ml_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {ml_metrics['profit_factor']:.2f}")
    print(f"Total Trades: {ml_metrics['total_trades']}")
    
    # ML-specific metrics
    if hasattr(ml_strategy, 'get_performance_summary'):
        ml_summary = ml_strategy.get_performance_summary()
        print(f"\nML Model Performance:")
        print(f"Prediction Accuracy: {ml_summary['accuracy']:.2%}")
        print(f"High-Confidence Accuracy: {ml_summary['confident_accuracy']:.2%}")
        print(f"Average Confidence: {ml_summary['avg_confidence']:.2f}")
    
    print("\n2. BENCHMARK COMPARISON")
    print("-"*40)
    print(f"{'Strategy':<15} {'Return':<12} {'Sharpe':<10} {'Max DD':<10} {'Trades':<8}")
    print("-"*60)
    
    # ML strategy
    print(f"{'ML Strategy':<15} {ml_metrics['total_return']:>10.2%} "
          f"{ml_metrics['sharpe_ratio']:>10.2f} {ml_metrics['max_drawdown']:>10.2%} "
          f"{ml_metrics['total_trades']:>8}")
    
    # Benchmarks
    for name, benchmark in benchmarks.items():
        bench_metrics = metrics_calc.calculate_metrics(benchmark)
        print(f"{name.upper():<15} {bench_metrics['total_return']:>10.2%} "
              f"{bench_metrics['sharpe_ratio']:>10.2f} {bench_metrics['max_drawdown']:>10.2%} "
              f"{bench_metrics['total_trades']:>8}")
    
    print("\n3. RISK ANALYSIS")
    print("-"*40)
    print(f"Value at Risk (95%): ${ml_metrics.get('var_95', 0):.2f}")
    print(f"Conditional VaR (95%): ${ml_metrics.get('cvar_95', 0):.2f}")
    print(f"Kelly Criterion: {ml_metrics.get('kelly_criterion', 0):.2%}")
    
    # Trade analysis
    if ml_results['trades']:
        trades_df = pd.DataFrame(ml_results['trades'])
        
        print("\n4. TRADE ANALYSIS")
        print("-"*40)
        
        if 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")
            print(f"Average Win: ${winning_trades['pnl'].mean():.2f}" if len(winning_trades) > 0 else "Average Win: N/A")
            print(f"Average Loss: ${losing_trades['pnl'].mean():.2f}" if len(losing_trades) > 0 else "Average Loss: N/A")
            print(f"Largest Win: ${winning_trades['pnl'].max():.2f}" if len(winning_trades) > 0 else "Largest Win: N/A")
            print(f"Largest Loss: ${losing_trades['pnl'].min():.2f}" if len(losing_trades) > 0 else "Largest Loss: N/A")
        
        if 'holding_period' in trades_df.columns:
            print(f"\nAverage Holding Period: {trades_df['holding_period'].mean():.1f} hours")
            print(f"Longest Trade: {trades_df['holding_period'].max():.1f} hours")
            print(f"Shortest Trade: {trades_df['holding_period'].min():.1f} hours")


def main():
    """Run complete ML trading demonstration."""
    print("="*80)
    print("ML TRADING WITH BACKTESTING DEMONSTRATION")
    print("="*80)
    
    # Configuration
    config = {
        'data': {
            'sequence_length': 50,
            'lookforward': 5,
            'threshold': 0.002,
            'train_split': 0.6,
            'val_split': 0.2,
            'test_split': 0.2
        },
        'models': {
            'lstm': {
                'enabled': True,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True
            },
            'gru': {
                'enabled': True,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2
            }
        },
        'training': {
            'batch_size': 32,
            'epochs': 20,  # Reduced for demo
            'learning_rate': 0.001,
            'early_stopping_patience': 5,
            'device': 'cpu'
        },
        'preprocessing': {
            'handle_missing': {'enabled': True, 'method': 'forward_fill'},
            'handle_infinite': {'enabled': True, 'method': 'clip'},
            'stationarity': {'enabled': True, 'method': 'pct_change'},
            'outliers': {'enabled': True, 'method': 'iqr'},
            'scaling': {'enabled': True, 'method': 'standard'}
        },
        'indicators': {
            'sma': {'enabled': True, 'periods': [10, 20, 50]},
            'ema': {'enabled': True, 'periods': [12, 26]},
            'rsi': {'enabled': True, 'period': 14},
            'macd': {'enabled': True},
            'bollinger': {'enabled': True},
            'atr': {'enabled': True}
        }
    }
    
    # 1. Prepare data
    df_all = prepare_data_for_demo(n_samples=5000)
    
    # Split data: 60% train, 20% validation (handled by pipeline), 20% test/backtest
    train_size = int(len(df_all) * 0.8)  # 80% for training+validation
    df_train = df_all.iloc[:train_size].copy()
    df_test = df_all.iloc[train_size:].copy()
    
    print(f"\nData split:")
    print(f"  Training set: {len(df_train)} samples")
    print(f"  Test/Backtest set: {len(df_test)} samples")
    
    # 2. Train ML models
    pipeline, training_results = train_ml_models(df_train, config)
    
    # Get trained models and preprocessor
    trained_models = pipeline.models
    preprocessor = pipeline.preprocessor
    feature_config = config
    
    # 3. Run backtest with ML strategy
    ml_results, ml_strategy = run_backtest_with_ml(
        df_test, trained_models, feature_config, preprocessor
    )
    
    # 4. Run benchmark strategies
    benchmarks = run_benchmark_strategies(df_test)
    
    # 5. Visualize results
    visualize_results(ml_results, benchmarks, ml_strategy)
    
    # 6. Print detailed report
    print_detailed_report(ml_results, benchmarks, ml_strategy)
    
    # 7. Save results
    results_summary = {
        'training_results': training_results,
        'ml_backtest': {
            'metrics': PerformanceMetrics().calculate_metrics(ml_results),
            'num_trades': len(ml_results['trades']),
            'final_capital': ml_results['equity_curve'][-1]
        },
        'benchmarks': {
            name: {
                'metrics': PerformanceMetrics().calculate_metrics(results),
                'num_trades': len(results['trades']),
                'final_capital': results['equity_curve'][-1]
            }
            for name, results in benchmarks.items()
        }
    }
    
    with open('ml_backtest_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("\n✓ Results saved to ml_backtest_results.json")
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    
    return ml_results, ml_strategy, benchmarks


if __name__ == "__main__":
    # Check for required packages
    try:
        import torch
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("❌ Missing required packages. Please install: pip install torch matplotlib")
        sys.exit(1)
    
    # Run demonstration
    ml_results, ml_strategy, benchmarks = main()