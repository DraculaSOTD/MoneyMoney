#!/usr/bin/env python3
"""
ML Training Pipeline Demonstration
Shows end-to-end training with multiple models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import the unified pipeline
from training.unified_ml_pipeline import UnifiedMLPipeline
from data.enhanced_data_loader import BinanceDataSource


def create_realistic_data(n_samples: int = 5000):
    """Create realistic crypto trading data for demonstration."""
    np.random.seed(42)
    
    # Generate time series
    timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq='1h')
    
    # Generate realistic price movement with trends
    # Add trend component
    trend = np.linspace(100, 150, n_samples) + np.sin(np.linspace(0, 4*np.pi, n_samples)) * 20
    
    # Add noise
    noise = np.random.normal(0, 2, n_samples)
    
    # Add volatility clusters
    volatility = np.ones(n_samples)
    for i in range(10):
        start = np.random.randint(0, n_samples-100)
        volatility[start:start+100] *= np.random.uniform(1.5, 3)
    
    prices = trend + noise * volatility
    
    # Ensure positive prices
    prices = np.maximum(prices, 10)
    
    # Create OHLC data
    df = pd.DataFrame({
        'Open time': timestamps.astype(np.int64) // 10**9 * 1000,
        'Open': prices * (1 + np.random.normal(0, 0.002, n_samples)),
        'High': prices * (1 + abs(np.random.normal(0, 0.01, n_samples))),
        'Low': prices * (1 - abs(np.random.normal(0, 0.01, n_samples))),
        'Close': prices,
        'Volume': np.random.lognormal(10, 1.5, n_samples),
        'Close time': (timestamps + pd.Timedelta(hours=1)).astype(np.int64) // 10**9 * 1000,
        'Quote asset volume': np.random.lognormal(15, 1.5, n_samples),
        'Number of trades': np.random.poisson(150, n_samples),
        'Taker buy base asset volume': np.random.lognormal(9.5, 1.5, n_samples),
        'Taker buy quote asset volume': np.random.lognormal(14.5, 1.5, n_samples),
        'Ignore': 0
    })
    
    # Add some market events (sharp moves)
    for _ in range(5):
        event_idx = np.random.randint(100, n_samples-100)
        event_size = np.random.choice([-1, 1]) * np.random.uniform(5, 15)
        df.loc[event_idx:event_idx+20, ['Open', 'High', 'Low', 'Close']] *= (1 + event_size/100)
        df.loc[event_idx:event_idx+20, 'Volume'] *= 3
    
    return df


def visualize_results(results: dict, save_path: str = 'ml_results_visualization.png'):
    """Create visualization of training results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ML Training Results', fontsize=16)
    
    # 1. Training curves
    ax1 = axes[0, 0]
    for model_name, model_results in results.items():
        history = model_results['history']
        ax1.plot(history['train_loss'], label=f'{model_name} train', alpha=0.7)
        ax1.plot(history['val_loss'], label=f'{model_name} val', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model accuracy comparison
    ax2 = axes[0, 1]
    accuracies = []
    model_names = []
    for model_name, model_results in results.items():
        accuracy = model_results['evaluation']['classification_report']['accuracy']
        accuracies.append(accuracy)
        model_names.append(model_name.upper())
    
    bars = ax2.bar(model_names, accuracies)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Model Performance Comparison')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 3. Confusion matrix for best model
    ax3 = axes[1, 0]
    best_model = max(results.items(), 
                    key=lambda x: x[1]['evaluation']['classification_report']['accuracy'])
    best_name, best_results = best_model
    
    conf_matrix = np.array(best_results['evaluation']['confusion_matrix'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sell', 'Buy', 'Hold'],
                yticklabels=['Sell', 'Buy', 'Hold'],
                ax=ax3)
    ax3.set_title(f'Confusion Matrix - {best_name.upper()}')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. F1 scores comparison
    ax4 = axes[1, 1]
    f1_scores = {}
    for model_name, model_results in results.items():
        report = model_results['evaluation']['classification_report']
        f1_scores[model_name] = {
            'sell': report['0']['f1-score'],
            'buy': report['1']['f1-score'],
            'macro': report['macro avg']['f1-score']
        }
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax4.bar(x - width, [f1_scores[m.lower()]['sell'] for m in model_names], 
            width, label='Sell', alpha=0.8)
    ax4.bar(x, [f1_scores[m.lower()]['buy'] for m in model_names], 
            width, label='Buy', alpha=0.8)
    ax4.bar(x + width, [f1_scores[m.lower()]['macro'] for m in model_names], 
            width, label='Macro Avg', alpha=0.8)
    
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Scores by Model and Class')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    
    return fig


def print_detailed_results(results: dict):
    """Print detailed results for each model."""
    print("\n" + "="*80)
    print("DETAILED MODEL RESULTS")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} Model:")
        print("-"*40)
        
        # Training info
        history = model_results['history']
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Best val accuracy: {max(history['val_accuracy']):.2f}%")
        print(f"Epochs trained: {len(history['train_loss'])}")
        
        # Test results
        eval_results = model_results['evaluation']
        report = eval_results['classification_report']
        
        print(f"\nTest Performance:")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  Precision (macro): {report['macro avg']['precision']:.4f}")
        print(f"  Recall (macro): {report['macro avg']['recall']:.4f}")
        print(f"  F1 (macro): {report['macro avg']['f1-score']:.4f}")
        
        # Per-class results
        print(f"\nPer-class F1 scores:")
        print(f"  Sell (0): {report['0']['f1-score']:.4f}")
        print(f"  Buy (1): {report['1']['f1-score']:.4f}")
        if '2' in report:
            print(f"  Hold (2): {report['2']['f1-score']:.4f}")
    
    # Best model
    best_model = max(results.items(), 
                    key=lambda x: x[1]['evaluation']['classification_report']['accuracy'])
    print(f"\n{'='*40}")
    print(f"BEST MODEL: {best_model[0].upper()}")
    print(f"Test Accuracy: {best_model[1]['evaluation']['classification_report']['accuracy']:.4f}")
    print(f"{'='*40}")


def main():
    """Run the ML training demonstration."""
    print("="*80)
    print("ML TRAINING PIPELINE DEMONSTRATION")
    print("="*80)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # 1. Generate realistic data
    print("\n1. Generating realistic crypto data...")
    df_binance = create_realistic_data(n_samples=5000)
    print(f"✓ Generated {len(df_binance)} hours of data")
    
    # 2. Configure pipeline
    print("\n2. Configuring ML pipeline...")
    config = {
        'data': {
            'sequence_length': 50,  # Shorter for demo
            'lookforward': 5,
            'threshold': 0.002,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        },
        'models': {
            'lstm': {
                'enabled': True,
                'hidden_size': 64,  # Smaller for demo
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True
            },
            'gru': {
                'enabled': True,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2
            },
            'cnn_lstm': {
                'enabled': True,
                'cnn_filters': [32, 64],  # Smaller for demo
                'kernel_sizes': [3, 5],
                'lstm_hidden': 64,
                'lstm_layers': 1
            }
        },
        'training': {
            'batch_size': 32,
            'epochs': 30,  # Fewer epochs for demo
            'learning_rate': 0.001,
            'early_stopping_patience': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
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
            'atr': {'enabled': True},
            'adx': {'enabled': True},
            'stochastic': {'enabled': True}
        }
    }
    
    # Create pipeline with custom config
    pipeline = UnifiedMLPipeline()
    pipeline.config = config
    
    print(f"✓ Pipeline configured")
    print(f"  - Models: {', '.join([k for k, v in config['models'].items() if v['enabled']])}")
    print(f"  - Device: {config['training']['device']}")
    
    # 3. Run training pipeline
    print("\n3. Running ML training pipeline...")
    print("-"*40)
    
    try:
        results = pipeline.run_pipeline(df_binance)
        
        # 4. Visualize results
        print("\n4. Creating visualizations...")
        visualize_results(results, 'ml_training_results.png')
        
        # 5. Print detailed results
        print_detailed_results(results)
        
        # 6. Save model comparison
        print("\n6. Model Comparison Summary:")
        print("-"*40)
        print(f"{'Model':<15} {'Accuracy':<10} {'F1 (macro)':<10} {'Training Time':<15}")
        print("-"*40)
        
        for model_name, model_results in results.items():
            acc = model_results['evaluation']['classification_report']['accuracy']
            f1 = model_results['evaluation']['classification_report']['macro avg']['f1-score']
            epochs = len(model_results['history']['train_loss'])
            print(f"{model_name.upper():<15} {acc:<10.4f} {f1:<10.4f} {epochs} epochs")
        
        print("\n✓ ML training demonstration complete!")
        print(f"✓ Models saved in current directory")
        print(f"✓ Results visualization saved as 'ml_training_results.png'")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Import torch here to check availability
    try:
        import torch
        print(f"PyTorch available: {torch.cuda.is_available()}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        print("❌ PyTorch not installed. Please install with: pip install torch")
        sys.exit(1)
    
    results = main()