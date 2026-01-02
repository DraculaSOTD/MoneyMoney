#!/usr/bin/env python3
"""
Demonstration of ML integration with enhanced features from the ML project.
Shows data preprocessing, feature engineering, labeling, and model training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import our modules
from data.enhanced_data_loader import EnhancedDataLoader
from features.decision_labeler import DecisionLabeler, LabelingMethod
from features.ml_feature_engineering import MLFeatureEngineering
from models.ml.lstm_models import BidirectionalLSTMModel
from models.ml.gru_models import GRUModel
from models.ml.hybrid_models import HybridCNNLSTM


def load_and_prepare_data():
    """Load data and compute indicators."""
    print("="*60)
    print("ML INTEGRATION DEMONSTRATION")
    print("="*60)
    print("\n1. Loading Data with Enhanced Features...")
    
    # Initialize loader
    loader = EnhancedDataLoader(data_dir="./data/historical")
    
    try:
        # Load data with indicators
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        df = loader.load_data_with_indicators(
            symbol="BTCUSDT",
            start_time=start_time,
            end_time=end_time,
            interval="1m",
            indicators=['macd', 'rsi', 'bollinger_bands', 'atr']
        )
        
        print(f"✓ Loaded {len(df)} rows with {len(df.columns)} features")
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data
        from data.data_loader import create_synthetic_data
        df = create_synthetic_data(
            symbol="BTCUSDT",
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            interval="1m",
            initial_price=50000.0,
            volatility=0.02
        )
        
        # Add indicators
        from features.enhanced_technical_indicators import EnhancedTechnicalIndicators
        df = EnhancedTechnicalIndicators.compute_all_indicators(df)
        print(f"✓ Created synthetic data with {len(df)} rows")
    
    return df


def create_ml_features(df):
    """Create ML-specific features."""
    print("\n2. Engineering ML Features...")
    
    # Initialize feature engineering
    ml_features = MLFeatureEngineering()
    
    # Create comprehensive ML features
    df_ml = ml_features.prepare_ml_features(df, {
        'percentage_features': True,
        'lagged_features': True,
        'rolling_features': True,
        'time_features': True,
        'indicator_features': True
    })
    
    print(f"✓ Created {len(df_ml.columns) - len(df.columns)} new features")
    
    # Show sample features
    new_features = [col for col in df_ml.columns if col not in df.columns]
    print("\nSample ML features created:")
    for feat in new_features[:10]:
        print(f"  • {feat}")
    
    return df_ml, ml_features


def create_labels(df):
    """Create trading labels for supervised learning."""
    print("\n3. Creating Trading Labels...")
    
    # Initialize labeler
    labeler = DecisionLabeler(method=LabelingMethod.PRICE_DIRECTION)
    
    # Create labels
    df_labeled = labeler.create_labels(
        df,
        lookforward=5,  # Look 5 periods ahead
        threshold=0.001  # 0.1% threshold
    )
    
    # Get label distribution
    dist = labeler.get_label_distribution(df_labeled)
    
    print("✓ Label distribution:")
    for decision, pct in dist['decision_percentages'].items():
        print(f"  • {decision}: {pct:.1f}%")
    
    # Validate labels
    validation = labeler.validate_labels(df_labeled)
    if validation['recommendations']:
        print("\nRecommendations:")
        for rec in validation['recommendations']:
            print(f"  ⚠️ {rec}")
    
    return df_labeled


def prepare_sequences(df, ml_features):
    """Prepare sequences for LSTM/GRU models."""
    print("\n4. Preparing Sequences for Deep Learning...")
    
    # Remove rows with NaN labels
    df_clean = df.dropna(subset=['label'])
    
    # Select features (exclude metadata and labels)
    exclude_cols = ['timestamp', 'symbol', 'label', 'decision', 'future_return']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    # Scale features
    df_scaled = ml_features.scale_features(df_clean[feature_cols])
    
    # Create sequences
    sequence_length = 50
    X, y = ml_features.create_sequences(
        pd.concat([df_scaled, df_clean[['label']]], axis=1),
        sequence_length=sequence_length,
        target_col='label',
        feature_cols=df_scaled.columns.tolist()
    )
    
    print(f"✓ Created sequences:")
    print(f"  • Input shape: {X.shape}")
    print(f"  • Target shape: {y.shape}")
    print(f"  • Features per timestep: {X.shape[2]}")
    
    return X, y, feature_cols


def train_models(X, y):
    """Train multiple ML models."""
    print("\n5. Training ML Models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"✓ Train/Test split:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Test samples: {len(X_test)}")
    
    # Models to train
    models = {
        'Bidirectional LSTM': BidirectionalLSTMModel({
            'lstm_units': [50, 50],
            'dense_units': [50],
            'dropout_rate': 0.2,
            'epochs': 10,  # Reduced for demo
            'batch_size': 32
        }),
        'GRU': GRUModel({
            'gru_units': [50, 50],
            'dense_units': [50],
            'dropout_rate': 0.2,
            'epochs': 10,
            'batch_size': 32
        }),
        'Hybrid CNN-LSTM': HybridCNNLSTM({
            'cnn_filters': [64],
            'lstm_units': [50],
            'dense_units': [50],
            'dropout_rate': 0.2,
            'epochs': 10,
            'batch_size': 32
        })
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n Training {name}...")
        
        # Train model
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            verbose=0
        )
        
        # Evaluate
        eval_results = model.evaluate(X_test, y_test)
        results[name] = eval_results
        
        print(f"  • Test Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  • Test Loss: {eval_results['loss']:.4f}")
    
    return models, results


def visualize_results(models, results, X_test, y_test):
    """Visualize model predictions and performance."""
    print("\n6. Visualizing Results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Plot 1: Model Accuracies
    ax = axes[0]
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    bars = ax.bar(model_names, accuracies)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 2: Confusion Matrix for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_name]
    
    ax = axes[1]
    cm = np.array(results[best_model_name]['confusion_matrix'])
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Add labels
    classes = ['Buy', 'Sell', 'Hold']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # Plot 3: Sample predictions
    ax = axes[2]
    sample_idx = min(100, len(y_test))
    predictions = best_model.predict(X_test[:sample_idx])
    
    ax.plot(y_test[:sample_idx], label='Actual', alpha=0.7)
    ax.plot(predictions, label='Predicted', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Class')
    ax.set_title('Sample Predictions')
    ax.legend()
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Buy', 'Sell', 'Hold'])
    
    # Plot 4: Classification Report
    ax = axes[3]
    ax.axis('off')
    
    # Create text summary
    report_text = f"Best Model: {best_model_name}\\n\\n"
    report_text += "Classification Report:\\n"
    report = results[best_model_name]['classification_report']
    
    for label in ['buy', 'sell', 'hold']:
        if label in report:
            metrics = report[label]
            report_text += f"\\n{label.upper()}:\\n"
            report_text += f"  Precision: {metrics['precision']:.3f}\\n"
            report_text += f"  Recall: {metrics['recall']:.3f}\\n"
            report_text += f"  F1-Score: {metrics['f1-score']:.3f}\\n"
    
    ax.text(0.1, 0.9, report_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    output_path = './examples/ml_results_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()


def main():
    """Run complete ML integration demo."""
    # Load data
    df = load_and_prepare_data()
    
    # Create ML features
    df_ml, ml_features = create_ml_features(df)
    
    # Create labels
    df_labeled = create_labels(df_ml)
    
    # Prepare sequences
    X, y, feature_names = prepare_sequences(df_labeled, ml_features)
    
    if len(X) < 100:
        print("\n⚠️ Insufficient data for meaningful model training.")
        print("Please ensure you have enough historical data.")
        return
    
    # Train models
    models, results = train_models(X, y)
    
    # Visualize results
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    visualize_results(models, results, X_test, y_test)
    
    # Summary
    print("\n" + "="*60)
    print("ML INTEGRATION COMPLETE")
    print("="*60)
    print("\nKey Accomplishments:")
    print("✅ Enhanced data loading with auto-format detection")
    print("✅ Comprehensive feature engineering (percentage, lags, rolling)")
    print("✅ Multiple labeling strategies for supervised learning")
    print("✅ Deep learning models (LSTM, GRU, CNN-LSTM)")
    print("✅ Model evaluation and visualization")
    print("\nFeatures from ML Project Successfully Integrated:")
    print("• Percentage change transformations")
    print("• Lagged features (T1, T2)")
    print("• 30-day cumulative returns")
    print("• RSI returns (RSI_Ret)")
    print("• Unix timestamp conversion")
    print("• Forward-looking labels")
    print("• Bidirectional LSTM architecture")
    print("• Model persistence and loading")


if __name__ == "__main__":
    main()