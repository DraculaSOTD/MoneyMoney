"""
Demonstration of Enhanced Temporal Fusion Transformer capabilities.

Shows:
- Multi-asset portfolio prediction
- Cross-asset attention mechanisms
- Regime-specific predictions
- Adaptive horizon selection
- Interpretability features
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

from enhanced_tft_model import EnhancedTemporalFusionTransformer


def generate_synthetic_multi_asset_data(n_samples: int = 1000, 
                                       n_assets: int = 5,
                                       n_timesteps: int = 192,
                                       n_features: int = 10):
    """Generate synthetic multi-asset time series data."""
    np.random.seed(42)
    
    # Generate correlated asset prices
    correlation_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation_matrix[i, j] = correlation_matrix[j, i] = np.random.uniform(0.3, 0.7)
    
    # Generate base returns
    base_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation_matrix * 0.02,  # 2% volatility
        size=(n_samples, n_timesteps)
    )
    
    # Create price series
    prices = np.zeros((n_samples, n_assets, n_timesteps))
    for i in range(n_assets):
        prices[:, i, :] = 100 * np.exp(np.cumsum(base_returns[:, :, i], axis=1))
    
    # Generate additional features
    temporal_features = np.zeros((n_samples, n_assets, n_timesteps, n_features))
    
    for asset_idx in range(n_assets):
        # Price (normalized)
        temporal_features[:, asset_idx, :, 0] = prices[:, asset_idx, :] / 100
        
        # Returns
        temporal_features[:, asset_idx, 1:, 1] = np.diff(prices[:, asset_idx, :], axis=1) / prices[:, asset_idx, :-1]
        
        # Moving averages
        for window in [5, 10, 20]:
            ma_idx = 2 + [5, 10, 20].index(window)
            for sample in range(n_samples):
                ma = pd.Series(prices[sample, asset_idx, :]).rolling(window).mean().fillna(method='bfill').values
                temporal_features[sample, asset_idx, :, ma_idx] = ma / 100
        
        # Volatility (20-period)
        for sample in range(n_samples):
            vol = pd.Series(temporal_features[sample, asset_idx, :, 1]).rolling(20).std().fillna(0.02).values
            temporal_features[sample, asset_idx, :, 5] = vol
        
        # Volume (synthetic)
        temporal_features[:, asset_idx, :, 6] = np.random.lognormal(10, 1, (n_samples, n_timesteps))
        
        # RSI (simplified)
        for sample in range(n_samples):
            returns = temporal_features[sample, asset_idx, :, 1]
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            
            avg_gains = pd.Series(gains).rolling(14).mean().fillna(0).values
            avg_losses = pd.Series(losses).rolling(14).mean().fillna(0).values
            
            rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
            rsi = 100 - (100 / (1 + rs))
            temporal_features[sample, asset_idx, :, 7] = rsi / 100
        
        # Market cap rank (static but included in temporal)
        temporal_features[:, asset_idx, :, 8] = (asset_idx + 1) / n_assets
        
        # Correlation with market (using asset 0 as market proxy)
        if asset_idx > 0:
            for sample in range(n_samples):
                corr = pd.Series(prices[sample, asset_idx, :]).rolling(30).corr(
                    pd.Series(prices[sample, 0, :])
                ).fillna(0.5).values
                temporal_features[sample, asset_idx, :, 9] = corr
        else:
            temporal_features[:, asset_idx, :, 9] = 1.0  # Market proxy has perfect correlation with itself
    
    # Generate static features
    static_features = np.zeros((n_samples, n_assets, 5))
    for asset_idx in range(n_assets):
        static_features[:, asset_idx, 0] = asset_idx / n_assets  # Asset type encoding
        static_features[:, asset_idx, 1] = np.random.uniform(0.5, 2.0)  # Beta
        static_features[:, asset_idx, 2] = np.random.uniform(10, 100)  # Market cap (billions)
        static_features[:, asset_idx, 3] = np.random.uniform(0.1, 0.5)  # Historical volatility
        static_features[:, asset_idx, 4] = np.random.uniform(-0.5, 0.5)  # Sentiment score
    
    # Generate market data for regime detection
    market_data = np.zeros((n_timesteps, 3))
    market_data[:, 0] = np.mean(prices[0, :, :], axis=0)  # Average price
    market_data[:, 1] = np.sum(temporal_features[0, :, :, 6], axis=0)  # Total volume
    market_data[:, 2] = np.mean(temporal_features[0, :, :, 5], axis=0)  # Average volatility
    
    return temporal_features, static_features, market_data, prices


def demonstrate_multi_asset_prediction():
    """Demonstrate multi-asset portfolio prediction."""
    print("\n" + "="*80)
    print("MULTI-ASSET PORTFOLIO PREDICTION DEMONSTRATION")
    print("="*80)
    
    # Generate data
    n_samples = 100
    n_assets = 5
    n_timesteps = 192  # 168 encoder + 24 decoder
    
    temporal_features, static_features, market_data, prices = generate_synthetic_multi_asset_data(
        n_samples, n_assets, n_timesteps
    )
    
    # Initialize model
    model = EnhancedTemporalFusionTransformer(
        n_encoder_steps=168,
        n_prediction_steps=24,
        n_features=10,
        n_static_features=5,
        n_assets=n_assets,
        hidden_size=128,
        lstm_layers=2,
        num_attention_heads=4,
        dropout_rate=0.1,
        enable_regime_specific=True,
        enable_cross_attention=True,
        enable_adaptive_horizon=True
    )
    
    print(f"\nModel initialized with {n_assets} assets")
    print(f"Total parameters: {sum(p.size for p in model.params.values()):,}")
    
    # Prepare inputs
    batch_idx = 0
    inputs = {
        'temporal_inputs': temporal_features[batch_idx:batch_idx+1],
        'static_inputs': static_features[batch_idx:batch_idx+1],
        'market_data': market_data,
        'portfolio_weights': np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # Custom weights
    }
    
    # Make predictions
    print("\n--- Making Multi-Asset Predictions ---")
    outputs = model.forward_multi_asset(inputs)
    
    print(f"Current market regime: {outputs['current_regime']}")
    print(f"Selected prediction horizon: {outputs['prediction_horizon']} minutes")
    
    # Display predictions for each asset
    print("\n--- Individual Asset Predictions (24-hour horizon) ---")
    for asset_idx, asset_pred in enumerate(outputs['asset_predictions']):
        q50_pred = asset_pred['q50'][0, -1]  # Last time step prediction
        q10_pred = asset_pred['q10'][0, -1] 
        q90_pred = asset_pred['q90'][0, -1]
        
        print(f"\nAsset {asset_idx + 1}:")
        print(f"  Median (Q50): {q50_pred:.4f}")
        print(f"  Lower bound (Q10): {q10_pred:.4f}")
        print(f"  Upper bound (Q90): {q90_pred:.4f}")
        print(f"  Uncertainty range: {(q90_pred - q10_pred):.4f}")
    
    # Portfolio prediction
    print("\n--- Portfolio-Level Prediction ---")
    portfolio_pred = outputs['portfolio_prediction']
    print(f"Portfolio median return: {portfolio_pred['q50'][0, -1]:.4f}")
    print(f"Portfolio uncertainty: {(portfolio_pred['q90'][0, -1] - portfolio_pred['q10'][0, -1]):.4f}")
    
    # Cross-asset attention
    if outputs['cross_asset_attention'] is not None:
        print("\n--- Cross-Asset Attention Weights ---")
        attention = outputs['cross_asset_attention'][0]  # First sample
        print("Asset correlation matrix from attention:")
        print(attention)
    
    return model, inputs, outputs


def demonstrate_regime_adaptation():
    """Demonstrate regime-specific predictions."""
    print("\n" + "="*80)
    print("REGIME-SPECIFIC ADAPTATION DEMONSTRATION")
    print("="*80)
    
    # Generate data with different regimes
    temporal_features, static_features, _, prices = generate_synthetic_multi_asset_data()
    
    # Create different market conditions
    market_conditions = {
        "trending_up": {
            "price_trend": 0.02,
            "volatility": 0.015,
            "volume_trend": 1.2
        },
        "trending_down": {
            "price_trend": -0.02,
            "volatility": 0.025,
            "volume_trend": 1.5
        },
        "volatile": {
            "price_trend": 0.0,
            "volatility": 0.04,
            "volume_trend": 2.0
        },
        "ranging": {
            "price_trend": 0.0,
            "volatility": 0.01,
            "volume_trend": 0.8
        }
    }
    
    # Initialize model
    model = EnhancedTemporalFusionTransformer(
        n_assets=5,
        enable_regime_specific=True
    )
    
    # Test predictions under different regimes
    results = {}
    
    for regime_name, conditions in market_conditions.items():
        print(f"\n--- Testing {regime_name.upper()} Market Regime ---")
        
        # Create market data for this regime
        n_steps = 192
        market_data = np.zeros((n_steps, 3))
        
        # Trending price
        base_price = 100
        trend = conditions["price_trend"]
        market_data[:, 0] = base_price * np.exp(np.cumsum(
            np.random.normal(trend, conditions["volatility"], n_steps)
        ))
        
        # Volume with trend
        market_data[:, 1] = 1000000 * conditions["volume_trend"] * np.random.lognormal(0, 0.3, n_steps)
        
        # Volatility
        market_data[:, 2] = conditions["volatility"]
        
        # Make predictions
        inputs = {
            'temporal_inputs': temporal_features[0:1],
            'static_inputs': static_features[0:1],
            'market_data': market_data
        }
        
        outputs = model.forward_multi_asset(inputs)
        
        # Store results
        results[regime_name] = {
            'regime': outputs['current_regime'],
            'horizon': outputs['prediction_horizon'],
            'predictions': outputs['asset_predictions'],
            'uncertainty': np.mean([
                pred['q90'][0, -1] - pred['q10'][0, -1] 
                for pred in outputs['asset_predictions']
            ])
        }
        
        print(f"  Detected regime: {outputs['current_regime']}")
        print(f"  Adaptive horizon: {outputs['prediction_horizon']} minutes")
        print(f"  Average uncertainty: {results[regime_name]['uncertainty']:.4f}")
    
    # Compare predictions across regimes
    print("\n--- Regime Comparison Summary ---")
    print("Regime        | Horizon | Avg Uncertainty")
    print("-" * 40)
    for regime, data in results.items():
        print(f"{regime:12} | {data['horizon']:7} | {data['uncertainty']:.4f}")
    
    return results


def demonstrate_interpretability():
    """Demonstrate model interpretability features."""
    print("\n" + "="*80)
    print("MODEL INTERPRETABILITY DEMONSTRATION")
    print("="*80)
    
    # Generate data
    temporal_features, static_features, market_data, _ = generate_synthetic_multi_asset_data()
    
    # Initialize model
    model = EnhancedTemporalFusionTransformer(
        n_assets=5,
        enable_regime_specific=True,
        enable_cross_attention=True
    )
    
    # Get interpretability outputs
    inputs = {
        'temporal_inputs': temporal_features[0:1],
        'static_inputs': static_features[0:1],
        'market_data': market_data
    }
    
    interpretability = model.get_interpretability_outputs(inputs)
    
    print(f"\nCurrent Market Regime: {interpretability['current_regime']}")
    print(f"Selected Prediction Horizon: {interpretability['selected_horizon']} minutes")
    
    # Feature importance for each asset
    print("\n--- Feature Importance by Asset ---")
    feature_names = ['Price', 'Returns', 'MA5', 'MA10', 'MA20', 'Volatility', 
                     'Volume', 'RSI', 'Market Cap Rank', 'Market Correlation']
    
    for asset_idx, importance in enumerate(interpretability['asset_feature_importance']):
        print(f"\nAsset {asset_idx + 1}:")
        
        # Temporal importance
        temp_imp = importance['temporal_past_importance'][0]  # First sample
        top_features_idx = np.argsort(temp_imp)[-3:][::-1]  # Top 3 features
        
        print("  Top temporal features:")
        for idx in top_features_idx:
            if idx < len(feature_names):
                print(f"    - {feature_names[idx]}: {temp_imp[idx]:.3f}")
    
    # Cross-asset attention analysis
    if interpretability['cross_asset_attention'] is not None:
        print("\n--- Cross-Asset Dependencies ---")
        attention = interpretability['cross_asset_attention'][0]  # First sample
        
        for i in range(len(attention)):
            # Find strongest dependencies
            deps = [(j, attention[i, j]) for j in range(len(attention)) if i != j]
            deps.sort(key=lambda x: x[1], reverse=True)
            
            if deps:
                print(f"\nAsset {i + 1} depends most on:")
                for asset_j, weight in deps[:2]:  # Top 2 dependencies
                    print(f"  - Asset {asset_j + 1}: {weight:.3f}")
    
    return interpretability


def demonstrate_training():
    """Demonstrate model training with multi-asset data."""
    print("\n" + "="*80)
    print("MULTI-ASSET TRAINING DEMONSTRATION")
    print("="*80)
    
    # Generate training data
    n_samples = 50
    temporal_features, static_features, market_data, prices = generate_synthetic_multi_asset_data(
        n_samples=n_samples
    )
    
    # Initialize model
    model = EnhancedTemporalFusionTransformer(
        n_assets=5,
        hidden_size=64,  # Smaller for faster demo
        enable_regime_specific=True,
        enable_cross_attention=True,
        enable_adaptive_horizon=True
    )
    
    # Training loop
    n_epochs = 5
    batch_size = 10
    
    print(f"\nTraining for {n_epochs} epochs with batch size {batch_size}")
    
    training_history = {
        'total_loss': [],
        'portfolio_loss': [],
        'asset_losses': []
    }
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        for batch_start in range(0, n_samples - batch_size, batch_size):
            # Prepare batch
            batch_temporal = temporal_features[batch_start:batch_start+batch_size]
            batch_static = static_features[batch_start:batch_start+batch_size]
            
            inputs = {
                'temporal_inputs': batch_temporal,
                'static_inputs': batch_static,
                'market_data': market_data
            }
            
            # Create synthetic targets (next period returns)
            targets = {}
            for asset_idx in range(5):
                for q, quantile in [(10, 0.1), (50, 0.5), (90, 0.9)]:
                    # Synthetic targets based on historical volatility
                    base_return = np.random.normal(0, 0.02, (batch_size, 24))
                    quantile_adjustment = np.random.normal(0, 0.01) * (quantile - 0.5)
                    targets[f'asset_{asset_idx}_q{q}'] = base_return + quantile_adjustment
            
            # Portfolio target
            targets['portfolio_q50'] = np.mean([
                targets[f'asset_{i}_q50'] for i in range(5)
            ], axis=0)
            
            # Training step
            losses = model.train_step(inputs, targets, learning_rate=0.001)
            epoch_losses.append(losses['total_loss'])
        
        # Record epoch statistics
        avg_loss = np.mean(epoch_losses)
        training_history['total_loss'].append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['total_loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Multi-Asset TFT Training Progress')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nTraining completed!")
    
    # Show adaptive horizon statistics
    if model.enable_adaptive_horizon:
        horizon_stats = model.horizon_selector.horizon_performance
        if horizon_stats:
            print("\n--- Adaptive Horizon Performance ---")
            for horizon, perfs in horizon_stats.items():
                if perfs:
                    print(f"Horizon {horizon}: Avg Performance {np.mean(perfs):.4f}")
    
    return model, training_history


def visualize_predictions():
    """Visualize multi-asset predictions."""
    print("\n" + "="*80)
    print("PREDICTION VISUALIZATION")
    print("="*80)
    
    # Generate data
    temporal_features, static_features, market_data, prices = generate_synthetic_multi_asset_data()
    
    # Initialize model
    model = EnhancedTemporalFusionTransformer(n_assets=5)
    
    # Make predictions
    inputs = {
        'temporal_inputs': temporal_features[0:1],
        'static_inputs': static_features[0:1],
        'market_data': market_data
    }
    
    outputs = model.forward_multi_asset(inputs)
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot predictions for each asset
    for asset_idx in range(5):
        ax = axes[asset_idx]
        
        # Historical prices
        historical = prices[0, asset_idx, -48:-24]  # Last 24 hours before prediction
        time_hist = np.arange(-24, 0)
        
        # Predictions
        pred_median = outputs['asset_predictions'][asset_idx]['q50'][0]
        pred_lower = outputs['asset_predictions'][asset_idx]['q10'][0]
        pred_upper = outputs['asset_predictions'][asset_idx]['q90'][0]
        time_pred = np.arange(0, 24)
        
        # Plot
        ax.plot(time_hist, historical, 'b-', label='Historical', linewidth=2)
        ax.plot(time_pred, pred_median * 100, 'r-', label='Prediction (Q50)', linewidth=2)
        ax.fill_between(time_pred, pred_lower * 100, pred_upper * 100, 
                        color='red', alpha=0.2, label='Q10-Q90 Range')
        
        ax.set_title(f'Asset {asset_idx + 1}')
        ax.set_xlabel('Hours')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Portfolio prediction
    ax = axes[5]
    portfolio_pred = outputs['portfolio_prediction']
    
    ax.plot(time_pred, portfolio_pred['q50'][0], 'g-', label='Portfolio Q50', linewidth=3)
    ax.fill_between(time_pred, portfolio_pred['q10'][0], portfolio_pred['q90'][0],
                    color='green', alpha=0.2, label='Portfolio Q10-Q90')
    
    ax.set_title('Portfolio Aggregate Prediction')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Multi-Asset Predictions - Regime: {outputs['current_regime']}", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("ENHANCED TEMPORAL FUSION TRANSFORMER DEMONSTRATION")
    print("="*80)
    
    # Multi-asset prediction
    model, inputs, outputs = demonstrate_multi_asset_prediction()
    
    # Regime adaptation
    regime_results = demonstrate_regime_adaptation()
    
    # Interpretability
    interpretability = demonstrate_interpretability()
    
    # Training demonstration
    trained_model, history = demonstrate_training()
    
    # Visualization
    visualize_predictions()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("✓ Multi-asset portfolio predictions")
    print("✓ Cross-asset attention mechanisms")
    print("✓ Regime-specific adaptations")
    print("✓ Adaptive horizon selection")
    print("✓ Comprehensive interpretability")
    print("✓ Multi-asset training workflow")


if __name__ == "__main__":
    main()