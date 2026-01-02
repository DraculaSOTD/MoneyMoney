# Enhanced Temporal Fusion Transformer (TFT) Guide

## Overview

The Enhanced TFT extends the base Temporal Fusion Transformer with advanced features specifically designed for multi-asset cryptocurrency trading:

- **Multi-Asset Support**: Joint modeling of multiple assets with cross-asset dependencies
- **Cross-Attention Mechanisms**: Capture relationships between different cryptocurrencies
- **Regime-Specific Predictions**: Adapt predictions based on market conditions
- **Adaptive Horizon Selection**: Dynamically adjust prediction horizon based on volatility

## Key Features

### 1. Multi-Asset Portfolio Modeling

The enhanced TFT can model multiple assets simultaneously, capturing their interdependencies:

```python
model = EnhancedTemporalFusionTransformer(
    n_assets=5,  # Model 5 cryptocurrencies together
    enable_cross_attention=True
)

# Input shape: (batch_size, n_assets, n_timesteps, n_features)
inputs = {
    'temporal_inputs': multi_asset_data,
    'static_inputs': asset_characteristics,
    'portfolio_weights': [0.3, 0.25, 0.2, 0.15, 0.1]
}

outputs = model.forward_multi_asset(inputs)
```

### 2. Market Regime Detection

Automatically detects and adapts to different market regimes:

- **Trending Up**: Bullish market with positive momentum
- **Trending Down**: Bearish market with negative momentum
- **Ranging**: Sideways market with mean reversion
- **Volatile**: High volatility environment
- **Unknown**: Uncertain market conditions

Each regime has specific parameters that adjust predictions accordingly.

### 3. Cross-Asset Attention

Captures dependencies between different assets using attention mechanisms:

```python
# Access cross-asset attention weights
attention_weights = outputs['cross_asset_attention']
# Shape: (batch_size, n_assets, n_assets)
```

This shows which assets influence each other's predictions.

### 4. Adaptive Horizon Selection

Dynamically adjusts prediction horizon based on:
- Current market volatility
- Detected market regime
- Asset correlations

Shorter horizons in volatile markets, longer horizons in stable conditions.

## Architecture Details

### Enhanced Components

1. **Asset Embeddings**: Each asset has learnable embeddings that capture its characteristics
2. **Regime-Specific Output Heads**: Separate prediction heads for each market regime
3. **Cross-Asset Interaction Network**: GRN-based network for modeling asset interactions
4. **Portfolio Aggregation Layer**: Combines individual predictions into portfolio-level forecasts

### Model Parameters

```python
EnhancedTemporalFusionTransformer(
    # Base TFT parameters
    n_encoder_steps=168,      # 7 days of hourly data
    n_prediction_steps=24,    # 24-hour forecast
    n_features=10,           # Number of input features
    n_static_features=5,     # Static features per asset
    hidden_size=160,         # Hidden layer dimension
    lstm_layers=2,           # Number of LSTM layers
    num_attention_heads=4,   # Multi-head attention
    dropout_rate=0.1,        # Dropout for regularization
    
    # Enhanced parameters
    n_assets=5,              # Number of assets to model
    enable_regime_specific=True,   # Use regime detection
    enable_cross_attention=True,   # Use cross-asset attention
    enable_adaptive_horizon=True   # Use adaptive horizons
)
```

## Input Features

### Temporal Features (per asset)
1. **Price**: Normalized price series
2. **Returns**: Period-over-period returns
3. **Moving Averages**: MA5, MA10, MA20
4. **Volatility**: Rolling standard deviation
5. **Volume**: Trading volume
6. **RSI**: Relative Strength Index
7. **Market Cap Rank**: Relative size indicator
8. **Market Correlation**: Correlation with market index

### Static Features (per asset)
1. **Asset Type**: Categorical encoding
2. **Beta**: Market sensitivity
3. **Market Cap**: Size in billions
4. **Historical Volatility**: Long-term volatility
5. **Sentiment Score**: Social/news sentiment

## Usage Examples

### Basic Multi-Asset Prediction

```python
# Initialize model
model = EnhancedTemporalFusionTransformer(n_assets=5)

# Prepare data
inputs = {
    'temporal_inputs': temporal_features,  # (batch, assets, time, features)
    'static_inputs': static_features,      # (batch, assets, static_features)
    'market_data': market_context         # Overall market data
}

# Get predictions
outputs = model.forward_multi_asset(inputs)

# Access results
asset_predictions = outputs['asset_predictions']  # List of predictions per asset
portfolio_prediction = outputs['portfolio_prediction']  # Aggregated portfolio
current_regime = outputs['current_regime']  # Detected market regime
```

### Training with Multi-Asset Data

```python
# Training step
targets = {
    'asset_0_q10': asset_0_lower_quantile,
    'asset_0_q50': asset_0_median,
    'asset_0_q90': asset_0_upper_quantile,
    # ... for each asset
    'portfolio_q50': portfolio_median_target
}

losses = model.train_step(inputs, targets, learning_rate=0.001)
```

### Interpretability Analysis

```python
# Get interpretability outputs
interpretability = model.get_interpretability_outputs(inputs)

# Feature importance per asset
for asset_idx, importance in enumerate(interpretability['asset_feature_importance']):
    print(f"Asset {asset_idx} top features:")
    print(importance['temporal_past_importance'])

# Cross-asset dependencies
attention_matrix = interpretability['cross_asset_attention']
```

## Best Practices

### 1. Data Preparation
- Normalize prices and volumes appropriately
- Ensure temporal alignment across assets
- Handle missing data before input
- Include relevant market-wide indicators

### 2. Portfolio Configuration
- Use realistic portfolio weights
- Consider rebalancing strategies
- Account for transaction costs
- Monitor concentration risk

### 3. Regime-Aware Trading
- Adjust position sizes based on regime
- Use wider stops in volatile regimes
- Reduce leverage in uncertain conditions
- Monitor regime transitions

### 4. Model Monitoring
- Track prediction accuracy by regime
- Monitor cross-asset attention patterns
- Validate adaptive horizon performance
- Check for concept drift

## Advanced Features

### Custom Regime Detection

```python
# Override regime detection
class CustomRegimeDetector(MarketRegimeDetector):
    def detect_regime(self, market_data):
        # Custom logic here
        return "custom_regime"

model.regime_detector = CustomRegimeDetector()
```

### Dynamic Horizon Adjustment

```python
# Access horizon selector
horizon = model.horizon_selector.select_horizon(
    volatility=0.03,
    regime="volatile",
    asset_correlation=0.6
)
```

### Portfolio Optimization Integration

```python
# Use predictions for portfolio optimization
predictions = outputs['asset_predictions']
expected_returns = [pred['q50'] for pred in predictions]
uncertainty = [pred['q90'] - pred['q10'] for pred in predictions]

# Feed to portfolio optimizer
optimal_weights = optimize_portfolio(expected_returns, uncertainty)
```

## Performance Considerations

### Memory Usage
- Multi-asset modeling increases memory requirements
- Consider batch size adjustments
- Use gradient accumulation for large portfolios

### Computational Efficiency
- Cross-attention scales with O(n²) for n assets
- Consider limiting attention to top correlated assets
- Use mixed precision training when possible

### Scalability
- Model supports up to ~20 assets efficiently
- For larger portfolios, consider hierarchical grouping
- Use asset clustering for dimension reduction

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Decrease hidden dimensions
   - Limit number of assets

2. **Unstable Training**
   - Check learning rate
   - Verify data normalization
   - Use gradient clipping

3. **Poor Cross-Asset Modeling**
   - Ensure sufficient correlation in data
   - Check static feature quality
   - Verify attention mechanism convergence

## Future Enhancements

Planned improvements include:

1. **Dynamic Asset Selection**: Automatically select relevant assets
2. **Hierarchical Attention**: Multi-level attention for asset groups
3. **Risk-Aware Predictions**: Incorporate risk metrics directly
4. **Real-Time Adaptation**: Online learning capabilities
5. **Multi-Frequency Modeling**: Combined short and long-term predictions

## References

- Original TFT Paper: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- Multi-Asset Modeling: Portfolio theory and cross-sectional dependencies
- Regime Detection: Hidden Markov Models and market microstructure
- Adaptive Forecasting: Dynamic horizon selection in financial markets