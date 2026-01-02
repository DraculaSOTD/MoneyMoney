# Live Trading Guide

This guide covers the new live trading capabilities that have been added to the Crypto ML Trading System.

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
```bash
# Create example .env file
python start_live_trading.py --create-env

# Copy and edit with your credentials
cp .env.example .env
nano .env
```

3. **Check configuration**:
```bash
python start_live_trading.py --check-env
```

4. **Start in demo mode** (recommended first):
```bash
python start_live_trading.py --demo
```

5. **Access the API**:
```bash
# In another terminal
curl http://localhost:8000/system/status
```

## 📋 New Components

### 1. **Exchange Connectors** (`exchanges/`)
- **BinanceConnector**: Full Binance API integration with WebSocket support
  - Spot trading
  - Real-time market data streams
  - Order management
  - Account information

### 2. **Real-Time Data Management** (`data_feeds/real_time_manager.py`)
- Manages live market data streams
- Buffers recent data for analysis
- Distributes data to ML models
- Optional Redis caching

### 3. **Execution Engine** (`trading/execution_engine.py`)
- Risk management enforcement
- Position tracking
- Order execution
- P&L calculation
- Automatic stop-loss/take-profit

### 4. **REST API** (`api/main.py`)
- FastAPI-based REST endpoints
- WebSocket support for real-time updates
- Authentication via bearer tokens
- Full trading system control

### 5. **Database Integration** (`database/`)
- PostgreSQL schemas for:
  - Trade history
  - Order tracking
  - Position management
  - Model predictions
  - Risk metrics
  - System alerts

### 6. **Security** (`security/key_manager.py`)
- Encrypted API key storage
- Multiple storage backends:
  - Environment variables
  - Encrypted files
  - System keyring
  - AWS Secrets Manager
  - HashiCorp Vault

### 7. **Decision Engine** (`decision_engine.py`)
- Combines ML predictions with technical analysis
- Multi-factor signal generation
- Position management rules
- Confidence-based sizing

### 8. **Live Trading Integration** (`live_trading_integration.py`)
- Orchestrates all components
- ML model integration
- Real-time prediction and execution
- Automatic model retraining

## 🔧 Configuration

### Environment Variables

**Required**:
- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `API_TOKEN`: Bearer token for API authentication
- `MASTER_PASSWORD`: Master password for encrypted storage (if using file/keyring)

**Optional**:
- `BINANCE_TESTNET`: Use testnet (default: true)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `TWITTER_API_KEY`: Twitter API for sentiment
- `NEWS_API_KEY`: News API for sentiment

### System Configuration

Edit `config/system_config.yaml`:

```yaml
# Risk Management
risk_management:
  max_position_size: 10000  # USD
  max_positions: 10
  max_daily_loss: 1000  # USD
  max_drawdown: 0.1  # 10%
  stop_loss_percent: 0.02  # 2%
  take_profit_percent: 0.05  # 5%

# Trading Settings
trading:
  symbols: 
    - BTCUSDT
    - ETHUSDT
  base_position_size: 1000  # USD
  min_confidence: 0.6
  allow_short: false

# ML Settings
ml:
  retrain_error_threshold: 0.02
  retrain_interval_hours: 24
```

## 🔌 API Endpoints

### System Control
- `POST /system/start` - Start trading system
- `POST /system/stop` - Stop trading system
- `GET /system/status` - Get system status

### Trading
- `POST /trading/order` - Place order
- `DELETE /trading/order/{order_id}` - Cancel order

### Positions
- `GET /positions` - Get all positions
- `POST /positions/close` - Close position

### Market Data
- `POST /data/subscribe` - Subscribe to market data
- `GET /data/{symbol}/ticker` - Get latest ticker
- `GET /data/{symbol}/klines` - Get recent klines

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates

## 🏃 Running Modes

### Demo Mode (Recommended for Testing)
```bash
python start_live_trading.py --demo
```
- Uses Binance testnet
- No real money at risk
- Full functionality testing

### Production Mode
```bash
# Set BINANCE_TESTNET=false in .env
python start_live_trading.py
```
- Real trading with actual funds
- Ensure proper risk limits
- Monitor closely

### API-Only Mode
```bash
python start_live_trading.py --no-api
```
- Runs trading system without REST API
- Useful for dedicated trading servers

## 📊 Monitoring

### Logs
- Console output for real-time monitoring
- `trading_system.log` for persistent logs

### Database Metrics
```sql
-- Recent trades
SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;

-- Current positions
SELECT * FROM positions WHERE status = 'open';

-- Risk metrics
SELECT * FROM risk_metrics ORDER BY timestamp DESC LIMIT 1;

-- Model performance
SELECT model_name, AVG(ABS(error)) as avg_error 
FROM model_predictions 
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY model_name;
```

### API Monitoring
```python
import requests

# Check system status
response = requests.get('http://localhost:8000/system/status')
print(response.json())

# Get current positions
headers = {'Authorization': 'Bearer your-token'}
response = requests.get('http://localhost:8000/positions', headers=headers)
print(response.json())
```

## 🔒 Security Best Practices

1. **API Keys**
   - Use read-only keys for testing
   - Enable IP whitelisting
   - Rotate keys regularly

2. **Database**
   - Use strong passwords
   - Enable SSL/TLS
   - Regular backups

3. **API Access**
   - Change default API_TOKEN
   - Use HTTPS in production
   - Implement rate limiting

4. **System**
   - Run as non-root user
   - Use firewall rules
   - Monitor for anomalies

## 🐛 Troubleshooting

### Common Issues

1. **"API key invalid"**
   - Check environment variables
   - Verify API key permissions
   - Ensure not using testnet keys on mainnet

2. **"Database connection failed"**
   - Check DATABASE_URL format
   - Verify PostgreSQL is running
   - Check network connectivity

3. **"WebSocket disconnected"**
   - Check internet connection
   - Verify firewall rules
   - Check rate limits

4. **"Model predictions unavailable"**
   - Ensure models are trained
   - Check model file paths
   - Verify sufficient historical data

### Debug Mode
```bash
# Set logging to DEBUG
export LOG_LEVEL=DEBUG
python start_live_trading.py
```

## 🚦 Next Steps

### Completed Features

✅ **Monitoring & Alerts**
   - Multi-channel alerting (Slack, Telegram, Email, Discord)
   - Real-time performance metrics
   - Prometheus metrics endpoint
   - Alert rate limiting and severity filtering

✅ **Alternative Data Integration**
   - Twitter sentiment analysis with engagement weighting
   - Reddit sentiment from multiple crypto subreddits
   - News sentiment from major financial sources
   - Combined sentiment scoring with trend analysis

### Remaining Features (Low Priority)

1. **Advanced Models**
   - Complete Transformer TFT implementation
   - Enhanced NLP models for sentiment
   - Multi-timeframe analysis

2. **Additional Exchanges**
   - Coinbase Pro
   - Kraken
   - Bybit

### Production Checklist

- [ ] Test thoroughly on testnet
- [ ] Set conservative risk limits
- [ ] Enable database backups
- [ ] Set up monitoring alerts
- [ ] Create disaster recovery plan
- [ ] Document trading strategies
- [ ] Review security measures
- [ ] Performance testing
- [ ] Gradual capital deployment

## 📚 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Live Trading System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   FastAPI   │    │  WebSocket   │    │   REST Client    │  │
│  │   Server    │◄───│   Clients    │    │   Applications   │  │
│  └──────┬──────┘    └──────────────┘    └──────────────────┘  │
│         │                                                        │
│  ┌──────▼──────────────────────────────────────────────────┐   │
│  │              Live Trading Integration                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │   │
│  │  │  Decision   │  │   ML        │  │   Execution    │  │   │
│  │  │  Engine     │  │   System    │  │   Engine       │  │   │
│  │  └─────────────┘  └─────────────┘  └────────────────┘  │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │               Real-Time Data Manager                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │   │
│  │  │   Binance   │  │  Coinbase   │  │   Other        │  │   │
│  │  │   Connector │  │  Connector  │  │   Exchanges    │  │   │
│  │  └─────────────┘  └─────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │   Database   │  │     Redis       │  │  Key Manager    │   │
│  │  PostgreSQL  │  │     Cache       │  │   Security      │   │
│  └──────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 💡 Tips

1. Start small - test with minimum position sizes
2. Monitor the first 24 hours closely
3. Keep detailed logs of system behavior
4. Gradually increase position sizes
5. Have a kill switch ready
6. Regular model performance reviews
7. Keep API keys secure and rotated

## 🤝 Contributing

To add new features:
1. Follow existing code patterns
2. Add comprehensive error handling
3. Include logging statements
4. Write unit tests
5. Update documentation
6. Test on testnet first

## 📞 Support

For issues:
1. Check logs in `trading_system.log`
2. Review system status endpoint
3. Verify environment configuration
4. Check database for errors
5. Review this guide

Happy Trading! 🚀