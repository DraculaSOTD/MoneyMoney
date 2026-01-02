# Monitoring & Alerting Guide

This guide covers the monitoring and alerting features of the Crypto ML Trading System.

## 🎯 Overview

The system provides comprehensive monitoring through:
- Real-time performance metrics
- Multi-channel alerting
- Prometheus metrics export
- API endpoints for monitoring
- Sentiment analysis tracking

## 📊 Metrics Collection

### Performance Metrics
- **Trading Statistics**: Win rate, profit factor, Sharpe ratio
- **P&L Tracking**: Daily and total P&L, drawdowns
- **Model Performance**: Prediction accuracy, error rates
- **System Health**: API latency, uptime, error counts

### Available Metrics

```python
# Prometheus metrics exposed at http://localhost:9090/metrics
trading_active_positions      # Active positions by exchange and symbol
trading_total_trades         # Total trades counter
trading_volume_usd          # Trading volume in USD
trading_portfolio_value_usd # Portfolio value gauge
trading_daily_pnl_usd      # Daily P&L gauge
model_prediction_error     # Model prediction error histogram
api_request_duration_seconds # API latency histogram
```

## 🚨 Alert Configuration

### Alert Channels

#### 1. Telegram (Recommended)
```bash
# Set environment variables
export TELEGRAM_BOT_TOKEN=your-bot-token
export TELEGRAM_CHAT_ID=your-chat-id
```

To get started:
1. Create a bot via @BotFather on Telegram
2. Get your chat ID by messaging the bot and visiting:
   `https://api.telegram.org/bot<token>/getUpdates`

#### 2. Slack
```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

#### 3. Email
```bash
export EMAIL_USERNAME=your-email@gmail.com
export EMAIL_PASSWORD=your-app-password
```

#### 4. Discord
```bash
export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK
```

### Alert Types

| Alert Type | Description | Default Severity |
|------------|-------------|------------------|
| `system_status` | System start/stop events | INFO |
| `trade_executed` | Order fills | INFO |
| `position_opened` | New positions | INFO |
| `position_closed` | Closed positions with P&L | INFO |
| `risk_limit` | Risk limits reached | WARNING |
| `large_drawdown` | Drawdown exceeds threshold | WARNING |
| `connection_lost` | Exchange disconnection | ERROR |
| `model_error` | ML model failures | ERROR |

### Alert Rules

Configure in `config/system_config.yaml`:

```yaml
alerting:
  rules:
    trade_executed:
      min_severity: info
      rate_limit_seconds: 30  # Max 1 alert per 30s
    large_drawdown:
      min_severity: warning
      rate_limit_seconds: 600  # Max 1 alert per 10m
```

## 📡 API Monitoring Endpoints

### Performance Metrics
```bash
# Get current performance
curl http://localhost:8000/monitoring/performance

# Response:
{
  "performance": {
    "total_trades": 150,
    "win_rate": "65.33%",
    "profit_factor": "2.15",
    "sharpe_ratio": "1.85",
    "max_drawdown": "8.5%"
  },
  "system_health": {
    "uptime": "2 days, 14:32:15",
    "active_positions": 3,
    "avg_api_latency_ms": "45.2"
  }
}
```

### Alert History
```bash
# Get recent alerts
curl http://localhost:8000/monitoring/alerts?limit=50
```

### Performance Report
```bash
# Get formatted performance report
curl http://localhost:8000/monitoring/report
```

### Export All Metrics
```bash
# Export comprehensive metrics JSON
curl http://localhost:8000/monitoring/metrics/export > metrics_export.json
```

## 🎭 Sentiment Monitoring

### Sentiment Sources
1. **Twitter**: Real-time crypto discussions
2. **Reddit**: Community sentiment from major subreddits
3. **News**: Financial news sentiment analysis

### Sentiment Metrics
- **Score**: -1 (bearish) to +1 (bullish)
- **Confidence**: 0 to 1 based on data volume
- **Trend**: Sentiment direction over time
- **Volatility**: Sentiment stability measure

### Configuration
```yaml
alternative_data:
  enabled: true
  twitter:
    enabled: true
    min_engagement: 10
  reddit:
    enabled: true
    subreddits:
      - cryptocurrency
      - CryptoMarkets
  news:
    enabled: true
    sources:
      - coindesk
      - cointelegraph
```

## 📈 Grafana Dashboard (Optional)

### Setup Grafana
```bash
# Run Grafana with Docker
docker run -d -p 3000:3000 grafana/grafana

# Add Prometheus data source
# URL: http://host.docker.internal:9090
```

### Import Dashboard
1. Go to Grafana (http://localhost:3000)
2. Import dashboard from `monitoring/grafana-dashboard.json`
3. Select Prometheus data source

### Key Panels
- Portfolio value over time
- Win rate and profit factor
- Position distribution
- Model prediction accuracy
- System health metrics

## 🔔 Alert Examples

### Telegram Alert Format
```
🚨 WARNING

Large Drawdown Detected

Portfolio drawdown: 12.5% (Max allowed: 10.0%)

📊 Details:
• Current Drawdown: 12.5%
• Max Drawdown: 10.0%

Type: large_drawdown
Time: 2024-01-15 14:32:45 UTC
```

### Slack Alert Format
```json
{
  "attachments": [{
    "color": "#ff9900",
    "title": "WARNING: Risk Limit Reached",
    "text": "Daily loss limit reached",
    "fields": [
      {"title": "Type", "value": "risk_limit"},
      {"title": "Current Loss", "value": "$1,050"},
      {"title": "Limit", "value": "$1,000"}
    ]
  }]
}
```

## 🛠️ Troubleshooting

### No Alerts Received
1. Check environment variables are set
2. Verify alert channel is enabled in config
3. Check logs for connection errors
4. Test webhook URLs manually

### Missing Metrics
1. Ensure Prometheus endpoint is enabled
2. Check metrics collector is running
3. Verify trades are being recorded

### High Alert Volume
1. Adjust rate limits in config
2. Increase severity thresholds
3. Filter by alert type

## 📝 Custom Alerts

Add custom alerts in your code:

```python
from monitoring.alerting import Alert, AlertType, AlertSeverity

# Create custom alert
custom_alert = Alert(
    type=AlertType.RISK_LIMIT,
    severity=AlertSeverity.WARNING,
    title="Custom Risk Alert",
    message=f"Custom metric exceeded: {value}",
    metadata={'metric': 'custom', 'value': value}
)

# Send alert
await alert_manager.send_alert(custom_alert)
```

## 🔍 Performance Analysis

### Daily Performance Report
The system generates comprehensive reports including:
- Trading statistics (wins/losses)
- Risk metrics (Sharpe, Sortino)
- Model performance by type
- System health indicators

### Access Reports
```python
# Via API
GET /monitoring/report

# Via console
python -c "from monitoring.metrics_collector import MetricsCollector; 
          mc = MetricsCollector(config); 
          print(mc.create_performance_report())"
```

## 🚀 Best Practices

1. **Alert Fatigue**: Set appropriate rate limits
2. **Channel Priority**: Use severity-based routing
3. **Metric Retention**: Configure data retention policies
4. **Dashboard Updates**: Refresh rates based on data frequency
5. **Sentiment Weight**: Adjust based on market conditions

## 📊 Example Monitoring Setup

```bash
# 1. Configure alerts (.env)
TELEGRAM_BOT_TOKEN=your-token
TELEGRAM_CHAT_ID=your-chat-id

# 2. Enable monitoring in config
monitoring:
  prometheus:
    enabled: true
alerting:
  telegram:
    enabled: true

# 3. Start system
python start_live_trading.py

# 4. Monitor via API
watch -n 5 'curl -s localhost:8000/monitoring/performance | jq'

# 5. Check Prometheus metrics
curl localhost:9090/metrics
```

Happy Monitoring! 📊🚨