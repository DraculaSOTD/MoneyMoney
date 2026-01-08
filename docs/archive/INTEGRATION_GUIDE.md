# MoneyMoney ↔ Trading Platform Integration Guide

## 🎯 Overview

This guide explains how the MoneyMoney frontend and Trading Platform backend are integrated to provide a complete AI-powered trading dashboard experience.

---

## 📋 Table of Contents

1. [Architecture](#architecture)
2. [Setup Instructions](#setup-instructions)
3. [API Endpoints](#api-endpoints)
4. [Data Flow](#data-flow)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     ADMIN PANEL (React)                          │
│                    http://localhost:5173                         │
│  - Data collection management                                    │
│  - Model training management                                     │
│  - Real-time WebSocket updates                                   │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    │ Admin API (/admin/*)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│              TRADING PLATFORM (FastAPI/Python)                   │
│                    http://localhost:8001                         │
│                                                                   │
│  Routers:                                                        │
│  - /admin/*        → Admin-only endpoints                       │
│  - /api/user/*     → User-facing endpoints (FILTERED)           │
│  - /data/*         → Data aggregation                           │
│  - /ws/*           → WebSocket real-time updates                │
│                                                                   │
│  Services:                                                       │
│  - DataAggregator  → Timeframe aggregation (1m→5m→1h→1D→1M)    │
│  - WebSocketManager → Real-time updates                         │
│  - BinanceConnector → Exchange data collection                  │
│                                                                   │
│  Database: PostgreSQL                                            │
│  - TradingProfile (has_data, models_trained flags)              │
│  - MarketData (1-minute OHLCV)                                  │
│  - DataCollectionJob, ModelTrainingJob                          │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    │ User API Proxy
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│               MONEYMONEY BACKEND (Node.js)                       │
│                    http://localhost:3000                         │
│                                                                   │
│  - User authentication (JWT)                                     │
│  - API proxy to Trading Platform                                │
│  - Fallback to local SQLite                                     │
│                                                                   │
│  Proxy Endpoints:                                                │
│  - /api/instruments → /api/user/instruments                     │
│  - /api/instruments/:symbol/data/:timeframe → Chart data        │
│  - /api/instruments/:symbol/predictions → ML predictions        │
│  - /api/instruments/:symbol/signals → Trading signals           │
│                                                                   │
│  Database: SQLite (fallback only)                                │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│            MONEYMONEY FRONTEND (HTML/JS/Chart.js)               │
│                    http://localhost:3000/dashboard              │
│                                                                   │
│  Features:                                                       │
│  - View ONLY trained instruments                                │
│  - Multi-timeframe charts (1m, 5m, 1h, 1D, 1M)                 │
│  - Live price updates (auto-refresh)                            │
│  - Model predictions display                                     │
│  - Trading signals display                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.8+** (for Trading Platform)
- **Node.js 14+** (for MoneyMoney)
- **PostgreSQL 12+** (for Trading Platform)
- **Binance API Key** (for data collection)

### Step 1: Trading Platform Setup

```bash
# Navigate to Trading Platform directory
cd "other platform/Trading"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your actual values:
# - DATABASE_URL
# - BINANCE_API_KEY
# - BINANCE_API_SECRET
# - JWT secrets

# Initialize database
# (Run your database migration scripts here)

# Start server
python api/main_simple.py

# Server should start on http://localhost:8001
```

### Step 2: MoneyMoney Setup

```bash
# Navigate to MoneyMoney directory
cd MoneyMoney

# Install dependencies
npm install

# Create .env file
cp .env.example .env
# Edit .env with your actual values:
# - PORT=3000
# - TRADING_API_URL=http://localhost:8001
# - JWT_SECRET

# Initialize SQLite database (automatic on first run)

# Start server
node server.js

# Server should start on http://localhost:3000
```

### Step 3: Admin Panel Setup (Optional)

```bash
# Navigate to admin panel directory
cd "other platform/Trading/frontend"

# Install dependencies
npm install

# Start development server
npm run dev

# Admin panel should start on http://localhost:5173
```

---

## 📡 API Endpoints

### Trading Platform User API (`/api/user/*`)

#### Get Trained Instruments
```http
GET /api/user/instruments
Authorization: Bearer <JWT_TOKEN>

Response:
[
  {
    "symbol": "BTCUSDT",
    "name": "Bitcoin",
    "category": "crypto",
    "has_data": true,
    "models_trained": true,
    "current_price": 45000.50,
    "change_percent": 2.5,
    "total_data_points": 43200,
    "data_interval": "1m"
  }
]
```

#### Get Chart Data
```http
GET /api/user/instruments/BTCUSDT/data/1h?limit=100
Authorization: Bearer <JWT_TOKEN>

Response:
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "candle_count": 100,
  "candles": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 45000.00,
      "high": 45500.00,
      "low": 44800.00,
      "close": 45200.00,
      "volume": 1234567.89
    }
  ]
}
```

#### Get Predictions
```http
GET /api/user/instruments/BTCUSDT/predictions
Authorization: Bearer <JWT_TOKEN>

Response:
[
  {
    "symbol": "BTCUSDT",
    "model_name": "ARIMA",
    "prediction_type": "price",
    "predicted_value": 46000.00,
    "confidence": 0.85,
    "timeframe": "1h",
    "timestamp": "2024-01-01T12:00:00Z"
  }
]
```

#### Get Trading Signals
```http
GET /api/user/instruments/BTCUSDT/signals
Authorization: Bearer <JWT_TOKEN>

Response:
{
  "symbol": "BTCUSDT",
  "signal": "buy",
  "confidence": 85,
  "entry_point": 45000.00,
  "take_profit": 47000.00,
  "stop_loss": 44000.00,
  "reasoning": "Strong bullish indicators from ML models",
  "generated_at": "2024-01-01T12:00:00Z"
}
```

### MoneyMoney Proxy API (`/api/*`)

#### Get Instruments (Proxy)
```http
GET /api/instruments
Authorization: Bearer <JWT_TOKEN>

# Proxies to: ${TRADING_API_URL}/api/user/instruments
# Falls back to SQLite if Trading Platform unavailable
```

#### Get Chart Data (Proxy)
```http
GET /api/instruments/BTCUSDT/data/1h?limit=100
Authorization: Bearer <JWT_TOKEN>

# Proxies to: ${TRADING_API_URL}/api/user/instruments/BTCUSDT/data/1h
```

#### Health Check
```http
GET /api/trading-platform/health
Authorization: Bearer <JWT_TOKEN>

Response:
{
  "tradingPlatform": {
    "status": "healthy",
    "service": "Trading Platform User API"
  },
  "moneyMoney": {
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

---

## 🔄 Data Flow

### Complete Workflow

#### 1. Admin Preparation (Admin Panel)
```
1. Admin logs in → http://localhost:5173
2. Admin collects data:
   - Select: Data Collection tab
   - Choose: BTCUSDT
   - Configure: 30 days back
   - Click: Start Collection
   - WebSocket: Real-time progress updates
   - Result: has_data = true

3. Admin trains models:
   - Select: Model Training tab
   - Choose: BTCUSDT
   - Mode: Auto-select (ARIMA, GARCH, GRU, CNN)
   - Click: Train Models
   - WebSocket: Real-time training progress
   - Result: models_trained = true
```

#### 2. User Consumption (MoneyMoney)
```
1. User logs in → http://localhost:3000/auth
2. User navigates to dashboard → http://localhost:3000/dashboard
3. Frontend JavaScript:
   - dashboard.js calls: loadInstruments()
   - API request: GET /api/instruments

4. MoneyMoney Backend:
   - Proxies to: http://localhost:8001/api/user/instruments
   - Trading Platform filters: has_data=true AND models_trained=true
   - Returns: [{ symbol: "BTCUSDT", ... }]

5. Frontend displays:
   - Dropdown shows: BTCUSDT ✓
   - User selects: BTCUSDT
   - dashboard.js calls: loadChartData('BTCUSDT', '1h')

6. Chart data fetch:
   - API request: GET /api/instruments/BTCUSDT/data/1h?limit=100
   - MoneyMoney proxies to Trading Platform
   - DataAggregator: Aggregates 1m → 1h data
   - Returns: 100 hourly candles

7. Chart renders:
   - Chart.js displays beautiful gradient chart
   - Auto-refresh every 60 seconds
   - User can switch timeframes (1D, 7D, 1M, 3M)
```

---

## 🧪 Testing

### Integration Test Checklist

#### Prerequisites Check
- [ ] Trading Platform running on port 8001
- [ ] MoneyMoney running on port 3000
- [ ] PostgreSQL database accessible
- [ ] Binance API credentials configured

#### Admin Flow
- [ ] Login to admin panel (admin/admin123)
- [ ] Navigate to Data Collection tab
- [ ] Select BTCUSDT from dropdown
- [ ] Start data collection (30 days)
- [ ] Verify WebSocket progress updates appear
- [ ] Wait for completion (100% progress)
- [ ] Check PostgreSQL: `SELECT has_data FROM trading_profiles WHERE symbol='BTCUSDT'` → should be `true`
- [ ] Navigate to Model Training tab
- [ ] Select BTCUSDT
- [ ] Click "Train Models" (auto-select mode)
- [ ] Verify WebSocket training updates appear
- [ ] Wait for all models to complete
- [ ] Check PostgreSQL: `SELECT models_trained FROM trading_profiles WHERE symbol='BTCUSDT'` → should be `true`

#### User Flow
- [ ] Open MoneyMoney: http://localhost:3000/auth
- [ ] Login with test user account
- [ ] Navigate to dashboard
- [ ] Verify BTCUSDT appears in instrument dropdown
- [ ] Verify NO untrained symbols appear
- [ ] Select BTCUSDT from dropdown
- [ ] Verify chart loads with data
- [ ] Click different timeframe buttons (1D, 7D, 1M, 3M)
- [ ] Verify chart updates for each timeframe
- [ ] Wait 60 seconds
- [ ] Verify chart auto-refreshes

#### API Tests
```bash
# Get token
TOKEN=$(curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password"}' \
  | jq -r '.token')

# Test instruments endpoint
curl http://localhost:3000/api/instruments \
  -H "Authorization: Bearer $TOKEN" \
  | jq '.'

# Should return array with BTCUSDT (and only trained symbols)

# Test chart data endpoint
curl "http://localhost:3000/api/instruments/BTCUSDT/data/1h?limit=10" \
  -H "Authorization: Bearer $TOKEN" \
  | jq '.'

# Should return 10 hourly candles

# Test health check
curl http://localhost:3000/api/trading-platform/health \
  -H "Authorization: Bearer $TOKEN" \
  | jq '.'

# Should show both systems healthy
```

#### Fallback Test
- [ ] Stop Trading Platform (Ctrl+C)
- [ ] Reload MoneyMoney dashboard
- [ ] Verify instruments still load (from SQLite fallback)
- [ ] Verify graceful error messages for chart data
- [ ] Restart Trading Platform
- [ ] Reload dashboard
- [ ] Verify chart data loads again

---

## 🔧 Troubleshooting

### Issue: Instruments Not Showing

**Symptoms:**
- MoneyMoney dashboard shows empty dropdown
- No instruments appear in search

**Diagnosis:**
```bash
# Check if Trading Platform is running
curl http://localhost:8001/

# Check MoneyMoney backend logs
# Look for: "Error fetching instruments from Trading Platform"

# Check database
psql -d trading_platform -c "SELECT symbol, has_data, models_trained FROM trading_profiles;"
```

**Solution:**
1. Verify Trading Platform is running on port 8001
2. Check `.env` file: `TRADING_API_URL=http://localhost:8001`
3. Ensure at least one symbol has both `has_data=true` AND `models_trained=true`
4. Check CORS configuration allows localhost:3000

---

### Issue: Chart Not Loading

**Symptoms:**
- Dropdown works, but chart shows loading spinner indefinitely
- Console errors about failed data fetch

**Diagnosis:**
```bash
# Check browser console for errors
# Check MoneyMoney server logs

# Test API directly
curl "http://localhost:3000/api/instruments/BTCUSDT/data/1h?limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Solution:**
1. Verify symbol has 1-minute data in PostgreSQL
2. Check DataAggregator is working: Trading Platform logs
3. Verify timeframe parameter is valid (1m, 5m, 1h, 1D, 1M)
4. Check user JWT token is valid and not expired

---

### Issue: WebSocket Not Connecting (Admin Panel)

**Symptoms:**
- Data collection/training starts but no progress updates
- Connection status shows "disconnected"

**Diagnosis:**
```bash
# Check WebSocket endpoint
curl http://localhost:8001/ws/stats

# Check admin token
echo $ADMIN_TOKEN | jwt decode -
```

**Solution:**
1. Verify Trading Platform WebSocket server is running
2. Check admin JWT token includes `"type": "admin"`
3. Verify firewall allows WebSocket connections
4. Check browser console for WebSocket errors

---

### Issue: Authentication Errors

**Symptoms:**
- "Unauthorized" errors
- Redirected to login page repeatedly

**Diagnosis:**
```javascript
// Browser console
localStorage.getItem('token')
// Should show JWT token

// Decode token
// Use: https://jwt.io
```

**Solution:**
1. Clear localStorage and login again
2. Verify JWT_SECRET matches between MoneyMoney and token generation
3. Check token expiration (default: 7 days)
4. Ensure Authorization header format: `Bearer <token>`

---

### Issue: Data Not Aggregating

**Symptoms:**
- 1m data works, but 1h/1D shows no data
- Empty candles array returned

**Diagnosis:**
```python
# Python shell
from services.data_aggregator import DataAggregator
from database.models import SessionLocal

db = SessionLocal()
agg = DataAggregator(db)
data = agg.get_aggregated_data('BTCUSDT', 1, '1h', limit=10)
print(len(data))
```

**Solution:**
1. Verify sufficient 1m data exists (need at least 60 candles for 1h)
2. Check DataAggregator logs for errors
3. Verify pandas resampling is working
4. Clear aggregation cache: POST /data/BTCUSDT/cache/clear

---

## 📚 Additional Resources

- **Trading Platform API Docs:** http://localhost:8001/docs (FastAPI auto-generated)
- **Admin Panel:** http://localhost:5173
- **MoneyMoney Dashboard:** http://localhost:3000/dashboard
- **Database Schema:** See `database/models.py`
- **WebSocket Events:** See `services/websocket_manager.py`

---

## 🆘 Support

If you encounter issues not covered in this guide:

1. Check server logs (both Trading Platform and MoneyMoney)
2. Check browser console for JavaScript errors
3. Verify all environment variables are set correctly
4. Ensure all services are running on correct ports
5. Check firewall/antivirus isn't blocking connections

---

## 📄 License

Copyright © 2024 DataPulse AI (Pty) Ltd. All rights reserved.
