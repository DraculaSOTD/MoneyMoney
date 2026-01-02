# MoneyMoney - AI-Powered Trading Dashboard

A professional AI-powered trading dashboard integrated with a complete machine learning trading platform. MoneyMoney provides real-time market data, ML-generated predictions, trading signals, and beautiful chart visualizations for crypto and forex traders.

## Overview

MoneyMoney is the **user-facing frontend** that integrates with the **Trading Platform backend**. This architecture provides:

- **User Dashboard**: Beautiful glassmorphism UI for viewing trained instruments and predictions
- **Admin Panel**: React-based admin interface for data collection and model training
- **ML Backend**: FastAPI service with PostgreSQL for data storage and model management
- **Real-time Updates**: WebSocket support for live progress tracking
- **Intelligent Fallback**: Local SQLite fallback when Trading Platform is unavailable

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MoneyMoney                               │
│  ┌───────────────────┐              ┌────────────────────────┐  │
│  │  User Dashboard   │              │   Admin Panel (React)  │  │
│  │  (HTML/CSS/JS)    │              │   - Data Collection    │  │
│  │  - Charts         │              │   - Model Training     │  │
│  │  - Predictions    │              │   - WebSocket Updates  │  │
│  └─────────┬─────────┘              └───────────┬────────────┘  │
│            │                                    │                │
│            └────────────┬───────────────────────┘                │
│                         │                                        │
│                  ┌──────▼────────┐                              │
│                  │  Node.js API  │                              │
│                  │  (Express)    │                              │
│                  │  - Proxy      │                              │
│                  │  - Auth       │                              │
│                  │  - Fallback   │                              │
│                  └──────┬────────┘                              │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          │ HTTP/WebSocket
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Trading Platform                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              FastAPI Backend (Python)                     │  │
│  │  - User API (/api/user/*)      - Admin API (/api/admin/*) │  │
│  │  - Data Aggregation            - Model Training           │  │
│  │  - Binance Integration         - WebSocket Manager        │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                            │                                     │
│                   ┌────────▼────────┐                           │
│                   │   PostgreSQL    │                           │
│                   │   - Profiles    │                           │
│                   │   - OHLCV Data  │                           │
│                   │   - Models      │                           │
│                   └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Automated Startup (Recommended)

```bash
cd /home/calvin/Websites/Trading\ Dashboard/MoneyMoney
./start-all.sh
```

This will automatically:
- Check for port conflicts (8001, 3000)
- Start Trading Platform on port 8001
- Start MoneyMoney on port 3000
- Open both in separate terminal windows
- Display access URLs

### Manual Startup

**Terminal 1 - Trading Platform:**
```bash
cd "/home/calvin/Websites/Trading Dashboard/other platform/Trading"
./start.sh
```

**Terminal 2 - MoneyMoney:**
```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./start.sh
```

### Access Points

- **User Dashboard**: http://localhost:3000
- **Admin Panel**: http://localhost:3000/admin
- **Trading Platform API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs

## Features

### 🚀 User Dashboard Features
- **Trained Instruments Only**: Users see ONLY instruments with collected data AND trained models
- **Interactive Charts**: Chart.js visualizations with gradient styling
- **Multiple Timeframes**: 1m, 5m, 1h, 1D, 1M chart views
- **Auto-Refresh**: Charts update every 60 seconds automatically
- **Connection Monitoring**: Visual indicator shows Trading Platform health
- **Real-time Prices**: Live price updates with 24h change percentages
- **ML Predictions**: AI-generated price predictions (coming soon)
- **Trading Signals**: Buy/sell signals with confidence scores (coming soon)

### 📊 Admin Panel Features
- **Data Collection**: Simplified interface for collecting market data from Binance
- **Model Training**: Train multiple ML models (ARIMA, GARCH, GRU_Attention, CNN_Pattern)
- **Real-time Progress**: WebSocket updates during data collection and training
- **Profile Management**: Manage trading profiles and instruments
- **Model Performance**: View accuracy metrics and training results

### 🤖 ML Models
- **ARIMA**: Time series forecasting
- **GARCH**: Volatility modeling
- **GRU_Attention**: Deep learning with attention mechanism
- **CNN_Pattern**: Pattern recognition using convolutional neural networks

### 🔒 Security
- **Separate Authentication**: Different JWT secrets for admin vs users
- **Secure Password Hashing**: bcrypt with 12 salt rounds
- **Token-based Auth**: JWT authentication with expiration
- **Protected Routes**: Middleware validation on all sensitive endpoints
- **CORS Protection**: Configured cross-origin resource sharing
- **Helmet.js**: HTTP security headers

## Technology Stack

### Frontend
- **User Dashboard**: HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Admin Panel**: React 19, Material-UI v7, WebSocket integration
- **Styling**: CSS custom properties, glassmorphism design, starfield background

### Backend
- **MoneyMoney API**: Node.js, Express.js, JWT authentication
- **Trading Platform**: FastAPI (Python), SQLAlchemy, WebSocket
- **Databases**: PostgreSQL (Trading Platform), SQLite (MoneyMoney fallback)

### Integrations
- **Exchange**: Binance API for market data
- **Data Aggregation**: Pandas for timeframe resampling (1m → 5m → 1h → 1D → 1M)
- **Real-time**: WebSocket for admin updates, polling for user dashboard

## Installation

### Prerequisites

**Node.js Environment:**
- Node.js (v16 or higher)
- npm (v7 or higher)

**Python Environment:**
- Python 3.8 or higher
- pip (Python package manager)
- virtualenv or venv

**Database:**
- PostgreSQL 12 or higher (for Trading Platform)
- SQLite3 (automatically included with Node.js and Python)

**API Keys:**
- Binance API key and secret (for live market data)

### Complete Setup

#### 1. Trading Platform Setup

```bash
# Navigate to Trading Platform
cd "/home/calvin/Websites/Trading Dashboard/other platform/Trading"

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create .env file from example
cp .env.example .env

# Edit .env with your credentials
nano .env  # Or use your preferred editor
```

**Required .env variables:**
```env
DATABASE_URL=postgresql://user:password@localhost:5432/trading_platform
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
JWT_SECRET=your_super_secret_jwt_key_change_this_in_production_2024
ADMIN_JWT_SECRET=admin_super_secret_jwt_key_change_this_in_production_2024
DEFAULT_DATA_INTERVAL=1m
CACHE_TTL_SECONDS=300
```

**Create PostgreSQL database:**
```bash
# Login to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE trading_platform;
\q
```

**Initialize database tables:**
```bash
# The tables will be created automatically when you first start the server
# Or you can run migrations if available
python -m alembic upgrade head  # If using Alembic
```

#### 2. MoneyMoney Setup

```bash
# Navigate to MoneyMoney
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"

# Install Node.js dependencies
npm install

# Create .env file from example
cp .env.example .env

# Edit .env if needed (defaults should work)
nano .env
```

**MoneyMoney .env variables:**
```env
PORT=3000
TRADING_API_URL=http://localhost:8001
JWT_SECRET=tradingdashboard_jwt_secret_key_2024
CHART_REFRESH_INTERVAL_MS=60000
```

#### 3. First Time Startup

**Option A: Automated (Recommended)**
```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./start-all.sh
```

**Option B: Manual**

Terminal 1:
```bash
cd "/home/calvin/Websites/Trading Dashboard/other platform/Trading"
./start.sh
```

Terminal 2:
```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./start.sh
```

#### 4. Create Admin Account

```bash
# Using curl
curl -X POST http://localhost:8001/api/admin/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "your_secure_password",
    "name": "Admin User"
  }'

# Or use the API documentation interface
# Open http://localhost:8001/docs
# Navigate to POST /api/admin/auth/register
# Click "Try it out" and fill in the details
```

#### 5. Initial Data Collection

1. Open admin panel: http://localhost:3000/admin
2. Login with admin credentials
3. Navigate to "Data Collection" tab
4. Select instruments (e.g., BTCUSDT, ETHUSDT)
5. Click "Start Collection" (this will collect 1m interval data)
6. Wait for "Data collection completed" message
7. Navigate to "Model Training" tab
8. Select the same instruments
9. Click "Train Models"
10. Wait for training to complete

Now users will be able to see these instruments with charts and predictions!

#### 6. Create User Account

```bash
# Register a user via MoneyMoney API
curl -X POST http://localhost:3000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "user_password"
  }'

# Or use the web interface
# Open http://localhost:3000/auth
# Click "Sign Up" and create an account
```

## Project Structure

```
MoneyMoney/
├── public/                        # Static files
│   ├── css/
│   │   ├── styles.css            # Main stylesheet (glassmorphism theme)
│   │   └── moneymoney.css        # Admin panel specific styles
│   └── js/
│       ├── main.js               # Core authentication & utilities
│       └── dashboard.js          # Chart integration & API calls
├── views/                        # HTML templates
│   ├── index.html                # Landing page
│   ├── auth.html                 # Login/Signup page
│   ├── dashboard.html            # User trading dashboard
│   ├── profile.html              # User profile management
│   └── admin.html                # Admin panel (React app root)
├── admin/                        # React Admin Panel (built files)
│   ├── src/
│   │   ├── components/
│   │   │   ├── SimplifiedDataCollection.tsx
│   │   │   ├── SmartModelTrainer.tsx
│   │   │   └── ... (other components)
│   │   └── hooks/
│   │       └── useWebSocket.ts   # WebSocket hook for real-time updates
│   └── build/                    # Production build
├── database/                     # SQLite database files (fallback)
├── server.js                     # Node.js Express server (API proxy)
├── start.sh                      # MoneyMoney startup script
├── start-all.sh                  # Master startup script (both services)
├── .env.example                  # Environment variables template
├── INTEGRATION_GUIDE.md          # Complete integration documentation
├── package.json                  # Dependencies and scripts
└── README.md                     # This file

other platform/Trading/           # Trading Platform (FastAPI)
├── api/
│   ├── routers/
│   │   ├── admin.py              # Admin-only endpoints
│   │   ├── admin_auth.py         # Admin authentication
│   │   ├── user_data.py          # User-facing API (CRITICAL)
│   │   ├── profiles.py           # Profile management
│   │   ├── trading.py            # Trading operations
│   │   ├── data.py               # Data collection
│   │   └── websocket.py          # WebSocket manager
│   ├── services/
│   │   ├── data_aggregator.py    # Timeframe aggregation (1m → 5m → 1h → 1D → 1M)
│   │   └── websocket_manager.py  # WebSocket connection manager
│   ├── models/
│   │   └── database.py           # SQLAlchemy models
│   └── main_simple.py            # FastAPI application
├── models/                       # ML model implementations
│   ├── arima_model.py
│   ├── garch_model.py
│   ├── gru_attention.py
│   └── cnn_pattern.py
├── start.sh                      # Trading Platform startup script
├── .env.example                  # Environment variables template
└── requirements.txt              # Python dependencies
```

## API Endpoints

### MoneyMoney API (Node.js - Port 3000)

#### User Authentication
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login (returns JWT token)

#### User Dashboard (Protected - Requires JWT)
- `GET /api/instruments` - Get trained instruments with filtering
  - **Query Params**: `?category=crypto|forex`
  - **Response**: Array of instruments with `has_data=true AND models_trained=true`

- `GET /api/instruments/:symbol/data/:timeframe` - Get OHLCV chart data
  - **Path Params**: `symbol` (e.g., BTCUSDT), `timeframe` (1m, 5m, 1h, 1D, 1M)
  - **Query Params**: `?limit=100` (number of candles)
  - **Response**: Aggregated OHLCV data for charting

- `GET /api/instruments/:symbol/predictions` - Get ML predictions
  - **Response**: Price predictions from trained models (placeholder)

- `GET /api/instruments/:symbol/signals` - Get trading signals
  - **Response**: Buy/sell signals with confidence scores (placeholder)

- `GET /api/instruments/:symbol/stats` - Get statistical summary
  - **Response**: Returns, volatility, Sharpe ratio, etc.

- `GET /api/trading-platform/health` - Check Trading Platform connection
  - **Response**: `{ tradingPlatform: { status: "healthy" | "unavailable" } }`

#### User Profile (Protected)
- `PUT /api/profile/update` - Update user profile
- `DELETE /api/profile/delete` - Delete user account

#### Public Routes
- `GET /` - Landing page
- `GET /auth` - Authentication page
- `GET /dashboard` - User dashboard (redirects if not authenticated)
- `GET /profile` - Profile page (redirects if not authenticated)
- `GET /admin` - Admin panel (separate authentication)

### Trading Platform API (FastAPI - Port 8001)

#### Admin Authentication
- `POST /api/admin/auth/register` - Create admin account
- `POST /api/admin/auth/login` - Admin login (returns admin JWT)

#### Admin Operations (Protected - Requires Admin JWT)
- `GET /api/admin/profiles` - Get all trading profiles
- `POST /api/admin/profiles` - Create new trading profile
- `GET /api/admin/profiles/:id/data-status` - Check data collection status
- `POST /api/data/collect` - Start data collection for instruments
- `POST /api/admin/train-models` - Train ML models for instruments

#### User API (Protected - Requires User JWT)
- `GET /api/user/instruments` - **CRITICAL ENDPOINT** - Get trained instruments only
  - **Filtering**: `WHERE has_data=true AND models_trained=true`
  - **Query Params**: `?category=crypto|forex`

- `GET /api/user/instruments/:symbol/data/:timeframe` - Get aggregated OHLCV data
  - Timeframes: `1m`, `5m`, `1h`, `1D`, `1M`
  - Returns candles aggregated from 1m base data

- `GET /api/user/instruments/:symbol/predictions` - ML model predictions
- `GET /api/user/instruments/:symbol/signals` - Trading signals
- `GET /api/user/instruments/:symbol/stats` - Statistical analysis
- `GET /api/user/health` - API health check

#### WebSocket
- `WS /api/ws/{client_id}` - Real-time updates for admin panel
  - Events: `data_collection_progress`, `model_training_progress`, `status_update`

#### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Database Schema

### PostgreSQL (Trading Platform)

#### trading_profiles Table
- `id` - Primary key
- `symbol` - Trading pair (e.g., BTCUSDT)
- `name` - Instrument name
- `profile_type` - crypto/forex/stock
- `exchange` - Exchange name (binance)
- `has_data` - Boolean flag (data collected)
- `models_trained` - Boolean flag (models trained)
- `last_data_update` - Timestamp of last data update
- `created_at` - Profile creation timestamp
- `updated_at` - Last update timestamp

#### ohlcv_data Table
- `id` - Primary key
- `profile_id` - Foreign key to trading_profiles
- `timestamp` - Candle timestamp (1-minute intervals)
- `open` - Opening price
- `high` - Highest price
- `low` - Lowest price
- `close` - Closing price
- `volume` - Trading volume
- `created_at` - Record creation timestamp

**Indexes**:
- `(profile_id, timestamp)` for fast queries
- `(timestamp)` for range queries

#### ml_models Table
- `id` - Primary key
- `profile_id` - Foreign key to trading_profiles
- `model_type` - arima/garch/gru_attention/cnn_pattern
- `model_path` - File path to saved model
- `accuracy` - Model accuracy score
- `parameters` - JSON parameters
- `trained_at` - Training timestamp
- `status` - training/completed/failed

#### admin_users Table
- `id` - Primary key
- `email` - Admin email (unique)
- `password_hash` - Bcrypt hashed password
- `name` - Admin name
- `created_at` - Account creation timestamp

### SQLite (MoneyMoney - Fallback)

#### users Table
- `id` - Primary key
- `email` - User email (unique)
- `password_hash` - Bcrypt hashed password (12 salt rounds)
- `subscription_status` - active/inactive
- `created_at` - Account creation date
- `updated_at` - Last update date

#### instruments Table (Fallback cache)
- `id` - Primary key
- `symbol` - Trading pair symbol
- `name` - Full instrument name
- `category` - crypto/forex
- `has_data` - 1/0 boolean
- `models_trained` - 1/0 boolean
- `price` - Current price (cached)
- `change_percent` - 24h change (cached)
- `last_updated` - Cache timestamp

## Data Flow

### Admin Workflow (Data Preparation)

1. **Login to Admin Panel**
   - Navigate to http://localhost:3000/admin
   - Login with admin credentials

2. **Collect Market Data**
   - Select instruments (e.g., BTCUSDT, ETHUSDT)
   - Click "Start Collection"
   - Data stored at **1-minute intervals** in PostgreSQL
   - Sets `has_data = true` on completion

3. **Train ML Models**
   - Select instruments with collected data
   - Click "Train Models"
   - Trains ARIMA, GARCH, GRU_Attention, CNN_Pattern
   - Sets `models_trained = true` on completion

4. **Monitor Progress**
   - WebSocket provides real-time updates
   - Progress bars show collection/training status

### User Workflow (Data Consumption)

1. **Login to Dashboard**
   - Navigate to http://localhost:3000
   - Sign up or login

2. **View Trained Instruments**
   - API calls: `GET /api/instruments`
   - MoneyMoney proxies to Trading Platform
   - **CRITICAL FILTER**: `WHERE has_data=true AND models_trained=true`
   - Only trained instruments appear in dropdown

3. **View Charts**
   - Select instrument from dropdown
   - Choose timeframe (1m, 5m, 1h, 1D, 1M)
   - Data aggregated on-demand from 1m base data
   - Chart updates every 60 seconds

4. **View Predictions & Signals**
   - Predictions tab shows ML model outputs
   - Signals tab shows buy/sell recommendations
   - Stats tab shows performance metrics

### Fallback Behavior

If Trading Platform is unavailable:
- MoneyMoney falls back to local SQLite database
- Connection status indicator shows "Trading Platform Unavailable"
- Limited functionality (cached data only)
- No real-time updates

## Usage Examples

### Admin: Collect Data for BTCUSDT

```bash
# Using curl
curl -X POST http://localhost:8001/api/data/collect \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["BTCUSDT"],
    "interval": "1m",
    "days": 30
  }'
```

### Admin: Train Models

```bash
curl -X POST http://localhost:8001/api/admin/train-models \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "profile_ids": [1, 2, 3],
    "model_types": ["arima", "garch", "gru_attention", "cnn_pattern"]
  }'
```

### User: Get Trained Instruments

```bash
curl http://localhost:3000/api/instruments \
  -H "Authorization: Bearer $USER_TOKEN"
```

### User: Get 1-Hour Chart Data

```bash
curl "http://localhost:3000/api/instruments/BTCUSDT/data/1h?limit=168" \
  -H "Authorization: Bearer $USER_TOKEN"
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using port 8001
lsof -i :8001

# Kill the process
kill -9 <PID>

# Or use the startup script which checks automatically
./start-all.sh
```

### Trading Platform Connection Failed

```bash
# Check Trading Platform health
curl http://localhost:8001/api/user/health

# Check logs
cd "/home/calvin/Websites/Trading Dashboard/other platform/Trading"
tail -f logs/app.log

# Restart Trading Platform
./start.sh
```

### No Instruments Showing in Dashboard

**Cause**: No instruments have both `has_data=true` AND `models_trained=true`

**Solution**:
1. Login to admin panel
2. Collect data for at least one instrument
3. Train models for that instrument
4. Refresh user dashboard

### PostgreSQL Connection Error

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Check database exists
psql -U postgres -l | grep trading_platform

# Create database if missing
psql -U postgres -c "CREATE DATABASE trading_platform;"
```

### WebSocket Connection Failed

**Cause**: CORS or WebSocket protocol issues

**Check**:
1. Both services are running
2. No proxy blocking WebSocket upgrade
3. Correct WebSocket URL in React app

### Chart Not Updating

**Cause**: Auto-refresh not working

**Solution**:
1. Check browser console for errors
2. Verify Trading Platform is healthy: http://localhost:3000/api/trading-platform/health
3. Check network tab for failed API calls
4. Verify JWT token is valid

## Production Deployment

### Environment Variables

**Trading Platform (.env)**:
```env
DATABASE_URL=postgresql://production_user:secure_password@db_host:5432/trading_platform
BINANCE_API_KEY=production_api_key
BINANCE_API_SECRET=production_api_secret
JWT_SECRET=super_secure_random_string_min_32_chars
ADMIN_JWT_SECRET=different_super_secure_random_string_min_32_chars
DEFAULT_DATA_INTERVAL=1m
CACHE_TTL_SECONDS=300
LOG_LEVEL=INFO
```

**MoneyMoney (.env)**:
```env
PORT=3000
TRADING_API_URL=https://api.yourtrading platform.com
JWT_SECRET=production_jwt_secret_min_32_chars
CHART_REFRESH_INTERVAL_MS=60000
NODE_ENV=production
```

### Security Checklist

- [ ] Change all default JWT secrets (32+ random characters)
- [ ] Use HTTPS for all production traffic
- [ ] Enable CORS only for trusted origins
- [ ] Set up database backups (daily recommended)
- [ ] Use strong PostgreSQL passwords
- [ ] Store Binance API keys securely (never commit to git)
- [ ] Enable rate limiting on all API endpoints
- [ ] Set up monitoring and alerting
- [ ] Use environment variables for all secrets
- [ ] Enable logging with rotation
- [ ] Set up firewall rules (only expose necessary ports)
- [ ] Use reverse proxy (nginx) for SSL termination
- [ ] Implement API versioning
- [ ] Add request validation and sanitization

### Deployment Recommendations

**Infrastructure**:
- Use Docker containers for easy deployment
- Set up load balancer for scaling
- Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
- Implement caching layer (Redis)
- Use CDN for static assets

**Monitoring**:
- Application monitoring (New Relic, DataDog)
- Error tracking (Sentry)
- Log aggregation (ELK stack, CloudWatch)
- Uptime monitoring (Pingdom, UptimeRobot)

**Database**:
- Regular backups (automated daily)
- Point-in-time recovery enabled
- Read replicas for scalability
- Connection pooling (PgBouncer)
- Index optimization

## Development

### Adding New Instruments

**Admin Panel Method**:
1. Login to admin panel
2. Navigate to "Profiles" or use data collection interface
3. Add new symbol (must match Binance symbol format)

**Direct Database Method**:
```sql
-- Insert into Trading Platform database
INSERT INTO trading_profiles (symbol, name, profile_type, exchange, has_data, models_trained)
VALUES ('ADAUSDT', 'Cardano', 'crypto', 'binance', false, false);

-- Then collect data and train models via admin panel
```

### Adding New Timeframes

Edit [data_aggregator.py](../other platform/Trading/api/services/data_aggregator.py):

```python
TIMEFRAME_MAP = {
    '1m': '1min',
    '5m': '5min',
    '15m': '15min',  # Add new timeframe
    '1h': '1h',
    '1D': '1D',
    '1M': '1M'
}
```

Update [dashboard.js](public/js/dashboard.js):
```javascript
// Add to timeframe buttons
<button data-timeframe="15m">15m</button>
```

### Adding New ML Models

1. **Create model file**: `other platform/Trading/models/your_model.py`
2. **Implement interface**:
```python
class YourModel:
    def train(self, data):
        # Training logic
        pass

    def predict(self, data):
        # Prediction logic
        pass

    def save(self, path):
        # Save model
        pass
```

3. **Register in training system**: Update model training logic to include your model

### Customizing UI Theme

Edit [styles.css](public/css/styles.css):

```css
:root {
    /* Primary colors */
    --color-primary-accent: #E1007A;     /* Pink gradient start */
    --color-secondary-accent: #8A2BE2;   /* Purple gradient end */

    /* Background */
    --color-background: #0A0A0F;
    --color-card-background: rgba(255, 255, 255, 0.03);

    /* Text */
    --color-text-primary: #FFFFFF;
    --color-text-secondary: rgba(255, 255, 255, 0.7);

    /* Glassmorphism */
    --glass-blur: blur(10px);
    --glass-opacity: 0.03;
}
```

## Testing

### Integration Testing

**Test complete workflow**:
```bash
# 1. Start both services
./start-all.sh

# 2. Create admin account
curl -X POST http://localhost:8001/api/admin/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@admin.com","password":"test123","name":"Test Admin"}'

# 3. Login admin
ADMIN_TOKEN=$(curl -X POST http://localhost:8001/api/admin/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@admin.com","password":"test123"}' | jq -r '.access_token')

# 4. Collect data
curl -X POST http://localhost:8001/api/data/collect \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbols":["BTCUSDT"],"interval":"1m","days":7}'

# 5. Train models
curl -X POST http://localhost:8001/api/admin/train-models \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"profile_ids":[1],"model_types":["arima"]}'

# 6. Create user account
curl -X POST http://localhost:3000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@user.com","password":"test123"}'

# 7. Login user
USER_TOKEN=$(curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@user.com","password":"test123"}' | jq -r '.token')

# 8. Verify user sees trained instrument
curl http://localhost:3000/api/instruments \
  -H "Authorization: Bearer $USER_TOKEN" | jq

# 9. Get chart data
curl "http://localhost:3000/api/instruments/BTCUSDT/data/1h?limit=24" \
  -H "Authorization: Bearer $USER_TOKEN" | jq
```

### Health Checks

```bash
# MoneyMoney
curl http://localhost:3000/

# Trading Platform
curl http://localhost:8001/api/user/health

# PostgreSQL
psql -U postgres -d trading_platform -c "SELECT COUNT(*) FROM trading_profiles;"
```

## FAQ

### Q: Why are no instruments showing in my dashboard?

**A**: Instruments only appear when BOTH conditions are met:
- `has_data = true` (data has been collected)
- `models_trained = true` (models have been trained)

Use admin panel to collect data and train models first.

### Q: Can I use this with exchanges other than Binance?

**A**: Yes, but you'll need to:
1. Implement adapter for new exchange API
2. Update data collection service
3. Ensure symbol format compatibility

### Q: How much historical data should I collect?

**A**: Recommendations:
- **Minimum**: 7 days for basic models
- **Recommended**: 30 days for better accuracy
- **Optimal**: 90+ days for deep learning models

### Q: What's the difference between admin and user JWT tokens?

**A**:
- **Admin JWT**: Signed with `ADMIN_JWT_SECRET`, full system access
- **User JWT**: Signed with `JWT_SECRET`, read-only access to trained data
- Completely separate authentication systems for security

### Q: Can I run both services on different servers?

**A**: Yes! Update `TRADING_API_URL` in MoneyMoney's `.env`:
```env
TRADING_API_URL=https://api.yourdomain.com
```

Ensure:
- CORS is properly configured
- Firewall allows traffic between servers
- HTTPS is used in production

### Q: How do I backup my data?

**A**:
```bash
# PostgreSQL backup
pg_dump -U postgres trading_platform > backup_$(date +%Y%m%d).sql

# SQLite backup
cp database/trading_dashboard.db backup_$(date +%Y%m%d).db

# Restore PostgreSQL
psql -U postgres trading_platform < backup_20251019.sql
```

## Known Limitations

1. **Predictions Endpoint**: Currently returns placeholder data
   - Implementation needed: Connect to actual model outputs

2. **Signals Endpoint**: Currently returns placeholder data
   - Implementation needed: Signal generation logic

3. **Sentiment Analysis**: Not yet implemented
   - Future: News sentiment integration
   - Future: Social media sentiment

4. **Training Queue**: No concurrent job management
   - Future: Priority queue system
   - Future: Job cancellation support

5. **Real-time Price Updates**: User dashboard uses polling (60s)
   - Future: WebSocket for real-time prices

## Roadmap

### Phase 5: Advanced Features (Planned)
- [ ] ML prediction integration with real model outputs
- [ ] Trading signal generation system
- [ ] Advanced training queue with priorities
- [ ] Job cancellation and retry logic

### Phase 6: Sentiment Analysis (Planned)
- [ ] News sentiment aggregator
- [ ] Social media sentiment integration
- [ ] Sentiment display in dashboard

### Phase 7: Performance Optimization (Planned)
- [ ] Redis caching layer
- [ ] WebSocket for user price updates
- [ ] Database query optimization
- [ ] API response caching

### Phase 8: Testing & Quality (Planned)
- [ ] Unit tests for all services
- [ ] Integration test suite
- [ ] Performance benchmarks
- [ ] Security audit

## Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Complete integration documentation
- **[API Documentation](http://localhost:8001/docs)** - Interactive API docs (when server running)
- **[.env.example](.env.example)** - Environment configuration template

## Support

For issues, questions, or contributions:
- Check the [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for detailed setup
- Review troubleshooting section above
- Check server logs for error details

## License

This project is proprietary. All rights reserved.

---

**MoneyMoney** - AI-Powered Trading Dashboard with Real-Time ML Predictions