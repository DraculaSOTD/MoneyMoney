# ✅ Setup Checklist

Use this checklist to verify your setup is complete and working.

## Before Running setup.sh

- [ ] PostgreSQL is installed
- [ ] PostgreSQL service is running (`sudo systemctl start postgresql`)
- [ ] You know your PostgreSQL username (usually `postgres`)
- [ ] You know your PostgreSQL password
- [ ] Python 3.8+ is installed
- [ ] Node.js 14+ is installed
- [ ] You're in the MoneyMoney directory

## Running setup.sh

```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./setup.sh
```

During setup, verify:

- [ ] ✅ PostgreSQL client found
- [ ] ✅ PostgreSQL service is running
- [ ] ✅ Python found
- [ ] ✅ Node.js found
- [ ] ✅ npm found
- [ ] ✅ PostgreSQL connection successful
- [ ] ✅ Database created or exists
- [ ] ✅ DATABASE_URL updated in .env
- [ ] ✅ Node.js dependencies installed
- [ ] ✅ Virtual environment created
- [ ] ✅ Python dependencies installed
- [ ] ✅ BTCUSDT data imported successfully
- [ ] ✅ Startup scripts are executable

## After setup.sh Completes

### File Structure

Verify these files/directories exist:

- [ ] `MoneyMoney/.env` (configured)
- [ ] `MoneyMoney/node_modules/` (exists and populated)
- [ ] `MoneyMoney/other platform/Trading/.env` (configured)
- [ ] `MoneyMoney/other platform/Trading/.env.backup` (backup of old config)
- [ ] `MoneyMoney/other platform/Trading/venv/` (virtual environment)
- [ ] `MoneyMoney/start-all.sh` (executable: `ls -l` shows `-rwxr-xr-x`)

### Database

Verify PostgreSQL database:

```bash
# Connect to database
psql -U postgres -d trading_platform

# Check tables exist
\dt

# Should show:
# - trading_profiles
# - market_data

# Check BTCUSDT data
SELECT symbol, has_data, total_data_points FROM trading_profiles WHERE symbol = 'BTCUSDT';

# Should show:
# BTCUSDT | t | 10082

# Check candle count
SELECT COUNT(*) FROM market_data WHERE symbol = 'BTCUSDT';

# Should show: 10082

# Exit
\q
```

Checklist:
- [ ] Database `trading_platform` exists
- [ ] Table `trading_profiles` exists
- [ ] Table `market_data` exists
- [ ] BTCUSDT profile has `has_data = true`
- [ ] BTCUSDT has 10,082 candles in `market_data`

## Starting Applications

### Start Command

```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./start-all.sh
```

Verify:
- [ ] Two terminal windows open
- [ ] Terminal 1: Trading Platform starts on port 8001
- [ ] Terminal 2: MoneyMoney starts on port 3000
- [ ] No error messages in either terminal

### Check Services

```bash
# Check if ports are listening
lsof -i :8001  # Trading Platform
lsof -i :3000  # MoneyMoney

# Test API endpoints
curl http://localhost:8001/     # Should return JSON
curl http://localhost:3000/     # Should return HTML
```

Checklist:
- [ ] Port 8001 is listening (Trading Platform)
- [ ] Port 3000 is listening (MoneyMoney)
- [ ] http://localhost:8001 responds
- [ ] http://localhost:3000 responds

## Testing the Application

### 1. Access Login Page

- [ ] Open browser: http://localhost:3000/auth
- [ ] Login page loads without errors
- [ ] No console errors in browser DevTools (F12)

### 2. Login

- [ ] Enter email: `test@example.com`
- [ ] Enter password: `testpassword`
- [ ] Click "Login"
- [ ] Redirects to dashboard: http://localhost:3000/dashboard

### 3. Dashboard Loads

- [ ] Dashboard page loads
- [ ] Instrument dropdown is visible
- [ ] Search bar is visible
- [ ] Chart area is visible

### 4. Select BTCUSDT

- [ ] Click "Select Instrument" dropdown
- [ ] "Bitcoin (BTCUSDT)" appears in the list
- [ ] Click "Bitcoin (BTCUSDT)"
- [ ] Chart loads with candlestick data
- [ ] No errors in browser console

### 5. View Chart

- [ ] Price chart displays candles
- [ ] X-axis shows timestamps
- [ ] Y-axis shows prices
- [ ] Chart is interactive (can zoom/pan)

### 6. Switch Timeframes

Click each timeframe button and verify chart updates:
- [ ] 1m (1-minute candles)
- [ ] 5m (5-minute candles)
- [ ] 1h (1-hour candles)
- [ ] 1D (1-day candles)
- [ ] 1M (1-month candles)

### 7. View Indicators ⭐ NEW FEATURE

- [ ] Click "Analysis" tab
- [ ] Indicators section appears
- [ ] Header shows "Technical Indicators"
- [ ] Count shows "70+ indicators"
- [ ] Indicators grouped by category:
  - [ ] Moving Averages (SMA, EMA, VWMA)
  - [ ] Oscillators & MACD (RSI, Stochastic, MACD)
  - [ ] Volatility Indicators (Bollinger Bands, ATR)
  - [ ] Trend Indicators (Parabolic SAR, ADX, Ichimoku)
  - [ ] Volume Indicators (CMF)
  - [ ] Support & Resistance (Pivot Points, R1-R3, S1-S3)
  - [ ] Divergences
  - [ ] Pattern Recognition

### 8. Check Indicator Values

- [ ] Each indicator shows a numerical value
- [ ] RSI value is between 0-100
- [ ] Bollinger Band values are present
- [ ] MACD components are displayed
- [ ] No "NaN" or "undefined" values

### 9. Check Color Coding

- [ ] Overbought indicators (RSI ≥70) are RED
- [ ] Oversold indicators (RSI ≤30) are GREEN
- [ ] Divergence indicators (if active) are YELLOW/ORANGE

### 10. Auto-Refresh

- [ ] Wait 60 seconds
- [ ] Chart should auto-refresh (you'll see a brief loading state)
- [ ] New data is loaded

### 11. Test API Directly

```bash
# Get auth token
TOKEN=$(curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpassword"}' \
  | jq -r '.token')

echo "Token: $TOKEN"

# Get instruments
curl http://localhost:3000/api/instruments \
  -H "Authorization: Bearer $TOKEN" \
  | jq '.'

# Should return array with BTCUSDT

# Get indicators
curl "http://localhost:3000/api/instruments/BTCUSDT/indicators?limit=200" \
  -H "Authorization: Bearer $TOKEN" \
  | jq '.total_indicators'

# Should return a number > 70
```

Checklist:
- [ ] Login API returns JWT token
- [ ] Instruments API returns BTCUSDT
- [ ] Indicators API returns 70+ indicators
- [ ] No 401/403/500 errors

## Common Issues

### ❌ Issue: Port already in use

**Solution:**
```bash
lsof -ti:3000 | xargs kill -9
lsof -ti:8001 | xargs kill -9
```

### ❌ Issue: Database connection failed

**Check:**
- [ ] PostgreSQL is running: `sudo systemctl status postgresql`
- [ ] DATABASE_URL is correct in `other platform/Trading/.env`
- [ ] Can connect manually: `psql -U postgres -d trading_platform`

### ❌ Issue: No instruments in dropdown

**Check:**
- [ ] Trading Platform is running (http://localhost:8001)
- [ ] BTCUSDT data was imported
- [ ] Run query: `SELECT * FROM trading_profiles;`

### ❌ Issue: Indicators tab is empty

**Check:**
- [ ] Trading Platform is running
- [ ] No errors in browser console
- [ ] API endpoint works: `curl localhost:8001/api/user/instruments/BTCUSDT/indicators`

### ❌ Issue: Python venv doesn't work

**Solution:**
```bash
cd "other platform/Trading"
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Success Criteria

✅ **Setup is successful if:**

1. Both services start without errors
2. You can login to MoneyMoney
3. BTCUSDT appears in instrument dropdown
4. Price chart displays candlestick data
5. **Analysis tab shows 70+ technical indicators**
6. All indicators have valid numerical values
7. No errors in browser console
8. No errors in terminal output

---

## 🎉 All Checked?

Congratulations! Your setup is complete and working perfectly.

**What's Next?**

- Explore different timeframes (1m, 5m, 1h, 1D, 1M)
- Review all 70+ technical indicators in the Analysis tab
- Check out indicator color coding for overbought/oversold signals
- Wait for auto-refresh to see real-time updates

**Need Help?**

- [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Full manual
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Architecture details
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Feature status

---

**Happy Trading! 💰📈**
