# ⚡ Quick Start

## 1. Setup (First Time Only)

```bash
# Navigate to MoneyMoney
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"

# Install Node dependencies
npm install

# Navigate to Trading Platform
cd "other platform/Trading"

# Create Python virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Import BTCUSDT data (10,082 candles)
python scripts/import_btcusdt_data.py
```

**⚠️ Important**: Edit `other platform/Trading/.env` and update the `DATABASE_URL` with your PostgreSQL credentials!

---

## 2. Start Applications

```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./start-all.sh
```

This opens two terminal windows:
- **Trading Platform** (Backend) on port 8001
- **MoneyMoney** (Frontend) on port 3000

---

## 3. Login & Test

1. Open browser: http://localhost:3000/auth
2. Login with:
   - **Email**: `test@example.com`
   - **Password**: `testpassword`
3. Select **"Bitcoin (BTCUSDT)"** from dropdown
4. Click **"Analysis"** tab to see 70+ technical indicators!

---

## 4. Stop Applications

Close both terminal windows or press `Ctrl+C` in each.

---

## URLs

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000/dashboard |
| **Login** | http://localhost:3000/auth |
| **API Docs** | http://localhost:8001/docs |

---

## Troubleshooting

**Port already in use?**
```bash
lsof -ti:3000 | xargs kill -9
lsof -ti:8001 | xargs kill -9
```

**PostgreSQL not running?**
```bash
sudo systemctl start postgresql
```

**Need full guide?** See [STARTUP_GUIDE.md](STARTUP_GUIDE.md)
