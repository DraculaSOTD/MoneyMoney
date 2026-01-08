# 🚀 Automated Setup

## Quick Setup (Recommended)

Run the automated setup script that handles everything:

```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./setup.sh
```

This single command will:
- ✅ Check all prerequisites
- ✅ Prompt for PostgreSQL credentials (securely)
- ✅ Create database if needed
- ✅ Update .env files automatically
- ✅ Install all Node.js dependencies
- ✅ Create Python virtual environment
- ✅ Install all Python dependencies
- ✅ Import 10,082 BTCUSDT candles
- ✅ Make startup scripts executable
- ✅ Verify everything is ready

**Time required:** 5-10 minutes (mostly automated)

---

## What You'll Need

The script will ask you for:

1. **PostgreSQL username** (default: `postgres`)
2. **PostgreSQL password** (your database password)
3. **Database name** (default: `trading_platform`)
4. **PostgreSQL host** (default: `localhost`)
5. **PostgreSQL port** (default: `5432`)

> **Note:** The password is entered securely (not shown on screen)

---

## Example Run

```bash
$ ./setup.sh

═══════════════════════════════════════════════════════════
  MoneyMoney & Trading Platform Setup
═══════════════════════════════════════════════════════════

This script will:
  1. Check prerequisites (PostgreSQL, Python, Node.js)
  2. Configure PostgreSQL database
  3. Install all dependencies
  4. Import BTCUSDT sample data
  5. Prepare applications for launch

Continue? (y/n) y

▶ Checking prerequisites...
✓ PostgreSQL client found: psql (PostgreSQL) 17.5
✓ PostgreSQL service is running
✓ Python found: 3.13.0
✓ Node.js found: v24.8.0
✓ npm found: 11.6.0

▶ Configuring PostgreSQL database...

ℹ Please provide your PostgreSQL credentials

PostgreSQL username [postgres]: postgres
PostgreSQL password: ********
Database name [trading_platform]: trading_platform
PostgreSQL host [localhost]: localhost
PostgreSQL port [5432]: 5432

▶ Testing PostgreSQL connection...
✓ PostgreSQL connection successful

▶ Checking if database 'trading_platform' exists...
⚠ Database 'trading_platform' does not exist
Create database 'trading_platform'? (y/n) y
✓ Database 'trading_platform' created

▶ Updating Trading Platform .env file...
✓ DATABASE_URL updated in Trading Platform .env

▶ Installing Node.js dependencies for MoneyMoney...
✓ Node.js dependencies installed

▶ Setting up Python virtual environment...
✓ Virtual environment created

▶ Installing Python dependencies (this may take a few minutes)...
✓ Python dependencies installed

▶ Importing BTCUSDT sample data...
ℹ This will import 10,082 1-minute candles into PostgreSQL
🚀 Starting BTCUSDT data import...
✓ BTCUSDT data imported successfully

▶ Making startup scripts executable...
✓ Startup scripts are executable

═══════════════════════════════════════════════════════════
  Setup Complete! 🎉
═══════════════════════════════════════════════════════════

All setup steps completed successfully!

Summary:
✓ PostgreSQL database configured: trading_platform
✓ Node.js dependencies installed
✓ Python virtual environment created
✓ Python dependencies installed
✓ BTCUSDT data imported (10,082 candles)
✓ Startup scripts ready

═══════════════════════════════════════════════════════════
  Next Steps
═══════════════════════════════════════════════════════════

1. Start the applications:
   ./start-all.sh

2. Access MoneyMoney dashboard:
   http://localhost:3000/auth

3. Login with test account:
   Email:    test@example.com
   Password: testpassword

4. Select BTCUSDT and click 'Analysis' tab to see 70+ indicators!
```

---

## After Setup

Once setup is complete, starting the applications is simple:

```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./start-all.sh
```

This opens two terminal windows:
- **Trading Platform** (port 8001)
- **MoneyMoney** (port 3000)

Then visit: **http://localhost:3000/auth**

---

## Troubleshooting

### "Permission denied" error

Make sure the script is executable:
```bash
chmod +x setup.sh
```

### "PostgreSQL connection failed"

Check if PostgreSQL is running:
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

### "Python module not found"

The script creates a virtual environment automatically. If you see this error, the venv may not have activated properly. Try manually:
```bash
cd "other platform/Trading"
source venv/bin/activate
pip install -r requirements.txt
```

### "Database already exists"

That's fine! The script will use the existing database. Make sure it's the correct one you want to use.

### Script hangs or fails

Check the log files:
- `/tmp/pip_install.log` - Python dependency installation
- `/tmp/import_data.log` - BTCUSDT data import

---

## What Gets Modified

The setup script makes these changes:

1. **Creates `.env` files** (if they don't exist):
   - `MoneyMoney/.env`
   - `MoneyMoney/other platform/Trading/.env`

2. **Updates DATABASE_URL** in Trading Platform `.env`:
   - Old .env is backed up to `.env.backup`

3. **Creates directories**:
   - `MoneyMoney/node_modules/` (npm packages)
   - `MoneyMoney/other platform/Trading/venv/` (Python virtual env)
   - `MoneyMoney/database/` (SQLite database)

4. **Makes scripts executable**:
   - `start.sh`
   - `start-all.sh`
   - `other platform/Trading/start.sh`

5. **Imports data to PostgreSQL**:
   - Creates `trading_profiles` table
   - Creates `market_data` table
   - Inserts BTCUSDT profile and 10,082 candles

---

## Manual Setup Alternative

If you prefer to set things up manually, see:
- [STARTUP_GUIDE.md](STARTUP_GUIDE.md) - Detailed manual setup
- [QUICK_START.md](QUICK_START.md) - Quick reference commands

---

## Re-running Setup

The script is safe to run multiple times. It will:
- Skip steps that are already complete
- Ask before overwriting configurations
- Backup existing `.env` files before updating

---

## Getting Help

- Full documentation: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- Detailed startup guide: [STARTUP_GUIDE.md](STARTUP_GUIDE.md)
- Quick reference: [QUICK_START.md](QUICK_START.md)
- Implementation status: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)

---

**Ready to start?** Just run:

```bash
./setup.sh
```

Happy trading! 💰📈
