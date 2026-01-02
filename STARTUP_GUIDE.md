# 🚀 Complete Startup Guide

## Quick Start

### Start Everything Automatically

From the MoneyMoney root directory, run:

```bash
./start-all.sh
```

This will open **3 terminal windows** and start:

1. **Trading Platform Backend** (Port 8001) - FastAPI + Python
2. **MoneyMoney Landing Page** (Port 3000) - Node.js server
3. **Admin Frontend** (Port 5173) - React + Vite

---

## 🌐 Access the Services

After running `./start-all.sh`, you can access:

### **Admin Portal (Main Application)**
- **Login Page:** http://localhost:5173/login
- **Dashboard:** http://localhost:5173/dashboard
- **Trading:** http://localhost:5173/trading
- **ML Models:** http://localhost:5173/models
- **Data Management:** http://localhost:5173/data

**Login Credentials:**
- Username: `admin`
- Password: `admin123`

### **MoneyMoney Landing Page**
- **Homepage:** http://localhost:3000

### **Backend API**
- **API Root:** http://localhost:8001
- **API Docs:** http://localhost:8001/docs
- **Interactive Docs:** http://localhost:8001/redoc

---

## 📋 What Each Service Does

### 1. Trading Platform Backend (Port 8001)
- FastAPI Python application
- Handles all API requests
- Connects to PostgreSQL database
- Manages ML model training and predictions
- Processes trading orders

### 2. MoneyMoney Landing Page (Port 3000)
- Marketing/landing page
- Node.js static server
- Public-facing website

### 3. Admin Frontend (Port 5173)
- React + TypeScript application
- Vite development server
- Admin control panel
- ML model training interface
- Trading dashboard

---

## 🔄 Start Services Individually

If you prefer to start services manually or one at a time:

### Backend API
```bash
cd "other platform/Trading"
./start.sh
```

### MoneyMoney Landing Page
```bash
./start.sh
```

### Admin Frontend
```bash
cd "other platform/Trading/frontend"
npm run dev
```

---

## 🛑 Stop Services

### Stop All Services
Close each of the 3 terminal windows, or press `Ctrl+C` in each terminal.

### Stop Individual Service
Find the process and kill it:

```bash
# Stop backend (port 8001)
kill $(lsof -t -i:8001)

# Stop MoneyMoney (port 3000)
kill $(lsof -t -i:3000)

# Stop admin frontend (port 5173)
kill $(lsof -t -i:5173)
```

---

## 🔧 Troubleshooting

### "Port already in use" error

The script will ask if you want to kill the existing process. Answer `y` to automatically stop it.

Alternatively, manually kill processes:
```bash
# Check what's using a port
lsof -i :5173

# Kill process on port
kill $(lsof -t -i:5173)
```

### Admin frontend doesn't start

Check if npm dependencies are installed:
```bash
cd "other platform/Trading/frontend"
npm install
```

### Can't access http://localhost:5173/login

Make sure:
1. The admin frontend terminal is running (check for "VITE" in the terminal)
2. No firewall is blocking port 5173
3. Backend is running on port 8001 (required for login to work)

### Login fails with "Invalid credentials"

Make sure:
1. Backend is running and accessible: `curl http://localhost:8001/`
2. PostgreSQL database is running
3. Admin user exists in database:
   ```bash
   psql -U postgres -d trading_platform -c "SELECT * FROM admin_users;"
   ```

---

## 💡 Tips

- **Hot Reload:** The admin frontend (Vite) supports hot reload - changes to code are instantly reflected
- **API Docs:** Use http://localhost:8001/docs to test API endpoints interactively
- **Logs:** Check each terminal window for logs and error messages
- **Database:** PostgreSQL must be running for the backend to work

---

## 📊 Service Dependencies

```
Admin Frontend (5173)
    ↓
Backend API (8001)
    ↓
PostgreSQL Database

MoneyMoney Landing (3000)
    ↓
(Independent - no backend required)
```

---

## 🎯 Common Workflows

### Train a Model
1. Start all services: `./start-all.sh`
2. Login at http://localhost:5173/login
3. Go to ML Models page
4. Select BTCUSDT profile
5. Choose model type and click "Train"

### View Trading Data
1. Login to admin portal
2. Go to Data Management
3. View BTCUSDT profile (10,081 candles)

### Check System Status
1. Login to admin portal
2. Go to Dashboard
3. View system metrics and performance

---

## 🚨 Important Notes

- ⚠️ Default password (`admin123`) should be changed in production
- ⚠️ Port 8001 must be accessible for admin frontend to work
- ⚠️ Close all 3 terminal windows when done to free up ports
- ⚠️ PostgreSQL must be running before starting services

---

## 📞 Need Help?

Check the logs in each terminal window for detailed error messages.

### Quick Health Check

```bash
# Check if services are running
lsof -i :8001  # Backend API
lsof -i :3000  # MoneyMoney Landing
lsof -i :5173  # Admin Frontend

# Test backend
curl http://localhost:8001/

# Test admin frontend
curl http://localhost:5173/
```

---

**Happy Trading! 🎉**
