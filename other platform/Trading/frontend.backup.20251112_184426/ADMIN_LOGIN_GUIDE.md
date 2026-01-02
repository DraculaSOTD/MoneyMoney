# Admin Login Guide

## 🔐 Login Credentials

**Default Admin Account:**
- **Username:** `admin`
- **Password:** `admin123`
- **Email:** `admin@tradingplatform.local`
- **Role:** Superuser (full access)

⚠️ **IMPORTANT:** Change the default password in production!

---

## 🚀 How to Access the Admin Portal

### 1. Start the Services

Make sure both the backend API and frontend are running:

```bash
# From MoneyMoney root directory
./start-all.sh

# OR manually:

# Terminal 1 - Start Backend API
cd "other platform/Trading"
./start.sh

# Terminal 2 - Start Admin Frontend
cd "other platform/Trading/frontend"
npm run dev
```

### 2. Access the Login Page

Open your browser and navigate to:
```
http://localhost:5173/login
```

### 3. Sign In

1. Enter **Username:** `admin`
2. Enter **Password:** `admin123`
3. Click **"Sign In"**

You'll be automatically redirected to the dashboard upon successful login.

---

## 📊 Available Pages After Login

Once logged in, you have access to:

### **Dashboard** (`/dashboard`)
- System overview
- Total P&L and performance metrics
- Active positions
- Recent trades

### **Trading** (`/trading`)
- Place market/limit orders
- View and manage positions
- Real-time price charts
- Technical indicators

### **ML Models** (`/models`) ⭐ **← Train Models Here!**
- Browse available ML models
- Configure training parameters
- Start model training jobs
- View training history and metrics
- Deploy trained models

### **Backtesting** (`/backtesting`)
- Configure backtest parameters
- Run strategy backtests
- View equity curves
- Performance reports

### **Data Management** (`/data`)
- View BTCUSDT profile (10,081 candles)
- Create new trading profiles
- Import historical data
- Monitor data quality

### **Monitoring** (`/monitoring`)
- System health metrics
- Performance reports
- Alert management

---

## 🤖 How to Train ML Models

### Step 1: Navigate to Models Page
After logging in, click **"ML Models"** in the sidebar or go directly to:
```
http://localhost:5173/models
```

### Step 2: Select Profile
Choose **BTCUSDT** from the profile dropdown selector at the top of the page.

### Step 3: Configure Training
1. Browse available model types:
   - LSTM/GRU (Deep Learning)
   - Random Forest
   - XGBoost
   - ARIMA/GARCH (Statistical)
   - Ensemble models

2. Click on a model to configure:
   - Set hyperparameters
   - Choose training epochs
   - Configure features

### Step 4: Start Training
1. Click **"Train Model"** button
2. Monitor real-time progress via WebSocket updates
3. View training metrics (loss, accuracy, etc.)

### Step 5: Deploy Model
Once training completes:
1. View performance metrics
2. Click **"Deploy"** to activate the model
3. Model will start generating predictions

---

## 👤 User Profile Features

The admin portal now displays:

### In the Header:
- **Username** displayed next to avatar
- **"Superuser"** badge (if you have superuser permissions)
- **Avatar** with user initials

### In the Profile Menu:
Click on your avatar to see:
- Full username
- Email address
- Superuser status badge
- **Logout** button

---

## 🔒 Security Features

### Authentication:
- ✅ JWT token-based authentication
- ✅ 8-hour token expiration
- ✅ Secure password hashing (bcrypt)
- ✅ Account lockout after 5 failed login attempts (30 minutes)

### Session Management:
- ✅ Token automatically added to all API requests
- ✅ Token verification on page load
- ✅ Auto-logout on token expiration
- ✅ Backend session invalidation on logout

### Error Handling:
- ✅ Clear error messages for invalid credentials
- ✅ Account locked warnings
- ✅ Session expired notifications
- ✅ Automatic redirect to login when unauthorized

---

## 🛠️ Troubleshooting

### "Invalid credentials" error
- Double-check username: `admin` (lowercase)
- Double-check password: `admin123`
- Make sure PostgreSQL database is running
- Verify backend API is running on port 8001

### "Session expired" message
- JWT tokens expire after 8 hours
- Simply log in again with your credentials

### "Account is locked" error
- Wait 30 minutes for automatic unlock
- Or ask a superuser admin to unlock your account

### Can't see BTCUSDT in dropdowns
- Make sure you ran the data import script
- Check PostgreSQL database has BTCUSDT profile
- Restart the backend API

### Login page shows but login fails
- Check browser console for errors
- Verify backend API is accessible at http://localhost:8001
- Test API directly: `curl http://localhost:8001/`

---

## 📝 API Endpoints Used

The login system uses these backend endpoints:

- **POST** `/admin/auth/login` - Login with username/password
- **GET** `/admin/auth/verify` - Verify JWT token is valid
- **POST** `/admin/auth/logout` - Invalidate session
- **GET** `/admin/auth/list` - List all admins (superuser only)

All endpoints are documented in the API docs:
```
http://localhost:8001/docs
```

---

## 🎯 Next Steps

After logging in successfully:

1. **Explore the Dashboard** - Get familiar with the system overview
2. **Check Data Management** - Verify BTCUSDT data is loaded
3. **Train Your First Model** - Go to ML Models page and start training
4. **View Results** - Monitor training progress and deploy models
5. **Start Trading** - Use deployed models for live trading signals

---

## 💡 Tips

- The session persists across page refreshes
- You'll stay logged in for 8 hours unless you log out manually
- The "Superuser" badge gives you full administrative access
- Check the system status indicator in the header (green = running)

---

## 🆘 Need Help?

If you encounter any issues:

1. Check the browser console for detailed error messages
2. Verify all services are running (`./start-all.sh`)
3. Check the backend logs for API errors
4. Ensure PostgreSQL is running and accessible

---

**Happy Trading! 🚀**
