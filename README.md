# TradingDashboard AI - Trading Signals Platform

A professional trading signals and market analysis platform designed for crypto and forex traders. TradingDashboard AI provides AI-powered trading signals, trend analysis, and model accuracy metrics to help traders make informed decisions.

## Features

### 🚀 Core Features
- **Real-time Trading Signals**: AI-generated buy/sell signals for crypto and forex
- **Market Analysis**: Comprehensive trend analysis and market sentiment
- **AI Models**: Multiple AI models with backtesting results and accuracy scores
- **Risk Management**: Take profit and stop loss recommendations
- **Mobile Responsive**: Optimized for all devices

### 📊 Trading Instruments
- **Cryptocurrency**: BTC/USD, ETH/USD, ADA/USD and more
- **Forex**: USD/GBP, GBP/USD and major currency pairs
- **Expandable**: Easy to add more trading instruments

### 👤 User Features
- **Secure Authentication**: JWT-based login system
- **User Profiles**: Account management and settings
- **Subscription Management**: $10/month premium membership
- **Usage Statistics**: Track your trading performance

## Technology Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Node.js with Express.js
- **Database**: SQLite3
- **Authentication**: JWT tokens with bcrypt hashing
- **Security**: Helmet.js, CORS, Rate limiting

## Installation

### Prerequisites
- Node.js (v16 or higher)
- npm (v7 or higher)

### Setup Instructions

1. **Clone and Navigate**
   ```bash
   cd MoneyMoney
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start the Server**
   ```bash
   npm start
   ```

4. **Development Mode** (with auto-restart)
   ```bash
   npm run dev
   ```

5. **Access the Application**
   - Open your browser to `http://localhost:3000`
   - The database will be automatically created and populated with sample data

## Project Structure

```
MoneyMoney/
├── public/                 # Static files
│   ├── css/
│   │   └── styles.css     # Main stylesheet
│   └── js/
│       └── main.js        # Frontend JavaScript
├── views/                 # HTML templates
│   ├── index.html         # Landing page
│   ├── auth.html          # Login/Signup page
│   ├── dashboard.html     # Main trading dashboard
│   └── profile.html       # User profile management
├── database/              # SQLite database files
├── server.js              # Main server file
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

## API Endpoints

### Authentication
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login

### Protected Routes (require authentication)
- `GET /api/instruments` - Get trading instruments and signals
- `GET /api/models` - Get AI model performance data
- `PUT /api/profile/update` - Update user profile
- `DELETE /api/profile/delete` - Delete user account

### Public Routes
- `GET /` - Landing page
- `GET /auth` - Authentication page
- `GET /dashboard` - Trading dashboard (redirects if not authenticated)
- `GET /profile` - Profile page (redirects if not authenticated)

## Database Schema

### Users Table
- `id` - Primary key
- `email` - User email (unique)
- `password_hash` - Bcrypt hashed password
- `subscription_status` - Subscription status
- `created_at` - Account creation date
- `updated_at` - Last update date

### Instruments Table
- `id` - Primary key
- `symbol` - Trading pair symbol (e.g., BTC/USD)
- `name` - Full instrument name
- `category` - crypto/forex
- `price` - Current price
- `change_percent` - Price change percentage
- `entry_point` - Recommended entry price
- `take_profit` - Take profit target
- `stop_loss` - Stop loss level
- `signal` - BUY/SELL signal
- `confidence` - AI confidence percentage
- `model_accuracy` - Model accuracy percentage

### Models Table
- `id` - Primary key
- `name` - Model name
- `description` - Model description
- `accuracy` - Model accuracy percentage
- `backtest_period` - Backtesting period
- `total_trades` - Total trades in backtest
- `win_rate` - Win rate percentage

## Security Features

- **Password Hashing**: bcrypt with salt rounds
- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: API rate limiting to prevent abuse
- **Input Validation**: Server-side validation for all inputs
- **CORS Protection**: Configured for secure cross-origin requests
- **Helmet.js**: Security headers and protections

## Customization

### Adding New Trading Instruments
1. Insert into the `instruments` table via SQL
2. Update the category options if needed
3. Ensure price data is regularly updated

### Adding New AI Models
1. Insert into the `models` table
2. Include backtesting results and accuracy metrics
3. Update model descriptions and performance data

### Styling Customization
- Edit `public/css/styles.css`
- Modify CSS custom properties in the `:root` section
- All colors and spacing use CSS variables for easy theming

## Production Deployment

### Environment Variables
Create a `.env` file with:
```
PORT=3000
JWT_SECRET=your-secure-jwt-secret-key
NODE_ENV=production
```

### Database
- SQLite is suitable for development and small-scale production
- For larger scale, consider migrating to PostgreSQL or MySQL
- Ensure regular database backups

### Security Recommendations
- Use HTTPS in production
- Set secure JWT secret key
- Configure proper CORS origins
- Enable additional rate limiting
- Regular security updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is proprietary. All rights reserved.

## Support

For support and questions, please contact the development team.

---

**TradingDashboard AI** - Empowering traders with AI-driven market insights.