# Crypto ML Trading System - Frontend

A comprehensive React-based frontend application for managing a cryptocurrency machine learning trading system.

## Features

### Core Features
- 🔐 **Authentication**: Token-based authentication system
- 📊 **Real-time Dashboard**: Live system metrics and performance monitoring
- 💹 **Trading Interface**: Order placement, position management, and market data visualization
- 🤖 **ML Model Management**: Train, deploy, and monitor machine learning models
- 📈 **Backtesting**: Configure and visualize backtesting results
- 💾 **Data Management**: Upload data, configure symbols, and monitor data quality
- 📡 **Real-time Updates**: WebSocket integration for live market data

### Technical Stack
- **React 18** with TypeScript
- **Redux Toolkit** for state management
- **RTK Query** for API integration
- **Material-UI (MUI) v5** for UI components
- **Socket.io** for WebSocket connections
- **Vite** for fast development and building
- **TradingView Lightweight Charts** (ready to integrate)
- **Recharts** for additional visualizations

## Getting Started

### Prerequisites
- Node.js 18+ 
- Backend API running on http://localhost:8000
- Valid API token for authentication

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure environment:
```bash
# Edit .env file to set your API URL
VITE_API_URL=http://localhost:8000
```

3. Start development server:
```bash
npm run dev
```

The application will be available at http://localhost:3000

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
src/
├── components/          # Reusable components
│   ├── auth/           # Authentication components
│   └── layout/         # Layout components
├── features/           # Redux slices for each feature
│   ├── auth/           # Authentication state
│   ├── system/         # System status state
│   ├── trading/        # Trading state
│   ├── marketData/     # Market data state
│   ├── mlModels/       # ML models state
│   └── backtesting/    # Backtesting state
├── pages/              # Main page components
├── services/           # API services (RTK Query)
├── store/              # Redux store configuration
└── theme/              # Material-UI theme

```

## Authentication

The system uses token-based authentication. On first visit, you'll be prompted to enter an API token that matches the one configured in your backend (`API_TOKEN` environment variable).

## Available Pages

1. **Dashboard**: System overview with key metrics and performance indicators
2. **Trading**: Place orders, manage positions, and view market data
3. **ML Models**: View model registry, start training jobs, compare model performance
4. **Backtesting**: Configure and run backtests, visualize results
5. **Data Management**: Manage symbols, upload data, monitor data quality
6. **Monitoring**: View detailed performance metrics, alerts, and reports

## API Integration

The frontend integrates with the backend API through RTK Query. All API endpoints are defined in the `services` directory:

- `systemApi.ts`: System management endpoints
- `tradingApi.ts`: Trading operations
- `marketDataApi.ts`: Market data subscriptions
- `monitoringApi.ts`: Performance monitoring
- Additional API services for ML models and backtesting

## WebSocket Integration

Real-time data is handled through WebSocket connections:
- Automatic reconnection on disconnect
- Subscribes to market data for selected symbols
- Updates positions and orders in real-time

## Development

### Adding New Features

1. Create a new Redux slice in `features/`
2. Add API endpoints in `services/`
3. Create page component in `pages/`
4. Add navigation item in `MainLayout.tsx`

### Styling

The application uses Material-UI with a custom dark theme optimized for trading interfaces. The theme is configured in `theme/theme.ts`.

## Next Steps

To complete the implementation:

1. **Trading Interface**: 
   - Implement order placement forms
   - Create position management tables
   - Add real-time order book display

2. **ML Model Management**:
   - Build model training configuration forms
   - Add training progress monitoring
   - Create model comparison charts

3. **Backtesting**:
   - Implement backtest configuration interface
   - Add equity curve visualization
   - Create trade analysis tools

4. **Data Management**:
   - Build file upload interface
   - Add symbol configuration manager
   - Create data quality dashboards

5. **Charts Integration**:
   - Integrate TradingView Lightweight Charts for candlestick charts
   - Add Recharts for performance metrics
   - Create custom trading indicators

## Deployment

For production deployment:

1. Build the application: `npm run build`
2. Serve the `dist` directory with any static file server
3. Configure environment variables on your server
4. Set up reverse proxy if needed

## License

[Your License Here]