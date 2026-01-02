"""
Yahoo Finance Connector
Handles fetching stock market data from Yahoo Finance
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Callable
from decimal import Decimal
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not installed. Install with: pip install yfinance")
    YFINANCE_AVAILABLE = False


@dataclass
class YahooTicker:
    """Stock ticker information"""
    symbol: str
    last_price: Decimal
    volume: Decimal
    high_24h: Decimal
    low_24h: Decimal
    open_price: Decimal
    timestamp: datetime


class YahooFinanceConnector:
    """
    Yahoo Finance connector for fetching stock market data.
    Similar interface to BinanceConnector for consistency.
    """

    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance library is required. Install with: pip install yfinance")

        logger.info("Yahoo Finance connector initialized")

    async def connect(self):
        """Placeholder for consistency with BinanceConnector"""
        pass

    async def disconnect(self):
        """Placeholder for consistency with BinanceConnector"""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    def get_ticker(self, symbol: str) -> YahooTicker:
        """
        Get current ticker information for a stock symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')

        Returns:
            YahooTicker object with current price and volume info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return YahooTicker(
                symbol=symbol,
                last_price=Decimal(str(info.get('currentPrice', 0))),
                volume=Decimal(str(info.get('volume', 0))),
                high_24h=Decimal(str(info.get('dayHigh', 0))),
                low_24h=Decimal(str(info.get('dayLow', 0))),
                open_price=Decimal(str(info.get('open', 0))),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    def get_historical_data(self, symbol: str, interval: str = '1d',
                          days_back: int = 30) -> pd.DataFrame:
        """
        Fetch historical data for a stock symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            days_back: Number of days to fetch

        Returns:
            DataFrame with OHLCV data

        Note:
            - 1m data is only available for last 7 days
            - 5m, 15m, 30m data available for ~60 days
            - 1h and above have longer history
        """
        try:
            ticker = yf.Ticker(symbol)

            # Calculate period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Fetch data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False  # Keep original OHLC values
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Rename columns to match Binance format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Rename index to 'timestamp'
            df.index.name = 'timestamp'

            logger.info(f"Fetched {len(df)} records for {symbol}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise

    def get_all_historical_data(self, symbol: str, interval: str = '1d',
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
        """
        Fetch ALL available historical data for a stock symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            start_date: Optional start date (defaults to max available)
            end_date: Optional end date (defaults to now)
            progress_callback: Optional callback function(current_count, total_estimated, status_message)

        Returns:
            DataFrame with all historical OHLCV data

        Note:
            Yahoo Finance data availability:
            - 1m interval: Last 7 days only
            - 5m, 15m, 30m: Last ~60 days
            - 1h: Last ~730 days
            - 1d, 1wk, 1mo: Multiple years (varies by stock)
        """
        try:
            ticker = yf.Ticker(symbol)

            # Set default dates
            if not end_date:
                end_date = datetime.now()

            if not start_date:
                # Default start dates based on interval
                interval_defaults = {
                    '1m': timedelta(days=7),      # Max 7 days for 1m
                    '5m': timedelta(days=60),     # ~60 days for 5m
                    '15m': timedelta(days=60),    # ~60 days for 15m
                    '30m': timedelta(days=60),    # ~60 days for 30m
                    '1h': timedelta(days=730),    # ~2 years for 1h
                    '1d': timedelta(days=365*20), # 20 years for daily
                    '1wk': timedelta(days=365*20),
                    '1mo': timedelta(days=365*20)
                }
                start_date = end_date - interval_defaults.get(interval, timedelta(days=365))
                logger.info(f"Using default start date: {start_date} for interval {interval}")

            logger.info(f"Fetching {symbol} data from {start_date} to {end_date} with interval {interval}")

            # Fetch all data in one request (Yahoo Finance handles this internally)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Rename columns to match Binance format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Rename index to 'timestamp'
            df.index.name = 'timestamp'

            # Progress callback
            if progress_callback:
                progress_callback(
                    len(df),
                    len(df),
                    f"Fetched {len(df):,} candles"
                )

            logger.info(f"Successfully fetched {len(df):,} total candles for {symbol}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching all historical data for {symbol}: {e}")
            raise

    def get_available_symbols(self, exchange: str = 'NASDAQ') -> list:
        """
        Get list of available stock symbols.
        Note: This is a simplified version. For production, use a proper symbol database.

        Args:
            exchange: Exchange name (NASDAQ, NYSE, etc.)

        Returns:
            List of stock symbols
        """
        # Common stock symbols (for demo purposes)
        common_stocks = {
            'NASDAQ': [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
                'AMD', 'INTC', 'CSCO', 'ADBE', 'PYPL', 'QCOM', 'TXN', 'AVGO'
            ],
            'NYSE': [
                'JPM', 'BAC', 'WMT', 'V', 'MA', 'JNJ', 'PG', 'KO',
                'DIS', 'NKE', 'BA', 'GE', 'IBM', 'GS', 'MS', 'C'
            ]
        }

        return common_stocks.get(exchange, [])

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and has data available.

        Args:
            symbol: Stock symbol to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got valid data
            return 'symbol' in info or 'shortName' in info

        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False

    def get_info(self, symbol: str) -> dict:
        """
        Get detailed information about a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            raise


# Convenience function for quick data fetch
def fetch_stock_data(symbol: str, interval: str = '1d', days_back: int = 30) -> pd.DataFrame:
    """
    Quick function to fetch stock data without creating connector instance.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
        days_back: Number of days to fetch

    Returns:
        DataFrame with OHLCV data
    """
    connector = YahooFinanceConnector()
    return connector.get_historical_data(symbol, interval, days_back)
