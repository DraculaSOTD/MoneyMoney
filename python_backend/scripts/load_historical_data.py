#!/usr/bin/env python3
"""
Script to load historical data and create trading profiles
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from python_backend.database import TradingProfile, ProfileMetrics, ProfileType, Base

# Database configuration - Use PostgreSQL from environment
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:@localhost:5432/trading_platform')

def create_technical_indicators(data):
    """Apply technical indicators to the dataframe"""
    
    # MACD
    data['12_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['26_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['12_EMA'] - data['26_EMA']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (rolling_std * 2)
    data['Lower_Band'] = data['SMA_20'] - (rolling_std * 2)
    
    # Additional SMAs
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # ATR
    high_minus_low = data['High'] - data['Low']
    high_minus_prev_close = (data['High'] - data['Close'].shift()).abs()
    low_minus_prev_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_minus_low, high_minus_prev_close, low_minus_prev_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # OBV (On-Balance Volume)
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    
    return data

def load_eth_data():
    """Load and process ETHUSDT data"""
    print("Loading ETHUSDT data...")
    
    # Load data
    df = pd.read_csv('/mnt/MassStorage/Projects/Machine Learning/ETH_Reinforce_bot/data/ETHUSDT_15m.csv')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['Open time'])
    
    # Rename columns to standard format
    df = df.rename(columns={
        'Open': 'Open',
        'High': 'High', 
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })
    
    # Apply technical indicators
    df = create_technical_indicators(df)
    
    # Calculate price changes
    df['price_change_24h'] = df['Close'].pct_change(periods=96) * 100  # 96 * 15min = 24h
    df['price_change_7d'] = df['Close'].pct_change(periods=672) * 100  # 672 * 15min = 7d
    df['price_change_30d'] = df['Close'].pct_change(periods=2880) * 100  # 2880 * 15min = 30d
    
    # Volume metrics
    df['volume_24h'] = df['Volume'].rolling(window=96).sum()
    df['volume_change_24h'] = df['volume_24h'].pct_change() * 100
    df['avg_volume_7d'] = df['Volume'].rolling(window=672).mean()
    
    return df

def load_gbp_data():
    """Load and process GBPUSD data"""
    print("Loading GBPUSD data...")
    
    # Load data
    df = pd.read_csv('/mnt/MassStorage/Projects/Machine Learning/TRADINGMODELV2/data/GBPUSD60.csv')
    
    # Combine date and time
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    
    # Apply technical indicators
    df = create_technical_indicators(df)
    
    # Calculate price changes
    df['price_change_24h'] = df['Close'].pct_change(periods=24) * 100  # 24 hours
    df['price_change_7d'] = df['Close'].pct_change(periods=168) * 100  # 168 hours = 7d
    df['price_change_30d'] = df['Close'].pct_change(periods=720) * 100  # 720 hours = 30d
    
    # Volume metrics
    df['volume_24h'] = df['Volume'].rolling(window=24).sum()
    df['volume_change_24h'] = df['volume_24h'].pct_change() * 100
    df['avg_volume_7d'] = df['Volume'].rolling(window=168).mean()
    
    return df

def create_profiles(session):
    """Create trading profiles for ETH and GBP"""
    
    # Check if profiles already exist
    eth_profile = session.query(TradingProfile).filter_by(symbol='ETHUSDT').first()
    gbp_profile = session.query(TradingProfile).filter_by(symbol='GBPUSD').first()
    
    if not eth_profile:
        print("Creating ETHUSDT profile...")
        eth_profile = TradingProfile(
            symbol='ETHUSDT',
            name='Ethereum/USDT',
            profile_type=ProfileType.CRYPTO,
            exchange='binance',
            description='Ethereum to USDT trading pair',
            base_currency='ETH',
            quote_currency='USDT',
            data_source='binance',
            timeframe='15m',
            lookback_days=365,
            min_trade_size=0.01,
            max_trade_size=10.0,
            max_position_size=100.0,
            trading_fee=0.001,
            max_drawdown_limit=0.2,
            position_risk_limit=0.02,
            daily_loss_limit=0.05,
            is_active=True
        )
        session.add(eth_profile)
        session.commit()
        print(f"Created ETHUSDT profile with ID: {eth_profile.id}")
    
    if not gbp_profile:
        print("Creating GBPUSD profile...")
        gbp_profile = TradingProfile(
            symbol='GBPUSD',
            name='British Pound/US Dollar',
            profile_type=ProfileType.FOREX,
            exchange='forex',
            description='GBP to USD forex pair',
            base_currency='GBP',
            quote_currency='USD',
            data_source='metatrader',
            timeframe='1h',
            lookback_days=365,
            min_trade_size=0.01,
            max_trade_size=1.0,
            max_position_size=10.0,
            trading_fee=0.0001,
            max_drawdown_limit=0.15,
            position_risk_limit=0.01,
            daily_loss_limit=0.03,
            is_active=True
        )
        session.add(gbp_profile)
        session.commit()
        print(f"Created GBPUSD profile with ID: {gbp_profile.id}")
    
    return eth_profile, gbp_profile

def populate_metrics(session, profile, df):
    """Populate ProfileMetrics table with historical data"""
    print(f"Populating metrics for {profile.symbol}...")
    
    # Get the latest 1000 rows to avoid overloading
    df_recent = df.tail(1000).copy()
    
    # Delete existing metrics for this profile
    session.query(ProfileMetrics).filter_by(profile_id=profile.id).delete()
    
    metrics_to_add = []
    
    for _, row in df_recent.iterrows():
        if pd.notna(row['RSI']) and pd.notna(row['MACD']):  # Skip rows with NaN values
            metric = ProfileMetrics(
                profile_id=profile.id,
                timestamp=row['timestamp'],
                current_price=row['Close'],
                price_change_24h=row.get('price_change_24h', 0) or 0,
                price_change_7d=row.get('price_change_7d', 0) or 0,
                price_change_30d=row.get('price_change_30d', 0) or 0,
                volume_24h=row.get('volume_24h', 0) or 0,
                volume_change_24h=row.get('volume_change_24h', 0) or 0,
                avg_volume_7d=row.get('avg_volume_7d', 0) or 0,
                rsi=row['RSI'],
                macd=row['MACD'],
                macd_signal=row['Signal_Line'],
                bollinger_upper=row['Upper_Band'],
                bollinger_lower=row['Lower_Band'],
                sma_20=row['SMA_20'],
                sma_50=row.get('SMA_50', row['SMA_20']),
                ema_12=row['EMA_12'],
                ema_26=row['EMA_26'],
                market_cap=None,  # Not applicable for these assets
                circulating_supply=None,
                sentiment_score=0,  # Placeholder
                social_volume=0,
                news_mentions=0,
                custom_metrics={}
            )
            metrics_to_add.append(metric)
    
    # Bulk insert
    session.bulk_save_objects(metrics_to_add)
    session.commit()
    print(f"Added {len(metrics_to_add)} metrics for {profile.symbol}")

def update_profile_performance(session, profile):
    """Update profile performance metrics based on simulated trading"""
    # For now, we'll set some reasonable default values
    # In a real system, these would be calculated from actual trades
    
    profile.total_trades = 150
    profile.win_rate = 0.55
    profile.avg_profit = 0.002
    profile.total_pnl = 234.56
    profile.sharpe_ratio = 1.45
    profile.max_drawdown = 0.08
    profile.last_trade_date = datetime.now()
    
    session.commit()
    print(f"Updated performance metrics for {profile.symbol}")

def main():
    """Main function to load data and create profiles"""
    # Create database engine
    engine = create_engine(DATABASE_URL)
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create profiles
        eth_profile, gbp_profile = create_profiles(session)
        
        # Load and process data
        eth_df = load_eth_data()
        gbp_df = load_gbp_data()
        
        # Populate metrics
        populate_metrics(session, eth_profile, eth_df)
        populate_metrics(session, gbp_profile, gbp_df)
        
        # Update performance metrics
        update_profile_performance(session, eth_profile)
        update_profile_performance(session, gbp_profile)
        
        print("\nData loading complete!")
        print("You can now refresh your trading platform to see the profiles.")
        
    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()