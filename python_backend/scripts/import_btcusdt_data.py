"""
Import BTCUSDT Data from CSV to Database
=========================================

One-time script to import the real BTCUSDT_1m.csv data into PostgreSQL database.
This makes the data available for the Trading Platform API and MoneyMoney dashboard.

Usage:
    python scripts/import_btcusdt_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
from python_backend.database.models import SessionLocal, TradingProfile, MarketData, ProfileType
from sqlalchemy import and_

def import_btcusdt_data():
    """Import BTCUSDT CSV data into PostgreSQL database."""

    print("=" * 80)
    print("BTCUSDT DATA IMPORT SCRIPT")
    print("=" * 80)

    # Initialize database session
    db = SessionLocal()

    try:
        # Step 1: Create or get TradingProfile for BTCUSDT
        print("\n[Step 1] Creating/finding TradingProfile for BTCUSDT...")

        profile = db.query(TradingProfile).filter(
            TradingProfile.symbol == 'BTCUSDT'
        ).first()

        if profile:
            print(f"  ✓ Found existing profile (ID: {profile.id})")

            # Check if data already exists
            existing_count = db.query(MarketData).filter(
                MarketData.symbol == 'BTCUSDT',
                MarketData.profile_id == profile.id
            ).count()

            if existing_count > 0:
                print(f"  ⚠ Warning: {existing_count} candles already exist for BTCUSDT")
                print(f"  Automatically deleting {existing_count} existing candles for fresh import...")
                db.query(MarketData).filter(
                    MarketData.symbol == 'BTCUSDT',
                    MarketData.profile_id == profile.id
                ).delete()
                db.commit()
                print("  ✓ Existing data deleted")
        else:
            print("  Creating new TradingProfile for BTCUSDT...")
            profile = TradingProfile(
                symbol='BTCUSDT',
                name='Bitcoin',
                profile_type=ProfileType.CRYPTO,
                exchange='binance',
                description='Bitcoin to USDT trading pair',
                base_currency='BTC',
                quote_currency='USDT',
                data_source='binance',
                timeframe='1h',
                lookback_days=365,
                min_trade_size=0.001,
                max_trade_size=1.0,
                max_position_size=10.0,
                trading_fee=0.001,
                max_drawdown_limit=0.2,
                position_risk_limit=0.02,
                daily_loss_limit=0.05,
                data_interval='1m',
                has_data=False,
                models_trained=False,
                total_data_points=0
            )
            db.add(profile)
            db.commit()
            print(f"  ✓ Created new profile (ID: {profile.id})")

        # Step 2: Load CSV data
        print("\n[Step 2] Loading BTCUSDT_1m.csv...")

        csv_path = Path(__file__).parent.parent / 'crypto_ml_trading' / 'data' / 'historical' / 'BTCUSDT_1m.csv'

        if not csv_path.exists():
            print(f"  ✗ ERROR: CSV file not found at {csv_path}")
            db.close()
            return

        print(f"  Reading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"  ✓ Loaded {len(df)} candles from CSV")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {', '.join(df.columns.tolist())}")

        # Display sample data
        print("\n  Sample data (first 3 rows):")
        print(df.head(3).to_string())

        # Step 3: Import data in batches
        print(f"\n[Step 3] Importing {len(df)} candles into database...")

        batch_size = 1000
        total_rows = len(df)
        inserted = 0

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]

            market_data_batch = []
            for _, row in batch.iterrows():
                market_data = MarketData(
                    symbol='BTCUSDT',
                    profile_id=profile.id,
                    timestamp=row['timestamp'],
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close']),
                    volume=float(row['volume'])
                )
                market_data_batch.append(market_data)

            # Bulk insert
            db.bulk_save_objects(market_data_batch)
            db.commit()

            inserted += len(market_data_batch)
            progress = (inserted / total_rows) * 100
            print(f"  Progress: {inserted}/{total_rows} ({progress:.1f}%)")

        print(f"  ✓ Successfully inserted {inserted} candles")

        # Step 4: Update TradingProfile flags
        print("\n[Step 4] Updating TradingProfile metadata...")

        profile.has_data = True
        profile.data_updated_at = datetime.utcnow()
        profile.total_data_points = len(df)

        # Calculate date range
        first_timestamp = df['timestamp'].min()
        last_timestamp = df['timestamp'].max()

        db.commit()

        print(f"  ✓ Updated profile:")
        print(f"    - has_data: True")
        print(f"    - total_data_points: {len(df)}")
        print(f"    - data_updated_at: {profile.data_updated_at}")
        print(f"    - date_range: {first_timestamp} to {last_timestamp}")

        # Step 5: Verify import
        print("\n[Step 5] Verifying import...")

        verify_count = db.query(MarketData).filter(
            MarketData.symbol == 'BTCUSDT',
            MarketData.profile_id == profile.id
        ).count()

        if verify_count == len(df):
            print(f"  ✓ Verification successful: {verify_count} candles in database")
        else:
            print(f"  ⚠ Warning: Expected {len(df)} candles, found {verify_count}")

        # Display latest candle
        latest = db.query(MarketData).filter(
            MarketData.symbol == 'BTCUSDT',
            MarketData.profile_id == profile.id
        ).order_by(MarketData.timestamp.desc()).first()

        if latest:
            print(f"\n  Latest candle in database:")
            print(f"    Timestamp: {latest.timestamp}")
            print(f"    Open: {latest.open_price:.2f}")
            print(f"    High: {latest.high_price:.2f}")
            print(f"    Low: {latest.low_price:.2f}")
            print(f"    Close: {latest.close_price:.2f}")
            print(f"    Volume: {latest.volume:.2f}")

        print("\n" + "=" * 80)
        print("IMPORT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Start the Trading Platform API: cd other_platform/Trading && ./start.sh")
        print("  2. Start MoneyMoney: cd MoneyMoney && ./start.sh")
        print("  3. Access dashboard: http://localhost:3000")
        print("  4. Select BTCUSDT to view the imported data")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR during import: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()

    finally:
        db.close()


if __name__ == '__main__':
    import_btcusdt_data()
