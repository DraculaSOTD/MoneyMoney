"""
Import existing CSV files into PostgreSQL database
Usage: python -m python_backend.scripts.import_csv_to_db
"""

import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python_backend.database.models import SessionLocal, MarketData, TradingProfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_csv_to_database(csv_path: str, symbol: str):
    """Import CSV file to database"""
    db = SessionLocal()
    try:
        # Read CSV
        logger.info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        logger.info(f"Loaded {len(df)} records from {csv_path}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        # Get or create profile
        profile = db.query(TradingProfile).filter(TradingProfile.symbol == symbol).first()
        if not profile:
            logger.info(f"Creating new profile for {symbol}")
            profile = TradingProfile(
                symbol=symbol,
                name=symbol,
                profile_type="crypto",
                exchange="binance",
                base_currency=symbol.replace("USDT", "").replace("BTC", ""),
                quote_currency="USDT" if "USDT" in symbol else "BTC",
                data_interval='1m',
                has_data=True,
                data_source='binance',
                timeframe='1m',
                lookback_days=365
            )
            db.add(profile)
            db.flush()
            logger.info(f"Created profile ID: {profile.id}")
        else:
            logger.info(f"Using existing profile ID: {profile.id}")

        # Check if data already exists
        existing_count = db.query(MarketData).filter(
            MarketData.symbol == symbol
        ).count()

        if existing_count > 0:
            logger.warning(f"⚠️  Data already exists for {symbol}: {existing_count} records")
            choice = input(f"Delete existing {existing_count} records and re-import? (y/n): ")
            if choice.lower() != 'y':
                logger.info("Skipping import")
                return

            logger.info(f"Deleting {existing_count} existing records...")
            db.query(MarketData).filter(MarketData.symbol == symbol).delete()
            db.commit()
            logger.info(f"✅ Deleted {existing_count} existing records")

        # Prepare records for bulk insert
        logger.info("Preparing records for database...")
        records = []
        for index, row in df.iterrows():
            record = MarketData(
                symbol=symbol,
                profile_id=profile.id,
                timestamp=index,
                open_price=float(row['open']),
                high_price=float(row['high']),
                low_price=float(row['low']),
                close_price=float(row['close']),
                volume=float(row['volume']),
                number_of_trades=None,  # Not in CSV
                quote_asset_volume=None  # Not in CSV
            )
            records.append(record)

        # Insert in batches for performance
        batch_size = 1000
        total_batches = (len(records) - 1) // batch_size + 1

        logger.info(f"Inserting {len(records)} records in {total_batches} batches...")
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            db.bulk_save_objects(batch)
            db.commit()
            batch_num = i // batch_size + 1
            logger.info(f"  ✅ Inserted batch {batch_num}/{total_batches} ({len(batch)} records)")

        # Update profile stats
        profile.total_data_points = len(records)
        profile.data_updated_at = datetime.utcnow()
        profile.has_data = True
        db.commit()

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ SUCCESS: Imported {len(records)} records for {symbol}")
        logger.info(f"   Profile ID: {profile.id}")
        logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"❌ Error importing CSV: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        db.close()

def main():
    """Main import function"""
    logger.info("\n" + "="*60)
    logger.info("CSV to Database Import Script")
    logger.info("="*60 + "\n")

    # Import BTCUSDT data
    csv_path = "python_backend/crypto_ml_trading/data/historical/BTCUSDT_1m.csv"

    if os.path.exists(csv_path):
        logger.info(f"📄 Found CSV file: {csv_path}")
        file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        logger.info(f"   File size: {file_size:.2f} MB\n")

        import_csv_to_database(csv_path, "BTCUSDT")
    else:
        logger.error(f"❌ CSV file not found: {csv_path}")
        logger.info("\nPlease ensure the CSV file exists at the specified path.")
        return

    logger.info("🎉 Import process complete!\n")

if __name__ == "__main__":
    main()
