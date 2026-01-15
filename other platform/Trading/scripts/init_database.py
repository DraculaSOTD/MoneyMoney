#!/usr/bin/env python3
"""
Database Initialization Script
==============================
Creates all tables and seeds initial data for fresh installations.
Idempotent - safe to run multiple times.

Usage:
    python scripts/init_database.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import bcrypt
from database.models import (
    Base, engine, SessionLocal,
    TradingProfile, ProfileType, AdminUser
)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def create_all_tables():
    """Create all database tables"""
    print("[1/4] Creating database tables...")
    # Tables are created on import in models.py, but let's be explicit
    Base.metadata.create_all(bind=engine)
    print("      OK - All tables created")


def create_default_admin():
    """Create default admin user if none exists"""
    print("[2/4] Creating default admin user...")
    db = SessionLocal()
    try:
        admin_count = db.query(AdminUser).count()
        if admin_count == 0:
            admin = AdminUser(
                username="admin",
                email="admin@trading.com",
                password_hash=hash_password("admin123"),
                full_name="Default Admin",
                is_superuser=True,
                is_active=True
            )
            db.add(admin)
            db.commit()
            print("      OK - Default admin created")
            print("           Email: admin@trading.com")
            print("           Password: admin123")
            print("           (CHANGE IN PRODUCTION!)")
        else:
            print(f"      OK - Admin user already exists ({admin_count} found)")
    except Exception as e:
        print(f"      ERROR - Failed to create admin: {e}")
        db.rollback()
    finally:
        db.close()


def create_btcusdt_profile():
    """Create BTCUSDT trading profile if it doesn't exist"""
    print("[3/4] Creating BTCUSDT trading profile...")
    db = SessionLocal()
    try:
        existing = db.query(TradingProfile).filter_by(symbol='BTCUSDT').first()
        if not existing:
            profile = TradingProfile(
                symbol='BTCUSDT',
                name='Bitcoin / USDT',
                profile_type=ProfileType.CRYPTO,
                exchange='binance',
                description='Bitcoin to USDT trading pair on Binance',
                base_currency='BTC',
                quote_currency='USDT',
                data_interval='1m',
                has_data=False,
                models_trained=False,
                total_data_points=0,
                is_active=True
            )
            db.add(profile)
            db.commit()
            print(f"      OK - BTCUSDT profile created (ID: {profile.id})")
        else:
            print(f"      OK - BTCUSDT profile already exists (ID: {existing.id})")
    except Exception as e:
        print(f"      ERROR - Failed to create profile: {e}")
        db.rollback()
    finally:
        db.close()


def verify_database():
    """Verify database setup is complete"""
    print("[4/4] Verifying database setup...")
    db = SessionLocal()
    try:
        # Count key tables
        from database.models import MarketData
        profile_count = db.query(TradingProfile).count()
        admin_count = db.query(AdminUser).count()

        print(f"      Trading profiles: {profile_count}")
        print(f"      Admin users: {admin_count}")
        print("      OK - Database verification complete")
    except Exception as e:
        print(f"      WARNING - Verification issue: {e}")
    finally:
        db.close()


def main():
    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    print()

    create_all_tables()
    create_default_admin()
    create_btcusdt_profile()
    verify_database()

    print()
    print("=" * 60)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 60)
    print()
    print("Next step: Run import_btcusdt_data.py to load historical data")
    print()


if __name__ == '__main__':
    main()
