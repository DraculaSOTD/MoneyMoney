"""
Recreate Database Tables
Drops all existing tables and creates new ones based on current models
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python_backend.database.models import Base, engine
from sqlalchemy import inspect

def recreate_database():
    """Drop all tables and recreate them"""

    print("=" * 60)
    print("DATABASE RECREATION SCRIPT")
    print("=" * 60)

    # Check existing tables
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    print(f"\nFound {len(existing_tables)} existing tables:")
    for table in existing_tables:
        print(f"  - {table}")

    # Confirm action
    print("\n⚠️  WARNING: This will DELETE ALL DATA in these tables!")
    response = input("\nType 'YES' to continue, or anything else to cancel: ")

    if response != 'YES':
        print("\n❌ Operation cancelled")
        return

    # Drop all tables
    print("\n🗑️  Dropping all tables...")
    Base.metadata.drop_all(engine)
    print("✅ All tables dropped")

    # Create all tables with new schema
    print("\n🏗️  Creating tables with new schema...")
    Base.metadata.create_all(engine)
    print("✅ All tables created")

    # Verify new tables
    inspector = inspect(engine)
    new_tables = inspector.get_table_names()

    print(f"\n✅ Successfully created {len(new_tables)} tables:")
    for table in new_tables:
        print(f"  - {table}")

    print("\n" + "=" * 60)
    print("✅ DATABASE RECREATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Restart the Python backend (should auto-restart)")
    print("2. Re-import BTCUSDT data if needed:")
    print("   python3 -m python_backend.scripts.import_csv_to_db")
    print("3. Test profile creation in the admin UI")
    print()

if __name__ == "__main__":
    recreate_database()
