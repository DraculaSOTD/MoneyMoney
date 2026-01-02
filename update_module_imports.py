#!/usr/bin/env python3
"""
Update imports in crypto_ml_trading, data_feeds, trading, monitoring, alternative_data
to use python_backend.* prefix for shared modules
"""
import os
import re
from pathlib import Path

def update_imports_in_file(filepath):
    """Update imports in a single Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern replacements for imports that need python_backend prefix
        replacements = [
            # exchanges.* → python_backend.exchanges.*
            (r'\bfrom exchanges\.', 'from python_backend.exchanges.'),
            (r'\bimport exchanges\.', 'import python_backend.exchanges.'),

            # database.* → python_backend.database.*
            (r'\bfrom database\.', 'from python_backend.database.'),
            (r'\bimport database\.', 'import python_backend.database.'),

            # security.* → python_backend.security.*
            (r'\bfrom security\.', 'from python_backend.security.'),
            (r'\bimport security\.', 'import python_backend.security.'),

            # services.* → python_backend.services.*
            (r'\bfrom services\.', 'from python_backend.services.'),
            (r'\bimport services\.', 'import python_backend.services.'),

            # api.* → python_backend.api.* (but be careful not to change api.main)
            (r'\bfrom api\.routers', 'from python_backend.api.routers'),

            # scripts.* → python_backend.scripts.*
            (r'\bfrom scripts\.', 'from python_backend.scripts.'),
            (r'\bimport scripts\.', 'import python_backend.scripts.'),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Update all Python files in the specified directories"""
    base_dir = Path(__file__).parent

    # Directories to update
    directories = [
        'crypto_ml_trading',
        'data_feeds',
        'trading',
        'monitoring',
        'alternative_data'
    ]

    updated_count = 0
    total_count = 0

    for dir_name in directories:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            print(f"Skipping {dir_name} (not found)")
            continue

        print(f"\nProcessing {dir_name}/...")

        # Find all Python files
        for py_file in dir_path.rglob('*.py'):
            # Skip __pycache__ directories
            if '__pycache__' in str(py_file):
                continue

            total_count += 1
            if update_imports_in_file(py_file):
                print(f"  Updated: {py_file.relative_to(base_dir)}")
                updated_count += 1

    print(f"\n{'='*60}")
    print(f"Total files scanned: {total_count}")
    print(f"Total files updated: {updated_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
