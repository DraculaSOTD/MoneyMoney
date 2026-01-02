#!/usr/bin/env python3
"""
Script to update imports from old structure to new python_backend structure
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

        # Pattern replacements for imports
        replacements = [
            # api.routers.* → python_backend.api.routers.*
            (r'\bfrom api\.routers\b', 'from python_backend.api.routers'),
            (r'\bimport api\.routers\b', 'import python_backend.api.routers'),

            # database.* → python_backend.database.*
            (r'\bfrom database\b', 'from python_backend.database'),
            (r'\bimport database\b', 'import python_backend.database'),

            # models.* → python_backend.models.*
            (r'\bfrom models\b', 'from python_backend.models'),
            (r'\bimport models\b', 'import python_backend.models'),

            # exchanges.* → python_backend.exchanges.*
            (r'\bfrom exchanges\b', 'from python_backend.exchanges'),
            (r'\bimport exchanges\b', 'import python_backend.exchanges'),

            # services.* → python_backend.services.*
            (r'\bfrom services\b', 'from python_backend.services'),
            (r'\bimport services\b', 'import python_backend.services'),

            # security.* → python_backend.security.*
            (r'\bfrom security\b', 'from python_backend.security'),
            (r'\bimport security\b', 'import python_backend.security'),

            # scripts.* → python_backend.scripts.*
            (r'\bfrom scripts\b', 'from python_backend.scripts'),
            (r'\bimport scripts\b', 'import python_backend.scripts'),
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
    """Update all Python files in the python_backend directory"""
    backend_dir = Path(__file__).parent
    updated_count = 0

    # Find all Python files
    for py_file in backend_dir.rglob('*.py'):
        # Skip this script itself and __pycache__ directories
        if py_file.name == 'update_imports.py' or '__pycache__' in str(py_file):
            continue

        if update_imports_in_file(py_file):
            print(f"Updated: {py_file.relative_to(backend_dir)}")
            updated_count += 1

    print(f"\nTotal files updated: {updated_count}")

if __name__ == "__main__":
    main()
