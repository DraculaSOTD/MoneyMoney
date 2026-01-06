-- Migration: Add UNIQUE constraint to market_data table
-- Purpose: Prevent duplicate (symbol, timestamp) entries
-- Date: 2025-11-04

-- Step 1: Identify and remove any existing duplicates
-- Keep the most recent record (highest id) for each (symbol, timestamp) pair
BEGIN;

-- Create a temporary table with the IDs to keep
CREATE TEMP TABLE market_data_to_keep AS
SELECT MAX(id) as id
FROM market_data
GROUP BY symbol, timestamp;

-- Delete duplicate records (keep only the highest ID for each symbol/timestamp)
DELETE FROM market_data
WHERE id NOT IN (SELECT id FROM market_data_to_keep);

-- Report how many duplicates were removed
DO $$
DECLARE
    deleted_count INTEGER;
BEGIN
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE 'Removed % duplicate records', deleted_count;
END $$;

-- Step 2: Add the UNIQUE constraint
ALTER TABLE market_data
ADD CONSTRAINT uq_market_data_symbol_timestamp
UNIQUE (symbol, timestamp);

COMMIT;

-- Verify the constraint was added
SELECT conname, contype, convalidated
FROM pg_constraint
WHERE conname = 'uq_market_data_symbol_timestamp';

-- Show summary
SELECT
    COUNT(*) as total_records,
    COUNT(DISTINCT symbol) as unique_symbols,
    MIN(timestamp) as earliest_data,
    MAX(timestamp) as latest_data
FROM market_data;
