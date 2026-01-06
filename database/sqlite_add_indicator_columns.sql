-- SQLite Migration: Add indicator tracking columns to trading_profiles
-- Purpose: Add has_indicators tracking and create indicator_data table
-- Date: 2025-11-07

BEGIN TRANSACTION;

-- Add new columns to trading_profiles
ALTER TABLE trading_profiles ADD COLUMN has_indicators BOOLEAN DEFAULT 0;
ALTER TABLE trading_profiles ADD COLUMN indicators_updated_at TIMESTAMP;

-- Create indicator_data table for SQLite
CREATE TABLE IF NOT EXISTS indicator_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_data_id INTEGER NOT NULL REFERENCES market_data(id) ON DELETE CASCADE,
    profile_id INTEGER NOT NULL REFERENCES trading_profiles(id) ON DELETE CASCADE,

    -- Moving Averages
    sma_10 REAL,
    sma_20 REAL,
    sma_50 REAL,
    sma_200 REAL,
    ema_12 REAL,
    ema_26 REAL,

    -- MACD
    macd REAL,
    macd_signal REAL,
    macd_histogram REAL,

    -- RSI
    rsi_14 REAL,

    -- Bollinger Bands
    bb_upper REAL,
    bb_middle REAL,
    bb_lower REAL,
    bb_width REAL,

    -- ATR
    atr_14 REAL,

    -- Stochastic
    stoch_k REAL,
    stoch_d REAL,

    -- ADX
    adx REAL,
    plus_di REAL,
    minus_di REAL,

    -- Volume
    obv REAL,

    -- Metadata
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config_version VARCHAR(50),

    UNIQUE(market_data_id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS ix_indicator_data_market_data ON indicator_data(market_data_id);
CREATE INDEX IF NOT EXISTS ix_indicator_data_profile ON indicator_data(profile_id);
CREATE INDEX IF NOT EXISTS ix_trading_profiles_has_indicators ON trading_profiles(has_indicators);

COMMIT;

-- Verify
SELECT 'indicator_data table' as verification, COUNT(*) as row_count FROM indicator_data;
SELECT 'profiles with indicators' as verification, COUNT(*) as count FROM trading_profiles WHERE has_indicators = 1;
