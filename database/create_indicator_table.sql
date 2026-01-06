-- Migration: Create indicator_data table and update trading_profiles
-- Purpose: Store calculated technical indicators for chart display
-- Date: 2025-11-06

BEGIN;

-- Create indicator_data table
CREATE TABLE IF NOT EXISTS indicator_data (
    id SERIAL PRIMARY KEY,
    market_data_id INTEGER NOT NULL REFERENCES market_data(id) ON DELETE CASCADE,
    profile_id INTEGER NOT NULL REFERENCES trading_profiles(id) ON DELETE CASCADE,

    -- Moving Averages
    sma_10 DOUBLE PRECISION,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,

    -- MACD
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,

    -- RSI
    rsi_14 DOUBLE PRECISION,

    -- Bollinger Bands
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,

    -- ATR
    atr_14 DOUBLE PRECISION,

    -- Stochastic
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,

    -- ADX
    adx DOUBLE PRECISION,
    plus_di DOUBLE PRECISION,
    minus_di DOUBLE PRECISION,

    -- Volume
    obv DOUBLE PRECISION,

    -- Metadata
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config_version VARCHAR(50),

    CONSTRAINT uq_indicator_market_data UNIQUE (market_data_id)
);

-- Create indexes
CREATE INDEX ix_indicator_data_market_data ON indicator_data(market_data_id);
CREATE INDEX ix_indicator_data_profile ON indicator_data(profile_id);

-- Add indicator tracking fields to trading_profiles
ALTER TABLE trading_profiles
    ADD COLUMN IF NOT EXISTS has_indicators BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS indicators_updated_at TIMESTAMP;

-- Create index on has_indicators for faster filtering
CREATE INDEX IF NOT EXISTS ix_trading_profiles_has_indicators ON trading_profiles(has_indicators);

COMMIT;

-- Verify tables created
SELECT
    'indicator_data' as table_name,
    COUNT(*) as row_count
FROM indicator_data
UNION ALL
SELECT
    'trading_profiles_indicator_fields' as table_name,
    COUNT(*) as profiles_with_indicators
FROM trading_profiles
WHERE has_indicators = TRUE;
