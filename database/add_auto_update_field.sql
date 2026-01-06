-- Add auto_update_enabled field to trading_profiles table
-- This enables/disables automatic minute-by-minute data updates

ALTER TABLE trading_profiles
ADD COLUMN IF NOT EXISTS auto_update_enabled BOOLEAN DEFAULT TRUE;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_profiles_auto_update
ON trading_profiles(auto_update_enabled)
WHERE auto_update_enabled = TRUE AND is_active = TRUE;

-- Set all existing profiles to auto-update by default
UPDATE trading_profiles
SET auto_update_enabled = TRUE
WHERE auto_update_enabled IS NULL;

COMMENT ON COLUMN trading_profiles.auto_update_enabled IS
'Enables automatic minute-by-minute updates for this profile';
