const axios = require('axios');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// Configuration
const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8001';
const SYNC_INTERVAL = 30000; // 30 seconds
const DB_PATH = path.join(__dirname, '..', 'database', 'tradingdashboard.db');

class DatabaseBridge {
    constructor() {
        this.db = new sqlite3.Database(DB_PATH);
        this.isRunning = false;
        this.syncCount = 0;
        this.lastSync = null;
    }

    /**
     * Start the database synchronization service
     */
    start() {
        if (this.isRunning) {
            console.log('⚠️  Database bridge already running');
            return;
        }

        this.isRunning = true;
        console.log('🔄 Database bridge service started');
        console.log(`   Syncing every ${SYNC_INTERVAL / 1000} seconds`);

        // Initial sync
        this.syncInstruments();

        // Schedule periodic syncs
        this.syncInterval = setInterval(() => {
            this.syncInstruments();
        }, SYNC_INTERVAL);
    }

    /**
     * Stop the database synchronization service
     */
    stop() {
        if (!this.isRunning) {
            return;
        }

        clearInterval(this.syncInterval);
        this.isRunning = false;
        console.log('⏹️  Database bridge service stopped');
    }

    /**
     * Sync trained instruments from Trading API to MoneyMoney SQLite
     */
    async syncInstruments() {
        try {
            console.log(`\\n[${new Date().toISOString()}] Starting instrument sync...`);

            // Fetch profiles from Trading API
            const response = await axios.get(`${TRADING_API_URL}/api/profiles`, {
                timeout: 5000
            });

            const profiles = response.data;

            // Filter only trained profiles with data
            const trainedProfiles = profiles.filter(p =>
                p.has_data &&
                p.models_trained &&
                p.total_data_points > 0
            );

            console.log(`   Found ${trainedProfiles.length} trained profiles`);

            let syncedCount = 0;
            let errorCount = 0;

            // Sync each profile to SQLite
            for (const profile of trainedProfiles) {
                try {
                    await this.upsertInstrument(profile);
                    syncedCount++;
                } catch (err) {
                    console.error(`   ❌ Error syncing ${profile.symbol}:`, err.message);
                    errorCount++;
                }
            }

            this.syncCount++;
            this.lastSync = new Date();

            console.log(`   ✅ Sync complete: ${syncedCount} synced, ${errorCount} errors`);
            console.log(`   Total syncs: ${this.syncCount}`);

        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                console.error('   ❌ Trading API not available (port 8002)');
            } else {
                console.error('   ❌ Sync error:', error.message);
            }
        }
    }

    /**
     * Insert or update an instrument in SQLite
     */
    upsertInstrument(profile) {
        return new Promise((resolve, reject) => {
            const sql = `
                INSERT INTO instruments (
                    symbol, name, category, price, change_percent,
                    has_data, models_trained, data_interval, total_data_points,
                    data_updated_at, last_training, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    price = excluded.price,
                    change_percent = excluded.change_percent,
                    has_data = excluded.has_data,
                    models_trained = excluded.models_trained,
                    data_interval = excluded.data_interval,
                    total_data_points = excluded.total_data_points,
                    data_updated_at = excluded.data_updated_at,
                    last_training = excluded.last_training,
                    last_updated = CURRENT_TIMESTAMP
            `;

            // Create unique constraint if it doesn't exist
            this.db.run(`CREATE UNIQUE INDEX IF NOT EXISTS idx_instruments_symbol ON instruments(symbol)`, (err) => {
                if (err && !err.message.includes('already exists')) {
                    console.error('Index creation error:', err.message);
                }
            });

            const params = [
                profile.symbol,
                profile.name,
                profile.profile_type || 'crypto',
                0, // price (will be updated by real-time data)
                0, // change_percent
                profile.has_data ? 1 : 0,
                profile.models_trained ? 1 : 0,
                profile.data_interval || '1m',
                profile.total_data_points || 0,
                profile.data_updated_at,
                profile.last_training
            ];

            this.db.run(sql, params, function (err) {
                if (err) {
                    reject(err);
                } else {
                    resolve(this.changes);
                }
            });
        });
    }

    /**
     * Get synchronization status
     */
    getStatus() {
        return {
            isRunning: this.isRunning,
            syncCount: this.syncCount,
            lastSync: this.lastSync,
            nextSync: this.isRunning && this.lastSync ?
                new Date(this.lastSync.getTime() + SYNC_INTERVAL) : null
        };
    }
}

// Create singleton instance
const dbBridge = new DatabaseBridge();

// Handle process termination
process.on('SIGINT', () => {
    console.log('\\nShutting down database bridge...');
    dbBridge.stop();
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\\nShutting down database bridge...');
    dbBridge.stop();
    process.exit(0);
});

module.exports = dbBridge;
