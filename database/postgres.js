/**
 * PostgreSQL Database Connection Module
 * Shared database connection with Python backend
 */

const { Pool } = require('pg');

// Create connection pool
const pool = new Pool({
    connectionString: process.env.DATABASE_URL || 'postgresql://postgres:@localhost:5432/trading_platform',
    max: 20, // Maximum number of clients in the pool
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
});

// Test connection on startup
pool.on('connect', () => {
    console.log('✅ Connected to PostgreSQL database');
});

pool.on('error', (err) => {
    console.error('❌ Unexpected PostgreSQL error:', err);
    process.exit(-1);
});

/**
 * Execute a query
 * @param {string} text - SQL query
 * @param {Array} params - Query parameters
 * @returns {Promise<Object>} Query result
 */
async function query(text, params) {
    const start = Date.now();
    try {
        const res = await pool.query(text, params);
        const duration = Date.now() - start;
        console.log('executed query', { text, duration, rows: res.rowCount });
        return res;
    } catch (error) {
        console.error('Query error:', error);
        throw error;
    }
}

/**
 * Get a client from the pool for transactions
 * @returns {Promise<PoolClient>}
 */
async function getClient() {
    return await pool.connect();
}

/**
 * Close the pool
 */
async function close() {
    await pool.end();
}

module.exports = {
    query,
    getClient,
    close,
    pool
};
