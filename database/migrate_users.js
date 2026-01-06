/**
 * Migrate users from SQLite to PostgreSQL
 * Run this script once to copy existing users
 */

const sqlite3 = require('sqlite3').verbose();
const { query, close } = require('./postgres');
const path = require('path');

// SQLite database path
const sqlitePath = path.join(__dirname, '../database/tradingdashboard.db');

async function migrateUsers() {
    console.log('Starting user migration from SQLite to PostgreSQL...\n');

    return new Promise((resolve, reject) => {
        // Connect to SQLite
        const db = new sqlite3.Database(sqlitePath, (err) => {
            if (err) {
                console.error('❌ Error connecting to SQLite:', err);
                reject(err);
                return;
            }
            console.log('✅ Connected to SQLite database');
        });

        // Get all users from SQLite
        db.all('SELECT * FROM users', [], async (err, rows) => {
            if (err) {
                console.error('❌ Error reading from SQLite:', err);
                db.close();
                reject(err);
                return;
            }

            console.log(`Found ${rows.length} users in SQLite\n`);

            try {
                let migrated = 0;
                let skipped = 0;

                for (const user of rows) {
                    try {
                        // Check if user already exists in PostgreSQL
                        const existingUser = await query(
                            'SELECT id FROM users WHERE email = $1',
                            [user.email]
                        );

                        if (existingUser.rows.length > 0) {
                            console.log(`⏭️  Skipping ${user.email} (already exists)`);
                            skipped++;
                            continue;
                        }

                        // Insert user into PostgreSQL
                        await query(
                            `INSERT INTO users (email, password_hash, role, full_name, subscription_status, created_at, updated_at)
                             VALUES ($1, $2, $3, $4, $5, $6, $7)`,
                            [
                                user.email,
                                user.password_hash,
                                user.role || 'user',
                                user.full_name,
                                user.subscription_status || 'active',
                                user.created_at,
                                user.updated_at
                            ]
                        );

                        console.log(`✅ Migrated user: ${user.email} (id: ${user.id})`);
                        migrated++;
                    } catch (error) {
                        console.error(`❌ Error migrating ${user.email}:`, error.message);
                    }
                }

                console.log(`\n📊 Migration Summary:`);
                console.log(`   Migrated: ${migrated}`);
                console.log(`   Skipped:  ${skipped}`);
                console.log(`   Total:    ${rows.length}`);

                db.close();
                await close();
                resolve({ migrated, skipped, total: rows.length });
            } catch (error) {
                console.error('❌ Migration error:', error);
                db.close();
                await close();
                reject(error);
            }
        });
    });
}

// Run migration if called directly
if (require.main === module) {
    migrateUsers()
        .then(() => {
            console.log('\n✅ Migration completed successfully');
            process.exit(0);
        })
        .catch((error) => {
            console.error('\n❌ Migration failed:', error);
            process.exit(1);
        });
}

module.exports = { migrateUsers };
