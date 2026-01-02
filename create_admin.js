const bcrypt = require('bcryptjs');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// Admin credentials
const adminEmail = 'admin@trading.com';
const adminPassword = 'admin123';

// Database path
const dbPath = path.join(__dirname, 'database', 'tradingdashboard.db');

// Connect to database
const db = new sqlite3.Database(dbPath, (err) => {
    if (err) {
        console.error('Error connecting to database:', err);
        process.exit(1);
    }
    console.log('Connected to SQLite database');
});

// Check if admin already exists
db.get('SELECT * FROM users WHERE email = ?', [adminEmail], async (err, row) => {
    if (err) {
        console.error('Error checking for admin:', err);
        db.close();
        process.exit(1);
    }

    if (row) {
        console.log(`\n✅ Admin user already exists: ${adminEmail}`);
        console.log(`   Role: ${row.role}`);

        // Update role to admin if not already
        if (row.role !== 'admin') {
            db.run('UPDATE users SET role = ? WHERE email = ?', ['admin', adminEmail], (err) => {
                if (err) {
                    console.error('Error updating role:', err);
                } else {
                    console.log(`   ✅ Updated role to 'admin'`);
                }
                db.close();
            });
        } else {
            db.close();
        }
    } else {
        // Hash password
        const saltRounds = 12;
        bcrypt.hash(adminPassword, saltRounds, (err, hash) => {
            if (err) {
                console.error('Error hashing password:', err);
                db.close();
                process.exit(1);
            }

            // Insert admin user
            const sql = `INSERT INTO users (email, password_hash, role, full_name, subscription_status)
                         VALUES (?, ?, ?, ?, ?)`;

            db.run(sql, [adminEmail, hash, 'admin', 'Admin User', 'active'], function(err) {
                if (err) {
                    console.error('Error creating admin:', err);
                } else {
                    console.log('\n🎉 Admin user created successfully!');
                    console.log('═══════════════════════════════════════');
                    console.log('Email:    admin@trading.com');
                    console.log('Password: admin123');
                    console.log('Role:     admin');
                    console.log('═══════════════════════════════════════');
                    console.log('\n⚠️  IMPORTANT: Change this password after first login!');
                    console.log('\nYou can now login at: http://localhost:3000/auth.html');
                }
                db.close();
            });
        });
    }
});
