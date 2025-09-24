const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            fontSrc: ["'self'", "https://fonts.gstatic.com"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'"]
        }
    }
}));

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Auth rate limiting
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5 // limit each IP to 5 auth requests per windowMs
});
app.use('/api/auth/', authLimiter);

// JWT Secret (in production, use environment variable)
const JWT_SECRET = process.env.JWT_SECRET || 'moneymoney_jwt_secret_key_2024';

// Database initialization
const db = new sqlite3.Database('./database/moneymoney.db', (err) => {
    if (err) {
        console.error('Error opening database:', err.message);
    } else {
        console.log('Connected to SQLite database');
        initializeDatabase();
    }
});

// Initialize database tables
function initializeDatabase() {
    // Users table
    db.run(`CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        subscription_status TEXT DEFAULT 'active',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);

    // Trading instruments table
    db.run(`CREATE TABLE IF NOT EXISTS instruments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        change_percent REAL NOT NULL,
        entry_point REAL,
        take_profit REAL,
        stop_loss REAL,
        signal TEXT,
        confidence INTEGER,
        model_accuracy INTEGER,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);

    // AI Models table
    db.run(`CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        accuracy REAL NOT NULL,
        backtest_period TEXT,
        total_trades INTEGER,
        win_rate REAL,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);

    // Populate sample data
    populateSampleData();
}

// Authentication middleware
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ message: 'Access token required' });
    }

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) {
            return res.status(403).json({ message: 'Invalid or expired token' });
        }
        req.user = user;
        next();
    });
}

// Routes
// Serve HTML pages
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

app.get('/auth', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'auth.html'));
});

app.get('/dashboard', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'dashboard.html'));
});

app.get('/profile', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'profile.html'));
});

app.get('/models', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'models.html'));
});

// API Routes
// User registration
app.post('/api/auth/signup', async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ message: 'Email and password are required' });
        }

        if (password.length < 6) {
            return res.status(400).json({ message: 'Password must be at least 6 characters long' });
        }

        // Check if user already exists
        db.get('SELECT * FROM users WHERE email = ?', [email], async (err, row) => {
            if (err) {
                return res.status(500).json({ message: 'Database error' });
            }

            if (row) {
                return res.status(400).json({ message: 'User already exists with this email' });
            }

            // Hash password
            const saltRounds = 12;
            const password_hash = await bcrypt.hash(password, saltRounds);

            // Create user
            db.run('INSERT INTO users (email, password_hash) VALUES (?, ?)',
                [email, password_hash],
                function(err) {
                    if (err) {
                        return res.status(500).json({ message: 'Failed to create user' });
                    }

                    // Generate JWT token
                    const token = jwt.sign(
                        { id: this.lastID, email: email },
                        JWT_SECRET,
                        { expiresIn: '7d' }
                    );

                    res.status(201).json({
                        message: 'User created successfully',
                        token: token,
                        user: { id: this.lastID, email: email }
                    });
                }
            );
        });
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// User login
app.post('/api/auth/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ message: 'Email and password are required' });
        }

        // Find user
        db.get('SELECT * FROM users WHERE email = ?', [email], async (err, row) => {
            if (err) {
                return res.status(500).json({ message: 'Database error' });
            }

            if (!row) {
                return res.status(400).json({ message: 'Invalid email or password' });
            }

            // Check password
            const isValidPassword = await bcrypt.compare(password, row.password_hash);
            if (!isValidPassword) {
                return res.status(400).json({ message: 'Invalid email or password' });
            }

            // Generate JWT token
            const token = jwt.sign(
                { id: row.id, email: row.email },
                JWT_SECRET,
                { expiresIn: '7d' }
            );

            res.json({
                message: 'Login successful',
                token: token,
                user: { id: row.id, email: row.email, created_at: row.created_at }
            });
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Get trading instruments (protected route)
app.get('/api/instruments', authenticateToken, (req, res) => {
    db.all('SELECT * FROM instruments ORDER BY symbol', (err, rows) => {
        if (err) {
            return res.status(500).json({ message: 'Database error' });
        }

        const instruments = rows.map(row => ({
            id: row.id,
            symbol: row.symbol,
            name: row.name,
            category: row.category,
            price: row.price.toFixed(4),
            change: row.change_percent,
            entryPoint: row.entry_point.toFixed(4),
            takeProfit: row.take_profit.toFixed(4),
            stopLoss: row.stop_loss.toFixed(4),
            signal: row.signal,
            confidence: row.confidence,
            modelAccuracy: row.model_accuracy
        }));

        res.json(instruments);
    });
});

// Get AI models (protected route)
app.get('/api/models', authenticateToken, (req, res) => {
    db.all('SELECT * FROM models ORDER BY accuracy DESC', (err, rows) => {
        if (err) {
            return res.status(500).json({ message: 'Database error' });
        }

        const models = rows.map(row => ({
            id: row.id,
            name: row.name,
            description: row.description,
            accuracy: row.accuracy,
            backtestPeriod: row.backtest_period,
            totalTrades: row.total_trades,
            winRate: row.win_rate,
            lastUpdated: new Date(row.last_updated).toLocaleDateString()
        }));

        res.json(models);
    });
});

// Update user profile (protected route)
app.put('/api/profile/update', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const { newEmail, currentPassword, newPassword } = req.body;

        // Get current user data
        db.get('SELECT * FROM users WHERE id = ?', [userId], async (err, user) => {
            if (err) {
                return res.status(500).json({ message: 'Database error' });
            }

            if (!user) {
                return res.status(404).json({ message: 'User not found' });
            }

            let updateFields = [];
            let updateValues = [];

            // Update email if provided
            if (newEmail && newEmail !== user.email) {
                // Check if email is already taken
                db.get('SELECT * FROM users WHERE email = ? AND id != ?', [newEmail, userId], (emailErr, existingUser) => {
                    if (emailErr) {
                        return res.status(500).json({ message: 'Database error' });
                    }

                    if (existingUser) {
                        return res.status(400).json({ message: 'Email is already taken' });
                    }

                    updateFields.push('email = ?');
                    updateValues.push(newEmail);

                    proceedWithUpdate();
                });
            } else {
                proceedWithUpdate();
            }

            async function proceedWithUpdate() {
                // Update password if provided
                if (currentPassword && newPassword) {
                    const isValidPassword = await bcrypt.compare(currentPassword, user.password_hash);
                    if (!isValidPassword) {
                        return res.status(400).json({ message: 'Current password is incorrect' });
                    }

                    const saltRounds = 12;
                    const newPasswordHash = await bcrypt.hash(newPassword, saltRounds);
                    updateFields.push('password_hash = ?');
                    updateValues.push(newPasswordHash);
                }

                if (updateFields.length === 0) {
                    return res.status(400).json({ message: 'No valid updates provided' });
                }

                updateFields.push('updated_at = CURRENT_TIMESTAMP');
                updateValues.push(userId);

                const updateQuery = `UPDATE users SET ${updateFields.join(', ')} WHERE id = ?`;

                db.run(updateQuery, updateValues, function(updateErr) {
                    if (updateErr) {
                        return res.status(500).json({ message: 'Failed to update profile' });
                    }

                    res.json({ message: 'Profile updated successfully' });
                });
            }
        });
    } catch (error) {
        console.error('Profile update error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Delete user account (protected route)
app.delete('/api/profile/delete', authenticateToken, (req, res) => {
    const userId = req.user.id;

    db.run('DELETE FROM users WHERE id = ?', [userId], function(err) {
        if (err) {
            return res.status(500).json({ message: 'Failed to delete account' });
        }

        if (this.changes === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        res.json({ message: 'Account deleted successfully' });
    });
});

// Create test account
async function createTestAccount() {
    // Check if test account already exists
    db.get('SELECT * FROM users WHERE email = ?', ['test@example.com'], async (err, row) => {
        if (err) {
            console.error('Error checking for test account:', err);
            return;
        }

        if (!row) {
            // Create test account with hashed password
            const bcrypt = require('bcryptjs');
            const saltRounds = 12;
            const passwordHash = await bcrypt.hash('testpassword', saltRounds);

            db.run('INSERT INTO users (email, password_hash, subscription_status) VALUES (?, ?, ?)',
                ['test@example.com', passwordHash, 'active'],
                function(err) {
                    if (err) {
                        console.error('Failed to create test account:', err);
                    } else {
                        console.log('Test account created successfully:');
                        console.log('  Email: test@example.com');
                        console.log('  Password: testpassword');
                    }
                }
            );
        } else {
            console.log('Test account already exists: test@example.com');
        }
    });
}

// Populate sample data
function populateSampleData() {
    // Create test account
    createTestAccount();

    // Check if data already exists
    db.get('SELECT COUNT(*) as count FROM instruments', (err, row) => {
        if (err || row.count > 0) return;

        const sampleInstruments = [
            {
                symbol: 'BTC/USD',
                name: 'Bitcoin',
                category: 'crypto',
                price: 43250.75,
                change_percent: 2.45,
                entry_point: 43100.00,
                take_profit: 45200.00,
                stop_loss: 41800.00,
                signal: 'BUY',
                confidence: 87,
                model_accuracy: 89
            },
            {
                symbol: 'ETH/USD',
                name: 'Ethereum',
                category: 'crypto',
                price: 2635.20,
                change_percent: -1.25,
                entry_point: 2650.00,
                take_profit: 2450.00,
                stop_loss: 2750.00,
                signal: 'SELL',
                confidence: 78,
                model_accuracy: 84
            },
            {
                symbol: 'USD/GBP',
                name: 'US Dollar / British Pound',
                category: 'forex',
                price: 0.7892,
                change_percent: 0.65,
                entry_point: 0.7885,
                take_profit: 0.8050,
                stop_loss: 0.7750,
                signal: 'BUY',
                confidence: 92,
                model_accuracy: 91
            },
            {
                symbol: 'ADA/USD',
                name: 'Cardano',
                category: 'crypto',
                price: 0.4823,
                change_percent: 3.12,
                entry_point: 0.4800,
                take_profit: 0.5200,
                stop_loss: 0.4600,
                signal: 'BUY',
                confidence: 82,
                model_accuracy: 86
            },
            {
                symbol: 'GBP/USD',
                name: 'British Pound / US Dollar',
                category: 'forex',
                price: 1.2675,
                change_percent: -0.82,
                entry_point: 1.2690,
                take_profit: 1.2450,
                stop_loss: 1.2850,
                signal: 'SELL',
                confidence: 85,
                model_accuracy: 88
            }
        ];

        sampleInstruments.forEach(instrument => {
            db.run(`INSERT INTO instruments (symbol, name, category, price, change_percent,
                    entry_point, take_profit, stop_loss, signal, confidence, model_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [instrument.symbol, instrument.name, instrument.category, instrument.price,
                 instrument.change_percent, instrument.entry_point, instrument.take_profit,
                 instrument.stop_loss, instrument.signal, instrument.confidence, instrument.model_accuracy]);
        });
    });

    // Sample AI models
    db.get('SELECT COUNT(*) as count FROM models', (err, row) => {
        if (err || row.count > 0) return;

        const sampleModels = [
            {
                name: 'Neural Trend Predictor',
                description: 'Advanced neural network for trend prediction and market sentiment analysis',
                accuracy: 89.5,
                backtest_period: '2 Years',
                total_trades: 1247,
                win_rate: 87.2
            },
            {
                name: 'Momentum Signal AI',
                description: 'Machine learning model focused on momentum-based trading signals',
                accuracy: 84.3,
                backtest_period: '18 Months',
                total_trades: 892,
                win_rate: 82.1
            },
            {
                name: 'Risk-Adjusted Portfolio',
                description: 'AI-driven portfolio optimization with dynamic risk management',
                accuracy: 91.7,
                backtest_period: '3 Years',
                total_trades: 2156,
                win_rate: 89.8
            }
        ];

        sampleModels.forEach(model => {
            db.run(`INSERT INTO models (name, description, accuracy, backtest_period,
                    total_trades, win_rate) VALUES (?, ?, ?, ?, ?, ?)`,
                [model.name, model.description, model.accuracy, model.backtest_period,
                 model.total_trades, model.win_rate]);
        });
    });
}

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ message: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Money Money server running on http://localhost:${PORT}`);
});