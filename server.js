// Load environment variables FIRST before any other imports
require('dotenv').config();

const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { Pool } = require('pg');
const path = require('path');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const axios = require('axios');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cookieParser = require('cookie-parser');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

// Trading Platform API configuration
const TRADING_API_URL = process.env.TRADING_API_URL || 'http://localhost:8002';

// Python Backend Process
let pythonBackend = null;

function startPythonBackend() {
    console.log('\n=== Starting Python Backend ===');

    // Use virtual environment Python from the correct backend location
    const backendPath = path.join(__dirname, 'other platform', 'Trading');
    const venvPython = path.join(backendPath, 'venv', 'bin', 'python');

    // Start uvicorn directly with the correct module
    pythonBackend = spawn(venvPython, [
        '-m', 'uvicorn',
        'api.main_simple:app',
        '--host', '0.0.0.0',
        '--port', '8002'
    ], {
        cwd: backendPath,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            PYTHONPATH: backendPath
        }
    });

    pythonBackend.stdout.on('data', (data) => {
        console.log(`[Python Backend] ${data.toString().trim()}`);
    });

    pythonBackend.stderr.on('data', (data) => {
        console.error(`[Python Backend Error] ${data.toString().trim()}`);
    });

    pythonBackend.on('close', (code) => {
        console.log(`[Python Backend] Process exited with code ${code}`);
        if (code !== 0 && code !== null) {
            console.log('[Python Backend] Restarting in 5 seconds...');
            setTimeout(startPythonBackend, 5000);
        }
    });

    pythonBackend.on('error', (error) => {
        console.error(`[Python Backend] Failed to start: ${error.message}`);
    });
}

// Handle process termination
process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server and Python backend');
    if (pythonBackend) {
        pythonBackend.kill('SIGTERM');
    }
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('SIGINT signal received: closing HTTP server and Python backend');
    if (pythonBackend) {
        pythonBackend.kill('SIGTERM');
    }
    process.exit(0);
});

// Middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com", "https://cdnjs.cloudflare.com"],
            fontSrc: ["'self'", "https://fonts.gstatic.com", "https://cdnjs.cloudflare.com"],
            scriptSrc: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://cdn.socket.io", "https://unpkg.com"],
            scriptSrcElem: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://cdn.socket.io", "https://unpkg.com"],
            scriptSrcAttr: ["'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'", "http://localhost:8002", "ws://localhost:8002"]
        }
    }
}));

app.use(cors());
app.use(express.json());
app.use(cookieParser());
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
const JWT_SECRET = process.env.JWT_SECRET || 'tradingdashboard_jwt_secret_key_2024';

// DEBUG: Log JWT secret info at startup
console.log('=' .repeat(80));
console.log('NODE.JS JWT SECRET DIAGNOSTIC:');
console.log(`  JWT_SECRET from env: ${process.env.JWT_SECRET ? process.env.JWT_SECRET.substring(0, 20) + '...' : 'NOT SET'}`);
console.log(`  JWT_SECRET being used (first 20 chars): ${JWT_SECRET.substring(0, 20)}...`);
console.log(`  JWT_SECRET length: ${JWT_SECRET.length} chars`);
console.log(`  Expected: 'tradingdashboard_jwt...' (57 chars from .env)`);
console.log(`  Using fallback: ${JWT_SECRET === 'tradingdashboard_jwt_secret_key_2024'}`);
console.log('=' .repeat(80));

// Database initialization - PostgreSQL connection pool
const db = new Pool({
    connectionString: process.env.DATABASE_URL,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
});

// Test database connection
db.query('SELECT NOW()', (err, res) => {
    if (err) {
        console.error('Error connecting to PostgreSQL database:', err.message);
    } else {
        console.log('Connected to PostgreSQL database');
        initializeDatabase();
    }
});

// Initialize database tables
async function initializeDatabase() {
    try {
        // Users table (PostgreSQL already has this from migrations)
        await db.query(`CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(50) DEFAULT 'user',
            full_name VARCHAR(255),
            subscription_status VARCHAR(50) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )`);

        // Trading instruments table (for MoneyMoney dashboard only)
        await db.query(`CREATE TABLE IF NOT EXISTS instruments (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(50) NOT NULL,
            name VARCHAR(255) NOT NULL,
            category VARCHAR(50) NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            change_percent DOUBLE PRECISION NOT NULL,
            entry_point DOUBLE PRECISION,
            take_profit DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            signal VARCHAR(20),
            confidence INTEGER,
            model_accuracy INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_data BOOLEAN DEFAULT FALSE,
            data_updated_at TIMESTAMP,
            models_trained BOOLEAN DEFAULT FALSE,
            last_training TIMESTAMP,
            data_interval VARCHAR(10) DEFAULT '1m',
            total_data_points INTEGER DEFAULT 0
        )`);

        // Add columns to existing users table if they don't exist
        await db.query(`ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'user'`).catch(() => {});
        await db.query(`ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255)`).catch(() => {});

        // Add columns to existing instruments table if they don't exist
        await db.query(`ALTER TABLE instruments ADD COLUMN IF NOT EXISTS has_data BOOLEAN DEFAULT FALSE`).catch(() => {});
        await db.query(`ALTER TABLE instruments ADD COLUMN IF NOT EXISTS data_updated_at TIMESTAMP`).catch(() => {});
        await db.query(`ALTER TABLE instruments ADD COLUMN IF NOT EXISTS models_trained BOOLEAN DEFAULT FALSE`).catch(() => {});
        await db.query(`ALTER TABLE instruments ADD COLUMN IF NOT EXISTS last_training TIMESTAMP`).catch(() => {});
        await db.query(`ALTER TABLE instruments ADD COLUMN IF NOT EXISTS data_interval VARCHAR(10) DEFAULT '1m'`).catch(() => {});
        await db.query(`ALTER TABLE instruments ADD COLUMN IF NOT EXISTS total_data_points INTEGER DEFAULT 0`).catch(() => {});

        // AI Models table
        await db.query(`CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            accuracy DOUBLE PRECISION NOT NULL,
            backtest_period VARCHAR(50),
            total_trades INTEGER,
            win_rate DOUBLE PRECISION,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )`);

        // Populate sample data AFTER all tables are created
        populateSampleData();
    } catch (error) {
        console.error('Error initializing database:', error);
    }
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

// Session-based authentication middleware (for HTML page navigation)
function authenticateSession(req, res, next) {
    const token = req.cookies.auth_session;

    if (!token) {
        return res.redirect('/auth');
    }

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) {
            res.clearCookie('auth_session');
            return res.redirect('/auth');
        }
        req.user = user;
        next();
    });
}

// Admin authentication middleware
function requireAdmin(req, res, next) {
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ message: 'Admin access required' });
    }
    next();
}

// Subscription authentication middleware (for user dashboard access)
async function requireActiveSubscription(req, res, next) {
    try {
        // Admins bypass subscription check
        if (req.user && req.user.role === 'admin') {
            return next();
        }

        const userId = req.user.id;

        // Check subscription status
        const result = await db.query(
            `SELECT subscription_status, subscription_expires_at
             FROM users WHERE id = $1`,
            [userId]
        );

        if (result.rows.length === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        const { subscription_status, subscription_expires_at } = result.rows[0];

        // Check if subscription is active
        const now = new Date();
        const expiresAt = subscription_expires_at ? new Date(subscription_expires_at) : null;

        // Allow access if status is 'active' AND (no expiration date OR expiration is in future)
        if (subscription_status === 'active' && (!expiresAt || expiresAt > now)) {
            // Subscription is active and not expired (or has no expiration = lifetime)
            return next();
        }

        // Subscription is inactive, cancelled, or expired
        if (expiresAt && expiresAt < now && subscription_status === 'active') {
            // Auto-update status to expired
            await db.query(
                `UPDATE users SET subscription_status = 'expired' WHERE id = $1`,
                [userId]
            );
        }

        // Redirect to subscription page or return error
        return res.status(403).json({
            message: 'Active subscription required',
            subscription_status: subscription_status === 'active' && expiresAt && expiresAt < now
                ? 'expired'
                : subscription_status,
            redirect: '/subscription'
        });

    } catch (error) {
        console.error('Subscription check error:', error);
        res.status(500).json({ message: 'Failed to verify subscription' });
    }
}

// Routes
// Serve HTML pages
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

app.get('/auth', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'auth.html'));
});

// User routes
app.get('/dashboard', authenticateSession, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'dashboard.html'));
});

app.get('/profile', authenticateSession, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'profile.html'));
});

app.get('/models', authenticateSession, requireActiveSubscription, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'models.html'));
});

// Subscription page (no subscription required to view this)
app.get('/subscription', authenticateSession, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'subscription.html'));
});

// Admin Panel Routes - Serve static HTML admin panel
// Uses cookie-based session authentication for page navigation
// Dashboard
app.get('/admin', authenticateSession, requireAdmin, (req, res) => {
    res.redirect('/admin/dashboard');
});

app.get('/admin/dashboard', authenticateSession, requireAdmin, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'admin', 'dashboard.html'));
});

// Data Collection
app.get('/admin/data-collection', authenticateSession, requireAdmin, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'admin', 'data-collection.html'));
});

// Data Management
app.get('/admin/data-management', authenticateSession, requireAdmin, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'admin', 'data-management.html'));
});

// ML Models
app.get('/admin/models', authenticateSession, requireAdmin, (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'admin', 'models.html'));
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
        const checkResult = await db.query('SELECT * FROM users WHERE email = $1', [email]);

        if (checkResult.rows.length > 0) {
            return res.status(400).json({ message: 'User already exists with this email' });
        }

        // Hash password
        const saltRounds = 12;
        const password_hash = await bcrypt.hash(password, saltRounds);

        // Create user with default 'user' role
        const insertResult = await db.query(
            'INSERT INTO users (email, password_hash, role) VALUES ($1, $2, $3) RETURNING id, email, role',
            [email, password_hash, 'user']
        );

        const newUser = insertResult.rows[0];

        // Generate JWT token with role
        const token = jwt.sign(
            { id: newUser.id, email: newUser.email, role: 'user' },
            JWT_SECRET,
            { expiresIn: '7d' }
        );

        res.status(201).json({
            message: 'User created successfully',
            token: token,
            role: 'user',
            user: { id: newUser.id, email: newUser.email, role: 'user' }
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
        const result = await db.query('SELECT * FROM users WHERE email = $1', [email]);

        if (result.rows.length === 0) {
            return res.status(400).json({ message: 'Invalid email or password' });
        }

        const user = result.rows[0];

        // Check password
        const isValidPassword = await bcrypt.compare(password, user.password_hash);
        if (!isValidPassword) {
            return res.status(400).json({ message: 'Invalid email or password' });
        }

        // Generate JWT token with role and type (for Python backend compatibility)
        const userRole = user.role || 'user';
        const token = jwt.sign(
            {
                id: user.id,
                email: user.email,
                role: userRole,
                type: userRole === 'admin' ? 'admin' : 'user'
            },
            JWT_SECRET,
            { expiresIn: '7d' }
        );

        // Set HTTP-only cookie for admin panel navigation
        res.cookie('auth_session', token, {
            httpOnly: true,
            secure: false, // Set to true in production with HTTPS
            maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
            sameSite: 'lax'
        });

        res.json({
            message: 'Login successful',
            token: token,
            role: user.role || 'user',
            user: {
                id: user.id,
                email: user.email,
                role: user.role || 'user',
                full_name: user.full_name,
                created_at: user.created_at
            }
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Get trading instruments (protected route)
// UPDATED: Now fetches from Trading Platform API profiles endpoint
// Note: Subscription check is now done on the frontend dashboard
app.get('/api/instruments', authenticateToken, async (req, res) => {
    try {
        // Fetch profiles from Trading Platform (no auth needed for this endpoint)
        const response = await axios.get(`${TRADING_API_URL}/api/profiles/`);

        // Transform profiles to MoneyMoney instrument format
        const instruments = response.data.map(item => ({
            id: item.symbol,
            symbol: item.symbol,
            name: item.name || item.symbol,
            category: item.profile_type || 'crypto',
            price: item.current_price ? parseFloat(item.current_price).toFixed(4) : '0.0000',
            change: item.price_change_24h ? parseFloat(item.price_change_24h).toFixed(2) : '0.00',
            change_percent: item.price_change_24h ? parseFloat(item.price_change_24h).toFixed(2) : '0.00',
            entryPoint: null,
            takeProfit: null,
            stopLoss: null,
            signal: 'hold',
            confidence: 0,
            modelAccuracy: item.win_rate ? Math.round(item.win_rate) : 0,
            hasData: item.is_active,
            modelsTrained: item.active_models > 0,
            dataUpdatedAt: item.updated_at,
            lastTraining: null,
            totalDataPoints: 0
        }));

        res.json(instruments);

    } catch (error) {
        console.error('Error fetching instruments from Trading Platform:', error.message);

        // Fallback to local database if Trading Platform is unavailable
        try {
            const result = await db.query('SELECT * FROM instruments WHERE has_data = true AND models_trained = true ORDER BY symbol');

            const instruments = result.rows.map(row => ({
                id: row.id,
                symbol: row.symbol,
                name: row.name,
                category: row.category,
                price: row.price ? row.price.toFixed(4) : '0.0000',
                change: row.change_percent ? row.change_percent.toFixed(2) : '0.00',
                entryPoint: row.entry_point ? row.entry_point.toFixed(4) : null,
                takeProfit: row.take_profit ? row.take_profit.toFixed(4) : null,
                stopLoss: row.stop_loss ? row.stop_loss.toFixed(4) : null,
                signal: row.signal || 'hold',
                confidence: row.confidence || 0,
                modelAccuracy: row.model_accuracy || 0,
                hasData: row.has_data,
                modelsTrained: row.models_trained
            }));

            res.json(instruments);
        } catch (dbError) {
            console.error('Database error:', dbError);
            res.status(500).json({ message: 'Database error' });
        }
    }
});

// Get AI models (protected route - requires active subscription)
app.get('/api/models', authenticateToken, requireActiveSubscription, async (req, res) => {
    try {
        const result = await db.query('SELECT * FROM models ORDER BY accuracy DESC');

        const models = result.rows.map(row => ({
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
    } catch (error) {
        console.error('Database error:', error);
        res.status(500).json({ message: 'Database error' });
    }
});

// Update user profile (protected route)
app.put('/api/profile/update', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const { newEmail, currentPassword, newPassword } = req.body;

        // Get current user data
        const userResult = await db.query('SELECT * FROM users WHERE id = $1', [userId]);

        if (userResult.rows.length === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        const user = userResult.rows[0];
        let updateFields = [];
        let updateValues = [];
        let paramCounter = 1;

        // Update email if provided
        if (newEmail && newEmail !== user.email) {
            // Check if email is already taken
            const emailCheck = await db.query('SELECT * FROM users WHERE email = $1 AND id != $2', [newEmail, userId]);

            if (emailCheck.rows.length > 0) {
                return res.status(400).json({ message: 'Email is already taken' });
            }

            updateFields.push(`email = $${paramCounter++}`);
            updateValues.push(newEmail);
        }

        // Update password if provided
        if (currentPassword && newPassword) {
            const isValidPassword = await bcrypt.compare(currentPassword, user.password_hash);
            if (!isValidPassword) {
                return res.status(400).json({ message: 'Current password is incorrect' });
            }

            const saltRounds = 12;
            const newPasswordHash = await bcrypt.hash(newPassword, saltRounds);
            updateFields.push(`password_hash = $${paramCounter++}`);
            updateValues.push(newPasswordHash);
        }

        if (updateFields.length === 0) {
            return res.status(400).json({ message: 'No valid updates provided' });
        }

        updateFields.push('updated_at = CURRENT_TIMESTAMP');
        updateValues.push(userId);

        const updateQuery = `UPDATE users SET ${updateFields.join(', ')} WHERE id = $${paramCounter}`;

        await db.query(updateQuery, updateValues);

        res.json({ message: 'Profile updated successfully' });
    } catch (error) {
        console.error('Profile update error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Delete user account (protected route)
app.delete('/api/profile/delete', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        const result = await db.query('DELETE FROM users WHERE id = $1', [userId]);

        if (result.rowCount === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        res.json({ message: 'Account deleted successfully' });
    } catch (error) {
        console.error('Delete account error:', error);
        res.status(500).json({ message: 'Failed to delete account' });
    }
});

// Create test account
async function createTestAccount() {
    try {
        // Check if test account already exists
        const result = await db.query('SELECT * FROM users WHERE email = $1', ['test@example.com']);

        if (result.rows.length === 0) {
            // Create test account with hashed password
            const saltRounds = 12;
            const passwordHash = await bcrypt.hash('testpassword', saltRounds);

            await db.query(
                'INSERT INTO users (email, password_hash, subscription_status) VALUES ($1, $2, $3)',
                ['test@example.com', passwordHash, 'active']
            );

            console.log('Test account created successfully:');
            console.log('  Email: test@example.com');
            console.log('  Password: testpassword');
        } else {
            console.log('Test account already exists: test@example.com');
        }
    } catch (error) {
        console.error('Error with test account:', error);
    }
}

// Populate sample data
async function populateSampleData() {
    try {
        // Create test account
        await createTestAccount();

        // Check if instruments data already exists
        const instrumentsResult = await db.query('SELECT COUNT(*) as count FROM instruments');
        if (instrumentsResult.rows[0].count > 0) {
            console.log('Sample instruments already exist');
        } else {
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

            for (const instrument of sampleInstruments) {
                await db.query(
                    `INSERT INTO instruments (symbol, name, category, price, change_percent,
                        entry_point, take_profit, stop_loss, signal, confidence, model_accuracy)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
                    [instrument.symbol, instrument.name, instrument.category, instrument.price,
                     instrument.change_percent, instrument.entry_point, instrument.take_profit,
                     instrument.stop_loss, instrument.signal, instrument.confidence, instrument.model_accuracy]
                );
            }
            console.log('Sample instruments populated');
        }

        // Check if models data already exists
        const modelsResult = await db.query('SELECT COUNT(*) as count FROM models');
        if (modelsResult.rows[0].count > 0) {
            console.log('Sample models already exist');
        } else {
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

            for (const model of sampleModels) {
                await db.query(
                    `INSERT INTO models (name, description, accuracy, backtest_period,
                        total_trades, win_rate) VALUES ($1, $2, $3, $4, $5, $6)`,
                    [model.name, model.description, model.accuracy, model.backtest_period,
                     model.total_trades, model.win_rate]
                );
            }
            console.log('Sample models populated');
        }
    } catch (error) {
        console.error('Error populating sample data:', error);
    }
}

// ==================== Trading Platform Proxy Endpoints ====================

// Get chart data for an instrument (proxy to Trading Platform)
app.get('/api/instruments/:symbol/data/:timeframe', authenticateToken, async (req, res) => {
    try {
        const { symbol, timeframe } = req.params;
        const { limit } = req.query;

        // Call the public API endpoint (no auth needed)
        const response = await axios.get(
            `${TRADING_API_URL}/api/public/data/${symbol}/${timeframe}`,
            {
                params: { limit: limit || 100 }
            }
        );

        // Response already in correct format
        res.json(response.data);

    } catch (error) {
        console.error('Error fetching chart data:', error.message);
        res.status(error.response?.status || 500).json({
            message: error.response?.data?.detail || 'Failed to fetch chart data'
        });
    }
});

// Get predictions for an instrument (proxy to Trading Platform)
app.get('/api/instruments/:symbol/predictions', authenticateToken, async (req, res) => {
    try {
        const { symbol } = req.params;

        // First get the profile ID for this symbol
        const profilesResponse = await axios.get(`${TRADING_API_URL}/api/profiles/`);
        const profile = profilesResponse.data.find(p => p.symbol === symbol);

        if (!profile) {
            return res.json([]);
        }

        // Get predictions from public endpoint
        const response = await axios.get(
            `${TRADING_API_URL}/api/public/profiles/${profile.id}/predictions`
        );

        res.json(response.data);

    } catch (error) {
        console.error('Error fetching predictions:', error.message);
        // Return empty array instead of error to prevent frontend issues
        res.json([]);
    }
});

// Get models for an instrument (proxy to Trading Platform)
app.get('/api/instruments/:symbol/models', authenticateToken, async (req, res) => {
    try {
        const { symbol } = req.params;

        // First get the profile ID for this symbol
        const profilesResponse = await axios.get(`${TRADING_API_URL}/api/profiles/`);
        const profile = profilesResponse.data.find(p => p.symbol === symbol);

        if (!profile) {
            return res.json([]);
        }

        // Get models from profiles endpoint
        const response = await axios.get(
            `${TRADING_API_URL}/api/profiles/${profile.id}/models`
        );

        res.json(response.data);

    } catch (error) {
        console.error('Error fetching models:', error.message);
        // Return empty array instead of error to prevent frontend issues
        res.json([]);
    }
});

// TODO: Implement actual trading signals calculation from predictions/indicators
// Currently returns default "hold" signal - backend integration pending
// Get trading signals for an instrument (proxy to Trading Platform)
app.get('/api/instruments/:symbol/signals', authenticateToken, async (req, res) => {
    try {
        const { symbol } = req.params;

        // Signals are not available in the Python backend, return default
        // In a real implementation, this would be calculated from predictions/indicators
        res.json({
            symbol: symbol,
            signal: 'hold',
            confidence: 0,
            entry_point: null,
            take_profit: null,
            stop_loss: null,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error fetching signals:', error.message);
        res.json({
            symbol: req.params.symbol,
            signal: 'hold',
            confidence: 0
        });
    }
});

// Get statistics for an instrument (proxy to Trading Platform)
app.get('/api/instruments/:symbol/stats', authenticateToken, async (req, res) => {
    try {
        const { symbol } = req.params;
        const { timeframe } = req.query;

        // Call the public API stats endpoint (no auth needed)
        const response = await axios.get(
            `${TRADING_API_URL}/api/public/data/${symbol}/${timeframe || '1D'}/stats`
        );

        res.json(response.data);

    } catch (error) {
        console.error('Error fetching stats:', error.message);
        // Return empty stats instead of error
        res.json({
            symbol: req.params.symbol,
            timeframe: req.query.timeframe || '1D',
            high: 0,
            low: 0,
            open: 0,
            close: 0,
            volume: 0
        });
    }
});

// TODO: Implement sentiment analysis backend integration
// Currently returns placeholder neutral sentiment data
// Get sentiment for an instrument (returns placeholder - sentiment not yet implemented)
app.get('/api/instruments/:symbol/sentiment', authenticateToken, async (req, res) => {
    try {
        const { symbol } = req.params;
        const { window } = req.query;

        // Sentiment analysis is not yet implemented in the backend
        // Return neutral placeholder data
        res.json({
            symbol: symbol,
            window: window || '1D',
            overall_sentiment: 0,
            sentiment_label: 'Neutral',
            sentiment_trend: 0,
            sentiment_volatility: 0,
            recommendation: 'Hold',
            avg_confidence: 0,
            data_points: 0,
            source_breakdown: {},
            latest_update: new Date().toISOString()
        });
    } catch (error) {
        console.error(`Error fetching sentiment for ${req.params.symbol}:`, error.message);
        res.json({
            symbol: req.params.symbol,
            overall_sentiment: 0,
            sentiment_label: 'Neutral'
        });
    }
});

// Get sentiment history for an instrument (returns empty - sentiment not yet implemented)
app.get('/api/instruments/:symbol/sentiment/history', authenticateToken, async (req, res) => {
    try {
        const { symbol } = req.params;

        // Sentiment history is not yet implemented
        res.json({
            symbol: symbol,
            history: [],
            total: 0
        });
    } catch (error) {
        console.error(`Error fetching sentiment history for ${req.params.symbol}:`, error.message);
        res.json({
            symbol: req.params.symbol,
            history: [],
            total: 0
        });
    }
});

// Get technical indicators for an instrument (proxy to Trading Platform)
app.get('/api/instruments/:symbol/indicators', authenticateToken, async (req, res) => {
    try {
        const { symbol } = req.params;
        const { limit } = req.query;

        // Call the public API indicators endpoint (no auth needed)
        const response = await axios.get(
            `${TRADING_API_URL}/api/public/data/${symbol}/indicators`,
            { params: { limit: limit || 200 } }
        );

        // Response already in correct format
        res.json(response.data);
    } catch (error) {
        console.error(`Error fetching indicators for ${req.params.symbol}:`, error.message);
        // Return empty indicators instead of error
        res.json({
            symbol: req.params.symbol,
            indicators: [],
            total_indicators: 0,
            timestamp: new Date().toISOString()
        });
    }
});

// Health check for Trading Platform connection
app.get('/api/trading-platform/health', authenticateToken, async (req, res) => {
    try {
        // Check if Trading Platform is responding (no auth needed)
        const response = await axios.get(`${TRADING_API_URL}/api/profiles/`, {
            timeout: 5000
        });

        res.json({
            tradingPlatform: {
                status: 'healthy',
                profiles: response.data.length,
                timestamp: new Date().toISOString()
            },
            moneyMoney: {
                status: 'healthy',
                timestamp: new Date().toISOString()
            }
        });

    } catch (error) {
        res.json({
            tradingPlatform: {
                status: 'unavailable',
                error: error.message
            },
            moneyMoney: {
                status: 'healthy',
                timestamp: new Date().toISOString()
            }
        });
    }
});

// ==================== PAYSTACK SUBSCRIPTION API ====================
// Paystack configuration
const PAYSTACK_SECRET_KEY = process.env.PAYSTACK_SECRET_KEY;
const PAYSTACK_API_URL = 'https://api.paystack.co';
const SUBSCRIPTION_AMOUNT = 20000; // ZAR 200.00 in cents (Paystack uses smallest currency unit)
const PLAN_CODE = 'PLN_o6aocIukczuw4dk'; // Actual Paystack plan code

// Helper: Verify Paystack webhook signature
function verifyPaystackSignature(payload, signature) {
    const crypto = require('crypto');
    const hash = crypto.createHmac('sha512', PAYSTACK_SECRET_KEY)
        .update(JSON.stringify(payload))
        .digest('hex');
    return hash === signature;
}

// Initialize subscription payment
app.post('/api/subscription/initialize', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        // Get user email from database
        const userResult = await db.query('SELECT email FROM users WHERE id = $1', [userId]);
        if (userResult.rows.length === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        const userEmail = userResult.rows[0].email;

        // Generate unique transaction reference
        const reference = `SUB_${userId}_${Date.now()}`;

        // Initialize Paystack transaction
        const paystackResponse = await axios.post(
            `${PAYSTACK_API_URL}/transaction/initialize`,
            {
                email: userEmail,
                amount: SUBSCRIPTION_AMOUNT, // ZAR 200.00 in cents
                currency: 'ZAR',
                reference: reference,
                plan: PLAN_CODE,
                callback_url: `${req.protocol}://${req.get('host')}/subscription-success`,
                metadata: {
                    user_id: userId,
                    subscription_type: 'monthly'
                }
            },
            {
                headers: {
                    Authorization: `Bearer ${PAYSTACK_SECRET_KEY}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        // Store pending payment in database
        await db.query(
            `INSERT INTO payment_history (user_id, reference, amount, currency, status, customer_email, payment_type)
             VALUES ($1, $2, $3, $4, $5, $6, $7)`,
            [userId, reference, 10.00, 'USD', 'pending', userEmail, 'subscription']
        );

        res.json({
            success: true,
            authorization_url: paystackResponse.data.data.authorization_url,
            access_code: paystackResponse.data.data.access_code,
            reference: reference
        });

    } catch (error) {
        console.error('Paystack initialization error:', error.response?.data || error.message);
        res.status(500).json({
            success: false,
            message: 'Failed to initialize payment',
            error: error.response?.data?.message || error.message
        });
    }
});

// Verify payment transaction
app.get('/api/subscription/verify/:reference', authenticateToken, async (req, res) => {
    try {
        const { reference } = req.params;
        const userId = req.user.id;

        // Verify with Paystack
        const paystackResponse = await axios.get(
            `${PAYSTACK_API_URL}/transaction/verify/${reference}`,
            {
                headers: {
                    Authorization: `Bearer ${PAYSTACK_SECRET_KEY}`
                }
            }
        );

        const transaction = paystackResponse.data.data;

        if (transaction.status === 'success') {
            // Calculate subscription dates
            const now = new Date();
            const expiresAt = new Date(now);
            expiresAt.setMonth(expiresAt.getMonth() + 1); // Add 1 month

            // Update user subscription status
            await db.query(
                `UPDATE users
                 SET subscription_status = 'active',
                     subscription_code = $1,
                     subscription_started_at = $2,
                     subscription_expires_at = $3,
                     auto_renew = true
                 WHERE id = $4`,
                [transaction.subscription_code || null, now, expiresAt, userId]
            );

            // Update payment history
            await db.query(
                `UPDATE payment_history
                 SET status = 'success',
                     paystack_reference = $1,
                     authorization_code = $2,
                     paid_at = $3,
                     metadata = $4
                 WHERE reference = $5`,
                [
                    transaction.reference,
                    transaction.authorization?.authorization_code || null,
                    now,
                    JSON.stringify(transaction),
                    reference
                ]
            );

            res.json({
                success: true,
                message: 'Subscription activated successfully',
                subscription: {
                    status: 'active',
                    started_at: now,
                    expires_at: expiresAt
                }
            });
        } else {
            // Update payment as failed
            await db.query(
                `UPDATE payment_history SET status = 'failed' WHERE reference = $1`,
                [reference]
            );

            res.json({
                success: false,
                message: 'Payment verification failed',
                status: transaction.status
            });
        }

    } catch (error) {
        console.error('Payment verification error:', error.response?.data || error.message);
        res.status(500).json({
            success: false,
            message: 'Failed to verify payment',
            error: error.response?.data?.message || error.message
        });
    }
});

// Get subscription status
app.get('/api/subscription/status', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        const result = await db.query(
            `SELECT subscription_status, subscription_started_at, subscription_expires_at, auto_renew
             FROM users WHERE id = $1`,
            [userId]
        );

        if (result.rows.length === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        const subscription = result.rows[0];

        // Check if subscription has expired
        const now = new Date();
        const expiresAt = subscription.subscription_expires_at ? new Date(subscription.subscription_expires_at) : null;

        let status = subscription.subscription_status || 'inactive';
        if (status === 'active' && expiresAt && expiresAt < now) {
            // Subscription has expired
            status = 'expired';
            await db.query(
                `UPDATE users SET subscription_status = 'expired' WHERE id = $1`,
                [userId]
            );
        }

        res.json({
            status: status,
            started_at: subscription.subscription_started_at,
            expires_at: subscription.subscription_expires_at,
            auto_renew: subscription.auto_renew,
            is_active: status === 'active',
            days_remaining: expiresAt && status === 'active'
                ? Math.ceil((expiresAt - now) / (1000 * 60 * 60 * 24))
                : 0
        });

    } catch (error) {
        console.error('Get subscription status error:', error);
        res.status(500).json({ message: 'Failed to get subscription status' });
    }
});

// Cancel subscription
app.post('/api/subscription/cancel', authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        // Get subscription code
        const userResult = await db.query(
            'SELECT subscription_code, subscription_status FROM users WHERE id = $1',
            [userId]
        );

        if (userResult.rows.length === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        const { subscription_code, subscription_status } = userResult.rows[0];

        if (subscription_status !== 'active') {
            return res.status(400).json({ message: 'No active subscription to cancel' });
        }

        // Disable auto-renewal in Paystack (if subscription code exists)
        if (subscription_code) {
            try {
                await axios.post(
                    `${PAYSTACK_API_URL}/subscription/disable`,
                    {
                        code: subscription_code,
                        token: subscription_code
                    },
                    {
                        headers: {
                            Authorization: `Bearer ${PAYSTACK_SECRET_KEY}`,
                            'Content-Type': 'application/json'
                        }
                    }
                );
            } catch (paystackError) {
                console.error('Paystack cancellation error:', paystackError.response?.data);
                // Continue even if Paystack fails - we still want to update our database
            }
        }

        // Update user subscription (keep active until expiry, but disable auto-renew)
        await db.query(
            `UPDATE users
             SET auto_renew = false,
                 subscription_status = 'cancelled'
             WHERE id = $1`,
            [userId]
        );

        res.json({
            success: true,
            message: 'Subscription cancelled. Access will remain until the end of the current billing period.'
        });

    } catch (error) {
        console.error('Cancel subscription error:', error);
        res.status(500).json({ message: 'Failed to cancel subscription' });
    }
});

// Paystack webhook handler
app.post('/api/payments/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
    try {
        const signature = req.headers['x-paystack-signature'];
        const payload = req.body;

        // Verify webhook signature
        if (!verifyPaystackSignature(payload, signature)) {
            console.error('Invalid Paystack webhook signature');
            return res.status(400).json({ error: 'Invalid signature' });
        }

        const event = JSON.parse(payload.toString());

        // Log webhook
        await db.query(
            `INSERT INTO webhook_logs (event_type, reference, payload, signature, ip_address)
             VALUES ($1, $2, $3, $4, $5)`,
            [
                event.event,
                event.data?.reference || null,
                event,
                signature,
                req.ip
            ]
        );

        // Handle different event types
        switch (event.event) {
            case 'charge.success':
                await handleChargeSuccess(event.data);
                break;

            case 'subscription.create':
                await handleSubscriptionCreate(event.data);
                break;

            case 'subscription.disable':
                await handleSubscriptionDisable(event.data);
                break;

            case 'subscription.not_renew':
                await handleSubscriptionNotRenew(event.data);
                break;

            default:
                console.log(`Unhandled webhook event: ${event.event}`);
        }

        // Mark webhook as processed
        await db.query(
            `UPDATE webhook_logs
             SET processed = true, processed_at = CURRENT_TIMESTAMP
             WHERE reference = $1 AND event_type = $2`,
            [event.data?.reference || null, event.event]
        );

        res.status(200).json({ success: true });

    } catch (error) {
        console.error('Webhook processing error:', error);

        // Log error
        try {
            await db.query(
                `UPDATE webhook_logs
                 SET processing_error = $1
                 WHERE reference = $2`,
                [error.message, event?.data?.reference || null]
            );
        } catch (dbError) {
            console.error('Failed to log webhook error:', dbError);
        }

        res.status(500).json({ error: 'Webhook processing failed' });
    }
});

// Webhook event handlers
async function handleChargeSuccess(data) {
    try {
        const reference = data.reference;

        // Update payment history
        await db.query(
            `UPDATE payment_history
             SET status = 'success',
                 paystack_reference = $1,
                 paid_at = CURRENT_TIMESTAMP,
                 metadata = $2
             WHERE reference = $3`,
            [data.reference, JSON.stringify(data), reference]
        );

        console.log(`Charge successful: ${reference}`);
    } catch (error) {
        console.error('Error handling charge success:', error);
    }
}

async function handleSubscriptionCreate(data) {
    try {
        const email = data.customer?.email;
        const subscriptionCode = data.subscription_code;

        // Find user by email
        const userResult = await db.query('SELECT id FROM users WHERE email = $1', [email]);

        if (userResult.rows.length > 0) {
            const userId = userResult.rows[0].id;

            // Update subscription
            await db.query(
                `UPDATE users
                 SET subscription_code = $1,
                     subscription_status = 'active'
                 WHERE id = $2`,
                [subscriptionCode, userId]
            );

            console.log(`Subscription created for user ${userId}: ${subscriptionCode}`);
        }
    } catch (error) {
        console.error('Error handling subscription create:', error);
    }
}

async function handleSubscriptionDisable(data) {
    try {
        const subscriptionCode = data.subscription_code;

        // Find user and update status
        await db.query(
            `UPDATE users
             SET subscription_status = 'cancelled',
                 auto_renew = false
             WHERE subscription_code = $1`,
            [subscriptionCode]
        );

        console.log(`Subscription disabled: ${subscriptionCode}`);
    } catch (error) {
        console.error('Error handling subscription disable:', error);
    }
}

async function handleSubscriptionNotRenew(data) {
    try {
        const subscriptionCode = data.subscription_code;

        // Mark subscription as expiring
        await db.query(
            `UPDATE users
             SET auto_renew = false
             WHERE subscription_code = $1`,
            [subscriptionCode]
        );

        console.log(`Subscription will not renew: ${subscriptionCode}`);
    } catch (error) {
        console.error('Error handling subscription not renew:', error);
    }
}

// Subscription success page
app.get('/subscription-success', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'subscription-success.html'));
});

// Proxy /api/* requests to Python backend (MUST be after Express API routes)
// This catches any /api/* requests that weren't handled by Express routes above
app.use('/api', createProxyMiddleware({
    target: TRADING_API_URL,
    changeOrigin: true,
    ws: true, // Enable WebSocket proxying
    logLevel: 'silent',
    onError: (err, req, res) => {
        console.error('API proxy error:', err.message);
        res.status(500).json({
            error: 'Trading API unavailable',
            message: 'Please ensure the Python backend is running'
        });
    },
}));

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ message: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
    console.log(`TradingDashboard AI server running on http://localhost:${PORT}`);

    // Start Python backend
    console.log('\\nStarting Python backend...');
    startPythonBackend();
});