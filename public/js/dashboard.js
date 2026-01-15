/**
 * Dashboard JavaScript - Trading Platform Integration
 * ===================================================
 *
 * Handles all dashboard functionality including:
 * - Loading trained instruments from Trading Platform
 * - Real-time chart updates with timeframe switching
 * - Model predictions and signals display
 * - Integration with MoneyMoney backend proxy
 */

// API Configuration
const API_BASE = window.location.origin; // http://localhost:3000
let currentChart = null;
let selectedInstrument = null;
let currentTimeframe = '1D';
let chartUpdateInterval = null;

// Authentication
function getAuthToken() {
    return localStorage.getItem('token');
}

function isAuthenticated() {
    return !!getAuthToken();
}

// API Helper Functions
async function apiCall(endpoint, options = {}) {
    const token = getAuthToken();

    if (!token) {
        window.location.href = '/auth';
        throw new Error('Not authenticated');
    }

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        }
    };

    const response = await fetch(`${API_BASE}${endpoint}`, {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...(options.headers || {})
        }
    });

    if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/auth';
        throw new Error('Unauthorized');
    }

    if (!response.ok) {
        const error = await response.json().catch(() => ({ message: 'Request failed' }));
        throw new Error(error.message || 'Request failed');
    }

    return response.json();
}

// Load Instruments (ONLY trained ones)
async function loadInstruments() {
    try {
        const instruments = await apiCall('/api/instruments');

        console.log(`Loaded ${instruments.length} trained instruments`);

        // Populate dropdown
        const dropdown = document.getElementById('instrumentsDropdown');
        if (dropdown) {
            dropdown.innerHTML = '<option value="">-- Select Instrument --</option>';

            instruments.forEach(instrument => {
                const option = document.createElement('option');
                option.value = instrument.symbol;
                option.textContent = `${instrument.name} (${instrument.symbol})`;
                option.dataset.instrument = JSON.stringify(instrument);
                dropdown.appendChild(option);
            });
        }

        // Populate search results
        const searchInput = document.getElementById('instrumentSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                filterInstruments(e.target.value, instruments);
            });
        }

        return instruments;

    } catch (error) {
        console.error('Error loading instruments:', error);
        window.MoneyMoney.showNotification('Failed to load instruments: ' + error.message, 'error');
        return [];
    }
}

// Filter instruments for search
function filterInstruments(query, instruments) {
    const dropdown = document.getElementById('searchDropdown');
    if (!dropdown) return;

    if (!query || query.length < 2) {
        dropdown.style.display = 'none';
        return;
    }

    const filtered = instruments.filter(inst =>
        inst.symbol.toLowerCase().includes(query.toLowerCase()) ||
        inst.name.toLowerCase().includes(query.toLowerCase())
    );

    if (filtered.length === 0) {
        dropdown.style.display = 'none';
        return;
    }

    dropdown.innerHTML = filtered.map(inst => `
        <div class="search-result-item" onclick="selectInstrumentFromSearch('${inst.symbol}')">
            <div>
                <strong>${inst.symbol}</strong> - ${inst.name}
                ${inst.modelsTrained ? '<span class="badge-trained">Trained</span>' : ''}
            </div>
            <div class="search-result-details">
                ${inst.price ? `$${inst.price}` : 'N/A'}
                ${inst.change ? `<span class="${inst.change >= 0 ? 'positive' : 'negative'}">${inst.change}%</span>` : ''}
            </div>
        </div>
    `).join('');

    dropdown.style.display = 'block';
}

// Select instrument from search
function selectInstrumentFromSearch(symbol) {
    const dropdown = document.getElementById('instrumentsDropdown');
    if (dropdown) {
        dropdown.value = symbol;
        dropdown.dispatchEvent(new Event('change'));
    }

    const searchDropdown = document.getElementById('searchDropdown');
    if (searchDropdown) {
        searchDropdown.style.display = 'none';
    }

    const searchInput = document.getElementById('instrumentSearch');
    if (searchInput) {
        searchInput.value = '';
    }
}

// Map frontend timeframes to API timeframes
function mapTimeframe(frontendTimeframe) {
    const mapping = {
        '1D': { apiTimeframe: '1h', limit: 24 },      // 24 hours of hourly data
        '7D': { apiTimeframe: '1h', limit: 168 },     // 7 days of hourly data
        '1M': { apiTimeframe: '1D', limit: 30 },      // 30 days of daily data
        '3M': { apiTimeframe: '1D', limit: 90 }       // 90 days of daily data
    };
    return mapping[frontendTimeframe] || { apiTimeframe: '1D', limit: 100 };
}

// Load chart data for selected instrument
async function loadChartData(symbol, timeframe, limit = null) {
    try {
        // Map the frontend timeframe to API timeframe
        const { apiTimeframe, limit: mappedLimit } = mapTimeframe(timeframe);
        const actualLimit = limit !== null ? limit : mappedLimit;

        const data = await apiCall(`/api/instruments/${symbol}/data/${apiTimeframe}?limit=${actualLimit}`);

        console.log(`Loaded ${data.total_candles} ${apiTimeframe} candles for ${symbol} (requested ${timeframe})`);

        return data;

    } catch (error) {
        console.error('Error loading chart data:', error);
        if (window.MoneyMoney && window.MoneyMoney.showNotification) {
            window.MoneyMoney.showNotification('Failed to load chart data: ' + error.message, 'error');
        }
        return null;
    }
}

// Update chart with new data
async function updateChart(symbol, timeframe) {
    const chartData = await loadChartData(symbol, timeframe);

    if (!chartData || !chartData.candles || chartData.candles.length === 0) {
        console.warn('No chart data available');
        return;
    }

    const canvas = document.getElementById('priceChart');
    if (!canvas) {
        console.error('Chart canvas not found');
        return;
    }

    const ctx = canvas.getContext('2d');

    // Load indicators data - await to ensure data is ready before rendering tabs
    try {
        await loadIndicators(symbol, 200);
    } catch (err) {
        console.warn('Failed to load indicators:', err);
    }

    // Destroy existing chart
    if (currentChart) {
        currentChart.destroy();
    }

    // Prepare data for Chart.js
    // Frontend timeframes: 1D, 7D, 1M, 3M
    // 1D and 7D use hourly data (show time), 1M and 3M use daily data (show date)
    const labels = chartData.candles.map(candle => {
        const date = new Date(candle.timestamp);
        if (timeframe === '1D' || timeframe === '7D') {
            // Hourly data - show date and time
            return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        } else if (timeframe === '1M' || timeframe === '3M') {
            // Daily data - show just date
            return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        } else {
            // Fallback for API timeframes
            return date.toLocaleDateString();
        }
    });

    const prices = chartData.candles.map(candle => parseFloat(candle.close));

    // Create gradient for line
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.8)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.2)');

    // Create chart
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `${symbol} Price`,
                data: prices,
                borderColor: '#3B82F6',
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: '#3B82F6',
                pointHoverBorderColor: '#FFFFFF',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(12, 11, 16, 0.95)',
                    titleColor: '#FFFFFF',
                    bodyColor: '#B0B0B0',
                    borderColor: 'rgba(59, 130, 246, 0.5)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `Price: $${context.parsed.y.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        borderColor: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#B0B0B0',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        borderColor: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#B0B0B0',
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });

    // Update title
    const title = document.getElementById('selectedInstrumentTitle');
    if (title) {
        title.textContent = `${symbol} - ${timeframe.toUpperCase()} Chart`;
    }
}

// Handle instrument selection
async function onInstrumentSelected(event) {
    const select = event.target;
    const symbol = select.value;

    if (!symbol) {
        return;
    }

    const instrumentData = JSON.parse(select.options[select.selectedIndex].dataset.instrument || '{}');
    selectedInstrument = { symbol, ...instrumentData };

    console.log('Selected instrument:', selectedInstrument);

    // Update chart
    await updateChart(symbol, currentTimeframe);

    // Load additional data (predictions, signals, etc.)
    loadInstrumentDetails(symbol);
}

// Load instrument details (predictions, signals, stats, models)
async function loadInstrumentDetails(symbol) {
    try {
        // Load models for this instrument
        const models = await apiCall(`/api/instruments/${symbol}/models`).catch(() => null);
        displayModels(models || []);

        // Load predictions
        const predictions = await apiCall(`/api/instruments/${symbol}/predictions`).catch(() => null);
        displayPredictions(predictions || []);

        // Load signals
        const signal = await apiCall(`/api/instruments/${symbol}/signals`).catch(() => null);
        if (signal) {
            displaySignal(signal);
        }

        // Load stats
        const stats = await apiCall(`/api/instruments/${symbol}/stats?timeframe=${currentTimeframe}`).catch(() => null);
        if (stats) {
            displayStats(stats);
        }

    } catch (error) {
        console.error('Error loading instrument details:', error);
    }
}

// Display models list in the Models tab
function displayModels(models) {
    const modelsListEl = document.getElementById('modelsList');
    const modelsCountEl = document.getElementById('modelsCount');

    if (!modelsListEl) {
        console.warn('modelsList container not found');
        return;
    }

    if (!models || models.length === 0) {
        modelsCountEl.textContent = '0 models';
        modelsListEl.innerHTML = `
            <div style="text-align: center; padding: 30px; color: var(--color-text-secondary);">
                <p style="margin: 0 0 10px 0;">No trained models for this instrument.</p>
                <p style="margin: 0; font-size: 0.9rem;">Go to the Models page to train a new model.</p>
            </div>
        `;
        return;
    }

    modelsCountEl.textContent = `${models.length} model${models.length > 1 ? 's' : ''}`;

    let html = '<div class="models-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px;">';

    models.forEach(model => {
        const isDeployed = model.is_deployed;
        const accuracy = model.accuracy ? model.accuracy.toFixed(1) : 'N/A';
        const modelType = model.model_type || 'Unknown';
        const trainedDate = model.trained_at ? new Date(model.trained_at).toLocaleDateString() : 'N/A';
        const statusClass = isDeployed ? 'deployed' : 'inactive';
        const statusText = isDeployed ? 'Deployed' : 'Inactive';
        const statusColor = isDeployed ? '#22c55e' : '#888';

        html += `
            <div class="model-card" style="background: var(--color-card-bg); border: 1px solid var(--color-card-border); border-radius: 8px; padding: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: var(--color-text-primary); font-size: 1rem;">${model.name || 'Unnamed Model'}</h4>
                    <span style="font-size: 0.75rem; padding: 3px 8px; border-radius: 12px; background: ${isDeployed ? 'rgba(34, 197, 94, 0.2)' : 'rgba(100, 100, 100, 0.2)'}; color: ${statusColor}; font-weight: 600;">
                        ${statusText}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: var(--color-text-secondary); margin-bottom: 10px;">
                    ${modelType} Model
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;">
                    <div>
                        <span style="color: var(--color-text-secondary);">Accuracy:</span>
                        <span style="font-weight: 600; color: ${parseFloat(accuracy) >= 60 ? '#22c55e' : parseFloat(accuracy) >= 50 ? '#f59e0b' : '#ef4444'};">${accuracy}%</span>
                    </div>
                    <div>
                        <span style="color: var(--color-text-secondary);">Trained:</span>
                        <span style="font-weight: 500;">${trainedDate}</span>
                    </div>
                </div>
                ${model.features_used ? `
                <div style="margin-top: 10px; font-size: 0.8rem; color: var(--color-text-secondary);">
                    Features: ${Array.isArray(model.features_used) ? model.features_used.slice(0, 3).join(', ') + (model.features_used.length > 3 ? '...' : '') : 'N/A'}
                </div>
                ` : ''}
            </div>
        `;
    });

    html += '</div>';
    modelsListEl.innerHTML = html;
}

// Display predictions in UI
function displayPredictions(predictions) {
    console.log('Predictions:', predictions);

    const container = document.getElementById('modelPredictions');
    if (!container) {
        console.warn('modelPredictions container not found');
        return;
    }

    if (!predictions || predictions.length === 0) {
        container.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 30px; color: var(--color-text-secondary);">
                <p style="margin: 0;">No predictions available yet.</p>
                <p style="margin: 5px 0 0 0; font-size: 0.85rem;">Deploy a model to start generating predictions.</p>
            </div>
        `;
        return;
    }

    // Group predictions by model
    const modelGroups = {};
    predictions.forEach(pred => {
        const modelName = pred.model_name || 'Unknown Model';
        if (!modelGroups[modelName]) {
            modelGroups[modelName] = [];
        }
        modelGroups[modelName].push(pred);
    });

    let html = '';
    Object.entries(modelGroups).forEach(([modelName, preds]) => {
        const latestPred = preds[0]; // Most recent prediction
        const predValue = latestPred.predicted_value;
        const confidence = latestPred.confidence ? (latestPred.confidence * 100).toFixed(0) : 'N/A';
        const predType = latestPred.prediction_type || 'price';

        // Determine signal based on prediction type
        let signal = 'HOLD';
        let signalClass = 'hold';
        if (predType === 'direction') {
            if (predValue > 0.6) {
                signal = 'BUY';
                signalClass = 'buy';
            } else if (predValue < 0.4) {
                signal = 'SELL';
                signalClass = 'sell';
            }
        } else if (predType === 'price' && latestPred.actual_value) {
            const diff = predValue - latestPred.actual_value;
            if (diff > 0) {
                signal = 'BUY';
                signalClass = 'buy';
            } else if (diff < 0) {
                signal = 'SELL';
                signalClass = 'sell';
            }
        }

        html += `
            <div class="card model-prediction-card">
                <div class="prediction-outcome ${signalClass}">
                    ${signal}
                </div>
                <h4 style="margin: 0 0 10px 0; color: var(--color-text-primary);">${modelName}</h4>
                <p style="margin: 0 0 15px 0; color: var(--color-text-secondary); font-size: 0.9rem;">
                    ${predType === 'price' ? 'Price prediction model' : 'Direction prediction model'}
                </p>
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.9rem;">Predicted:</span>
                        <span style="font-weight: 600; color: var(--color-primary-accent);">
                            ${predType === 'price' ? '$' + parseFloat(predValue).toLocaleString(undefined, {maximumFractionDigits: 2}) : (predValue * 100).toFixed(1) + '%'}
                        </span>
                    </div>
                    ${latestPred.actual_value ? `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.9rem;">Actual:</span>
                        <span style="font-weight: 600;">
                            ${predType === 'price' ? '$' + parseFloat(latestPred.actual_value).toLocaleString(undefined, {maximumFractionDigits: 2}) : (latestPred.actual_value * 100).toFixed(1) + '%'}
                        </span>
                    </div>
                    ` : ''}
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 0.9rem;">Confidence:</span>
                        <span style="font-weight: 600; color: var(--color-primary-accent);">${confidence}%</span>
                    </div>
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: var(--color-text-secondary);">
                    Last updated: ${latestPred.timestamp ? new Date(latestPred.timestamp).toLocaleString() : 'N/A'}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Display trading signal in UI
function displaySignal(signal) {
    console.log('Signal:', signal);
    // TODO: Update UI with signal (buy/sell/hold indicator)
}

// Display statistics in UI
function displayStats(stats) {
    console.log('Stats:', stats);

    // Update Current Price
    const currentPriceEl = document.getElementById('currentPrice');
    if (currentPriceEl && stats.close) {
        currentPriceEl.textContent = '$' + parseFloat(stats.close).toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 4
        });
    }

    // Update Price Change
    const priceChangeEl = document.getElementById('priceChange');
    if (priceChangeEl && stats.change_percent !== undefined) {
        const change = parseFloat(stats.change_percent);
        priceChangeEl.textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
        priceChangeEl.style.color = change >= 0 ? '#22c55e' : '#ef4444';
    }

    // Update Support Level (low of the period)
    const supportEl = document.getElementById('supportLevel');
    if (supportEl && stats.low) {
        supportEl.textContent = '$' + parseFloat(stats.low).toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 4
        });
    }

    // Update Resistance Level (high of the period)
    const resistanceEl = document.getElementById('resistanceLevel');
    if (resistanceEl && stats.high) {
        resistanceEl.textContent = '$' + parseFloat(stats.high).toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 4
        });
    }
}

// Handle timeframe selection
function onTimeframeSelected(event) {
    const buttons = document.querySelectorAll('.timeframe-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    currentTimeframe = event.target.dataset.timeframe;

    if (selectedInstrument) {
        updateChart(selectedInstrument.symbol, currentTimeframe);
    }
}

// Auto-refresh chart data
function startAutoRefresh(intervalMs = 60000) { // Default: 1 minute
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
    }

    chartUpdateInterval = setInterval(() => {
        if (selectedInstrument) {
            console.log('Auto-refreshing chart data...');
            updateChart(selectedInstrument.symbol, currentTimeframe);
        }
    }, intervalMs);
}

function stopAutoRefresh() {
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
        chartUpdateInterval = null;
    }
}

// Logout handler
function handleLogout() {
    localStorage.removeItem('token');
    window.location.href = '/';
}

// Check subscription status
async function checkSubscriptionStatus() {
    try {
        const status = await apiCall('/api/subscription/status');
        return status;
    } catch (error) {
        console.error('Error checking subscription status:', error);
        return { is_active: false, status: 'unknown' };
    }
}

// Show subscription expired message
function showSubscriptionExpiredMessage() {
    const dropdown = document.getElementById('instrumentsDropdown');
    if (dropdown) {
        dropdown.disabled = true;
        dropdown.innerHTML = '<option value="">-- Subscription Required --</option>';
    }

    // Create subscription banner if it doesn't exist
    let banner = document.getElementById('subscriptionBanner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'subscriptionBanner';
        banner.style.cssText = `
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.2));
            border: 1px solid rgba(59, 130, 246, 0.5);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
        `;
        banner.innerHTML = `
            <div>
                <strong style="color: #3B82F6;">Subscription Expired</strong>
                <p style="margin: 5px 0 0 0; color: #B0B0B0; font-size: 14px;">
                    Your subscription has expired. Renew to access trading data and predictions.
                </p>
            </div>
            <a href="/subscription" class="btn btn-primary" style="padding: 10px 20px; text-decoration: none;">
                Renew Subscription
            </a>
        `;

        // Insert at the top of the main content area
        const mainContent = document.querySelector('.dashboard-content') || document.querySelector('main') || document.body.firstChild;
        if (mainContent && mainContent.parentNode) {
            mainContent.parentNode.insertBefore(banner, mainContent);
        }
    }
}

// Initialize dashboard
async function initDashboard() {
    console.log('Initializing dashboard...');

    // Check authentication
    if (!isAuthenticated()) {
        window.location.href = '/auth';
        return;
    }

    // Set up logout button
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    // Check subscription status before loading instruments
    const subscriptionStatus = await checkSubscriptionStatus();
    console.log('Subscription status:', subscriptionStatus);

    if (!subscriptionStatus.is_active) {
        showSubscriptionExpiredMessage();
        console.log('Dashboard initialized with expired subscription');
        return; // Don't load instruments if subscription is expired
    }

    // Load instruments
    await loadInstruments();

    // Set up event listeners
    const instrumentDropdown = document.getElementById('instrumentsDropdown');
    if (instrumentDropdown) {
        instrumentDropdown.addEventListener('change', onInstrumentSelected);
    }

    const timeframeButtons = document.querySelectorAll('.timeframe-btn');
    timeframeButtons.forEach(btn => {
        btn.addEventListener('click', onTimeframeSelected);
    });

    // Start auto-refresh
    startAutoRefresh();

    console.log('Dashboard initialized successfully');
}

// Connection status monitoring
let connectionStatusInterval = null;

async function checkConnection() {
    const statusElement = document.getElementById('connectionStatus');
    const textElement = document.getElementById('connectionText');

    if (!statusElement || !textElement) return;

    try {
        const response = await fetch(`${API_BASE}/api/trading-platform/health`, {
            headers: {
                'Authorization': `Bearer ${getAuthToken()}`
            }
        });

        if (response.ok) {
            const data = await response.json();

            if (data.tradingPlatform && data.tradingPlatform.status === 'healthy') {
                statusElement.classList.remove('disconnected');
                statusElement.classList.add('connected');
                textElement.textContent = 'Connected';
            } else {
                statusElement.classList.remove('connected');
                statusElement.classList.add('disconnected');
                textElement.textContent = 'Trading Platform Unavailable';
            }
        } else {
            statusElement.classList.remove('connected');
            statusElement.classList.add('disconnected');
            textElement.textContent = 'Disconnected';
        }
    } catch (error) {
        statusElement.classList.remove('connected');
        statusElement.classList.add('disconnected');
        textElement.textContent = 'Offline';
    }
}

function startConnectionMonitoring() {
    // Check immediately
    checkConnection();

    // Check every 30 seconds
    if (connectionStatusInterval) {
        clearInterval(connectionStatusInterval);
    }

    connectionStatusInterval = setInterval(checkConnection, 30000);
}

function stopConnectionMonitoring() {
    if (connectionStatusInterval) {
        clearInterval(connectionStatusInterval);
        connectionStatusInterval = null;
    }
}

/**
 * Load technical indicators for selected instrument
 */
async function loadIndicators(symbol, limit = 200) {
    try {
        console.log(`Loading indicators for ${symbol}...`);

        const indicators = await apiCall(`/api/instruments/${symbol}/indicators?limit=${limit}`);

        if (indicators) {
            displayIndicators(indicators);
        }

        return indicators;

    } catch (error) {
        console.error('Error loading indicators:', error);
        return null;
    }
}

/**
 * Display technical indicators in the Analysis tab
 */
function displayIndicators(indicators) {
    // Update the Analysis tab content
    const analysisTab = document.getElementById('analysisTab');

    if (!analysisTab) {
        console.warn('Analysis tab not found in DOM');
        return;
    }

    // Also update RSI in the summary tab
    const rsiIndicator = indicators.indicators.find(i => i.name.toUpperCase().includes('RSI'));
    if (rsiIndicator) {
        const rsiValueEl = document.getElementById('rsiValue');
        const rsiStatusEl = document.getElementById('rsiStatus');
        if (rsiValueEl) {
            rsiValueEl.textContent = parseFloat(rsiIndicator.value).toFixed(1);
        }
        if (rsiStatusEl) {
            const rsi = parseFloat(rsiIndicator.value);
            if (rsi < 30) {
                rsiStatusEl.textContent = 'Oversold';
                rsiStatusEl.style.color = '#22c55e';
            } else if (rsi > 70) {
                rsiStatusEl.textContent = 'Overbought';
                rsiStatusEl.style.color = '#ef4444';
            } else {
                rsiStatusEl.textContent = 'Neutral';
                rsiStatusEl.style.color = '#f59e0b';
            }
        }
    }

    // Group indicators by category
    const categorized = {};
    indicators.indicators.forEach(indicator => {
        if (!categorized[indicator.category]) {
            categorized[indicator.category] = [];
        }
        categorized[indicator.category].push(indicator);
    });

    // Category labels
    const categoryLabels = {
        'moving_average': 'Moving Averages',
        'oscillator': 'Oscillators & MACD',
        'volatility': 'Volatility Indicators',
        'trend': 'Trend Indicators',
        'momentum': 'Momentum Indicators',
        'volume': 'Volume Indicators',
        'support_resistance': 'Support & Resistance',
        'divergence': 'Divergences',
        'pattern': 'Pattern Recognition',
        'other': 'Other Indicators'
    };

    // Sort categories in preferred order
    const categoryOrder = [
        'moving_average',
        'oscillator',
        'volatility',
        'trend',
        'momentum',
        'volume',
        'support_resistance',
        'divergence',
        'pattern',
        'other'
    ];

    // Build HTML
    let indicatorsHtml = `
        <div class="indicators-container">
            <div class="indicators-header">
                <h3>Technical Indicators</h3>
                <div class="indicators-meta">
                    <span>${indicators.total_indicators} indicators</span>
                    <span class="separator">•</span>
                    <span>${new Date(indicators.timestamp).toLocaleString()}</span>
                </div>
            </div>
    `;

    categoryOrder.forEach(category => {
        if (categorized[category] && categorized[category].length > 0) {
            const categoryName = categoryLabels[category] || category;
            const categoryIndicators = categorized[category];

            indicatorsHtml += `
                <div class="indicator-category">
                    <h4 class="category-title">${categoryName} <span class="category-count">(${categoryIndicators.length})</span></h4>
                    <div class="indicator-grid">
            `;

            categoryIndicators.forEach(indicator => {
                const value = indicator.value !== null && indicator.value !== undefined
                    ? formatIndicatorValue(indicator.value, indicator.name)
                    : 'N/A';

                const valueClass = getIndicatorValueClass(indicator.value, indicator.name);

                indicatorsHtml += `
                    <div class="indicator-item ${valueClass}">
                        <span class="indicator-name">${formatIndicatorName(indicator.name)}</span>
                        <span class="indicator-value">${value}</span>
                    </div>
                `;
            });

            indicatorsHtml += `
                    </div>
                </div>
            `;
        }
    });

    indicatorsHtml += `</div>`;

    analysisTab.innerHTML = indicatorsHtml;

    // Also update the summary tab with indicator analysis
    displayIndicatorSummary(indicators);
}

/**
 * Display indicator summary in the Analysis Summary tab
 * Provides a formatted overview of key indicators with signals
 */
function displayIndicatorSummary(indicators) {
    const indicatorMap = {};
    indicators.indicators.forEach(ind => {
        indicatorMap[ind.name] = ind.value;
    });

    // Helper to get value by name pattern
    const getVal = (pattern) => {
        const key = Object.keys(indicatorMap).find(k => k.includes(pattern));
        return key ? indicatorMap[key] : null;
    };

    // Colors
    const bullish = '#22c55e';
    const bearish = '#ef4444';
    const neutral = '#f59e0b';

    // Calculate overall signal based on multiple indicators
    let bullishCount = 0;
    let bearishCount = 0;
    let totalSignals = 0;

    // Moving Averages Analysis
    const priceSma20 = getVal('Price vs SMA20');
    const priceSma50 = getVal('Price vs SMA50');
    const ema12 = getVal('EMA 12');
    const ema26 = getVal('EMA 26');

    // SMA 20 Signal
    const sma20SignalEl = document.getElementById('sma20Signal');
    if (sma20SignalEl && priceSma20 !== null) {
        if (priceSma20 > 0) {
            sma20SignalEl.textContent = '▲ Above';
            sma20SignalEl.style.color = bullish;
            bullishCount++;
        } else {
            sma20SignalEl.textContent = '▼ Below';
            sma20SignalEl.style.color = bearish;
            bearishCount++;
        }
        totalSignals++;
    }

    // SMA 50 Signal
    const sma50SignalEl = document.getElementById('sma50Signal');
    if (sma50SignalEl && priceSma50 !== null) {
        if (priceSma50 > 0) {
            sma50SignalEl.textContent = '▲ Above';
            sma50SignalEl.style.color = bullish;
            bullishCount++;
        } else {
            sma50SignalEl.textContent = '▼ Below';
            sma50SignalEl.style.color = bearish;
            bearishCount++;
        }
        totalSignals++;
    }

    // EMA Signal (12 vs 26 crossover)
    const emaSignalEl = document.getElementById('emaSignal');
    if (emaSignalEl && ema12 !== null && ema26 !== null) {
        if (ema12 > ema26) {
            emaSignalEl.textContent = '▲ Bullish';
            emaSignalEl.style.color = bullish;
            bullishCount++;
        } else {
            emaSignalEl.textContent = '▼ Bearish';
            emaSignalEl.style.color = bearish;
            bearishCount++;
        }
        totalSignals++;
    }

    // MA Trend overall
    const maTrendEl = document.getElementById('maTrend');
    if (maTrendEl) {
        const maScore = (priceSma20 !== null && priceSma20 > 0 ? 1 : 0) +
                        (priceSma50 !== null && priceSma50 > 0 ? 1 : 0) +
                        (ema12 !== null && ema26 !== null && ema12 > ema26 ? 1 : 0);
        if (maScore >= 2) {
            maTrendEl.textContent = '▲ BULLISH';
            maTrendEl.style.color = bullish;
        } else if (maScore <= 1) {
            maTrendEl.textContent = '▼ BEARISH';
            maTrendEl.style.color = bearish;
        } else {
            maTrendEl.textContent = '— NEUTRAL';
            maTrendEl.style.color = neutral;
        }
    }

    // Oscillators Analysis
    const rsi = getVal('RSI');
    const macdHist = getVal('MACD Histogram');
    const stochK = getVal('Stochastic %K');
    const stochD = getVal('Stochastic %D');

    // RSI Signal
    const rsiSignalEl = document.getElementById('rsiSignal');
    if (rsiSignalEl && rsi !== null) {
        if (rsi < 30) {
            rsiSignalEl.textContent = `${rsi.toFixed(0)} Oversold`;
            rsiSignalEl.style.color = bullish;
            bullishCount++;
        } else if (rsi > 70) {
            rsiSignalEl.textContent = `${rsi.toFixed(0)} Overbought`;
            rsiSignalEl.style.color = bearish;
            bearishCount++;
        } else {
            rsiSignalEl.textContent = `${rsi.toFixed(0)} Neutral`;
            rsiSignalEl.style.color = neutral;
        }
        totalSignals++;
    }

    // MACD Signal
    const macdSignalEl = document.getElementById('macdSignal');
    if (macdSignalEl && macdHist !== null) {
        if (macdHist > 0) {
            macdSignalEl.textContent = '▲ Bullish';
            macdSignalEl.style.color = bullish;
            bullishCount++;
        } else {
            macdSignalEl.textContent = '▼ Bearish';
            macdSignalEl.style.color = bearish;
            bearishCount++;
        }
        totalSignals++;
    }

    // Stochastic Signal
    const stochSignalEl = document.getElementById('stochSignal');
    if (stochSignalEl && stochK !== null) {
        if (stochK < 20) {
            stochSignalEl.textContent = `${stochK.toFixed(0)} Oversold`;
            stochSignalEl.style.color = bullish;
            bullishCount++;
        } else if (stochK > 80) {
            stochSignalEl.textContent = `${stochK.toFixed(0)} Overbought`;
            stochSignalEl.style.color = bearish;
            bearishCount++;
        } else {
            stochSignalEl.textContent = `${stochK.toFixed(0)} Neutral`;
            stochSignalEl.style.color = neutral;
        }
        totalSignals++;
    }

    // Oscillator Bias
    const oscillatorBiasEl = document.getElementById('oscillatorBias');
    if (oscillatorBiasEl) {
        let oscBullish = 0;
        let oscBearish = 0;
        if (rsi !== null) { rsi < 40 ? oscBullish++ : (rsi > 60 ? oscBearish++ : null); }
        if (macdHist !== null) { macdHist > 0 ? oscBullish++ : oscBearish++; }
        if (stochK !== null) { stochK < 40 ? oscBullish++ : (stochK > 60 ? oscBearish++ : null); }

        if (oscBullish > oscBearish) {
            oscillatorBiasEl.textContent = '▲ BULLISH';
            oscillatorBiasEl.style.color = bullish;
        } else if (oscBearish > oscBullish) {
            oscillatorBiasEl.textContent = '▼ BEARISH';
            oscillatorBiasEl.style.color = bearish;
        } else {
            oscillatorBiasEl.textContent = '— NEUTRAL';
            oscillatorBiasEl.style.color = neutral;
        }
    }

    // Volatility & Volume Analysis
    const bbPct = getVal('Bollinger %B');
    const atr = getVal('ATR');
    const volRatio = getVal('Volume Ratio');
    const adx = getVal('ADX');

    // Bollinger %B Signal
    const bbSignalEl = document.getElementById('bbSignal');
    if (bbSignalEl && bbPct !== null) {
        if (bbPct < 0.2) {
            bbSignalEl.textContent = 'Near Lower';
            bbSignalEl.style.color = bullish;
        } else if (bbPct > 0.8) {
            bbSignalEl.textContent = 'Near Upper';
            bbSignalEl.style.color = bearish;
        } else {
            bbSignalEl.textContent = 'Middle';
            bbSignalEl.style.color = neutral;
        }
    }

    // ATR Value
    const atrValueEl = document.getElementById('atrValue');
    if (atrValueEl && atr !== null) {
        atrValueEl.textContent = atr.toFixed(2);
        atrValueEl.style.color = 'var(--color-text-primary)';
    }

    // Volume Signal
    const volumeSignalEl = document.getElementById('volumeSignal');
    if (volumeSignalEl && volRatio !== null) {
        if (volRatio > 1.5) {
            volumeSignalEl.textContent = '▲ High';
            volumeSignalEl.style.color = bullish;
        } else if (volRatio < 0.5) {
            volumeSignalEl.textContent = '▼ Low';
            volumeSignalEl.style.color = bearish;
        } else {
            volumeSignalEl.textContent = '— Normal';
            volumeSignalEl.style.color = neutral;
        }
    }

    // ADX Trend
    const adxTrendEl = document.getElementById('adxTrend');
    if (adxTrendEl && adx !== null) {
        if (adx > 25) {
            adxTrendEl.textContent = `${adx.toFixed(0)} Strong`;
            adxTrendEl.style.color = bullish;
        } else if (adx < 20) {
            adxTrendEl.textContent = `${adx.toFixed(0)} Weak`;
            adxTrendEl.style.color = neutral;
        } else {
            adxTrendEl.textContent = `${adx.toFixed(0)} Moderate`;
            adxTrendEl.style.color = neutral;
        }
    }

    // Overall Signal Badge
    const signalBadge = document.getElementById('signalBadge');
    const signalStrengthValue = document.getElementById('signalStrengthValue');

    if (signalBadge) {
        if (totalSignals > 0) {
            const bullishPct = (bullishCount / totalSignals) * 100;
            const bearishPct = (bearishCount / totalSignals) * 100;

            if (bullishPct >= 60) {
                signalBadge.textContent = 'BULLISH';
                signalBadge.style.background = 'rgba(34, 197, 94, 0.2)';
                signalBadge.style.color = bullish;
                signalBadge.style.border = `2px solid ${bullish}`;
            } else if (bearishPct >= 60) {
                signalBadge.textContent = 'BEARISH';
                signalBadge.style.background = 'rgba(239, 68, 68, 0.2)';
                signalBadge.style.color = bearish;
                signalBadge.style.border = `2px solid ${bearish}`;
            } else {
                signalBadge.textContent = 'NEUTRAL';
                signalBadge.style.background = 'rgba(245, 158, 11, 0.2)';
                signalBadge.style.color = neutral;
                signalBadge.style.border = `2px solid ${neutral}`;
            }

            if (signalStrengthValue) {
                const strength = Math.max(bullishPct, bearishPct);
                signalStrengthValue.textContent = `${strength.toFixed(0)}%`;
                signalStrengthValue.style.color = strength >= 70 ? bullish : (strength >= 50 ? neutral : 'var(--color-text-secondary)');
            }
        } else {
            // No indicators data available - show NO DATA state
            signalBadge.textContent = 'NO DATA';
            signalBadge.style.background = 'rgba(100, 100, 100, 0.2)';
            signalBadge.style.color = 'var(--color-text-secondary)';
            signalBadge.style.border = '2px solid var(--color-text-secondary)';

            if (signalStrengthValue) {
                signalStrengthValue.textContent = '--';
                signalStrengthValue.style.color = 'var(--color-text-secondary)';
            }
        }
    }
}

/**
 * Format indicator name for display
 */
function formatIndicatorName(name) {
    // Replace underscores with spaces
    return name.replace(/_/g, ' ');
}

/**
 * Format indicator value based on indicator type
 */
function formatIndicatorValue(value, indicatorName) {
    // For percentage indicators
    if (indicatorName.includes('%') || indicatorName.toUpperCase().includes('RSI')) {
        return value.toFixed(2) + '%';
    }

    // For divergence (boolean)
    if (indicatorName.toLowerCase().includes('divergence')) {
        return value === 1 ? 'Yes' : 'No';
    }

    // For prices and general values
    if (Math.abs(value) >= 1000) {
        return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    return value.toFixed(4);
}

/**
 * Get CSS class for indicator value (for coloring)
 */
function getIndicatorValueClass(value, indicatorName) {
    if (value === null || value === undefined) {
        return '';
    }

    const name = indicatorName.toUpperCase();

    // RSI: Overbought/Oversold
    if (name.includes('RSI')) {
        if (value >= 70) return 'value-overbought';
        if (value <= 30) return 'value-oversold';
        return '';
    }

    // Stochastic: Similar to RSI
    if (name.includes('%K') || name.includes('%D')) {
        if (value >= 80) return 'value-overbought';
        if (value <= 20) return 'value-oversold';
        return '';
    }

    // Divergence: Highlight if present
    if (name.includes('DIVERGENCE')) {
        return value === 1 ? 'value-divergence' : '';
    }

    return '';
}

/**
 * Hide indicators display
 */
function hideIndicators() {
    const analysisTab = document.getElementById('analysis');
    if (analysisTab) {
        analysisTab.innerHTML = '<p class="text-muted">Select an instrument to view technical indicators.</p>';
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
    stopConnectionMonitoring();
    if (currentChart) {
        currentChart.destroy();
    }
});

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initDashboard();
        startConnectionMonitoring();
    });
} else {
    initDashboard();
    startConnectionMonitoring();
}

// Export for global access
window.Dashboard = {
    loadInstruments,
    loadChartData,
    updateChart,
    selectInstrumentFromSearch,
    handleLogout,
    startAutoRefresh,
    stopAutoRefresh,
    checkConnection,
    startConnectionMonitoring,
    stopConnectionMonitoring,
    loadIndicators,
    displayIndicators,
    hideIndicators
};
