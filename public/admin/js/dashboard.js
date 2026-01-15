/**
 * Dashboard JavaScript
 * Complete implementation with profile metrics, charts, tables, and real-time updates
 */

// Global state
let state = {
    selectedProfileId: null,
    selectedProfile: null,
    profiles: [],
    systemStatus: null,
    performance: null,
    positions: [],
    latestMetrics: null,
    metricsHistory: [],
    predictions: [],
    models: [],
    chart: null,
    chartSeries: null
};

// Polling intervals
let systemStatusInterval = null;
let performanceInterval = null;
let positionsInterval = null;
let metricsInterval = null;

/**
 * Initialize dashboard
 */
async function initializeDashboard() {
    try {
        // Check authentication
        const token = localStorage.getItem('adminToken');
        if (!token) {
            window.location.href = '/auth';
            return;
        }

        // Load initial data
        await Promise.all([
            loadProfiles(),
            loadSystemStatus(),
            loadPerformance(),
            loadPositions()
        ]);

        // Setup event listeners
        setupEventListeners();

        // Start polling intervals
        startPolling();

        // Setup WebSocket auto-refresh for profiles table
        setupProfilesTableAutoRefresh();

        // Show initial state
        updateUI();

    } catch (error) {
        console.error('Dashboard initialization error:', error);
        if (error.message && (error.message.includes('401') || error.message.includes('403'))) {
            window.location.href = '/auth';
        }
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    const profileSelector = document.getElementById('profileSelector');
    profileSelector.addEventListener('change', onProfileChange);
}

/**
 * Load all profiles
 */
async function loadProfiles() {
    try {
        const data = await apiService.getProfiles({ is_active: true });
        state.profiles = data.profiles || data || [];

        // Populate profile selector
        const profileSelector = document.getElementById('profileSelector');
        profileSelector.innerHTML = '<option value="">Select a trading profile...</option>';

        state.profiles.forEach(profile => {
            const option = document.createElement('option');
            option.value = profile.id;
            option.textContent = `${profile.name} (${profile.symbol})`;
            profileSelector.appendChild(option);
        });

        // Render profiles table
        renderProfilesTable();

    } catch (error) {
        console.error('Error loading profiles:', error);
    }
}

/**
 * Load system status
 */
async function loadSystemStatus() {
    try {
        state.systemStatus = await apiService.getSystemStatus();
    } catch (error) {
        console.error('Error loading system status:', error);
        // Set default if endpoint doesn't exist
        state.systemStatus = { status: 'UNKNOWN', active_positions: 0, total_pnl: 0 };
    }
}

/**
 * Load performance metrics
 */
async function loadPerformance() {
    try {
        state.performance = await apiService.getPerformance({ period: '24h' });
    } catch (error) {
        console.error('Error loading performance:', error);
        // Set defaults if endpoint doesn't exist
        state.performance = { total_pnl: 0, total_trades: 0, win_rate: 0 };
    }
}

/**
 * Load active positions
 */
async function loadPositions() {
    try {
        state.positions = await apiService.getPositions('OPEN');
        if (state.positions && state.positions.length > 0) {
            renderPositionsTable();
        }
    } catch (error) {
        console.error('Error loading positions:', error);
        state.positions = [];
    }
}

/**
 * Handle profile selection change
 */
async function onProfileChange() {
    const profileSelector = document.getElementById('profileSelector');
    const profileId = profileSelector.value;

    if (!profileId) {
        state.selectedProfileId = null;
        state.selectedProfile = null;
        state.latestMetrics = null;
        state.metricsHistory = [];
        state.predictions = [];
        state.models = [];
        updateUI();
        return;
    }

    state.selectedProfileId = parseInt(profileId);
    state.selectedProfile = state.profiles.find(p => p.id === state.selectedProfileId);

    // Load profile-specific data
    await Promise.all([
        loadLatestMetrics(),
        loadMetricsHistory(),
        loadPredictions(),
        loadModels()
    ]);

    updateUI();
}

/**
 * Load latest metrics for selected profile
 */
async function loadLatestMetrics() {
    if (!state.selectedProfileId) return;

    try {
        state.latestMetrics = await apiService.getLatestMetrics(state.selectedProfileId);
    } catch (error) {
        console.error('Error loading latest metrics:', error);
        state.latestMetrics = null;
    }
}

/**
 * Load metrics history for chart
 */
async function loadMetricsHistory() {
    if (!state.selectedProfileId) return;

    try {
        const data = await apiService.getMetricsHistory(state.selectedProfileId, { hours: 24 });
        state.metricsHistory = data.metrics || data || [];
        renderChart();
    } catch (error) {
        console.error('Error loading metrics history:', error);
        state.metricsHistory = [];
    }
}

/**
 * Load predictions for selected profile
 */
async function loadPredictions() {
    if (!state.selectedProfileId) return;

    try {
        const data = await apiService.getPredictions(state.selectedProfileId, { hours: 6 });
        state.predictions = data.predictions || data || [];
    } catch (error) {
        console.error('Error loading predictions:', error);
        state.predictions = [];
    }
}

/**
 * Load models for selected profile
 */
async function loadModels() {
    if (!state.selectedProfileId) return;

    try {
        const data = await apiService.getModels(state.selectedProfileId);
        state.models = (data.models || data || []).filter(m => m.is_deployed);
    } catch (error) {
        console.error('Error loading models:', error);
        state.models = [];
    }
}

/**
 * Update UI based on current state
 */
function updateUI() {
    // Update info alert
    const infoAlert = document.getElementById('infoAlert');
    const profileMetricsSection = document.getElementById('profileMetricsSection');
    const chartSection = document.getElementById('chartSection');

    if (!state.selectedProfileId) {
        infoAlert.style.display = 'flex';
        profileMetricsSection.style.display = 'none';
        chartSection.style.display = 'none';
    } else {
        infoAlert.style.display = 'none';
        profileMetricsSection.style.display = 'block';
        chartSection.style.display = 'block';
        renderProfileMetrics();
        renderModelsAndPredictions();
    }

    // Update system metrics
    updateSystemMetrics();

    // Update positions section
    const positionsSection = document.getElementById('positionsSection');
    if (state.positions && state.positions.length > 0) {
        positionsSection.style.display = 'block';
    } else {
        positionsSection.style.display = 'none';
    }
}

/**
 * Render profile-specific metrics
 */
function renderProfileMetrics() {
    if (!state.selectedProfile) return;

    const title = document.getElementById('profileMetricsTitle');
    title.textContent = `Profile Metrics - ${state.selectedProfile.name}`;

    const grid = document.getElementById('profileMetricsGrid');

    const pnlValue = state.selectedProfile.total_pnl || 0;
    const pnlColor = pnlValue >= 0 ? 'success' : 'error';

    const winRate = (state.selectedProfile.win_rate || 0) * 100;
    const sharpe = state.selectedProfile.sharpe_ratio || 0;
    const sharpeColor = sharpe > 1.5 ? 'success' : 'warning';

    const currentPrice = state.latestMetrics?.current_price || state.selectedProfile.current_price || 0;
    const priceChange24h = state.latestMetrics?.price_change_24h || state.selectedProfile.price_change_24h || 0;
    const priceChangeColor = priceChange24h >= 0 ? 'success' : 'error';

    grid.innerHTML = `
        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-title">Profile P&L</span>
                <i class="fas fa-wallet stat-icon ${pnlColor}"></i>
            </div>
            <div class="stat-value" style="color: var(--color-${pnlColor});">$${pnlValue.toFixed(2)}</div>
            <div class="stat-subtitle">${state.selectedProfile.symbol}</div>
        </div>

        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-title">Win Rate</span>
                <i class="fas fa-chart-line stat-icon primary"></i>
            </div>
            <div class="stat-value">${winRate.toFixed(1)}%</div>
            <div class="stat-subtitle">${state.selectedProfile.total_trades || 0} trades</div>
        </div>

        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-title">Sharpe Ratio</span>
                <i class="fas fa-chart-bar stat-icon ${sharpeColor}"></i>
            </div>
            <div class="stat-value" style="color: var(--color-${sharpeColor});">${sharpe.toFixed(2)}</div>
            <div class="stat-subtitle">Risk-adjusted returns</div>
        </div>

        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-title">Current Price</span>
                <i class="fas fa-dollar-sign stat-icon primary"></i>
            </div>
            <div class="stat-value">$${currentPrice.toFixed(4)}</div>
            <div class="stat-change ${priceChangeColor}">
                ${priceChange24h >= 0 ? '+' : ''}${priceChange24h.toFixed(2)}%
            </div>
        </div>
    `;
}

/**
 * Update system-wide metrics
 */
function updateSystemMetrics() {
    // Total P&L
    const totalPnl = state.performance?.total_pnl || 0;
    const totalPnlEl = document.getElementById('totalPnl');
    totalPnlEl.textContent = `$${totalPnl.toFixed(2)}`;
    totalPnlEl.style.color = totalPnl >= 0 ? 'var(--color-success)' : 'var(--color-error)';

    // Active Positions
    const activePositions = state.positions?.length || 0;
    document.getElementById('activePositions').textContent = activePositions;

    // System Status
    const systemStatus = state.systemStatus?.status || 'UNKNOWN';
    document.getElementById('systemStatus').textContent = systemStatus;

    // Active Profiles
    const activeProfilesCount = state.profiles.filter(p => p.is_active).length;
    document.getElementById('activeProfiles').textContent = activeProfilesCount;
    document.getElementById('totalProfilesText').textContent = `${state.profiles.length} total`;
}

/**
 * Render price chart using lightweight-charts
 */
function renderChart() {
    if (!state.metricsHistory || state.metricsHistory.length === 0) {
        document.getElementById('priceChart').style.display = 'none';
        document.getElementById('noChartData').style.display = 'flex';
        return;
    }

    document.getElementById('priceChart').style.display = 'block';
    document.getElementById('noChartData').style.display = 'none';

    const chartContainer = document.getElementById('priceChart');
    const chartTitle = document.getElementById('chartTitle');
    chartTitle.textContent = `Price History - ${state.selectedProfile?.symbol || ''}`;

    // Clear previous chart
    chartContainer.innerHTML = '';

    // Create new chart
    state.chart = LightweightCharts.createChart(chartContainer, {
        layout: {
            background: { color: 'transparent' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2B2B43' },
            horzLines: { color: '#363C4E' },
        },
        width: chartContainer.clientWidth,
        height: 350,
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
        },
    });

    // Add line series
    state.chartSeries = state.chart.addLineSeries({
        color: '#3B82F6',
        lineWidth: 2,
    });

    // Prepare data
    const chartData = state.metricsHistory
        .map(m => ({
            time: new Date(m.timestamp).getTime() / 1000,
            value: m.current_price
        }))
        .sort((a, b) => a.time - b.time);

    state.chartSeries.setData(chartData);
    state.chart.timeScale().fitContent();

    // Handle window resize
    const resizeObserver = new ResizeObserver(entries => {
        if (state.chart) {
            state.chart.applyOptions({ width: chartContainer.clientWidth });
        }
    });
    resizeObserver.observe(chartContainer);
}

/**
 * Render models and predictions panel
 */
function renderModelsAndPredictions() {
    const modelsContent = document.getElementById('modelsContent');

    let html = '';

    // Deployed Models Section
    if (state.models && state.models.length > 0) {
        html += `
            <div style="margin-bottom: 24px;">
                <h4 style="color: var(--color-text-secondary); font-size: 0.9rem; margin-bottom: 12px;">
                    Deployed Models (${state.models.length})
                </h4>
        `;

        state.models.slice(0, 3).forEach(model => {
            const accuracy = model.test_accuracy ? model.test_accuracy.toFixed(1) : 'N/A';
            html += `
                <div class="model-item">
                    <div class="model-name">${model.model_name}</div>
                    <div class="model-details">
                        <span>Type: ${model.model_type.toUpperCase()}</span>
                        <span>Accuracy: ${accuracy}%</span>
                    </div>
                </div>
            `;
        });

        html += '</div>';
    } else {
        html += `
            <div style="margin-bottom: 24px;">
                <h4 style="color: var(--color-text-secondary); font-size: 0.9rem; margin-bottom: 12px;">
                    Deployed Models
                </h4>
                <p style="color: var(--color-text-secondary); font-size: 0.9rem;">No deployed models</p>
            </div>
        `;
    }

    // Predictions Section
    if (state.predictions && state.predictions.length > 0) {
        html += `
            <div>
                <h4 style="color: var(--color-text-secondary); font-size: 0.9rem; margin-bottom: 12px;">
                    Latest Predictions
                </h4>
        `;

        state.predictions.slice(0, 2).forEach(pred => {
            const signal = pred.signal || 'HOLD';
            const signalClass = signal.toLowerCase();
            const target = pred.price_prediction?.toFixed(4) || 'N/A';
            const confidence = (pred.confidence || 0) * 100;

            html += `
                <div class="prediction-item">
                    <div class="prediction-header">
                        <div class="prediction-target">Target: $${target}</div>
                        <span class="signal-badge ${signalClass}">${signal}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--color-text-secondary); margin-bottom: 8px;">
                        ${pred.prediction_horizon || '1h'} horizon
                    </div>
                    ${pred.confidence ? `
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%;"></div>
                        </div>
                    ` : ''}
                </div>
            `;
        });

        html += '</div>';
    }

    modelsContent.innerHTML = html || '<p style="color: var(--color-text-secondary);">No data available</p>';
}

/**
 * Render profiles summary table
 */
function renderProfilesTable() {
    const tbody = document.getElementById('profilesTableBody');

    if (!state.profiles || state.profiles.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; padding: 40px; color: var(--color-text-secondary);">
                    No profiles found
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = state.profiles.slice(0, 10).map(profile => {
        // Current Price - show N/A if missing
        const currentPrice = profile.current_price ? profile.current_price.toFixed(4) : 'N/A';

        // 24h Change - show em dash if missing, otherwise format with +/-
        const priceChange = profile.price_change_24h;
        const hasChange = priceChange !== null && priceChange !== undefined;
        const priceChangeClass = hasChange && priceChange >= 0 ? 'text-success' : 'text-error';
        const priceChangeText = hasChange ? `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%` : '—';

        // Total P&L - show em dash if null or show actual value (including 0.00)
        const pnl = profile.total_pnl;
        const hasPnl = pnl !== null && pnl !== undefined;
        const pnlClass = hasPnl && pnl >= 0 ? 'text-success' : 'text-error';
        const pnlText = hasPnl ? `$${pnl.toFixed(2)}` : '—';

        // Win Rate - show em dash if no trades, otherwise show percentage
        const winRate = profile.win_rate;
        const hasWinRate = winRate !== null && winRate !== undefined && profile.total_trades > 0;
        const winRateText = hasWinRate ? `${(winRate * 100).toFixed(1)}%` : '—';

        const activeModels = profile.active_models || 0;
        const deployedModels = profile.deployed_models || 0;

        return `
            <tr>
                <td><strong>${profile.name}</strong></td>
                <td><span class="status-badge default">${profile.symbol}</span></td>
                <td>${profile.profile_type.toUpperCase()}</td>
                <td>$${currentPrice}</td>
                <td class="${priceChangeClass}">${priceChangeText}</td>
                <td class="${pnlClass}">${pnlText}</td>
                <td>${winRateText}</td>
                <td>
                    ${activeModels > 0 ? `<span class="status-badge default">${activeModels}</span>` : ''}
                    ${deployedModels > 0 ? `<span class="status-badge success">${deployedModels}</span>` : ''}
                    ${activeModels === 0 && deployedModels === 0 ? '<span style="color: var(--color-text-secondary);">0</span>' : ''}
                </td>
            </tr>
        `;
    }).join('');
}

/**
 * Render positions table
 */
function renderPositionsTable() {
    const tbody = document.getElementById('positionsTableBody');

    if (!state.positions || state.positions.length === 0) {
        return;
    }

    tbody.innerHTML = state.positions.slice(0, 5).map(position => {
        const pnl = position.unrealized_pnl || 0;
        const pnlClass = pnl >= 0 ? 'text-success' : 'text-error';

        const pnlPercent = position.entry_price && position.quantity
            ? ((pnl / (position.quantity * position.entry_price)) * 100).toFixed(2)
            : '0.00';

        const sideClass = position.side === 'BUY' ? 'success' : 'error';

        return `
            <tr>
                <td><strong>${position.symbol}</strong></td>
                <td><span class="status-badge ${sideClass}">${position.side}</span></td>
                <td>${position.quantity}</td>
                <td>$${position.entry_price.toFixed(4)}</td>
                <td>$${position.current_price?.toFixed(4) || '-'}</td>
                <td class="${pnlClass}">$${pnl.toFixed(2)}</td>
                <td class="${pnlClass}">${pnlPercent}%</td>
            </tr>
        `;
    }).join('');
}

/**
 * Start polling intervals
 */
function startPolling() {
    // System status - every 5 seconds
    systemStatusInterval = setInterval(async () => {
        await loadSystemStatus();
        updateSystemMetrics();
    }, 5000);

    // Performance - every 30 seconds
    performanceInterval = setInterval(async () => {
        await loadPerformance();
        updateSystemMetrics();
    }, 30000);

    // Positions - every 10 seconds
    positionsInterval = setInterval(async () => {
        await loadPositions();
        updateUI();
    }, 10000);

    // Metrics for selected profile - every 10 seconds
    metricsInterval = setInterval(async () => {
        if (state.selectedProfileId) {
            await loadLatestMetrics();
            renderProfileMetrics();
        }
    }, 10000);
}

/**
 * Stop polling intervals
 */
function stopPolling() {
    if (systemStatusInterval) clearInterval(systemStatusInterval);
    if (performanceInterval) clearInterval(performanceInterval);
    if (positionsInterval) clearInterval(positionsInterval);
    if (metricsInterval) clearInterval(metricsInterval);
}

/**
 * Setup auto-refresh for profiles table via WebSocket
 */
let lastProfilesRefresh = 0;
const PROFILES_REFRESH_DEBOUNCE = 2000; // 2 seconds debounce

function setupProfilesTableAutoRefresh() {
    // Listen for preprocessing completed events (fired when data updates)
    websocketService.on('preprocessingCompleted', async (data) => {
        // Debounce to prevent too many rapid updates
        const now = Date.now();
        if (now - lastProfilesRefresh < PROFILES_REFRESH_DEBOUNCE) {
            return;
        }
        lastProfilesRefresh = now;

        try {
            // Reload profiles data (which will trigger table re-render)
            await loadProfiles();
            console.log('✓ Profiles table auto-refreshed after data update');
        } catch (error) {
            console.error('Error auto-refreshing profiles table:', error);
        }
    });
}

/**
 * Refresh dashboard
 */
async function refreshDashboard() {
    const refreshBtn = event.target.closest('.icon-btn');
    const icon = refreshBtn.querySelector('i');

    icon.style.animation = 'spin 1s linear infinite';

    try {
        await Promise.all([
            loadProfiles(),
            loadSystemStatus(),
            loadPerformance(),
            loadPositions()
        ]);

        if (state.selectedProfileId) {
            await Promise.all([
                loadLatestMetrics(),
                loadMetricsHistory(),
                loadPredictions(),
                loadModels()
            ]);
        }

        updateUI();
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
    } finally {
        setTimeout(() => {
            icon.style.animation = '';
        }, 1000);
    }
}

/**
 * Cleanup on page unload
 */
window.addEventListener('beforeunload', () => {
    stopPolling();
    // Remove WebSocket listener
    websocketService.off('preprocessingCompleted');
    if (state.chart) {
        state.chart.remove();
    }
});

/**
 * Initialize on DOMContentLoaded
 */
document.addEventListener('DOMContentLoaded', initializeDashboard);
