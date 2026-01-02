/**
 * Data Management Page
 * Based on: MoneyMoney/other platform/Trading/frontend/src/pages/DataManagement.tsx
 */

// State
let state = {
    profiles: [],
    editingProfile: null,
    activeTab: 'profiles',
    dataSummary: []
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializePage();
    setupWebSocketListeners();
});

async function initializePage() {
    // Load profiles
    await loadProfiles();

    // Setup event listeners
    setupEventListeners();

    // Load data summary
    await loadDataSummary();
}

function setupEventListeners() {
    // Action buttons
    document.getElementById('newProfileBtn').addEventListener('click', () => openProfileDialog());
    document.getElementById('refreshBtn').addEventListener('click', refreshData);

    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tab = e.currentTarget.dataset.tab;
            switchTab(tab);
        });
    });
}

async function loadProfiles() {
    try {
        const profiles = await apiService.getProfiles({});
        state.profiles = profiles;
        renderProfilesTable();
    } catch (error) {
        showNotification('Failed to load profiles: ' + error.message, 'error');
    }
}

async function loadDataSummary() {
    try {
        const symbols = await apiService.getAvailableSymbols();
        state.dataSummary = symbols;
        renderDataSummary();
    } catch (error) {
        console.error('Failed to load data summary:', error);
    }
}

function renderProfilesTable() {
    const tbody = document.getElementById('profilesTableBody');
    const noProfiles = document.getElementById('noProfiles');

    if (state.profiles.length === 0) {
        tbody.innerHTML = '';
        noProfiles.style.display = 'block';
        return;
    }

    noProfiles.style.display = 'none';

    tbody.innerHTML = state.profiles.map(profile => `
        <tr>
            <td><strong>${profile.symbol}</strong></td>
            <td>${profile.name}</td>
            <td><span class="status-badge default">${profile.profile_type.toUpperCase()}</span></td>
            <td>${profile.exchange}</td>
            <td>
                ${profile.is_active ?
                    '<span class="status-badge success"><i class="fas fa-check-circle"></i> Active</span>' :
                    '<span class="status-badge warning"><i class="fas fa-pause-circle"></i> Inactive</span>'
                }
            </td>
            <td>${profile.model_count || 0}</td>
            <td>${profile.updated_at ? formatDateTimeFull(profile.updated_at) : 'N/A'}</td>
            <td>
                <div class="action-btns">
                    ${profile.has_data ? `
                        <button class="icon-btn" onclick="preprocessProfile('${profile.symbol}')" title="Calculate Indicators" style="background-color: rgba(59, 130, 246, 0.2); color: #3b82f6;">
                            <i class="fas fa-calculator"></i>
                        </button>
                    ` : ''}
                    <button class="icon-btn primary" onclick="editProfile(${profile.id})" title="Edit Profile">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="icon-btn danger" onclick="deleteProfile(${profile.id})" title="Delete Profile">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function renderDataSummary() {
    const container = document.getElementById('dataSummaryGrid');
    const noData = document.getElementById('noDataSummary');

    if (!state.dataSummary || state.dataSummary.length === 0) {
        container.innerHTML = '';
        noData.style.display = 'block';
        return;
    }

    noData.style.display = 'none';

    container.innerHTML = state.dataSummary.map(item => `
        <div class="summary-card glassmorphism">
            <h3>
                <i class="fas fa-chart-line"></i>
                ${item.symbol}
            </h3>
            <div class="summary-stats">
                <div class="summary-stat">
                    <span class="summary-stat-label">Total Records</span>
                    <span class="summary-stat-value">${formatNumber(item.record_count || 0)}</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-label">Start Date</span>
                    <span class="summary-stat-value">${item.start_date ? formatDateTimeFull(item.start_date) : 'N/A'}</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-label">End Date</span>
                    <span class="summary-stat-value">${item.end_date ? formatDateTimeFull(item.end_date) : 'N/A'}</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-label">Last Updated</span>
                    <span class="summary-stat-value">${item.last_updated ? formatDateTimeFull(item.last_updated) : 'N/A'}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function switchTab(tabName) {
    state.activeTab = tabName;

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        const contentId = content.id.replace('tab-', '');
        if (contentId === tabName) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

async function refreshData() {
    const btn = document.getElementById('refreshBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i>';

    try {
        await Promise.all([
            loadProfiles(),
            loadDataSummary()
        ]);
        showNotification('Data refreshed successfully', 'success');
    } catch (error) {
        showNotification('Failed to refresh data', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sync-alt"></i>';
    }
}

// Profile Dialog Functions
function openProfileDialog(profile = null) {
    state.editingProfile = profile;

    // Update dialog title
    document.getElementById('dialogTitle').textContent = profile ? 'Edit Profile' : 'New Profile';

    // Populate form
    const form = document.getElementById('profileForm');
    if (profile) {
        // Edit mode - populate with existing data
        form.symbol.value = profile.symbol;
        form.name.value = profile.name;
        form.profile_type.value = profile.profile_type;
        form.exchange.value = profile.exchange;
        form.description.value = profile.description || '';
        form.base_currency.value = profile.base_currency;
        form.quote_currency.value = profile.quote_currency;
        form.data_source.value = profile.data_source;
        form.timeframe.value = profile.timeframe;
        form.lookback_days.value = profile.lookback_days;
    } else {
        // Create mode - reset form
        form.reset();
        form.profile_type.value = 'crypto';
        form.exchange.value = 'binance';
        form.data_source.value = 'binance';
        form.timeframe.value = '1h';
        form.lookback_days.value = 365;
    }

    document.getElementById('profileDialog').style.display = 'flex';
}

function closeProfileDialog() {
    document.getElementById('profileDialog').style.display = 'none';
    state.editingProfile = null;
}

async function saveProfile() {
    const form = document.getElementById('profileForm');

    // Validate form
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    // Collect form data
    const formData = {
        symbol: form.symbol.value,
        name: form.name.value,
        profile_type: form.profile_type.value,
        exchange: form.exchange.value,
        description: form.description.value,
        base_currency: form.base_currency.value,
        quote_currency: form.quote_currency.value,
        data_source: form.data_source.value,
        timeframe: form.timeframe.value,
        lookback_days: parseInt(form.lookback_days.value)
    };

    // Disable save button
    const saveBtn = document.getElementById('saveProfileBtn');
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';

    try {
        if (state.editingProfile) {
            // Update existing profile
            await apiService.updateProfile(state.editingProfile.id, formData);
            showNotification('Profile updated successfully', 'success');
        } else {
            // Create new profile
            await apiService.createProfile(formData);
            showNotification('Profile created successfully', 'success');
        }

        closeProfileDialog();
        await loadProfiles();
    } catch (error) {
        showNotification('Failed to save profile: ' + error.message, 'error');
    } finally {
        saveBtn.disabled = false;
        saveBtn.innerHTML = '<i class="fas fa-save"></i> Save Profile';
    }
}

function editProfile(profileId) {
    const profile = state.profiles.find(p => p.id === profileId);
    if (profile) {
        openProfileDialog(profile);
    }
}

function deleteProfile(profileId) {
    const profile = state.profiles.find(p => p.id === profileId);
    if (!profile) return;

    showConfirmDialog(
        `Are you sure you want to delete the profile "${profile.name}" (${profile.symbol})? This action cannot be undone.`,
        async () => {
            try {
                await apiService.deleteProfile(profileId);
                showNotification('Profile deleted successfully', 'success');
                await loadProfiles();
            } catch (error) {
                showNotification('Failed to delete profile: ' + error.message, 'error');
            }
        }
    );
}

function setupWebSocketListeners() {
    // Check if websocketService is available
    if (typeof websocketService === 'undefined') {
        console.warn('WebSocket service not available');
        return;
    }

    // Preprocessing events
    websocketService.on('preprocessingStarted', (data) => {
        console.log('Preprocessing started:', data);
        showProgressNotification(data.symbol, 0, 'Starting indicator calculation...');
    });

    websocketService.on('preprocessingProgress', (data) => {
        console.log('Preprocessing progress:', data);
        const { symbol, progress, current_stage, records_processed, total_records } = data;

        let stageText = current_stage || `Processing ${progress}%`;
        if (records_processed && total_records) {
            stageText += ` (${records_processed.toLocaleString()}/${total_records.toLocaleString()} records)`;
        }

        showProgressNotification(symbol, progress, stageText);
    });

    websocketService.on('preprocessingCompleted', (data) => {
        console.log('Preprocessing completed:', data);
        const { symbol, total_records, indicators_count } = data;
        hideProgressNotification(symbol);
        showNotification(
            `Successfully calculated ${indicators_count} indicators for ${symbol} (${total_records.toLocaleString()} records)`,
            'success'
        );
        // Reload profiles to show updated status
        loadProfiles();
    });

    websocketService.on('preprocessingFailed', (data) => {
        console.error('Preprocessing failed:', data);
        const { symbol, error } = data;
        hideProgressNotification(symbol);
        showNotification(`Indicator calculation failed for ${symbol}: ${error}`, 'error');
    });
}

function showProgressNotification(symbol, progress, stage) {
    const notificationId = `progress-${symbol}`;
    let notification = document.getElementById(notificationId);

    if (!notification) {
        // Create progress notification element
        notification = document.createElement('div');
        notification.id = notificationId;
        notification.className = 'notification info progress-notification';
        notification.innerHTML = `
            <div class="notification-header">
                <strong>${symbol}</strong>
                <button class="close-btn" onclick="hideProgressNotification('${symbol}')">&times;</button>
            </div>
            <div class="notification-body">
                <div class="progress-text"></div>
                <div class="progress-bar-container">
                    <div class="progress-bar"></div>
                </div>
            </div>
        `;

        // Add to notification container
        let container = document.querySelector('.notification-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'notification-container';
            document.body.appendChild(container);
        }
        container.appendChild(notification);
    }

    // Update progress
    const progressText = notification.querySelector('.progress-text');
    const progressBar = notification.querySelector('.progress-bar');

    progressText.textContent = stage;
    progressBar.style.width = `${progress}%`;
}

function hideProgressNotification(symbol) {
    const notificationId = `progress-${symbol}`;
    const notification = document.getElementById(notificationId);
    if (notification) {
        notification.remove();
    }
}

async function preprocessProfile(symbol) {
    const profile = state.profiles.find(p => p.symbol === symbol);
    if (!profile) return;

    if (!profile.has_data) {
        showNotification('No data available for this profile. Please collect data first.', 'error');
        return;
    }

    try {
        const response = await apiService.preprocessData(symbol, false);

        if (response.status === 'skipped') {
            showNotification(`Indicators already calculated for ${symbol}. Use recalculate if needed.`, 'info');
        } else if (response.status === 'started') {
            // Show initial progress notification
            showProgressNotification(symbol, 0, 'Initializing indicator calculation...');
            // WebSocket events will handle the rest
        }
    } catch (error) {
        hideProgressNotification(symbol);
        showNotification(`Failed to calculate indicators: ${error.message}`, 'error');
    }
}

// ==================== DATA QUALITY TAB FUNCTIONS ====================

async function initQualityTab() {
    // Populate profile selector
    const select = document.getElementById('qualityProfileSelect');
    if (!select) return;

    // Clear existing options except first
    select.innerHTML = '<option value="">-- Choose a profile --</option>';

    // Add profiles that have data
    state.profiles
        .filter(p => p.has_data)
        .forEach(profile => {
            const option = document.createElement('option');
            option.value = profile.symbol;
            option.textContent = `${profile.symbol} - ${profile.name}`;
            select.appendChild(option);
        });

    // Add change event listener
    select.addEventListener('change', async (e) => {
        const symbol = e.target.value;
        if (symbol) {
            await loadDataQuality(symbol);
        } else {
            // Show empty state
            document.getElementById('qualitySummary').style.display = 'none';
            document.getElementById('noQuality').style.display = 'block';
        }
    });
}

async function loadDataQuality(symbol) {
    try {
        // Show loading state
        showLoading();

        // Fetch quality metrics and data preview in parallel
        const [qualityData, previewData] = await Promise.all([
            apiService.getDataQuality(symbol),
            apiService.getMarketDataPreview(symbol, 100)
        ]);

        // Hide empty state, show summary
        document.getElementById('noQuality').style.display = 'none';
        document.getElementById('qualitySummary').style.display = 'block';

        // Render quality summary
        renderQualitySummary(qualityData);

        // Render data table
        renderDataTable(previewData);

        hideLoading();
    } catch (error) {
        hideLoading();
        showNotification(`Failed to load data quality: ${error.message}`, 'error');
    }
}

function renderQualitySummary(data) {
    // Update Total Records
    document.getElementById('totalRecords').textContent = data.total_records.toLocaleString();

    // Update Completeness
    document.getElementById('completeness').textContent = `${data.completeness_pct}%`;

    // Update Quality Score with color coding
    const scoreEl = document.getElementById('qualityScore');
    scoreEl.textContent = data.quality_score;
    scoreEl.setAttribute('data-grade', data.quality_score);

    // Update Missing Data
    document.getElementById('missingData').textContent = data.missing_data_points.toLocaleString();

    // Update Outliers
    document.getElementById('outliers').textContent = data.outlier_count.toLocaleString();

    // Update Date Range
    if (data.date_range && data.date_range.first && data.date_range.last) {
        const firstDate = new Date(data.date_range.first).toLocaleDateString();
        const lastDate = new Date(data.date_range.last).toLocaleDateString();
        document.getElementById('dateRange').textContent = `${firstDate} → ${lastDate}`;
    } else {
        document.getElementById('dateRange').textContent = '-';
    }
}

function renderDataTable(data) {
    const tbody = document.getElementById('marketDataBody');
    tbody.innerHTML = '';

    if (!data.data || data.data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem;">No data available</td></tr>';
        return;
    }

    data.data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.timestamp}</td>
            <td>${formatPrice(row.open)}</td>
            <td>${formatPrice(row.high)}</td>
            <td>${formatPrice(row.low)}</td>
            <td>${formatPrice(row.close)}</td>
            <td>${formatVolume(row.volume)}</td>
            <td>${row.number_of_trades.toLocaleString()}</td>
        `;
        tbody.appendChild(tr);
    });
}

function formatPrice(price) {
    if (price === null || price === undefined) return '-';
    // Format with appropriate decimal places
    if (price < 1) {
        return price.toFixed(8);
    } else if (price < 100) {
        return price.toFixed(4);
    } else {
        return price.toFixed(2);
    }
}

function formatVolume(volume) {
    if (volume === null || volume === undefined) return '-';
    if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    } else {
        return volume.toFixed(2);
    }
}

// Update switchTab function to initialize quality tab
const originalSwitchTab = switchTab;
switchTab = function(tabName) {
    originalSwitchTab(tabName);

    // Initialize quality tab when switched to
    if (tabName === 'quality') {
        initQualityTab();
    }
};
