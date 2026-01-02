/**
 * ML Models Page
 * Based on: MoneyMoney/other platform/Trading/frontend/src/pages/Models.tsx
 */

// State
let state = {
    selectedProfileId: null,
    selectedProfile: null,
    profiles: [],
    models: [],
    trainingHistory: [],
    activeTab: 'overview',
    isTraining: false
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializePage();
});

async function initializePage() {
    // Load profiles
    await loadProfiles();

    // Setup event listeners
    setupEventListeners();

    // Connect WebSocket
    websocketService.connect();

    // Wait for connection and subscribe to model training channel
    websocketService.on('connectionStatus', (connected) => {
        if (connected) {
            console.log('WebSocket connected, subscribing to model_training channel...');
            // Add small delay to ensure WebSocket handshake is complete
            setTimeout(() => {
                websocketService.subscribe(['model_training']);
                console.log('Subscription request sent for model_training channel');
            }, 100);
        }
    });

    // Subscribe to model training events
    websocketService.on('modelTrainingProgress', handleTrainingProgress);
    websocketService.on('modelTrainingCompleted', handleTrainingCompleted);
    websocketService.on('modelTrainingFailed', handleTrainingFailed);

    // Check for autoTrain parameter
    const urlParams = new URLSearchParams(window.location.search);
    const autoTrain = urlParams.get('autoTrain');
    const profileId = urlParams.get('profileId');

    if (autoTrain === 'true' && profileId) {
        state.selectedProfileId = parseInt(profileId);
        document.getElementById('profileSelect').value = profileId;
        await onProfileChange();
        openTrainingDialog();
    }
}

async function loadProfiles() {
    try {
        const profiles = await apiService.getProfiles({ is_active: true });
        state.profiles = profiles;

        const profileSelect = document.getElementById('profileSelect');
        profileSelect.innerHTML = '<option value="">Select a profile...</option>';

        profiles.forEach(profile => {
            const option = document.createElement('option');
            option.value = profile.id;
            option.textContent = `${profile.name} (${profile.symbol}) - ${profile.exchange}`;
            profileSelect.appendChild(option);
        });
    } catch (error) {
        showNotification('Failed to load profiles: ' + error.message, 'error');
    }
}

function setupEventListeners() {
    // Profile selection
    document.getElementById('profileSelect').addEventListener('change', onProfileChange);

    // Action buttons
    document.getElementById('trainModelBtn').addEventListener('click', openTrainingDialog);
    document.getElementById('refreshBtn').addEventListener('click', refreshData);

    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tab = e.currentTarget.dataset.tab;
            switchTab(tab);
        });
    });
}

async function onProfileChange() {
    const profileId = parseInt(document.getElementById('profileSelect').value);

    if (profileId) {
        state.selectedProfileId = profileId;
        state.selectedProfile = state.profiles.find(p => p.id === profileId);

        document.getElementById('trainModelBtn').disabled = false;
        document.getElementById('noProfileSelected').style.display = 'none';
        document.getElementById('modelsTable').style.display = 'block';

        await loadModels();
        await loadTrainingHistory();
    } else {
        state.selectedProfileId = null;
        state.selectedProfile = null;

        document.getElementById('trainModelBtn').disabled = true;
        document.getElementById('noProfileSelected').style.display = 'block';
        document.getElementById('modelsTable').style.display = 'none';
    }
}

async function loadModels() {
    if (!state.selectedProfileId) return;

    try {
        const models = await apiService.getModels(state.selectedProfileId);
        state.models = models;
        renderModelsTable();
    } catch (error) {
        showNotification('Failed to load models: ' + error.message, 'error');
    }
}

async function loadTrainingHistory() {
    if (!state.selectedProfileId) return;

    try {
        const history = await apiService.getTrainingHistory(state.selectedProfileId);
        state.trainingHistory = history;
        renderTrainingHistory();
    } catch (error) {
        console.error('Failed to load training history:', error);
    }
}

function renderModelsTable() {
    const tbody = document.getElementById('modelsTableBody');
    const noModels = document.getElementById('noModels');

    if (state.models.length === 0) {
        tbody.innerHTML = '';
        noModels.style.display = 'block';
        return;
    }

    noModels.style.display = 'none';

    tbody.innerHTML = state.models.map(model => `
        <tr>
            <td>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <strong>${model.model_name}</strong>
                    ${model.is_primary ? '<span class="status-badge info" style="font-size: 0.7rem;">PRIMARY</span>' : ''}
                </div>
            </td>
            <td>
                <span class="status-badge default">${model.model_type.toUpperCase()}</span>
            </td>
            <td>
                ${getStatusBadge(model.status)}
            </td>
            <td>${formatPerformance(model.test_accuracy)}%</td>
            <td>${model.test_sharpe ? model.test_sharpe.toFixed(2) : 'N/A'}</td>
            <td>${model.last_trained ? formatDateTime(model.last_trained) : 'Never'}</td>
            <td>
                <div class="action-btns">
                    ${model.status === 'trained' && !model.is_deployed ? `
                        <button class="icon-btn success" onclick="deployModel(${model.id})" title="Deploy Model">
                            <i class="fas fa-rocket"></i>
                        </button>
                    ` : ''}
                    ${model.is_deployed ? `
                        <button class="icon-btn danger" onclick="undeployModel(${model.id})" title="Undeploy Model">
                            <i class="fas fa-stop"></i>
                        </button>
                    ` : ''}
                    <button class="icon-btn primary" onclick="viewModelDetails(${model.id})" title="View Details">
                        <i class="fas fa-info-circle"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function renderTrainingHistory() {
    const container = document.getElementById('trainingHistoryList');
    const noHistory = document.getElementById('noTrainingHistory');

    if (state.trainingHistory.length === 0) {
        container.innerHTML = '';
        noHistory.style.display = 'block';
        return;
    }

    noHistory.style.display = 'none';

    container.innerHTML = state.trainingHistory.map(history => `
        <div class="history-item">
            <div class="history-header">
                <span class="history-title">${history.model_name} - Run #${history.id}</span>
                ${getStatusBadge(history.status)}
            </div>
            <div class="history-details">
                <div class="history-detail-item">
                    <span class="history-detail-label">Started</span>
                    <span class="history-detail-value">${formatDateTimeFull(history.started_at)}</span>
                </div>
                <div class="history-detail-item">
                    <span class="history-detail-label">Duration</span>
                    <span class="history-detail-value">${history.duration ? formatDuration(history.duration) : 'N/A'}</span>
                </div>
                <div class="history-detail-item">
                    <span class="history-detail-label">Accuracy</span>
                    <span class="history-detail-value">${history.accuracy ? (history.accuracy * 100).toFixed(2) + '%' : 'N/A'}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function getStatusBadge(status) {
    const statusMap = {
        'deployed': { class: 'success', icon: 'rocket', text: 'Deployed' },
        'trained': { class: 'info', icon: 'check-circle', text: 'Trained' },
        'training': { class: 'warning', icon: 'spinner fa-spin', text: 'Training' },
        'failed': { class: 'error', icon: 'exclamation-circle', text: 'Failed' },
        'pending': { class: 'default', icon: 'clock', text: 'Pending' },
        'completed': { class: 'success', icon: 'check-circle', text: 'Completed' },
        'running': { class: 'warning', icon: 'spinner fa-spin', text: 'Running' }
    };

    const config = statusMap[status] || { class: 'default', icon: 'question', text: status };

    return `<span class="status-badge ${config.class}">
        <i class="fas fa-${config.icon}"></i>
        ${config.text}
    </span>`;
}

function formatPerformance(value) {
    if (value === null || value === undefined) return 'N/A';
    return (value * 100).toFixed(2);
}

function formatDuration(seconds) {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
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
    if (!state.selectedProfileId) return;

    const btn = document.getElementById('refreshBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i>';

    try {
        await Promise.all([
            loadModels(),
            loadTrainingHistory()
        ]);
        showNotification('Data refreshed successfully', 'success');
    } catch (error) {
        showNotification('Failed to refresh data', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sync-alt"></i>';
    }
}

// Training Dialog Functions
function openTrainingDialog() {
    if (!state.selectedProfile) return;

    document.getElementById('trainingSymbol').textContent = state.selectedProfile.symbol;
    document.getElementById('trainingDialog').style.display = 'flex';
    document.getElementById('trainingProgress').style.display = 'none';
    document.getElementById('startTrainingBtn').style.display = 'inline-flex';

    // Reset checkboxes
    document.querySelectorAll('input[name="modelType"]').forEach(cb => {
        cb.checked = true;
        cb.disabled = false;
    });
}

function closeTrainingDialog() {
    document.getElementById('trainingDialog').style.display = 'none';
    state.isTraining = false;
}

async function startTraining() {
    if (!state.selectedProfile) return;

    // Get selected models
    const selectedModels = Array.from(document.querySelectorAll('input[name="modelType"]:checked'))
        .map(cb => cb.value);

    if (selectedModels.length === 0) {
        showNotification('Please select at least one model to train', 'warning');
        return;
    }

    state.isTraining = true;

    // Reset model progress tracking for new training session
    state.modelProgress = {};

    // Hide start button, show progress
    document.getElementById('startTrainingBtn').style.display = 'none';
    document.getElementById('trainingProgress').style.display = 'block';

    // Disable checkboxes
    document.querySelectorAll('input[name="modelType"]').forEach(cb => {
        cb.disabled = true;
    });

    try {
        const response = await apiService.startModelTraining(
            state.selectedProfile.symbol,
            selectedModels
        );

        showNotification(`Training started for ${selectedModels.length} models`, 'success');

        // Update progress UI
        document.getElementById('trainingStatus').textContent = `Training ${selectedModels.length} models...`;
        document.getElementById('trainingPercent').textContent = '0%';
        document.getElementById('trainingProgressBar').style.width = '0%';

    } catch (error) {
        showNotification('Failed to start training: ' + error.message, 'error');
        closeTrainingDialog();
    }
}

function handleTrainingProgress(data) {
    console.log('Training progress:', data);

    if (!state.isTraining) return;

    // Track progress per model for aggregate calculation
    if (!state.modelProgress) {
        state.modelProgress = {};
    }

    // Update progress for this specific model
    if (data.model_name) {
        state.modelProgress[data.model_name] = data.progress || 0;
    }

    // Calculate aggregate progress across all models
    const modelNames = Object.keys(state.modelProgress);
    const totalProgress = Object.values(state.modelProgress).reduce((sum, p) => sum + p, 0);
    const averageProgress = modelNames.length > 0 ? Math.floor(totalProgress / modelNames.length) : 0;

    // Update progress bar with aggregate progress
    document.getElementById('trainingPercent').textContent = `${averageProgress}%`;
    document.getElementById('trainingProgressBar').style.width = `${averageProgress}%`;

    // Update status message with current model info
    const statusText = data.model_name
        ? `Training ${data.model_name}: ${data.message || data.status || 'In progress...'}`
        : (data.message || 'Training in progress...');

    document.getElementById('trainingStatus').textContent = statusText;
}

function handleTrainingCompleted(data) {
    console.log('Training completed:', data);

    state.isTraining = false;

    // Update progress to 100%
    document.getElementById('trainingPercent').textContent = '100%';
    document.getElementById('trainingProgressBar').style.width = '100%';
    document.getElementById('trainingStatus').textContent = data.message || 'Training completed!';

    // Reset model progress tracking
    state.modelProgress = {};

    // Close dialog after delay
    setTimeout(() => {
        closeTrainingDialog();
        refreshData();
        showNotification('Model training completed successfully!', 'success');
    }, 2000);
}

function handleTrainingFailed(data) {
    console.error('Training failed:', data);

    state.isTraining = false;

    // Reset model progress tracking
    state.modelProgress = {};

    showNotification('Training failed: ' + (data.error || data.message || 'Unknown error'), 'error');
    closeTrainingDialog();
}

// Model Action Functions
async function deployModel(modelId) {
    if (!state.selectedProfileId) return;

    try {
        await apiService.deployModel(state.selectedProfileId, modelId);
        showNotification('Model deployed successfully', 'success');
        await loadModels();
    } catch (error) {
        showNotification('Failed to deploy model: ' + error.message, 'error');
    }
}

async function undeployModel(modelId) {
    if (!state.selectedProfileId) return;

    showConfirmDialog(
        'Are you sure you want to undeploy this model?',
        async () => {
            try {
                await apiService.undeployModel(state.selectedProfileId, modelId);
                showNotification('Model undeployed successfully', 'success');
                await loadModels();
            } catch (error) {
                showNotification('Failed to undeploy model: ' + error.message, 'error');
            }
        }
    );
}

function viewModelDetails(modelId) {
    const model = state.models.find(m => m.id === modelId);
    if (!model) return;

    const details = `
        <strong>Model Name:</strong> ${model.model_name}<br>
        <strong>Type:</strong> ${model.model_type}<br>
        <strong>Version:</strong> ${model.model_version}<br>
        <strong>Status:</strong> ${model.status}<br>
        <strong>Test Accuracy:</strong> ${formatPerformance(model.test_accuracy)}%<br>
        <strong>Test Sharpe:</strong> ${model.test_sharpe ? model.test_sharpe.toFixed(2) : 'N/A'}<br>
        <strong>Validation Accuracy:</strong> ${formatPerformance(model.validation_accuracy)}%<br>
        <strong>Training Samples:</strong> ${model.training_samples || 'N/A'}<br>
        <strong>Last Trained:</strong> ${model.last_trained ? formatDateTimeFull(model.last_trained) : 'Never'}<br>
        <strong>Is Primary:</strong> ${model.is_primary ? 'Yes' : 'No'}<br>
        <strong>Is Deployed:</strong> ${model.is_deployed ? 'Yes' : 'No'}
    `;

    showNotification(details, 'info');
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    websocketService.off('modelTrainingProgress', handleTrainingProgress);
    websocketService.off('modelTrainingCompleted', handleTrainingCompleted);
    websocketService.off('modelTrainingFailed', handleTrainingFailed);
});
