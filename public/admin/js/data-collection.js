/**
 * Data Collection Page
 * Based on: MoneyMoney/other platform/Trading/frontend/src/pages/DataCollection.tsx
 */

// State
let state = {
    activeStep: 0,
    selectedProfileId: null,
    selectedProfile: null,
    profiles: [],
    daysBack: null,  // Will be set from selected profile's lookback_days
    exchange: 'binance',
    isCollecting: false,
    collectionProgress: null
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

    // Subscribe to data collection events
    websocketService.on('dataCollectionProgress', handleCollectionProgress);
    websocketService.on('dataCollectionCompleted', handleCollectionCompleted);
    websocketService.on('dataCollectionFailed', handleCollectionFailed);
}

async function loadProfiles() {
    try {
        const profiles = await apiService.getProfiles({ is_active: true });
        state.profiles = profiles;

        const profileSelect = document.getElementById('profileSelect');
        profileSelect.innerHTML = '<option value="">Select a trading profile...</option>';

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
    document.getElementById('profileSelect').addEventListener('change', (e) => {
        const profileId = parseInt(e.target.value);
        if (profileId) {
            state.selectedProfileId = profileId;
            state.selectedProfile = state.profiles.find(p => p.id === profileId);
            updateSelectedProfileInfo();
            updateNavigationButtons();
        } else {
            state.selectedProfileId = null;
            state.selectedProfile = null;
            document.getElementById('selectedProfileInfo').style.display = 'none';
            updateNavigationButtons();
        }
    });

    // Exchange selection
    document.getElementById('exchangeSelect').addEventListener('change', (e) => {
        state.exchange = e.target.value;
    });

    // Navigation buttons
    document.getElementById('backBtn').addEventListener('click', handleBack);
    document.getElementById('nextBtn').addEventListener('click', handleNext);
    document.getElementById('startCollectionBtn').addEventListener('click', handleStartCollection);

    // Complete step buttons
    document.getElementById('trainModelsBtn').addEventListener('click', () => {
        window.location.href = `/admin/models?autoTrain=true&profileId=${state.selectedProfileId}`;
    });

    document.getElementById('collectMoreBtn').addEventListener('click', resetWizard);
}

function updateSelectedProfileInfo() {
    if (!state.selectedProfile) return;

    document.getElementById('infoSymbol').textContent = state.selectedProfile.symbol;
    document.getElementById('infoType').textContent = state.selectedProfile.profile_type.toUpperCase();
    document.getElementById('infoExchange').textContent = state.selectedProfile.exchange;

    const hasDataElement = document.getElementById('infoHasData');
    if (state.selectedProfile.has_data) {
        hasDataElement.innerHTML = '<span style="color: var(--color-success);">✓ Has Data</span>';
    } else {
        hasDataElement.innerHTML = '<span style="color: var(--color-text-secondary);">No Data</span>';
    }

    // Set days back and exchange from profile settings
    state.daysBack = state.selectedProfile.lookback_days || 30;
    state.exchange = state.selectedProfile.data_source || state.selectedProfile.exchange || 'binance';

    // Update display to show profile's lookback period
    document.getElementById('profileLookbackDays').textContent = state.daysBack;

    // Update exchange dropdown to match profile
    const exchangeSelect = document.getElementById('exchangeSelect');
    if (exchangeSelect) {
        exchangeSelect.value = state.exchange;
    }

    document.getElementById('selectedProfileInfo').style.display = 'block';
}

function updateNavigationButtons() {
    const backBtn = document.getElementById('backBtn');
    const nextBtn = document.getElementById('nextBtn');
    const startBtn = document.getElementById('startCollectionBtn');

    // Back button
    if (state.activeStep === 0 || state.isCollecting) {
        backBtn.style.display = 'none';
    } else if (state.activeStep < 3) {
        backBtn.style.display = 'inline-flex';
    } else {
        backBtn.style.display = 'none';
    }

    // Next button (step 0 only)
    if (state.activeStep === 0) {
        nextBtn.style.display = 'inline-flex';
        nextBtn.disabled = !state.selectedProfileId;
        startBtn.style.display = 'none';
    } else if (state.activeStep === 1) {
        nextBtn.style.display = 'none';
        startBtn.style.display = 'inline-flex';
        startBtn.disabled = !state.selectedProfileId || state.isCollecting;
    } else {
        nextBtn.style.display = 'none';
        startBtn.style.display = 'none';
    }
}

function handleNext() {
    if (state.activeStep === 0 && !state.selectedProfileId) {
        showError('Please select a symbol first');
        return;
    }

    hideError();
    setActiveStep(state.activeStep + 1);
}

function handleBack() {
    setActiveStep(state.activeStep - 1);
}

async function handleStartCollection() {
    if (!state.selectedProfile) return;

    // Show custom confirmation dialog for all data collections
    showDataCollectionWarning(
        {
            symbol: state.selectedProfile.symbol,
            daysBack: state.daysBack,
            interval: '1m',
            profileName: state.selectedProfile.name
        },
        async () => {
            // User confirmed - proceed with collection
            await startDataCollectionProcess();
        },
        () => {
            // User cancelled - do nothing
            console.log('Data collection cancelled by user');
        }
    );
}

async function startDataCollectionProcess() {
    state.isCollecting = true;
    updateNavigationButtons();
    hideError();

    try {
        const response = await apiService.startDataCollection(
            state.selectedProfile.symbol,
            state.daysBack,
            state.exchange
        );

        state.collectionProgress = {
            jobId: response.job_id,
            status: 'pending',
            progress: 0,
            currentStage: 'initializing',
            totalRecords: 0
        };

        document.getElementById('collectingSymbol').textContent = state.selectedProfile.symbol;
        setActiveStep(2);
    } catch (error) {
        showError(error.message || 'Failed to start data collection');
        state.isCollecting = false;
        updateNavigationButtons();
    }
}

function handleCollectionProgress(data) {
    console.log('Collection progress:', data);

    state.collectionProgress = {
        jobId: data.job_id,
        status: data.status,
        progress: data.progress,
        currentStage: data.current_stage || data.status,
        totalRecords: data.total_records || 0
    };

    updateProgressUI();
}

function handleCollectionCompleted(data) {
    console.log('Collection completed:', data);

    state.collectionProgress = {
        jobId: data.job_id,
        status: 'completed',
        progress: 100,
        currentStage: 'completed',
        totalRecords: data.total_records
    };

    state.isCollecting = false;
    updateProgressUI();

    // Move to completion step
    setTimeout(() => {
        document.getElementById('totalRecords').textContent = data.total_records.toLocaleString();
        document.getElementById('completedSymbol').textContent = state.selectedProfile?.symbol || '';
        setActiveStep(3);
        showNotification('Data collection completed successfully!', 'success');
    }, 1000);
}

function handleCollectionFailed(data) {
    console.error('Collection failed:', data);

    state.collectionProgress = {
        jobId: data.job_id,
        status: 'failed',
        progress: 0,
        currentStage: 'failed',
        totalRecords: 0,
        errorMessage: data.error
    };

    showError(data.error || 'Data collection failed');
    state.isCollecting = false;
    updateNavigationButtons();
}

function updateProgressUI() {
    if (!state.collectionProgress) return;

    // Hide initializing, show progress
    document.getElementById('initializing').style.display = 'none';
    document.getElementById('collectionProgress').style.display = 'block';

    // Update progress percentage
    document.getElementById('progressPercent').textContent = `${state.collectionProgress.progress}%`;
    document.getElementById('progressBarFill').style.width = `${state.collectionProgress.progress}%`;

    // Update stage name and icon
    const stageName = state.collectionProgress.currentStage.charAt(0).toUpperCase() +
                     state.collectionProgress.currentStage.slice(1);
    document.getElementById('stageName').textContent = stageName;

    const stageIcon = document.getElementById('stageIcon');
    stageIcon.className = ''; // Clear classes

    // Update icon based on stage
    switch (state.collectionProgress.currentStage) {
        case 'fetching':
            stageIcon.className = 'fas fa-cloud-download-alt';
            break;
        case 'preprocessing':
            stageIcon.className = 'fas fa-cogs fa-spin';
            break;
        case 'storing':
            stageIcon.className = 'fas fa-database';
            break;
        case 'completed':
            stageIcon.className = 'fas fa-check-circle';
            break;
        case 'failed':
            stageIcon.className = 'fas fa-exclamation-circle';
            break;
        default:
            stageIcon.className = 'fas fa-spinner fa-spin';
    }

    // Update stage indicators
    const fetchingCard = document.getElementById('stage-fetching');
    const preprocessingCard = document.getElementById('stage-preprocessing');
    const storingCard = document.getElementById('stage-storing');

    // Remove all active classes
    [fetchingCard, preprocessingCard, storingCard].forEach(card => {
        card.classList.remove('active');
    });

    // Activate current stage
    if (state.collectionProgress.progress >= 10 && state.collectionProgress.progress < 40) {
        fetchingCard.classList.add('active');
    } else if (state.collectionProgress.progress >= 40 && state.collectionProgress.progress < 70) {
        preprocessingCard.classList.add('active');
    } else if (state.collectionProgress.progress >= 70) {
        storingCard.classList.add('active');
    }

    // Update records count
    if (state.collectionProgress.totalRecords > 0) {
        document.getElementById('recordsProcessed').textContent =
            state.collectionProgress.totalRecords.toLocaleString();
        document.getElementById('progressInfo').style.display = 'flex';
    }
}

function setActiveStep(step) {
    state.activeStep = step;

    // Update stepper UI
    document.querySelectorAll('.stepper .step').forEach((stepEl, index) => {
        stepEl.classList.remove('active', 'completed');

        if (index < step) {
            stepEl.classList.add('completed');
        } else if (index === step) {
            stepEl.classList.add('active');
        }
    });

    // Update step content
    document.querySelectorAll('.step-content').forEach((content, index) => {
        if (index === step) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });

    // Update navigation buttons
    updateNavigationButtons();

    // Reset progress UI if going back to step 2
    if (step === 2 && !state.collectionProgress) {
        document.getElementById('initializing').style.display = 'flex';
        document.getElementById('collectionProgress').style.display = 'none';
    }
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorAlert').style.display = 'flex';
}

function hideError() {
    document.getElementById('errorAlert').style.display = 'none';
}

function resetWizard() {
    state.activeStep = 0;
    state.selectedProfileId = null;
    state.selectedProfile = null;
    state.isCollecting = false;
    state.collectionProgress = null;

    document.getElementById('profileSelect').value = '';
    document.getElementById('selectedProfileInfo').style.display = 'none';
    state.daysBack = null;

    document.getElementById('initializing').style.display = 'flex';
    document.getElementById('collectionProgress').style.display = 'none';

    setActiveStep(0);
    hideError();
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    websocketService.off('dataCollectionProgress', handleCollectionProgress);
    websocketService.off('dataCollectionCompleted', handleCollectionCompleted);
    websocketService.off('dataCollectionFailed', handleCollectionFailed);
});
