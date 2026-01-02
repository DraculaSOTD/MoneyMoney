/**
 * UI Helper Functions
 * Reusable utility functions for UI operations
 */

// ==================== Notifications ====================

function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 10000; display: flex; flex-direction: column; gap: 10px;';
        document.body.appendChild(container);
    }

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;

    const icon = {
        success: '<i class="fas fa-check-circle"></i>',
        error: '<i class="fas fa-exclamation-circle"></i>',
        warning: '<i class="fas fa-exclamation-triangle"></i>',
        info: '<i class="fas fa-info-circle"></i>'
    }[type] || '<i class="fas fa-info-circle"></i>';

    notification.innerHTML = `
        ${icon}
        <span>${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;

    notification.style.cssText = `
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 20px;
        background-color: var(--color-card-background);
        border: 1px solid var(--color-card-border);
        border-left: 4px solid ${type === 'success' ? 'var(--color-success)' : type === 'error' ? 'var(--color-error)' : type === 'warning' ? 'var(--color-warning)' : 'var(--color-primary-accent)'};
        border-radius: 8px;
        backdrop-filter: blur(10px);
        color: var(--color-text-primary);
        min-width: 300px;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        animation: slideInRight 0.3s ease;
    `;

    container.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// ==================== Loading Overlay ====================

function showLoading(message = 'Loading...') {
    // Remove existing overlay
    hideLoading();

    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <p>${message}</p>
        </div>
    `;

    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(12, 11, 16, 0.8);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    `;

    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

// ==================== Formatting Functions ====================

function formatNumber(num) {
    if (num === null || num === undefined) return 'N/A';
    return num.toLocaleString('en-US');
}

function formatCurrency(num, decimals = 2) {
    if (num === null || num === undefined) return '$0.00';
    return '$' + num.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatPercent(num, decimals = 2) {
    if (num === null || num === undefined) return '0.00%';
    return num.toFixed(decimals) + '%';
}

function formatDateTime(date) {
    if (!date) return 'N/A';

    const d = typeof date === 'string' ? new Date(date) : date;

    if (isNaN(d.getTime())) return 'Invalid Date';

    const now = new Date();
    const diff = now - d;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (seconds < 60) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;

    return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: d.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
    });
}

function formatDateTimeFull(date) {
    if (!date) return 'N/A';
    const d = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(d.getTime())) return 'Invalid Date';

    return d.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// ==================== UI Component Creators ====================

function createChip(label, color = 'default') {
    const colors = {
        success: 'var(--color-success)',
        error: 'var(--color-error)',
        warning: 'var(--color-warning)',
        info: 'var(--color-primary-accent)',
        default: 'var(--color-text-secondary)'
    };

    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = label;
    chip.style.cssText = `
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: ${colors[color] || colors.default}20;
        border: 1px solid ${colors[color] || colors.default};
        color: ${colors[color] || colors.default};
    `;

    return chip;
}

function createProgressBar(progress, color = 'primary') {
    const colors = {
        primary: 'linear-gradient(90deg, var(--color-primary-accent), var(--color-secondary-accent))',
        success: 'var(--color-success)',
        warning: 'var(--color-warning)',
        error: 'var(--color-error)'
    };

    const container = document.createElement('div');
    container.className = 'progress-bar-container';
    container.style.cssText = `
        width: 100%;
        height: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        overflow: hidden;
    `;

    const bar = document.createElement('div');
    bar.className = 'progress-bar';
    bar.style.cssText = `
        height: 100%;
        width: ${progress}%;
        background: ${colors[color] || colors.primary};
        border-radius: 5px;
        transition: width 0.3s ease;
    `;

    container.appendChild(bar);
    return container;
}

function createButton(text, variant = 'primary', icon = null) {
    const button = document.createElement('button');
    button.className = `btn btn-${variant}`;

    if (icon) {
        button.innerHTML = `<i class="${icon}"></i> ${text}`;
    } else {
        button.textContent = text;
    }

    return button;
}

// ==================== Confirmation Dialog ====================

function showConfirmDialog(message, onConfirm, onCancel = null) {
    const overlay = document.createElement('div');
    overlay.className = 'dialog-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9998;
    `;

    const dialog = document.createElement('div');
    dialog.className = 'confirm-dialog';
    dialog.style.cssText = `
        background-color: var(--color-card-background);
        border: 1px solid var(--color-card-border);
        border-radius: 16px;
        padding: 30px;
        min-width: 400px;
        max-width: 500px;
        backdrop-filter: blur(10px);
    `;

    dialog.innerHTML = `
        <h3 style="margin-bottom: 20px; color: var(--color-text-primary);">Confirm Action</h3>
        <p style="margin-bottom: 30px; color: var(--color-text-secondary);">${message}</p>
        <div style="display: flex; gap: 12px; justify-content: flex-end;">
            <button class="btn btn-secondary" id="confirm-cancel">Cancel</button>
            <button class="btn btn-primary" id="confirm-ok">Confirm</button>
        </div>
    `;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    document.getElementById('confirm-ok').onclick = () => {
        overlay.remove();
        if (onConfirm) onConfirm();
    };

    document.getElementById('confirm-cancel').onclick = () => {
        overlay.remove();
        if (onCancel) onCancel();
    };

    // Click outside to cancel
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            overlay.remove();
            if (onCancel) onCancel();
        }
    };
}

// ==================== Data Collection Warning Dialog ====================

function showDataCollectionWarning(options, onConfirm, onCancel = null) {
    const {
        symbol,
        daysBack,
        interval = '1m',
        profileName
    } = options;

    // Calculate estimates
    const candlesEstimate = daysBack * 1440; // 1440 minutes per day for 1m intervals
    const timeEstimateMinutes = Math.ceil(candlesEstimate / 60000); // Rough estimate: 1000 candles/second
    const storageMB = Math.ceil((candlesEstimate * 200) / 1024 / 1024); // ~200 bytes per candle

    // Determine warning level
    let warningLevel = 'info';
    let warningIcon = 'fa-info-circle';
    let warningColor = 'var(--color-primary-accent)';

    if (daysBack > 365) {
        warningLevel = 'warning';
        warningIcon = 'fa-exclamation-triangle';
        warningColor = 'var(--color-warning)';
    }

    if (daysBack > 730) {
        warningLevel = 'caution';
        warningIcon = 'fa-exclamation-circle';
        warningColor = 'var(--color-error)';
    }

    const overlay = document.createElement('div');
    overlay.className = 'dialog-overlay data-collection-warning-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(8px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9998;
        animation: fadeIn 0.3s ease;
    `;

    const dialog = document.createElement('div');
    dialog.className = 'data-collection-warning-dialog';
    dialog.style.cssText = `
        background: linear-gradient(135deg,
            var(--color-card-background) 0%,
            rgba(59, 130, 246, 0.1) 100%);
        border: 2px solid ${warningColor};
        border-radius: 20px;
        padding: 0;
        min-width: 500px;
        max-width: 600px;
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5),
                    0 0 20px ${warningColor}40;
        animation: slideInScale 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    `;

    dialog.innerHTML = `
        <div style="
            background: linear-gradient(90deg, ${warningColor}20, transparent);
            border-bottom: 1px solid var(--color-card-border);
            padding: 25px 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        ">
            <div style="
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: ${warningColor}30;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                color: ${warningColor};
            ">
                <i class="fas ${warningIcon}"></i>
            </div>
            <div>
                <h2 style="margin: 0; color: var(--color-text-primary); font-size: 1.5rem;">
                    Data Collection Request
                </h2>
                <p style="margin: 5px 0 0; color: var(--color-text-secondary); font-size: 0.9rem;">
                    ${profileName || symbol} - ${interval} interval
                </p>
            </div>
        </div>

        <div style="padding: 30px;">
            <div style="
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--color-card-border);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 25px;
            ">
                <h3 style="
                    margin: 0 0 15px 0;
                    color: var(--color-text-primary);
                    font-size: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                ">
                    <i class="fas fa-chart-bar" style="color: var(--color-primary-accent);"></i>
                    Collection Details
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div class="stat-item">
                        <div class="stat-label">Time Period</div>
                        <div class="stat-value">${daysBack.toLocaleString()} days</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Estimated Candles</div>
                        <div class="stat-value">${candlesEstimate.toLocaleString()}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Download Time</div>
                        <div class="stat-value">~${timeEstimateMinutes} min</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Storage Required</div>
                        <div class="stat-value">~${storageMB} MB</div>
                    </div>
                </div>
            </div>

            ${daysBack > 365 ? `
            <div style="
                background: ${warningColor}15;
                border: 1px solid ${warningColor};
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 25px;
                display: flex;
                align-items: start;
                gap: 12px;
            ">
                <i class="fas fa-info-circle" style="color: ${warningColor}; margin-top: 2px;"></i>
                <div style="flex: 1;">
                    <p style="margin: 0; color: var(--color-text-primary); font-size: 0.9rem; line-height: 1.5;">
                        <strong>Large Data Request:</strong> This collection will run in the background and may take a while.
                        You can monitor progress in the Data Management section.
                    </p>
                </div>
            </div>
            ` : ''}

            <div style="display: flex; gap: 12px; justify-content: flex-end;">
                <button class="btn btn-secondary" id="data-warning-cancel" style="
                    padding: 12px 24px;
                    border-radius: 10px;
                    font-weight: 600;
                ">
                    <i class="fas fa-times"></i> Cancel
                </button>
                <button class="btn btn-primary" id="data-warning-confirm" style="
                    padding: 12px 24px;
                    border-radius: 10px;
                    font-weight: 600;
                    background: linear-gradient(90deg, var(--color-primary-accent), var(--color-secondary-accent));
                ">
                    <i class="fas fa-download"></i> Start Collection
                </button>
            </div>
        </div>
    `;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    document.getElementById('data-warning-confirm').onclick = () => {
        overlay.style.animation = 'fadeOut 0.2s ease';
        setTimeout(() => {
            overlay.remove();
            if (onConfirm) onConfirm();
        }, 200);
    };

    document.getElementById('data-warning-cancel').onclick = () => {
        overlay.style.animation = 'fadeOut 0.2s ease';
        setTimeout(() => {
            overlay.remove();
            if (onCancel) onCancel();
        }, 200);
    };

    // Click outside to cancel
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            overlay.style.animation = 'fadeOut 0.2s ease';
            setTimeout(() => {
                overlay.remove();
                if (onCancel) onCancel();
            }, 200);
        }
    };
}
