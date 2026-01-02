/**
 * API Service
 * Centralized API client for all backend requests
 */

class APIService {
    constructor() {
        this.baseUrl = 'http://localhost:8002';
    }

    getAuthHeaders() {
        const token = localStorage.getItem('adminToken');
        return {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        };
    }

    async handleResponse(response) {
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`;

            // Enhanced error logging
            console.error('API Error Details:', {
                status: response.status,
                statusText: response.statusText,
                url: response.url,
                errorData: errorData,
                message: errorMessage
            });

            // Check for authentication errors
            if (response.status === 401) {
                console.error('Authentication failed - Token may be invalid or expired');
                console.log('Current token:', localStorage.getItem('adminToken') ? 'Token exists' : 'No token found');

                // Optional: redirect to login if token is invalid
                // window.location.href = '/admin/login.html';
            }

            throw new Error(errorMessage);
        }
        return response.json();
    }

    async handleError(error) {
        console.error('API Error:', error);
        console.error('Error stack:', error.stack);
        throw error;
    }

    // ==================== Profile Management ====================

    async getProfiles(params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const url = `${this.baseUrl}/api/profiles/${queryString ? '?' + queryString : ''}`;

            const response = await fetch(url, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getProfile(profileId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async createProfile(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify(data)
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async updateProfile(profileId, data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}`, {
                method: 'PUT',
                headers: this.getAuthHeaders(),
                body: JSON.stringify(data)
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async deleteProfile(profileId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}`, {
                method: 'DELETE',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    // ==================== Data Collection ====================

    async startDataCollection(symbol, daysBack = 30, exchange = 'binance') {
        try {
            const response = await fetch(`${this.baseUrl}/admin/data/collect/${symbol}?days_back=${daysBack}&exchange=${exchange}`, {
                method: 'POST',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getDataCollectionStatus(jobId) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/data/status/${jobId}`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getAvailableSymbols() {
        try {
            const response = await fetch(`${this.baseUrl}/admin/data/available`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async deleteSymbolData(symbol) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/data/${symbol}`, {
                method: 'DELETE',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async preprocessData(symbol, recalculate = false) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/data/preprocess/${symbol}?recalculate=${recalculate}`, {
                method: 'POST',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    // ==================== Model Management ====================

    async getModels(profileId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}/models`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async createModel(profileId, data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}/models`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify(data)
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async deployModel(profileId, modelId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}/models/${modelId}/deploy`, {
                method: 'PUT',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async undeployModel(profileId, modelId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}/models/${modelId}/undeploy`, {
                method: 'PUT',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async startModelTraining(symbol, models = null) {
        try {
            const body = models ? { symbol, models } : { symbol };

            const response = await fetch(`${this.baseUrl}/admin/models/train/${symbol}`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify(body)
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getAllModelStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/admin/models/status`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getSymbolModelStatus(symbol) {
        try {
            const response = await fetch(`${this.baseUrl}/admin/models/status/${symbol}`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getTrainingHistory(profileId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}/training-history`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    // ==================== System Status ====================

    async getSystemStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/system/status`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async startSystem(config = {}) {
        try {
            const response = await fetch(`${this.baseUrl}/system/start`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify(config)
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async stopSystem() {
        try {
            const response = await fetch(`${this.baseUrl}/system/stop`, {
                method: 'POST',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    // ==================== Performance & Monitoring ====================

    async getPerformance(params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const url = `${this.baseUrl}/monitoring/performance${queryString ? '?' + queryString : ''}`;

            const response = await fetch(url, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getAlerts(params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const url = `${this.baseUrl}/monitoring/alerts${queryString ? '?' + queryString : ''}`;

            const response = await fetch(url, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async generateReport(params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const url = `${this.baseUrl}/monitoring/report${queryString ? '?' + queryString : ''}`;

            const response = await fetch(url, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    // ==================== Trading & Positions ====================

    async getPositions(status = null) {
        try {
            const url = status
                ? `${this.baseUrl}/positions?status=${status}`
                : `${this.baseUrl}/positions`;

            const response = await fetch(url, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async closePosition(positionId, quantity = null) {
        try {
            const body = quantity ? { position_id: positionId, quantity } : { position_id: positionId };

            const response = await fetch(`${this.baseUrl}/positions/close`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify(body)
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async placeOrder(orderData) {
        try {
            const response = await fetch(`${this.baseUrl}/trading/order`, {
                method: 'POST',
                headers: this.getAuthHeaders(),
                body: JSON.stringify(orderData)
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async cancelOrder(orderId) {
        try {
            const response = await fetch(`${this.baseUrl}/trading/order/${orderId}`, {
                method: 'DELETE',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    // ==================== Metrics & Predictions ====================

    async getLatestMetrics(profileId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/profiles/${profileId}/metrics/latest`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getMetricsHistory(profileId, params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const url = `${this.baseUrl}/api/profiles/${profileId}/metrics/history${queryString ? '?' + queryString : ''}`;

            const response = await fetch(url, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getPredictions(profileId, params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const url = `${this.baseUrl}/api/profiles/${profileId}/predictions${queryString ? '?' + queryString : ''}`;

            const response = await fetch(url, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    // ==================== Data Quality ====================

    async getDataQuality(symbol) {
        try {
            const response = await fetch(`${this.baseUrl}/api/data-quality/${symbol}`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getMarketDataPreview(symbol, limit = 100) {
        try {
            const response = await fetch(`${this.baseUrl}/api/data-quality/${symbol}/preview?limit=${limit}`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }

    async getDataSummary() {
        try {
            const response = await fetch(`${this.baseUrl}/admin/data/summary`, {
                method: 'GET',
                headers: this.getAuthHeaders()
            });

            return await this.handleResponse(response);
        } catch (error) {
            return this.handleError(error);
        }
    }
}

// Create singleton instance
const apiService = new APIService();
