/**
 * WebSocket Service
 * Handles real-time communication with the backend using native WebSocket API
 * Connects to FastAPI WebSocket endpoints
 */

class WebSocketService {
    constructor() {
        this.socket = null;
        this.reconnectInterval = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.eventListeners = new Map();
        this.connected = false;
        this.baseUrl = 'ws://localhost:8002';
        this.pingInterval = null;
    }

    connect() {
        const token = localStorage.getItem('adminToken');

        if (!token) {
            console.error('No auth token available');
            return;
        }

        // Close existing connection if any
        if (this.socket) {
            this.socket.close();
        }

        // Connect to FastAPI WebSocket admin endpoint with token as query parameter
        // Using /ws/admin instead of /ws/data-collection to support multiple channel subscriptions
        const wsUrl = `${this.baseUrl}/ws/admin?token=${encodeURIComponent(token)}`;

        try {
            this.socket = new WebSocket(wsUrl);

            this.socket.onopen = () => {
                console.log('WebSocket connected');
                this.connected = true;
                this.reconnectAttempts = 0;

                if (this.reconnectInterval) {
                    clearInterval(this.reconnectInterval);
                    this.reconnectInterval = null;
                }

                // Start ping to keep connection alive
                this.startPing();

                this.emit('connectionStatus', true);
            };

            this.socket.onclose = (event) => {
                console.log('WebSocket disconnected', event.code, event.reason);
                this.connected = false;
                this.stopPing();
                this.emit('connectionStatus', false);

                // Only attempt reconnect if not a normal closure
                if (event.code !== 1000) {
                    this.attemptReconnect();
                }
            };

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.emit('error', error);
            };

            this.socket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.attemptReconnect();
        }
    }

    handleMessage(message) {
        const { type, ...data } = message;

        switch (type) {
            // Connection and subscription events
            case 'welcome':
                console.log('WebSocket welcome:', data);
                this.emit('welcome', data);
                break;

            case 'subscribed':
                console.log('Subscribed to channel:', data.channel);
                this.emit('subscribed', data);
                break;

            case 'unsubscribed':
                console.log('Unsubscribed from channel:', data.channel);
                this.emit('unsubscribed', data);
                break;

            case 'error':
                console.error('WebSocket error message:', data);
                this.emit('wsError', data);
                break;

            // Data collection events
            case 'data_collection_started':
                console.log('Data collection started:', data);
                this.emit('dataCollectionStarted', data);
                break;

            case 'data_collection_progress':
                console.log('Data collection progress:', data);
                this.emit('dataCollectionProgress', data);
                break;

            case 'data_collection_completed':
                console.log('Data collection completed:', data);
                this.emit('dataCollectionCompleted', data);
                break;

            case 'data_collection_failed':
                console.error('Data collection failed:', data);
                this.emit('dataCollectionFailed', data);
                break;

            // Preprocessing events
            case 'preprocessing_started':
                console.log('Preprocessing started:', data);
                this.emit('preprocessingStarted', data);
                break;

            case 'preprocessing_progress':
                console.log('Preprocessing progress:', data);
                this.emit('preprocessingProgress', data);
                break;

            case 'preprocessing_completed':
                console.log('Preprocessing completed:', data);
                this.emit('preprocessingCompleted', data);
                break;

            case 'preprocessing_failed':
                console.error('Preprocessing failed:', data);
                this.emit('preprocessingFailed', data);
                break;

            // Model training events
            case 'model_training_started':
                console.log('Model training started:', data);
                this.emit('modelTrainingStarted', data);
                break;

            case 'model_training_progress':
                console.log('Model training progress:', data);
                this.emit('modelTrainingProgress', data);
                break;

            case 'model_training_completed':
                console.log('Model training completed:', data);
                this.emit('modelTrainingCompleted', data);
                break;

            case 'model_training_failed':
                console.error('Model training failed:', data);
                this.emit('modelTrainingFailed', data);
                break;

            // Market data events (for future use)
            case 'market_data':
                this.emit('marketData', data);
                break;

            case 'position_update':
                this.emit('positionUpdate', data);
                break;

            // Ping/pong for keep-alive
            case 'ping':
                this.send({ type: 'pong' });
                break;

            case 'pong':
                // Received pong response
                break;

            default:
                console.warn('Unknown message type:', type);
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.close(1000, 'Client disconnecting');
            this.socket = null;
        }

        this.stopPing();

        if (this.reconnectInterval) {
            clearInterval(this.reconnectInterval);
            this.reconnectInterval = null;
        }

        this.connected = false;
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('maxReconnectAttemptsReached', this.maxReconnectAttempts);
            return;
        }

        if (!this.reconnectInterval) {
            this.reconnectInterval = setInterval(() => {
                this.reconnectAttempts++;
                console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                this.connect();
            }, 5000);
        }
    }

    startPing() {
        this.stopPing();
        this.pingInterval = setInterval(() => {
            if (this.connected) {
                this.send({ type: 'ping' });
            }
        }, 15000); // Ping every 15 seconds
    }

    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    send(data) {
        if (!this.socket || !this.connected) {
            console.error('WebSocket not connected');
            return false;
        }

        try {
            this.socket.send(JSON.stringify(data));
            return true;
        } catch (error) {
            console.error('Error sending WebSocket message:', error);
            return false;
        }
    }

    subscribe(channels) {
        if (!Array.isArray(channels)) {
            channels = [channels];
        }

        return this.send({
            type: 'subscribe',
            channels: channels
        });
    }

    unsubscribe(channels) {
        if (!Array.isArray(channels)) {
            channels = [channels];
        }

        return this.send({
            type: 'unsubscribe',
            channels: channels
        });
    }

    // Event emitter methods
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(callback);
    }

    off(event, callback) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.delete(callback);
        }
    }

    emit(event, data) {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    isConnected() {
        return this.connected;
    }
}

// Create singleton instance
const websocketService = new WebSocketService();

// Auto-connect when admin token is available
if (localStorage.getItem('adminToken')) {
    // Wait a bit for page to load
    setTimeout(() => {
        websocketService.connect();
    }, 500);
}
