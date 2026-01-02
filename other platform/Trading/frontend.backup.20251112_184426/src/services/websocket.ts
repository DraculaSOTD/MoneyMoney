import { io, Socket } from 'socket.io-client';
import { store } from '../store';
import { updateData, setConnectionStatus } from '../features/marketData/marketDataSlice';
import { updatePosition } from '../features/trading/tradingSlice';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectInterval: ReturnType<typeof setInterval> | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private eventListeners: Map<string, Set<Function>> = new Map();

  connect() {
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
    const token = store.getState().auth.token;

    if (!token) {
      console.error('No auth token available');
      return;
    }

    this.socket = io(API_BASE_URL, {
      path: '/ws',
      transports: ['websocket'],
      auth: {
        token: token,
      },
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      store.dispatch(setConnectionStatus(true));
      this.reconnectAttempts = 0;
      if (this.reconnectInterval) {
        clearInterval(this.reconnectInterval);
        this.reconnectInterval = null;
      }
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      store.dispatch(setConnectionStatus(false));
      this.attemptReconnect();
    });

    this.socket.on('market_data', (data: any) => {
      store.dispatch(updateData({
        symbol: data.symbol,
        price: data.data.last_price,
        bid: data.data.bid,
        ask: data.data.ask,
        volume: data.data.volume,
        change24h: data.data.change_24h || 0,
        changePercent24h: data.data.change_percent_24h || 0,
        high24h: data.data.high_24h || 0,
        low24h: data.data.low_24h || 0,
        timestamp: Date.now(),
      }));
    });

    this.socket.on('position_update', (position: any) => {
      store.dispatch(updatePosition(position));
    });

    this.socket.on('error', (error: any) => {
      console.error('WebSocket error:', error);
    });

    // Data collection events
    this.socket.on('data_collection_started', (data: any) => {
      console.log('Data collection started:', data);
      this.emit('dataCollectionStarted', data);
    });

    this.socket.on('data_collection_progress', (data: any) => {
      console.log('Data collection progress:', data);
      this.emit('dataCollectionProgress', data);
    });

    this.socket.on('data_collection_completed', (data: any) => {
      console.log('Data collection completed:', data);
      this.emit('dataCollectionCompleted', data);
    });

    this.socket.on('data_collection_failed', (data: any) => {
      console.error('Data collection failed:', data);
      this.emit('dataCollectionFailed', data);
    });

    // Model training events
    this.socket.on('model_training_started', (data: any) => {
      console.log('Model training started:', data);
      this.emit('modelTrainingStarted', data);
    });

    this.socket.on('model_training_progress', (data: any) => {
      console.log('Model training progress:', data);
      this.emit('modelTrainingProgress', data);
    });

    this.socket.on('model_training_completed', (data: any) => {
      console.log('Model training completed:', data);
      this.emit('modelTrainingCompleted', data);
    });

    this.socket.on('model_training_failed', (data: any) => {
      console.error('Model training failed:', data);
      this.emit('modelTrainingFailed', data);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    if (this.reconnectInterval) {
      clearInterval(this.reconnectInterval);
      this.reconnectInterval = null;
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
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

  subscribe(symbols: string[]) {
    if (!this.socket || !this.socket.connected) {
      console.error('WebSocket not connected');
      return;
    }

    this.socket.emit('subscribe', {
      type: 'subscribe',
      symbols: symbols,
    });
  }

  unsubscribe(symbols: string[]) {
    if (!this.socket || !this.socket.connected) {
      console.error('WebSocket not connected');
      return;
    }

    this.socket.emit('unsubscribe', {
      type: 'unsubscribe',
      symbols: symbols,
    });
  }

  // Event emitter methods for data collection and model training
  on(event: string, callback: Function) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function) {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  private emit(event: string, data: any) {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  }
}

export const websocketService = new WebSocketService();