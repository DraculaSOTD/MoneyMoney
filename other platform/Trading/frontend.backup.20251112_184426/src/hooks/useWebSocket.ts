/**
 * WebSocket Hook
 * ==============
 *
 * Custom React hook for managing WebSocket connections with auto-reconnection,
 * message handling, and subscription management.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export interface UseWebSocketOptions {
  url: string;
  token: string | null;
  autoConnect?: boolean;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

export interface UseWebSocketReturn {
  status: WebSocketStatus;
  sendMessage: (message: object) => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
  connect: () => void;
  disconnect: () => void;
  lastMessage: WebSocketMessage | null;
  isConnected: boolean;
}

/**
 * Custom hook for WebSocket connections
 *
 * @example
 * ```typescript
 * const { status, sendMessage, subscribe, lastMessage } = useWebSocket({
 *   url: 'ws://localhost:8000/ws/admin',
 *   token: authToken,
 *   onMessage: (msg) => console.log('Received:', msg),
 * });
 *
 * useEffect(() => {
 *   if (status === 'connected') {
 *     subscribe('data_collection');
 *   }
 * }, [status]);
 * ```
 */
export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    token,
    autoConnect = true,
    reconnect = true,
    reconnectInterval = 3000,
    reconnectAttempts = 10,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const [status, setStatus] = useState<WebSocketStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const pingInterval = useRef<NodeJS.Timeout | null>(null);

  // Clear reconnect timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
  }, []);

  // Clear ping interval
  const clearPingInterval = useCallback(() => {
    if (pingInterval.current) {
      clearInterval(pingInterval.current);
      pingInterval.current = null;
    }
  }, []);

  // Send message
  const sendMessage = useCallback((message: object) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }, []);

  // Subscribe to channel
  const subscribe = useCallback((channel: string) => {
    sendMessage({ action: 'subscribe', channel });
  }, [sendMessage]);

  // Unsubscribe from channel
  const unsubscribe = useCallback((channel: string) => {
    sendMessage({ action: 'unsubscribe', channel });
  }, [sendMessage]);

  // Start ping interval to keep connection alive
  const startPingInterval = useCallback(() => {
    clearPingInterval();
    pingInterval.current = setInterval(() => {
      sendMessage({ action: 'ping' });
    }, 30000); // Ping every 30 seconds
  }, [sendMessage, clearPingInterval]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!token) {
      console.warn('No token provided for WebSocket connection');
      return;
    }

    if (ws.current && (ws.current.readyState === WebSocket.CONNECTING || ws.current.readyState === WebSocket.OPEN)) {
      console.warn('WebSocket is already connected or connecting');
      return;
    }

    setStatus('connecting');
    clearReconnectTimeout();

    try {
      const wsUrl = `${url}?token=${encodeURIComponent(token)}`;
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setStatus('connected');
        reconnectCount.current = 0;
        startPingInterval();

        if (onConnect) {
          onConnect();
        }
      };

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);

          if (onMessage) {
            onMessage(message);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('error');

        if (onError) {
          onError(error);
        }
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setStatus('disconnected');
        clearPingInterval();

        if (onDisconnect) {
          onDisconnect();
        }

        // Attempt reconnection
        if (reconnect && reconnectCount.current < reconnectAttempts) {
          reconnectCount.current += 1;
          console.log(`Reconnecting... (Attempt ${reconnectCount.current}/${reconnectAttempts})`);

          reconnectTimeout.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (reconnectCount.current >= reconnectAttempts) {
          console.error('Max reconnection attempts reached');
        }
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setStatus('error');
    }
  }, [url, token, reconnect, reconnectAttempts, reconnectInterval, onConnect, onMessage, onError, onDisconnect, clearReconnectTimeout, clearPingInterval, startPingInterval]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    clearReconnectTimeout();
    clearPingInterval();

    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }

    setStatus('disconnected');
  }, [clearReconnectTimeout, clearPingInterval]);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect && token) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, token]); // Only reconnect when token changes

  return {
    status,
    sendMessage,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
    lastMessage,
    isConnected: status === 'connected',
  };
}


/**
 * Specialized hook for data collection WebSocket
 */
export function useDataCollectionWebSocket(token: string | null) {
  const [progress, setProgress] = useState<{
    [jobId: string]: {
      progress: number;
      status: string;
      stage?: string;
      symbol: string;
      error?: string;
    };
  }>({});

  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type.includes('data_collection')) {
      const { job_id, symbol, progress: prog, status, stage, error } = message;

      if (job_id) {
        setProgress((prev) => ({
          ...prev,
          [job_id]: {
            progress: prog || 0,
            status: status || 'unknown',
            stage,
            symbol,
            error,
          },
        }));
      }
    }
  }, []);

  const ws = useWebSocket({
    url: 'ws://localhost:8000/ws/data-collection',
    token,
    onMessage: handleMessage,
  });

  return {
    ...ws,
    progress,
  };
}


/**
 * Specialized hook for model training WebSocket
 */
export function useModelTrainingWebSocket(token: string | null) {
  const [trainingStatus, setTrainingStatus] = useState<{
    [jobId: string]: {
      progress: number;
      status: string;
      symbol: string;
      model_name: string;
      accuracy?: number;
      error?: string;
    };
  }>({});

  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type.includes('model_training')) {
      const { job_id, symbol, model_name, progress: prog, status, accuracy, error } = message;

      if (job_id) {
        setTrainingStatus((prev) => ({
          ...prev,
          [job_id]: {
            progress: prog || 0,
            status: status || 'unknown',
            symbol,
            model_name,
            accuracy,
            error,
          },
        }));
      }
    }
  }, []);

  const ws = useWebSocket({
    url: 'ws://localhost:8000/ws/model-training',
    token,
    onMessage: handleMessage,
  });

  return {
    ...ws,
    trainingStatus,
  };
}


/**
 * Specialized hook for admin panel with multiple channels
 */
export function useAdminWebSocket(token: string | null) {
  const [dataCollectionEvents, setDataCollectionEvents] = useState<WebSocketMessage[]>([]);
  const [modelTrainingEvents, setModelTrainingEvents] = useState<WebSocketMessage[]>([]);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type.includes('data_collection')) {
      setDataCollectionEvents((prev) => [...prev.slice(-50), message]); // Keep last 50 messages
    } else if (message.type.includes('model_training')) {
      setModelTrainingEvents((prev) => [...prev.slice(-50), message]);
    }
  }, []);

  const ws = useWebSocket({
    url: 'ws://localhost:8000/ws/admin',
    token,
    onMessage: handleMessage,
  });

  return {
    ...ws,
    dataCollectionEvents,
    modelTrainingEvents,
  };
}
