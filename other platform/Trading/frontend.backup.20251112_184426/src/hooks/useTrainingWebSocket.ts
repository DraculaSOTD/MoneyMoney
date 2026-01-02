import { useEffect, useRef, useCallback, useState } from 'react';
import { useDispatch } from 'react-redux';
import { 
  setDataFetchJob, 
  setPreprocessJob, 
  updateTrainingJob,
  type DataPipelineJob
} from '../features/mlModels/mlModelsSlice';

interface WebSocketMessage {
  type: 'connection' | 'job_update' | 'pong';
  job_id?: string;
  status?: string;
  progress?: number;
  current_step?: string;
  message?: string;
  result?: any;
  error?: string;
}

interface UseTrainingWebSocketOptions {
  onJobUpdate?: (jobId: string, update: DataPipelineJob) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

export const useTrainingWebSocket = (options: UseTrainingWebSocketOptions = {}) => {
  const {
    onJobUpdate,
    autoReconnect = true,
    reconnectInterval = 5000,
  } = options;

  const dispatch = useDispatch();
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const pingInterval = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const connect = useCallback(() => {
    try {
      // Use the same base URL as the API, but with ws:// protocol
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/api/ml/ws/training`;
      
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket connected for training updates');
        setIsConnected(true);
        setConnectionError(null);
        
        // Start ping interval to keep connection alive
        pingInterval.current = setInterval(() => {
          if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000); // Ping every 30 seconds
      };

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          if (message.type === 'job_update' && message.job_id) {
            const jobUpdate: DataPipelineJob = {
              job_id: message.job_id,
              status: message.status as any,
              progress: message.progress || 0,
              current_step: message.current_step || '',
              message: message.message || '',
              result: message.result,
              error: message.error,
            };

            // Determine job type based on job_id or current_step
            if (message.current_step?.includes('Fetching') || 
                message.current_step?.includes('Downloading')) {
              dispatch(setDataFetchJob(jobUpdate));
            } else if (message.current_step?.includes('Preprocessing') || 
                      message.current_step?.includes('Feature')) {
              dispatch(setPreprocessJob(jobUpdate));
            } else if (message.current_step?.includes('Training')) {
              // For training jobs, update the training job state
              dispatch(updateTrainingJob({
                id: message.job_id,
                status: message.status as any,
                progress: message.progress || 0,
                model_type: 'unknown', // This would come from the actual implementation
                start_time: new Date().toISOString(),
                logs: [message.message || ''],
              }));
            }

            // Call custom callback if provided
            if (onJobUpdate) {
              onJobUpdate(message.job_id, jobUpdate);
            }
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionError('WebSocket connection error');
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        // Clear ping interval
        if (pingInterval.current) {
          clearInterval(pingInterval.current);
          pingInterval.current = null;
        }
        
        // Attempt to reconnect if enabled
        if (autoReconnect && !reconnectTimeout.current) {
          reconnectTimeout.current = setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            reconnectTimeout.current = null;
            connect();
          }, reconnectInterval);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionError('Failed to connect to WebSocket');
    }
  }, [dispatch, onJobUpdate, autoReconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (pingInterval.current) {
      clearInterval(pingInterval.current);
      pingInterval.current = null;
    }
    
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const subscribeToJob = useCallback((jobId: string) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: 'subscribe',
        job_id: jobId,
      }));
    }
  }, []);

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    connectionError,
    subscribeToJob,
    reconnect: connect,
    disconnect,
  };
};