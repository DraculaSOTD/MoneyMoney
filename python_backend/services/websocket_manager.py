"""
WebSocket Manager
=================

Manages WebSocket connections for real-time updates in the admin panel.
Supports broadcasting progress updates for data collection and model training.
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types"""
    # Data collection
    DATA_COLLECTION_STARTED = "data_collection_started"
    DATA_COLLECTION_PROGRESS = "data_collection_progress"
    DATA_COLLECTION_COMPLETED = "data_collection_completed"
    DATA_COLLECTION_FAILED = "data_collection_failed"

    # Indicator preprocessing
    PREPROCESSING_STARTED = "preprocessing_started"
    PREPROCESSING_PROGRESS = "preprocessing_progress"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    PREPROCESSING_FAILED = "preprocessing_failed"

    # Model training
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_PROGRESS = "model_training_progress"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    MODEL_TRAINING_FAILED = "model_training_failed"

    # General
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts messages to connected clients.
    """

    def __init__(self):
        # Store active connections
        self.active_connections: Set[WebSocket] = set()

        # Store connections by subscription type
        self.data_collection_subscribers: Set[WebSocket] = set()
        self.preprocessing_subscribers: Set[WebSocket] = set()
        self.model_training_subscribers: Set[WebSocket] = set()

        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

        # Keep track of last messages for reconnecting clients
        self.recent_messages: Dict[str, list] = {
            'data_collection': [],
            'preprocessing': [],
            'model_training': []
        }
        self.max_recent_messages = 50

        logger.info("WebSocket ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Optional client identifier
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            'client_id': client_id or f"client_{id(websocket)}",
            'connected_at': datetime.utcnow(),
            'subscriptions': []
        }

        logger.info(f"WebSocket connected: {self.connection_metadata[websocket]['client_id']}")

        # Send connection confirmation
        await self.send_personal_message(websocket, {
            'type': MessageType.CONNECTED,
            'message': 'Connected to Trading Platform WebSocket',
            'timestamp': datetime.utcnow().isoformat(),
            'client_id': self.connection_metadata[websocket]['client_id']
        })

    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.data_collection_subscribers:
            self.data_collection_subscribers.remove(websocket)

        if websocket in self.preprocessing_subscribers:
            self.preprocessing_subscribers.remove(websocket)

        if websocket in self.model_training_subscribers:
            self.model_training_subscribers.remove(websocket)

        client_id = self.connection_metadata.get(websocket, {}).get('client_id', 'unknown')

        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        logger.info(f"WebSocket disconnected: {client_id}")

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """
        Send a message to a specific WebSocket connection.

        Args:
            websocket: Target WebSocket
            message: Message dictionary
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, subscribers: Optional[Set[WebSocket]] = None):
        """
        Broadcast a message to all or specific subscribers.

        Args:
            message: Message dictionary
            subscribers: Specific set of subscribers (None = all active connections)
        """
        targets = subscribers if subscribers is not None else self.active_connections

        # Store in recent messages for reconnecting clients
        msg_category = self._get_message_category(message.get('type'))
        if msg_category:
            self.recent_messages[msg_category].append(message)
            # Keep only recent messages
            if len(self.recent_messages[msg_category]) > self.max_recent_messages:
                self.recent_messages[msg_category] = self.recent_messages[msg_category][-self.max_recent_messages:]

        # Broadcast to all targets
        disconnected = set()
        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def _get_message_category(self, message_type: Optional[str]) -> Optional[str]:
        """Determine message category for storage"""
        if not message_type:
            return None

        if 'data_collection' in message_type:
            return 'data_collection'
        elif 'preprocessing' in message_type:
            return 'preprocessing'
        elif 'model_training' in message_type:
            return 'model_training'

        return None

    async def subscribe_to_data_collection(self, websocket: WebSocket):
        """Subscribe a connection to data collection updates"""
        self.data_collection_subscribers.add(websocket)

        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]['subscriptions'].append('data_collection')

        logger.info(f"Client subscribed to data_collection: {self.connection_metadata.get(websocket, {}).get('client_id')}")

        # Send recent messages to catch up
        for msg in self.recent_messages['data_collection'][-10:]:  # Last 10 messages
            await self.send_personal_message(websocket, msg)

    async def subscribe_to_preprocessing(self, websocket: WebSocket):
        """Subscribe a connection to preprocessing updates"""
        self.preprocessing_subscribers.add(websocket)

        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]['subscriptions'].append('preprocessing')

        logger.info(f"Client subscribed to preprocessing: {self.connection_metadata.get(websocket, {}).get('client_id')}")

        # Send recent messages to catch up
        for msg in self.recent_messages['preprocessing'][-10:]:
            await self.send_personal_message(websocket, msg)

    async def subscribe_to_model_training(self, websocket: WebSocket):
        """Subscribe a connection to model training updates"""
        self.model_training_subscribers.add(websocket)

        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]['subscriptions'].append('model_training')

        logger.info(f"Client subscribed to model_training: {self.connection_metadata.get(websocket, {}).get('client_id')}")

        # Send recent messages to catch up
        for msg in self.recent_messages['model_training'][-10:]:
            await self.send_personal_message(websocket, msg)

    async def broadcast_data_collection_progress(self, job_id: str, symbol: str,
                                                 progress: int, status: str,
                                                 stage: Optional[str] = None,
                                                 error: Optional[str] = None):
        """
        Broadcast data collection progress update.

        Args:
            job_id: Job identifier
            symbol: Trading symbol
            progress: Progress percentage (0-100)
            status: Job status
            stage: Current stage (fetching/preprocessing/storing)
            error: Error message if failed
        """
        data_payload = {
            'job_id': job_id,
            'symbol': symbol,
            'progress': progress,
            'status': status,
            'current_stage': stage,  # Frontend expects "current_stage" not "stage"
            'total_records': 0,  # Placeholder for frontend compatibility
            'timestamp': datetime.utcnow().isoformat()
        }

        if error:
            data_payload['error'] = error

        message = {
            'type': MessageType.DATA_COLLECTION_PROGRESS,
            'data': data_payload
        }

        await self.broadcast(message, self.data_collection_subscribers)
        logger.debug(f"Broadcast data collection progress: {symbol} - {progress}%")

    async def broadcast_data_collection_started(self, job_id: str, symbol: str, days_back: int):
        """Broadcast data collection started event"""
        message = {
            'type': MessageType.DATA_COLLECTION_STARTED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'days_back': days_back,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.data_collection_subscribers)
        logger.info(f"Broadcast data collection started: {symbol}")

    async def broadcast_data_collection_completed(self, job_id: str, symbol: str,
                                                  total_records: int):
        """Broadcast data collection completed event"""
        message = {
            'type': MessageType.DATA_COLLECTION_COMPLETED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'total_records': total_records,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.data_collection_subscribers)
        logger.info(f"Broadcast data collection completed: {symbol} ({total_records} records)")

    async def broadcast_data_collection_failed(self, job_id: str, symbol: str, error: str):
        """Broadcast data collection failed event"""
        message = {
            'type': MessageType.DATA_COLLECTION_FAILED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'error': error,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.data_collection_subscribers)
        logger.warning(f"Broadcast data collection failed: {symbol} - {error}")

    async def broadcast_preprocessing_started(self, job_id: str, symbol: str, total_records: int):
        """Broadcast preprocessing started event"""
        message = {
            'type': MessageType.PREPROCESSING_STARTED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'total_records': total_records,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.preprocessing_subscribers)
        logger.info(f"Broadcast preprocessing started: {symbol} ({total_records} records)")

    async def broadcast_preprocessing_progress(self, job_id: str, symbol: str,
                                              progress: int, status: str,
                                              stage: Optional[str] = None,
                                              records_processed: Optional[int] = None,
                                              total_records: Optional[int] = None):
        """
        Broadcast preprocessing progress update.

        Args:
            job_id: Job identifier
            symbol: Trading symbol
            progress: Progress percentage (0-100)
            status: Job status
            stage: Current stage (loading/calculating/storing)
            records_processed: Number of records processed so far
            total_records: Total number of records to process
        """
        data_payload = {
            'job_id': job_id,
            'symbol': symbol,
            'progress': progress,
            'status': status,
            'current_stage': stage,
            'timestamp': datetime.utcnow().isoformat()
        }

        if records_processed is not None:
            data_payload['records_processed'] = records_processed
        if total_records is not None:
            data_payload['total_records'] = total_records

        message = {
            'type': MessageType.PREPROCESSING_PROGRESS,
            'data': data_payload
        }

        await self.broadcast(message, self.preprocessing_subscribers)
        logger.debug(f"Broadcast preprocessing progress: {symbol} - {progress}% ({stage})")

    async def broadcast_preprocessing_completed(self, job_id: str, symbol: str,
                                               total_records: int, indicators_count: int):
        """Broadcast preprocessing completed event"""
        message = {
            'type': MessageType.PREPROCESSING_COMPLETED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'total_records': total_records,
                'indicators_count': indicators_count,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.preprocessing_subscribers)
        logger.info(f"Broadcast preprocessing completed: {symbol} ({total_records} records, {indicators_count} indicators)")

    async def broadcast_preprocessing_failed(self, job_id: str, symbol: str, error: str):
        """Broadcast preprocessing failed event"""
        message = {
            'type': MessageType.PREPROCESSING_FAILED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'error': error,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.preprocessing_subscribers)
        logger.warning(f"Broadcast preprocessing failed: {symbol} - {error}")

    async def broadcast_model_training_progress(self, job_id: str, symbol: str,
                                               model_name: str, progress: int,
                                               status: str, accuracy: Optional[float] = None,
                                               error: Optional[str] = None):
        """
        Broadcast model training progress update.

        Args:
            job_id: Job identifier
            symbol: Trading symbol
            model_name: Model being trained
            progress: Progress percentage (0-100)
            status: Job status
            accuracy: Model accuracy if available
            error: Error message if failed
        """
        data_obj = {
            'job_id': job_id,
            'symbol': symbol,
            'model_name': model_name,
            'progress': progress,
            'message': status,  # Frontend expects 'message' field
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        }

        if accuracy is not None:
            data_obj['accuracy'] = accuracy

        if error:
            data_obj['error'] = error

        message = {
            'type': MessageType.MODEL_TRAINING_PROGRESS,
            'data': data_obj
        }

        subscriber_count = len(self.model_training_subscribers)
        await self.broadcast(message, self.model_training_subscribers)
        logger.info(f"📡 Broadcast training progress to {subscriber_count} subscribers: {symbol}/{model_name} - {progress}% - {status}")

    async def broadcast_model_training_started(self, job_id: str, symbol: str,
                                              models: list):
        """Broadcast model training started event"""
        message = {
            'type': MessageType.MODEL_TRAINING_STARTED,
            'job_id': job_id,
            'symbol': symbol,
            'models': models,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.broadcast(message, self.model_training_subscribers)
        logger.info(f"Broadcast model training started: {symbol} - {models}")

    async def broadcast_model_training_completed(self, job_id: str, symbol: str,
                                                model_name: str, accuracy: float):
        """Broadcast model training completed event"""
        message = {
            'type': MessageType.MODEL_TRAINING_COMPLETED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'model_name': model_name,
                'accuracy': accuracy,
                'message': f"Training completed for {model_name}",
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.model_training_subscribers)
        logger.info(f"Broadcast model training completed: {symbol}/{model_name} - {accuracy:.2f}%")

    async def broadcast_model_training_failed(self, job_id: str, symbol: str,
                                             model_name: str, error: str):
        """Broadcast model training failed event"""
        message = {
            'type': MessageType.MODEL_TRAINING_FAILED,
            'data': {
                'job_id': job_id,
                'symbol': symbol,
                'model_name': model_name,
                'error': error,
                'message': f"Training failed for {model_name}: {error}",
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast(message, self.model_training_subscribers)
        logger.warning(f"Broadcast model training failed: {symbol}/{model_name} - {error}")

    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'data_collection_subscribers': len(self.data_collection_subscribers),
            'preprocessing_subscribers': len(self.preprocessing_subscribers),
            'model_training_subscribers': len(self.model_training_subscribers),
            'recent_messages_count': {
                'data_collection': len(self.recent_messages['data_collection']),
                'preprocessing': len(self.recent_messages['preprocessing']),
                'model_training': len(self.recent_messages['model_training'])
            }
        }


# Global connection manager instance
connection_manager = ConnectionManager()
