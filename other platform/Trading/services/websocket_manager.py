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
        self.model_training_subscribers: Set[WebSocket] = set()

        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

        # Keep track of last messages for reconnecting clients
        self.recent_messages: Dict[str, list] = {
            'data_collection': [],
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
        # Check if connection is still in our active set before sending
        if websocket not in self.active_connections:
            logger.debug("Attempted to send to disconnected WebSocket, skipping")
            return

        try:
            await websocket.send_json(message)
        except Exception as e:
            # Log at debug level to reduce noise during normal disconnections
            logger.debug(f"Error sending message (connection may have closed): {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, subscribers: Optional[Set[WebSocket]] = None):
        """
        Broadcast a message to all or specific subscribers.

        Args:
            message: Message dictionary
            subscribers: Specific set of subscribers (None = all active connections)
        """
        # Create a copy to avoid modification during iteration
        targets = set(subscribers) if subscribers is not None else self.active_connections.copy()

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
            # Skip if already disconnected
            if connection not in self.active_connections:
                continue
            try:
                await connection.send_json(message)
            except Exception as e:
                # Log at debug level to reduce noise during normal disconnections
                logger.debug(f"Error broadcasting (connection may have closed): {e}")
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
        message = {
            'type': MessageType.DATA_COLLECTION_PROGRESS,
            'job_id': job_id,
            'symbol': symbol,
            'progress': progress,
            'status': status,
            'stage': stage,
            'timestamp': datetime.utcnow().isoformat()
        }

        if error:
            message['error'] = error

        await self.broadcast(message, self.data_collection_subscribers)
        logger.debug(f"Broadcast data collection progress: {symbol} - {progress}%")

    async def broadcast_data_collection_started(self, job_id: str, symbol: str, days_back: int):
        """Broadcast data collection started event"""
        message = {
            'type': MessageType.DATA_COLLECTION_STARTED,
            'job_id': job_id,
            'symbol': symbol,
            'days_back': days_back,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.broadcast(message, self.data_collection_subscribers)
        logger.info(f"Broadcast data collection started: {symbol}")

    async def broadcast_data_collection_completed(self, job_id: str, symbol: str,
                                                  total_data_points: int):
        """Broadcast data collection completed event"""
        message = {
            'type': MessageType.DATA_COLLECTION_COMPLETED,
            'job_id': job_id,
            'symbol': symbol,
            'total_data_points': total_data_points,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.broadcast(message, self.data_collection_subscribers)
        logger.info(f"Broadcast data collection completed: {symbol} ({total_data_points} points)")

    async def broadcast_data_collection_failed(self, job_id: str, symbol: str, error: str):
        """Broadcast data collection failed event"""
        message = {
            'type': MessageType.DATA_COLLECTION_FAILED,
            'job_id': job_id,
            'symbol': symbol,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.broadcast(message, self.data_collection_subscribers)
        logger.warning(f"Broadcast data collection failed: {symbol} - {error}")

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
        message = {
            'type': MessageType.MODEL_TRAINING_PROGRESS,
            'job_id': job_id,
            'symbol': symbol,
            'model_name': model_name,
            'progress': progress,
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        }

        if accuracy is not None:
            message['accuracy'] = accuracy

        if error:
            message['error'] = error

        await self.broadcast(message, self.model_training_subscribers)
        logger.debug(f"Broadcast model training progress: {symbol}/{model_name} - {progress}%")

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
            'job_id': job_id,
            'symbol': symbol,
            'model_name': model_name,
            'accuracy': accuracy,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.broadcast(message, self.model_training_subscribers)
        logger.info(f"Broadcast model training completed: {symbol}/{model_name} - {accuracy:.2f}%")

    async def broadcast_model_training_failed(self, job_id: str, symbol: str,
                                             model_name: str, error: str):
        """Broadcast model training failed event"""
        message = {
            'type': MessageType.MODEL_TRAINING_FAILED,
            'job_id': job_id,
            'symbol': symbol,
            'model_name': model_name,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }

        await self.broadcast(message, self.model_training_subscribers)
        logger.warning(f"Broadcast model training failed: {symbol}/{model_name} - {error}")

    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'data_collection_subscribers': len(self.data_collection_subscribers),
            'model_training_subscribers': len(self.model_training_subscribers),
            'recent_messages_count': {
                'data_collection': len(self.recent_messages['data_collection']),
                'model_training': len(self.recent_messages['model_training'])
            }
        }


# Global connection manager instance
connection_manager = ConnectionManager()
