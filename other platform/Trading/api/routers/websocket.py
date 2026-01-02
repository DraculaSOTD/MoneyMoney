"""
WebSocket Router
================

WebSocket endpoints for real-time updates in the admin panel.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Optional
import logging
import json
import asyncio

from services.websocket_manager import connection_manager, MessageType
from api.routers.admin_auth import get_current_admin_from_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


async def verify_admin_websocket(websocket: WebSocket, token: str) -> Optional[dict]:
    """
    Verify admin authentication for WebSocket connection.

    Args:
        websocket: WebSocket connection
        token: JWT token

    Returns:
        Admin info dict if authenticated, None otherwise
    """
    try:
        admin = await get_current_admin_from_token(token)
        return {
            'admin_id': admin.id,
            'username': admin.username,
            'is_superuser': admin.is_superuser
        }
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return None


@router.websocket("/admin")
async def admin_websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT authentication token")
):
    """
    Main admin WebSocket endpoint for all real-time updates.

    **Connection URL:**
    ```
    ws://localhost:8000/ws/admin?token=YOUR_JWT_TOKEN
    ```

    **Message Format (Client → Server):**
    ```json
    {
        "action": "subscribe",
        "channel": "data_collection" | "model_training"
    }
    ```

    ```json
    {
        "action": "unsubscribe",
        "channel": "data_collection" | "model_training"
    }
    ```

    ```json
    {
        "action": "ping"
    }
    ```

    **Message Format (Server → Client):**

    Data Collection Progress:
    ```json
    {
        "type": "data_collection_progress",
        "job_id": "abc123",
        "symbol": "BTCUSDT",
        "progress": 45,
        "status": "fetching",
        "stage": "fetching",
        "timestamp": "2024-01-01T12:00:00Z"
    }
    ```

    Model Training Progress:
    ```json
    {
        "type": "model_training_progress",
        "job_id": "def456",
        "symbol": "ETHUSDT",
        "model_name": "ARIMA",
        "progress": 75,
        "status": "training",
        "accuracy": 85.5,
        "timestamp": "2024-01-01T12:00:00Z"
    }
    ```
    """
    # Verify authentication
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return

    admin_info = await verify_admin_websocket(websocket, token)
    if not admin_info:
        return

    # Connect the client
    client_id = f"admin_{admin_info['username']}"
    await connection_manager.connect(websocket, client_id)

    try:
        # Send welcome message with available channels
        await connection_manager.send_personal_message(websocket, {
            'type': 'welcome',
            'message': f"Welcome {admin_info['username']}!",
            'available_channels': ['data_collection', 'model_training'],
            'instructions': {
                'subscribe': 'Send {"action": "subscribe", "channel": "data_collection"}',
                'unsubscribe': 'Send {"action": "unsubscribe", "channel": "data_collection"}',
                'ping': 'Send {"action": "ping"}'
            }
        })

        # Keep connection alive and handle messages
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get('action')

                if action == 'subscribe':
                    channel = message.get('channel')

                    if channel == 'data_collection':
                        await connection_manager.subscribe_to_data_collection(websocket)
                        await connection_manager.send_personal_message(websocket, {
                            'type': 'subscribed',
                            'channel': 'data_collection',
                            'message': 'Subscribed to data collection updates'
                        })

                    elif channel == 'model_training':
                        await connection_manager.subscribe_to_model_training(websocket)
                        await connection_manager.send_personal_message(websocket, {
                            'type': 'subscribed',
                            'channel': 'model_training',
                            'message': 'Subscribed to model training updates'
                        })

                    else:
                        await connection_manager.send_personal_message(websocket, {
                            'type': 'error',
                            'message': f'Unknown channel: {channel}'
                        })

                elif action == 'unsubscribe':
                    channel = message.get('channel')

                    if channel == 'data_collection':
                        if websocket in connection_manager.data_collection_subscribers:
                            connection_manager.data_collection_subscribers.remove(websocket)
                        await connection_manager.send_personal_message(websocket, {
                            'type': 'unsubscribed',
                            'channel': 'data_collection',
                            'message': 'Unsubscribed from data collection updates'
                        })

                    elif channel == 'model_training':
                        if websocket in connection_manager.model_training_subscribers:
                            connection_manager.model_training_subscribers.remove(websocket)
                        await connection_manager.send_personal_message(websocket, {
                            'type': 'unsubscribed',
                            'channel': 'model_training',
                            'message': 'Unsubscribed from model training updates'
                        })

                elif action == 'ping':
                    await connection_manager.send_personal_message(websocket, {
                        'type': MessageType.PONG,
                        'message': 'pong'
                    })

                elif action == 'get_stats':
                    stats = connection_manager.get_stats()
                    await connection_manager.send_personal_message(websocket, {
                        'type': 'stats',
                        'data': stats
                    })

                else:
                    await connection_manager.send_personal_message(websocket, {
                        'type': 'error',
                        'message': f'Unknown action: {action}'
                    })

            except json.JSONDecodeError:
                await connection_manager.send_personal_message(websocket, {
                    'type': 'error',
                    'message': 'Invalid JSON format'
                })

            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await connection_manager.send_personal_message(websocket, {
                    'type': 'error',
                    'message': 'Internal server error'
                })

    except WebSocketDisconnect:
        logger.info(f"Admin WebSocket disconnected: {client_id}")
        connection_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(websocket)


@router.websocket("/data-collection")
async def data_collection_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT authentication token")
):
    """
    Dedicated WebSocket endpoint for data collection updates only.

    **Connection URL:**
    ```
    ws://localhost:8000/ws/data-collection?token=YOUR_JWT_TOKEN
    ```

    Automatically subscribes to data collection channel.
    """
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return

    admin_info = await verify_admin_websocket(websocket, token)
    if not admin_info:
        return

    client_id = f"admin_{admin_info['username']}_datacollection"
    await connection_manager.connect(websocket, client_id)

    # Auto-subscribe to data collection
    await connection_manager.subscribe_to_data_collection(websocket)

    try:
        # Keep connection alive
        while True:
            data = await websocket.receive_text()

            # Handle ping/pong
            try:
                message = json.loads(data)
                if message.get('action') == 'ping':
                    await connection_manager.send_personal_message(websocket, {
                        'type': MessageType.PONG,
                        'message': 'pong'
                    })
            except:
                pass

    except WebSocketDisconnect:
        logger.info(f"Data collection WebSocket disconnected: {client_id}")
        connection_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"Data collection WebSocket error: {e}")
        connection_manager.disconnect(websocket)


@router.websocket("/model-training")
async def model_training_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT authentication token")
):
    """
    Dedicated WebSocket endpoint for model training updates only.

    **Connection URL:**
    ```
    ws://localhost:8000/ws/model-training?token=YOUR_JWT_TOKEN
    ```

    Automatically subscribes to model training channel.
    """
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return

    admin_info = await verify_admin_websocket(websocket, token)
    if not admin_info:
        return

    client_id = f"admin_{admin_info['username']}_modeltraining"
    await connection_manager.connect(websocket, client_id)

    # Auto-subscribe to model training
    await connection_manager.subscribe_to_model_training(websocket)

    try:
        # Keep connection alive
        while True:
            data = await websocket.receive_text()

            # Handle ping/pong
            try:
                message = json.loads(data)
                if message.get('action') == 'ping':
                    await connection_manager.send_personal_message(websocket, {
                        'type': MessageType.PONG,
                        'message': 'pong'
                    })
            except:
                pass

    except WebSocketDisconnect:
        logger.info(f"Model training WebSocket disconnected: {client_id}")
        connection_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"Model training WebSocket error: {e}")
        connection_manager.disconnect(websocket)


@router.get("/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics.

    Returns current connection counts and subscriber information.
    """
    return connection_manager.get_stats()
