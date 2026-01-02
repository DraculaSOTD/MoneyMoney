from dotenv import load_dotenv
load_dotenv()  # Load .env file before other imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
# import socketio  # Temporarily disabled - venv issue
from python_backend.api.routers import profiles, trading, admin_auth, admin, data, websocket, user_data, data_quality
from python_backend.security.admin_manager import init_admin_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Platform API", version="1.0.0")

# Socket.IO temporarily disabled due to venv import issues
# Will be re-enabled after fixing virtual environment setup
# sio = socketio.AsyncServer(
#     async_mode='asgi',
#     cors_allowed_origins='*',
#     logger=True,
#     engineio_logger=False
# )

# Wrap FastAPI app with Socket.IO
# socket_app = socketio.ASGIApp(
#     socketio_server=sio,
#     other_asgi_app=app,
#     socketio_path='/ws/socket.io'
# )

# Temporarily use app directly without Socket.IO wrapper
socket_app = app

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(profiles.router)
app.include_router(trading.router)
app.include_router(admin_auth.router)  # Admin authentication
app.include_router(admin.router)        # Admin management
app.include_router(data.router)         # Data aggregation (timeframes)
app.include_router(websocket.router)    # WebSocket real-time updates
app.include_router(user_data.router)    # User-facing API (MoneyMoney integration)
app.include_router(data_quality.router) # Data quality monitoring

@app.get("/")
async def root():
    return {
        "name": "Trading Platform API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "profiles": "/api/profiles/*",
            "trading": "/api/trading/*",
            "data": "/data/*",
            "user": "/api/user/*",
            "admin_auth": "/admin/auth/*",
            "admin": "/admin/*",
            "data_quality": "/api/data-quality/*",
            "websocket": "ws://localhost:8000/ws/*"
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Trading API...")
    # Initialize admin system (create default admin if none exists)
    init_admin_system()
    logger.info("Admin system initialized")

    # Initialize automatic data updater
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from python_backend.services.data_updater import data_updater

        scheduler = AsyncIOScheduler()

        # Schedule data updates every minute
        scheduler.add_job(
            data_updater.update_all_active_profiles,
            'cron',
            minute='*',  # Every minute
            id='auto_data_update',
            replace_existing=True,
            max_instances=1  # Prevent overlapping runs
        )

        scheduler.start()
        app.state.scheduler = scheduler
        logger.info("✓ Automatic data updater initialized (runs every minute)")
    except ImportError:
        logger.warning("⚠ APScheduler not installed - automatic updates disabled")
        logger.warning("  Install with: sudo pacman -S python-apscheduler")
    except Exception as e:
        logger.error(f"❌ Failed to initialize scheduler: {e}")

    # Socket.IO temporarily disabled
    # app.state.sio = sio

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Trading API...")

    # Shutdown scheduler if running
    if hasattr(app.state, 'scheduler'):
        try:
            app.state.scheduler.shutdown(wait=False)
            logger.info("✓ Scheduler shut down")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")

# ==================== Socket.IO Event Handlers (TEMPORARILY DISABLED) ====================
# Will be re-enabled after fixing venv issues

# @sio.event
# async def connect(sid, environ, auth):
#     """Handle Socket.IO connection with JWT authentication"""
#     try:
#         token = auth.get('token') if auth else None
#         if not token:
#             logger.warning(f"Socket.IO connection rejected - no token: {sid}")
#             return False
#         from python_backend.api.routers.admin_auth import verify_admin_token
#         payload = verify_admin_token(token)
#         await sio.save_session(sid, {
#             'admin_id': payload['sub'],
#             'username': payload.get('username', 'unknown')
#         })
#         logger.info(f"✅ Socket.IO connected: {sid} (admin: {payload.get('username', 'unknown')})")
#         return True
#     except Exception as e:
#         logger.error(f"❌ Socket.IO auth failed for {sid}: {e}")
#         return False

# @sio.event
# async def disconnect(sid):
#     """Handle Socket.IO disconnection"""
#     try:
#         session = await sio.get_session(sid)
#         username = session.get('username', 'unknown')
#         logger.info(f"Socket.IO disconnected: {sid} (admin: {username})")
#     except:
#         logger.info(f"Socket.IO disconnected: {sid}")

# @sio.event
# async def subscribe(sid, data):
#     """Handle subscription requests"""
#     try:
#         session = await sio.get_session(sid)
#         logger.info(f"Subscribe request from {session.get('username', sid)}: {data}")
#     except Exception as e:
#         logger.error(f"Subscribe error: {e}")

# @sio.event
# async def unsubscribe(sid, data):
#     """Handle unsubscription requests"""
#     try:
#         session = await sio.get_session(sid)
#         logger.info(f"Unsubscribe request from {session.get('username', sid)}: {data}")
#     except Exception as e:
#         logger.error(f"Unsubscribe error: {e}")