from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
from api.routers import profiles, trading, admin_auth, admin, data, websocket, user_data, data_quality, public_data
from security.admin_manager import init_admin_system
from services.data_updater import data_updater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Platform API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174", "http://localhost:3010"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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
app.include_router(public_data.router)  # Public API for Node.js backend

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
            "websocket": "ws://localhost:8002/ws/*",
            "system": "/system/status",
            "monitoring": "/monitoring/performance"
        }
    }

@app.get("/system/status")
async def get_system_status():
    """
    Get system status information
    Simplified endpoint for dashboard monitoring
    """
    return {
        "status": "RUNNING",
        "timestamp": datetime.utcnow().isoformat(),
        "data_manager": True,
        "execution_engine": False,
        "metrics": {
            "uptime": "running",
            "active_profiles": 1,
            "scheduler_running": data_updater.scheduler is not None and data_updater.scheduler.running
        }
    }

@app.get("/monitoring/performance")
async def get_monitoring_performance(period: str = "24h"):
    """
    Get monitoring performance metrics
    Simplified endpoint for dashboard monitoring
    """
    return {
        "period": period,
        "performance": {
            "total_pnl": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0
        },
        "system_health": {
            "status": "healthy",
            "api_latency_ms": 10,
            "database_connected": True
        },
        "model_performance": {
            "active_models": 0,
            "avg_accuracy": 0.0
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Trading API...")
    # Initialize admin system (create default admin if none exists)
    init_admin_system()
    logger.info("Admin system initialized")

    # Start auto-update scheduler
    data_updater.start_scheduler()
    logger.info("Auto-update scheduler initialized")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Trading API...")

    # Stop scheduler
    if data_updater.scheduler:
        data_updater.scheduler.shutdown()
        logger.info("Auto-update scheduler stopped")