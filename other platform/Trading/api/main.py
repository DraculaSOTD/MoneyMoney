from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import asyncio
import json
import logging
import os
from enum import Enum

# from exchanges.binance_connector import OrderSide, OrderType
# from data_feeds.real_time_manager import RealTimeDataManager, DataType
# from trading.execution_engine import ExecutionEngine, RiskLimits, PositionStatus
from api.routers import profiles, ml_pipeline
from database.models import OrderSide, OrderType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto ML Trading API", version="1.0.0")

# Include routers
app.include_router(profiles.router)
app.include_router(ml_pipeline.router, prefix="/api/ml", tags=["ML Pipeline"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

class SystemStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    INITIALIZING = "initializing"

class OrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    strategy: Optional[str] = None
    exchange: str = "binance"

class PositionResponse(BaseModel):
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    status: str

class SystemConfig(BaseModel):
    max_position_size: float = 10000
    max_positions: int = 10
    max_daily_loss: float = 1000
    max_drawdown: float = 0.1
    max_exposure: float = 50000
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05

class DataSubscription(BaseModel):
    symbols: List[str]
    data_types: List[str]
    interval: str = "1m"

class GlobalState:
    def __init__(self):
        self.data_manager: Optional[RealTimeDataManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.system_status = SystemStatus.STOPPED
        self.websocket_connections: List[WebSocket] = []
        self.api_key: Optional[str] = None
        self.ml_system = None  # Will be initialized when main.py integration is added

state = GlobalState()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    expected_token = os.getenv("API_TOKEN", "your-secret-token")
    
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    
    return token

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Trading API...")
    state.api_key = os.getenv("API_KEY", "demo-key")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Trading API...")
    if state.data_manager:
        await state.data_manager.stop()
    state.system_status = SystemStatus.STOPPED

@app.get("/")
async def root():
    return {
        "name": "Crypto ML Trading API",
        "status": state.system_status,
        "version": "1.0.0",
        "endpoints": {
            "system": "/system/*",
            "trading": "/trading/*",
            "data": "/data/*",
            "positions": "/positions/*",
            "profiles": "/api/profiles/*"
        }
    }

@app.post("/system/start", dependencies=[Depends(verify_token)])
async def start_system(config: SystemConfig):
    try:
        if state.system_status == SystemStatus.RUNNING:
            return {"status": "already_running"}
        
        state.system_status = SystemStatus.INITIALIZING
        
        risk_limits = RiskLimits(
            max_position_size=Decimal(str(config.max_position_size)),
            max_positions=config.max_positions,
            max_daily_loss=Decimal(str(config.max_daily_loss)),
            max_drawdown=Decimal(str(config.max_drawdown)),
            max_exposure=Decimal(str(config.max_exposure)),
            stop_loss_pct=Decimal(str(config.stop_loss_pct)),
            take_profit_pct=Decimal(str(config.take_profit_pct))
        )
        
        state.data_manager = RealTimeDataManager(redis_url=os.getenv("REDIS_URL"))
        await state.data_manager.start()
        
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        await state.data_manager.add_binance_exchange(api_key, api_secret, testnet)
        
        state.execution_engine = ExecutionEngine(risk_limits)
        await state.execution_engine.initialize(state.data_manager)
        
        state.system_status = SystemStatus.RUNNING
        
        return {
            "status": "started",
            "config": config.dict(),
            "testnet": testnet
        }
        
    except Exception as e:
        state.system_status = SystemStatus.ERROR
        logger.error(f"System start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/stop", dependencies=[Depends(verify_token)])
async def stop_system():
    if state.system_status != SystemStatus.RUNNING:
        return {"status": "not_running"}
    
    if state.data_manager:
        await state.data_manager.stop()
    
    state.system_status = SystemStatus.STOPPED
    return {"status": "stopped"}

@app.get("/system/status")
async def get_system_status():
    metrics = {}
    if state.execution_engine:
        metrics = state.execution_engine.get_metrics()
    
    return {
        "status": state.system_status,
        "timestamp": datetime.now().isoformat(),
        "data_manager": state.data_manager is not None,
        "execution_engine": state.execution_engine is not None,
        "metrics": metrics
    }

@app.post("/trading/order", dependencies=[Depends(verify_token)])
async def place_order(request: OrderRequest):
    if state.system_status != SystemStatus.RUNNING:
        raise HTTPException(status_code=400, detail="System not running")
    
    try:
        order = await state.execution_engine.place_order(
            request.exchange,
            request.symbol,
            OrderSide(request.side),
            OrderType(request.order_type),
            Decimal(str(request.quantity)),
            Decimal(str(request.price)) if request.price else None,
            request.strategy
        )
        
        if not order:
            raise HTTPException(status_code=400, detail="Order rejected")
        
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "status": order.status.value,
            "quantity": float(order.quantity),
            "executed_qty": float(order.executed_qty),
            "price": float(order.price) if order.price else None,
            "timestamp": order.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/trading/order/{order_id}", dependencies=[Depends(verify_token)])
async def cancel_order(order_id: str, symbol: str, exchange: str = "binance"):
    if state.system_status != SystemStatus.RUNNING:
        raise HTTPException(status_code=400, detail="System not running")
    
    try:
        order = await state.execution_engine.cancel_order(exchange, symbol, order_id)
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return {
            "order_id": order.order_id,
            "status": order.status.value,
            "message": "Order cancelled"
        }
        
    except Exception as e:
        logger.error(f"Order cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions", dependencies=[Depends(verify_token)])
async def get_positions(status: Optional[str] = None):
    if not state.execution_engine:
        return {"positions": []}
    
    position_status = PositionStatus(status) if status else None
    positions = state.execution_engine.get_positions(position_status)
    
    return {
        "positions": [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "quantity": float(p.quantity),
                "entry_price": float(p.entry_price),
                "current_price": float(p.current_price),
                "unrealized_pnl": float(p.unrealized_pnl),
                "realized_pnl": float(p.realized_pnl),
                "status": p.status.value,
                "entry_time": p.entry_time.isoformat(),
                "strategy": p.strategy
            }
            for p in positions
        ],
        "total_pnl": float(state.execution_engine.get_total_pnl()) if state.execution_engine else 0
    }

@app.post("/positions/close", dependencies=[Depends(verify_token)])
async def close_position(symbol: str, side: str, quantity: Optional[float] = None):
    if state.system_status != SystemStatus.RUNNING:
        raise HTTPException(status_code=400, detail="System not running")
    
    try:
        order = await state.execution_engine.close_position(
            symbol, 
            OrderSide(side),
            Decimal(str(quantity)) if quantity else None
        )
        
        if not order:
            raise HTTPException(status_code=404, detail="Position not found")
        
        return {
            "order_id": order.order_id,
            "status": order.status.value,
            "message": "Position closed"
        }
        
    except Exception as e:
        logger.error(f"Position close failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/subscribe", dependencies=[Depends(verify_token)])
async def subscribe_data(subscription: DataSubscription):
    if state.system_status != SystemStatus.RUNNING:
        raise HTTPException(status_code=400, detail="System not running")
    
    try:
        data_types = [DataType(dt) for dt in subscription.data_types]
        
        await state.data_manager.start_multiple_streams(
            "binance",
            subscription.symbols,
            data_types,
            subscription.interval
        )
        
        return {
            "status": "subscribed",
            "symbols": subscription.symbols,
            "data_types": subscription.data_types
        }
        
    except Exception as e:
        logger.error(f"Data subscription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/snapshot")
async def get_market_snapshot():
    if not state.data_manager:
        return {"snapshot": {}}
    
    return {"snapshot": state.data_manager.get_market_snapshot()}

@app.get("/data/{symbol}/ticker")
async def get_ticker(symbol: str):
    if not state.data_manager:
        raise HTTPException(status_code=400, detail="Data manager not initialized")
    
    ticker = state.data_manager.get_latest_ticker(symbol)
    
    if not ticker:
        raise HTTPException(status_code=404, detail="Ticker not found")
    
    return {
        "symbol": ticker.symbol,
        "bid": float(ticker.bid_price),
        "ask": float(ticker.ask_price),
        "last": float(ticker.last_price),
        "volume": float(ticker.volume),
        "timestamp": ticker.timestamp.isoformat()
    }

@app.get("/data/{symbol}/klines")
async def get_klines(symbol: str, limit: int = 100):
    if not state.data_manager:
        raise HTTPException(status_code=400, detail="Data manager not initialized")
    
    klines_df = state.data_manager.get_latest_klines(symbol, limit)
    
    if klines_df.empty:
        return {"klines": []}
    
    klines = []
    for idx, row in klines_df.iterrows():
        klines.append({
            "timestamp": idx.isoformat(),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume']
        })
    
    return {"klines": klines}

@app.get("/monitoring/performance")
async def get_performance_metrics():
    if not state.ml_system:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    # Get metrics from live trading integration
    from live_trading_integration import LiveTradingIntegration
    if hasattr(state, 'integration') and state.integration.metrics_collector:
        return {
            "performance": state.integration.metrics_collector.get_performance_summary(),
            "system_health": state.integration.metrics_collector.get_system_health(),
            "model_performance": state.integration.metrics_collector.get_model_performance()
        }
    else:
        return {"error": "Metrics collector not available"}

@app.get("/monitoring/alerts")
async def get_alerts(limit: int = 100):
    if hasattr(state, 'integration') and state.integration.alert_manager:
        return {
            "alerts": state.integration.alert_manager.get_alert_history(limit),
            "channels": state.integration.alert_manager.get_channel_status()
        }
    else:
        return {"alerts": [], "channels": {}}

@app.get("/monitoring/report")
async def get_performance_report():
    if hasattr(state, 'integration') and state.integration.metrics_collector:
        report = state.integration.metrics_collector.create_performance_report()
        return {"report": report}
    else:
        return {"report": "Metrics collector not available"}

@app.get("/monitoring/metrics/export")
async def export_metrics():
    if hasattr(state, 'integration') and state.integration.metrics_collector:
        metrics_json = state.integration.metrics_collector.export_metrics_json()
        return JSONResponse(content=json.loads(metrics_json))
    else:
        return {"error": "Metrics collector not available"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.websocket_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("type") == "subscribe":
                symbols = msg.get("symbols", [])
                for symbol in symbols:
                    def create_callback(ws, sym):
                        def callback(market_data):
                            asyncio.create_task(ws.send_json({
                                "type": "market_data",
                                "symbol": sym,
                                "data": {
                                    "bid": float(market_data.data.bid_price),
                                    "ask": float(market_data.data.ask_price),
                                    "last": float(market_data.data.last_price),
                                    "volume": float(market_data.data.volume),
                                    "timestamp": market_data.timestamp.isoformat()
                                }
                            }))
                        return callback
                    
                    if state.data_manager:
                        state.data_manager.subscribe(
                            symbol, DataType.TICKER, 
                            create_callback(websocket, symbol)
                        )
                        
                await websocket.send_json({
                    "type": "subscribed",
                    "symbols": symbols
                })
                
    except WebSocketDisconnect:
        state.websocket_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        state.websocket_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)