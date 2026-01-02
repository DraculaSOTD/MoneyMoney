from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal
import uuid

from database import get_session, Order, Position, Trade, OrderSide, OrderType, OrderStatus, PositionStatus, ProfileMetrics, TradingProfile
from database.models import SessionLocal

router = APIRouter(prefix="", tags=["trading"])

class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'LIMIT' or 'MARKET'
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    strategy: Optional[str] = None
    exchange: str = "binance"

class OrderResponse(BaseModel):
    order_id: str
    client_order_id: str
    symbol: str
    exchange: str
    side: str
    type: str
    status: str
    price: Optional[float]
    quantity: float
    executed_qty: float
    timestamp: datetime
    strategy: Optional[str]
    
    class Config:
        orm_mode = True

class PositionResponse(BaseModel):
    id: int
    symbol: str
    exchange: str
    side: str
    entry_price: float
    quantity: float
    current_price: float
    entry_time: datetime
    exit_time: Optional[datetime]
    status: str
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    strategy: Optional[str]
    
    class Config:
        orm_mode = True

class PositionsResponse(BaseModel):
    positions: List[PositionResponse]
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_value: float

class ClosePositionRequest(BaseModel):
    position_id: int
    quantity: Optional[float] = None  # If None, close entire position

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_price(db: Session, symbol: str) -> float:
    """Get the current price from the latest metrics"""
    latest_metric = db.query(ProfileMetrics).join(
        TradingProfile, ProfileMetrics.profile_id == TradingProfile.id
    ).filter(
        TradingProfile.symbol == symbol
    ).order_by(ProfileMetrics.timestamp.desc()).first()
    
    if not latest_metric:
        raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")
    
    return latest_metric.current_price

@router.post("/trading/order", response_model=OrderResponse)
async def place_order(order: OrderRequest, db: Session = Depends(get_db)):
    """Place a new order"""
    # Validate symbol exists
    profile = db.query(TradingProfile).filter(TradingProfile.symbol == order.symbol).first()
    if not profile:
        raise HTTPException(status_code=404, detail=f"Trading profile not found for {order.symbol}")
    
    # Get current price
    current_price = get_current_price(db, order.symbol)
    
    # For market orders, use current price
    if order.order_type == "MARKET":
        execution_price = current_price
    else:
        execution_price = order.price
        if not execution_price:
            raise HTTPException(status_code=400, detail="Price is required for limit orders")
    
    # Create order
    db_order = Order(
        order_id=str(uuid.uuid4()),
        client_order_id=f"CLI_{uuid.uuid4()}",
        symbol=order.symbol,
        exchange=order.exchange,
        side=OrderSide[order.side.upper()],
        type=OrderType[order.order_type.upper()],
        status=OrderStatus.NEW,
        price=execution_price,
        quantity=order.quantity,
        executed_qty=0,
        strategy=order.strategy
    )
    
    db.add(db_order)
    
    # Simulate immediate execution for market orders
    if order.order_type == "MARKET":
        # Mark order as filled
        db_order.status = OrderStatus.FILLED
        db_order.executed_qty = order.quantity
        
        # Create or update position
        existing_position = db.query(Position).filter(
            Position.symbol == order.symbol,
            Position.status == PositionStatus.OPEN,
            Position.side == OrderSide[order.side.upper()]
        ).first()
        
        if existing_position:
            # Update existing position (average price)
            total_value = (existing_position.quantity * existing_position.entry_price) + (order.quantity * execution_price)
            total_quantity = existing_position.quantity + order.quantity
            existing_position.quantity = total_quantity
            existing_position.entry_price = total_value / total_quantity
            existing_position.current_price = current_price
            existing_position.unrealized_pnl = (current_price - existing_position.entry_price) * total_quantity * (1 if order.side == "BUY" else -1)
        else:
            # Create new position
            new_position = Position(
                symbol=order.symbol,
                exchange=order.exchange,
                side=OrderSide[order.side.upper()],
                entry_price=execution_price,
                quantity=order.quantity,
                current_price=current_price,
                status=PositionStatus.OPEN,
                unrealized_pnl=0,
                realized_pnl=0,
                fees=order.quantity * execution_price * profile.trading_fee,
                strategy=order.strategy
            )
            db.add(new_position)
        
        # Create trade record
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=order.symbol,
            exchange=order.exchange,
            order_id=db_order.order_id,
            side=OrderSide[order.side.upper()],
            price=execution_price,
            quantity=order.quantity,
            commission=order.quantity * execution_price * profile.trading_fee,
            commission_asset=profile.quote_currency,
            strategy=order.strategy
        )
        db.add(trade)
    
    db.commit()
    db.refresh(db_order)
    
    return OrderResponse(
        order_id=db_order.order_id,
        client_order_id=db_order.client_order_id,
        symbol=db_order.symbol,
        exchange=db_order.exchange,
        side=db_order.side.value,
        type=db_order.type.value,
        status=db_order.status.value,
        price=db_order.price,
        quantity=db_order.quantity,
        executed_qty=db_order.executed_qty,
        timestamp=db_order.timestamp,
        strategy=db_order.strategy
    )

@router.get("/positions", response_model=PositionsResponse)
async def get_positions(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all positions"""
    query = db.query(Position)
    
    if status:
        query = query.filter(Position.status == PositionStatus[status.upper()])
    else:
        # Default to open positions
        query = query.filter(Position.status == PositionStatus.OPEN)
    
    if symbol:
        query = query.filter(Position.symbol == symbol)
    
    positions = query.all()
    
    # Update current prices and calculate P&L
    total_unrealized_pnl = 0
    total_realized_pnl = 0
    total_value = 0
    
    position_responses = []
    for position in positions:
        # Get current price
        current_price = get_current_price(db, position.symbol)
        position.current_price = current_price
        
        # Calculate unrealized P&L
        if position.status == PositionStatus.OPEN:
            if position.side == OrderSide.BUY:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity - position.fees
            else:  # SELL
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity - position.fees
            
            total_unrealized_pnl += position.unrealized_pnl
            total_value += position.quantity * current_price
        
        total_realized_pnl += position.realized_pnl
        
        position_responses.append(PositionResponse(
            id=position.id,
            symbol=position.symbol,
            exchange=position.exchange,
            side=position.side.value,
            entry_price=position.entry_price,
            quantity=position.quantity,
            current_price=position.current_price,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            status=position.status.value,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=position.unrealized_pnl,
            fees=position.fees,
            strategy=position.strategy
        ))
    
    db.commit()  # Save updated prices
    
    return PositionsResponse(
        positions=position_responses,
        total_unrealized_pnl=total_unrealized_pnl,
        total_realized_pnl=total_realized_pnl,
        total_value=total_value
    )

@router.post("/positions/close", response_model=PositionResponse)
async def close_position(request: ClosePositionRequest, db: Session = Depends(get_db)):
    """Close a position"""
    position = db.query(Position).filter(Position.id == request.position_id).first()
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    
    if position.status != PositionStatus.OPEN:
        raise HTTPException(status_code=400, detail="Position is not open")
    
    # Get current price
    current_price = get_current_price(db, position.symbol)
    
    # Calculate realized P&L
    close_quantity = request.quantity or position.quantity
    if close_quantity > position.quantity:
        raise HTTPException(status_code=400, detail="Cannot close more than position quantity")
    
    # Get profile for fees
    profile = db.query(TradingProfile).filter(TradingProfile.symbol == position.symbol).first()
    close_fees = close_quantity * current_price * profile.trading_fee
    
    if position.side == OrderSide.BUY:
        realized_pnl = (current_price - position.entry_price) * close_quantity - position.fees - close_fees
    else:  # SELL
        realized_pnl = (position.entry_price - current_price) * close_quantity - position.fees - close_fees
    
    # Create closing order
    close_order = Order(
        order_id=str(uuid.uuid4()),
        client_order_id=f"CLOSE_{uuid.uuid4()}",
        symbol=position.symbol,
        exchange=position.exchange,
        side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
        type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        price=current_price,
        quantity=close_quantity,
        executed_qty=close_quantity,
        strategy=position.strategy
    )
    db.add(close_order)
    
    # Create closing trade
    trade = Trade(
        trade_id=str(uuid.uuid4()),
        symbol=position.symbol,
        exchange=position.exchange,
        order_id=close_order.order_id,
        side=close_order.side,
        price=current_price,
        quantity=close_quantity,
        commission=close_fees,
        commission_asset=profile.quote_currency,
        strategy=position.strategy
    )
    db.add(trade)
    
    # Update position
    if close_quantity == position.quantity:
        # Full close
        position.status = PositionStatus.CLOSED
        position.exit_time = datetime.utcnow()
        position.quantity = 0
    else:
        # Partial close
        position.status = PositionStatus.PARTIAL
        position.quantity -= close_quantity
    
    position.realized_pnl += realized_pnl
    position.fees += close_fees
    position.current_price = current_price
    position.unrealized_pnl = 0 if position.quantity == 0 else (
        (current_price - position.entry_price) * position.quantity * (1 if position.side == OrderSide.BUY else -1)
    )
    
    db.commit()
    db.refresh(position)
    
    return PositionResponse(
        id=position.id,
        symbol=position.symbol,
        exchange=position.exchange,
        side=position.side.value,
        entry_price=position.entry_price,
        quantity=position.quantity,
        current_price=position.current_price,
        entry_time=position.entry_time,
        exit_time=position.exit_time,
        status=position.status.value,
        realized_pnl=position.realized_pnl,
        unrealized_pnl=position.unrealized_pnl,
        fees=position.fees,
        strategy=position.strategy
    )

@router.delete("/trading/order/{order_id}")
async def cancel_order(order_id: str, db: Session = Depends(get_db)):
    """Cancel an order"""
    order = db.query(Order).filter(Order.order_id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    if order.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
        raise HTTPException(status_code=400, detail="Order cannot be cancelled")
    
    order.status = OrderStatus.CANCELED
    db.commit()
    
    return {"message": "Order cancelled successfully"}