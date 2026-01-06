import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from decimal import Decimal

from database.models import (
    Trade, Order, Position, ModelPrediction, SystemMetrics,
    RiskMetrics, Alert, BacktestResult, DataQuality,
    OrderStatus, PositionStatus, get_session
)

logger = logging.getLogger(__name__)

class TradingRepository:
    def __init__(self, database_url: str):
        self.session = get_session(database_url)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
        
    def save_trade(self, trade_data: Dict[str, Any]) -> Trade:
        trade = Trade(
            trade_id=trade_data['trade_id'],
            symbol=trade_data['symbol'],
            exchange=trade_data.get('exchange', 'binance'),
            order_id=trade_data['order_id'],
            side=trade_data['side'],
            price=float(trade_data['price']),
            quantity=float(trade_data['quantity']),
            commission=float(trade_data.get('commission', 0)),
            commission_asset=trade_data.get('commission_asset', 'USDT'),
            timestamp=trade_data.get('timestamp', datetime.utcnow()),
            strategy=trade_data.get('strategy')
        )
        self.session.add(trade)
        self.session.commit()
        return trade
        
    def save_order(self, order_data: Dict[str, Any]) -> Order:
        order = Order(
            order_id=order_data['order_id'],
            client_order_id=order_data.get('client_order_id'),
            symbol=order_data['symbol'],
            exchange=order_data.get('exchange', 'binance'),
            side=order_data['side'],
            type=order_data['type'],
            status=order_data['status'],
            price=float(order_data.get('price', 0)),
            quantity=float(order_data['quantity']),
            executed_qty=float(order_data.get('executed_qty', 0)),
            timestamp=order_data.get('timestamp', datetime.utcnow()),
            update_time=order_data.get('update_time', datetime.utcnow()),
            strategy=order_data.get('strategy')
        )
        self.session.add(order)
        self.session.commit()
        return order
        
    def update_order_status(self, order_id: str, status: OrderStatus, 
                           executed_qty: Optional[float] = None) -> Optional[Order]:
        order = self.session.query(Order).filter_by(order_id=order_id).first()
        if order:
            order.status = status
            order.update_time = datetime.utcnow()
            if executed_qty is not None:
                order.executed_qty = executed_qty
            self.session.commit()
        return order
        
    def save_position(self, position_data: Dict[str, Any]) -> Position:
        position = Position(
            symbol=position_data['symbol'],
            exchange=position_data.get('exchange', 'binance'),
            side=position_data['side'],
            entry_price=float(position_data['entry_price']),
            quantity=float(position_data['quantity']),
            current_price=float(position_data.get('current_price', position_data['entry_price'])),
            entry_time=position_data.get('entry_time', datetime.utcnow()),
            status=position_data.get('status', PositionStatus.OPEN),
            strategy=position_data.get('strategy')
        )
        self.session.add(position)
        self.session.commit()
        return position
        
    def update_position(self, position_id: int, current_price: float,
                       unrealized_pnl: float) -> Optional[Position]:
        position = self.session.query(Position).get(position_id)
        if position:
            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl
            self.session.commit()
        return position
        
    def close_position(self, position_id: int, exit_price: float,
                      realized_pnl: float, fees: float) -> Optional[Position]:
        position = self.session.query(Position).get(position_id)
        if position:
            position.status = PositionStatus.CLOSED
            position.exit_time = datetime.utcnow()
            position.current_price = exit_price
            position.realized_pnl = realized_pnl
            position.fees = fees
            position.unrealized_pnl = 0
            self.session.commit()
        return position
        
    def get_open_positions(self, symbol: Optional[str] = None,
                          strategy: Optional[str] = None) -> List[Position]:
        query = self.session.query(Position).filter_by(status=PositionStatus.OPEN)
        
        if symbol:
            query = query.filter_by(symbol=symbol)
        if strategy:
            query = query.filter_by(strategy=strategy)
            
        return query.all()
        
    def save_model_prediction(self, prediction_data: Dict[str, Any]) -> ModelPrediction:
        prediction = ModelPrediction(
            model_name=prediction_data['model_name'],
            symbol=prediction_data['symbol'],
            timestamp=prediction_data.get('timestamp', datetime.utcnow()),
            prediction=float(prediction_data['prediction']),
            confidence=float(prediction_data.get('confidence', 0)),
            features=prediction_data.get('features', {}),
            actual_price=float(prediction_data['actual_price']) if 'actual_price' in prediction_data else None,
            error=float(prediction_data['error']) if 'error' in prediction_data else None
        )
        self.session.add(prediction)
        self.session.commit()
        return prediction
        
    def get_recent_predictions(self, model_name: str, symbol: str,
                             hours: int = 24) -> List[ModelPrediction]:
        since = datetime.utcnow() - timedelta(hours=hours)
        return self.session.query(ModelPrediction).filter(
            and_(
                ModelPrediction.model_name == model_name,
                ModelPrediction.symbol == symbol,
                ModelPrediction.timestamp >= since
            )
        ).order_by(ModelPrediction.timestamp.desc()).all()
        
    def save_system_metric(self, metric_type: str, metric_name: str,
                          value: float, metadata: Optional[Dict] = None):
        metric = SystemMetrics(
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {}
        )
        self.session.add(metric)
        self.session.commit()
        return metric
        
    def save_risk_metrics(self, metrics: Dict[str, Any]) -> RiskMetrics:
        risk_metric = RiskMetrics(
            portfolio_value=float(metrics['portfolio_value']),
            daily_pnl=float(metrics['daily_pnl']),
            total_pnl=float(metrics['total_pnl']),
            sharpe_ratio=float(metrics.get('sharpe_ratio', 0)),
            max_drawdown=float(metrics.get('max_drawdown', 0)),
            current_drawdown=float(metrics.get('current_drawdown', 0)),
            var_95=float(metrics.get('var_95', 0)),
            cvar_95=float(metrics.get('cvar_95', 0)),
            exposure=float(metrics.get('exposure', 0)),
            open_positions=int(metrics.get('open_positions', 0))
        )
        self.session.add(risk_metric)
        self.session.commit()
        return risk_metric
        
    def get_risk_metrics_history(self, hours: int = 24) -> List[RiskMetrics]:
        since = datetime.utcnow() - timedelta(hours=hours)
        return self.session.query(RiskMetrics).filter(
            RiskMetrics.timestamp >= since
        ).order_by(RiskMetrics.timestamp).all()
        
    def save_alert(self, alert_type: str, severity: str,
                  message: str, metadata: Optional[Dict] = None) -> Alert:
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            metadata=metadata or {}
        )
        self.session.add(alert)
        self.session.commit()
        return alert
        
    def get_unacknowledged_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        query = self.session.query(Alert).filter_by(acknowledged=False)
        
        if severity:
            query = query.filter_by(severity=severity)
            
        return query.order_by(Alert.timestamp.desc()).all()
        
    def acknowledge_alert(self, alert_id: int) -> Optional[Alert]:
        alert = self.session.query(Alert).get(alert_id)
        if alert:
            alert.acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            self.session.commit()
        return alert
        
    def save_backtest_result(self, result_data: Dict[str, Any]) -> BacktestResult:
        result = BacktestResult(
            run_id=result_data['run_id'],
            strategy_name=result_data['strategy_name'],
            start_date=result_data['start_date'],
            end_date=result_data['end_date'],
            initial_capital=float(result_data['initial_capital']),
            final_capital=float(result_data['final_capital']),
            total_return=float(result_data['total_return']),
            sharpe_ratio=float(result_data.get('sharpe_ratio', 0)),
            max_drawdown=float(result_data.get('max_drawdown', 0)),
            win_rate=float(result_data.get('win_rate', 0)),
            total_trades=int(result_data.get('total_trades', 0)),
            parameters=result_data.get('parameters', {}),
            detailed_results=result_data.get('detailed_results', {})
        )
        self.session.add(result)
        self.session.commit()
        return result
        
    def get_best_backtest_results(self, strategy_name: str,
                                 metric: str = 'total_return',
                                 limit: int = 10) -> List[BacktestResult]:
        query = self.session.query(BacktestResult).filter_by(strategy_name=strategy_name)
        
        if metric == 'total_return':
            query = query.order_by(BacktestResult.total_return.desc())
        elif metric == 'sharpe_ratio':
            query = query.order_by(BacktestResult.sharpe_ratio.desc())
        elif metric == 'win_rate':
            query = query.order_by(BacktestResult.win_rate.desc())
            
        return query.limit(limit).all()
        
    def save_data_quality(self, symbol: str, data_type: str,
                         quality_metrics: Dict[str, Any]) -> DataQuality:
        quality = DataQuality(
            symbol=symbol,
            data_type=data_type,
            missing_points=int(quality_metrics.get('missing_points', 0)),
            outliers=int(quality_metrics.get('outliers', 0)),
            quality_score=float(quality_metrics.get('quality_score', 1.0)),
            issues=quality_metrics.get('issues', [])
        )
        self.session.add(quality)
        self.session.commit()
        return quality
        
    def get_trade_statistics(self, start_date: datetime, end_date: datetime,
                           symbol: Optional[str] = None,
                           strategy: Optional[str] = None) -> Dict[str, Any]:
        query = self.session.query(Trade).filter(
            and_(
                Trade.timestamp >= start_date,
                Trade.timestamp <= end_date
            )
        )
        
        if symbol:
            query = query.filter_by(symbol=symbol)
        if strategy:
            query = query.filter_by(strategy=strategy)
            
        trades = query.all()
        
        if not trades:
            return {
                'total_trades': 0,
                'total_volume': 0,
                'total_commission': 0,
                'symbols_traded': []
            }
            
        total_volume = sum(t.price * t.quantity for t in trades)
        total_commission = sum(t.commission for t in trades)
        symbols = list(set(t.symbol for t in trades))
        
        return {
            'total_trades': len(trades),
            'total_volume': total_volume,
            'total_commission': total_commission,
            'symbols_traded': symbols,
            'trades_by_symbol': {
                sym: len([t for t in trades if t.symbol == sym])
                for sym in symbols
            }
        }
        
    def cleanup_old_data(self, days_to_keep: int = 30):
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Delete old system metrics
        self.session.query(SystemMetrics).filter(
            SystemMetrics.timestamp < cutoff_date
        ).delete()
        
        # Delete old alerts that are acknowledged
        self.session.query(Alert).filter(
            and_(
                Alert.timestamp < cutoff_date,
                Alert.acknowledged == True
            )
        ).delete()
        
        # Delete old model predictions
        self.session.query(ModelPrediction).filter(
            ModelPrediction.timestamp < cutoff_date
        ).delete()
        
        self.session.commit()
        logger.info(f"Cleaned up data older than {days_to_keep} days")