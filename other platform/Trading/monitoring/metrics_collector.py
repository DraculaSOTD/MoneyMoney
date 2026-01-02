import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
from prometheus_client import Gauge, Counter, Histogram, Info, generate_latest
import aiohttp

logger = logging.getLogger(__name__)

# Prometheus metrics
system_status = Info('trading_system_status', 'Trading system status')
active_positions = Gauge('trading_active_positions', 'Number of active positions', ['exchange', 'symbol'])
total_trades = Counter('trading_total_trades', 'Total number of trades', ['exchange', 'symbol', 'side'])
trade_volume = Counter('trading_volume_usd', 'Total trading volume in USD', ['exchange', 'symbol'])
portfolio_value = Gauge('trading_portfolio_value_usd', 'Total portfolio value in USD')
daily_pnl = Gauge('trading_daily_pnl_usd', 'Daily P&L in USD')
total_pnl = Gauge('trading_total_pnl_usd', 'Total P&L in USD')
model_prediction_error = Histogram('model_prediction_error', 'Model prediction error', ['model_name'])
api_latency = Histogram('api_request_duration_seconds', 'API request latency', ['endpoint', 'method'])
websocket_messages = Counter('websocket_messages_total', 'Total WebSocket messages', ['type'])
risk_metric_value = Gauge('trading_risk_metric', 'Risk metric values', ['metric_name'])

@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_time: timedelta = timedelta()
    
@dataclass
class SystemMetrics:
    uptime: timedelta = timedelta()
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    api_requests: int = 0
    websocket_connections: int = 0
    database_queries: int = 0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    warning_count: int = 0

class MetricsCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = datetime.utcnow()
        
        # Performance tracking
        self.trades_history: deque = deque(maxlen=1000)
        self.pnl_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.equity_curve: List[float] = []
        self.daily_returns: deque = deque(maxlen=365)
        
        # Real-time metrics
        self.current_positions: Dict[str, Any] = {}
        self.model_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.api_latencies: deque = deque(maxlen=1000)
        
        # Alerts tracking
        self.risk_violations: deque = deque(maxlen=100)
        
        # Grafana/Prometheus endpoint
        self.metrics_server_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start metrics collection and optional Prometheus endpoint"""
        if self.config.get('monitoring', {}).get('prometheus', {}).get('enabled', False):
            port = self.config['monitoring']['prometheus'].get('port', 9090)
            self.metrics_server_task = asyncio.create_task(self._run_metrics_server(port))
            logger.info(f"Started Prometheus metrics server on port {port}")
            
    async def stop(self):
        """Stop metrics collection"""
        if self.metrics_server_task:
            self.metrics_server_task.cancel()
            try:
                await self.metrics_server_task
            except asyncio.CancelledError:
                pass
                
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade"""
        self.trades_history.append({
            'timestamp': datetime.utcnow(),
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'quantity': trade_data['quantity'],
            'price': trade_data['price'],
            'pnl': trade_data.get('pnl', 0),
            'fees': trade_data.get('fees', 0)
        })
        
        # Update Prometheus metrics
        total_trades.labels(
            exchange=trade_data.get('exchange', 'binance'),
            symbol=trade_data['symbol'],
            side=trade_data['side']
        ).inc()
        
        trade_volume.labels(
            exchange=trade_data.get('exchange', 'binance'),
            symbol=trade_data['symbol']
        ).inc(trade_data['quantity'] * trade_data['price'])
        
        # Update performance metrics
        self._update_performance_metrics()
        
    def update_position(self, position_data: Dict[str, Any]):
        """Update position information"""
        key = f"{position_data['symbol']}_{position_data['side']}"
        self.current_positions[key] = position_data
        
        # Update Prometheus metrics
        active_positions.labels(
            exchange=position_data.get('exchange', 'binance'),
            symbol=position_data['symbol']
        ).set(len([p for p in self.current_positions.values() 
                  if p['symbol'] == position_data['symbol'] and p['status'] == 'open']))
                  
    def record_model_prediction(self, model_name: str, prediction: float, 
                              actual: Optional[float] = None):
        """Record model prediction for accuracy tracking"""
        self.model_predictions[model_name].append({
            'timestamp': datetime.utcnow(),
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual) if actual else None
        })
        
        if actual:
            model_prediction_error.labels(model_name=model_name).observe(abs(prediction - actual))
            
    def record_api_latency(self, endpoint: str, method: str, latency_ms: float):
        """Record API request latency"""
        self.api_latencies.append({
            'timestamp': datetime.utcnow(),
            'endpoint': endpoint,
            'method': method,
            'latency_ms': latency_ms
        })
        
        api_latency.labels(endpoint=endpoint, method=method).observe(latency_ms / 1000)
        
    def record_risk_metric(self, metric_name: str, value: float):
        """Record risk management metric"""
        risk_metric_value.labels(metric_name=metric_name).set(value)
        
        # Check for violations
        limits = self.config.get('risk_management', {})
        if metric_name == 'drawdown' and value > limits.get('max_drawdown', 0.1):
            self.risk_violations.append({
                'timestamp': datetime.utcnow(),
                'metric': metric_name,
                'value': value,
                'limit': limits['max_drawdown']
            })
            
    def update_portfolio_value(self, value: float):
        """Update total portfolio value"""
        portfolio_value.set(value)
        self.equity_curve.append(value)
        
        # Calculate returns
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
            
    def update_pnl(self, daily: float, total: float):
        """Update P&L metrics"""
        daily_pnl.set(daily)
        total_pnl.set(total)
        
        self.pnl_history.append({
            'timestamp': datetime.utcnow(),
            'daily_pnl': daily,
            'total_pnl': total
        })
        
    def _update_performance_metrics(self):
        """Calculate performance metrics from trade history"""
        if not self.trades_history:
            return
            
        trades_df = pd.DataFrame(list(self.trades_history))
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # P&L metrics
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        # Calculate ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        # Sharpe ratio (simplified)
        if self.daily_returns:
            returns = np.array(list(self.daily_returns))
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
            
        # Max drawdown
        if self.equity_curve:
            equity = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_dd = abs(np.min(drawdown))
        else:
            max_dd = 0
            
        self.performance_metrics = PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd
        )
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        self._update_performance_metrics()
        
        return {
            'total_trades': self.performance_metrics.total_trades,
            'win_rate': f"{self.performance_metrics.win_rate:.2%}",
            'profit_factor': f"{self.performance_metrics.profit_factor:.2f}",
            'sharpe_ratio': f"{self.performance_metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.performance_metrics.max_drawdown:.2%}",
            'avg_win': f"${self.performance_metrics.avg_win:.2f}",
            'avg_loss': f"${self.performance_metrics.avg_loss:.2f}"
        }
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        uptime = datetime.utcnow() - self.start_time
        
        # Calculate error rate
        total_requests = len(self.api_latencies)
        error_rate = 0  # Would need to track errors
        
        # Calculate average latency
        if self.api_latencies:
            avg_latency = np.mean([l['latency_ms'] for l in self.api_latencies])
        else:
            avg_latency = 0
            
        return {
            'status': 'healthy',  # Would implement health checks
            'uptime': str(uptime),
            'total_api_requests': total_requests,
            'avg_api_latency_ms': f"{avg_latency:.2f}",
            'error_rate': f"{error_rate:.2%}",
            'active_positions': len([p for p in self.current_positions.values() if p.get('status') == 'open']),
            'risk_violations': len(self.risk_violations)
        }
        
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get model prediction accuracy metrics"""
        model_stats = {}
        
        for model_name, predictions in self.model_predictions.items():
            valid_predictions = [p for p in predictions if p['error'] is not None]
            
            if valid_predictions:
                errors = [p['error'] for p in valid_predictions]
                model_stats[model_name] = {
                    'predictions_count': len(valid_predictions),
                    'avg_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'max_error': np.max(errors),
                    'mae': np.mean(np.abs(errors)),
                    'rmse': np.sqrt(np.mean(np.square(errors)))
                }
            else:
                model_stats[model_name] = {
                    'predictions_count': 0,
                    'status': 'no_data'
                }
                
        return model_stats
        
    def export_metrics_json(self) -> str:
        """Export all metrics as JSON"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'performance': self.get_performance_summary(),
            'system_health': self.get_system_health(),
            'model_performance': self.get_model_performance(),
            'current_positions': [
                {k: v for k, v in pos.items() if k != '_id'}
                for pos in self.current_positions.values()
            ],
            'recent_trades': [
                {k: v.isoformat() if isinstance(v, datetime) else v for k, v in trade.items()}
                for trade in list(self.trades_history)[-10:]
            ]
        }
        
        return json.dumps(metrics, indent=2)
        
    async def _run_metrics_server(self, port: int):
        """Run Prometheus metrics endpoint"""
        async def handle_metrics(request):
            metrics = generate_latest()
            return aiohttp.web.Response(text=metrics.decode('utf-8'), 
                                      content_type='text/plain')
                                      
        app = aiohttp.web.Application()
        app.router.add_get('/metrics', handle_metrics)
        
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        # Keep server running
        while True:
            await asyncio.sleep(3600)
            
    def create_performance_report(self) -> str:
        """Create a formatted performance report"""
        self._update_performance_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                 TRADING PERFORMANCE REPORT                    ║
║                 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}                    ║
╚══════════════════════════════════════════════════════════════╝

📊 TRADING STATISTICS
├─ Total Trades: {self.performance_metrics.total_trades}
├─ Winning Trades: {self.performance_metrics.winning_trades}
├─ Losing Trades: {self.performance_metrics.losing_trades}
├─ Win Rate: {self.performance_metrics.win_rate:.2%}
└─ Profit Factor: {self.performance_metrics.profit_factor:.2f}

💰 PROFIT & LOSS
├─ Average Win: ${self.performance_metrics.avg_win:.2f}
├─ Average Loss: ${self.performance_metrics.avg_loss:.2f}
├─ Largest Win: ${self.performance_metrics.largest_win:.2f}
└─ Largest Loss: ${self.performance_metrics.largest_loss:.2f}

📈 RISK METRICS
├─ Sharpe Ratio: {self.performance_metrics.sharpe_ratio:.2f}
├─ Sortino Ratio: {self.performance_metrics.sortino_ratio:.2f}
├─ Max Drawdown: {self.performance_metrics.max_drawdown:.2%}
└─ Current Drawdown: {self._calculate_current_drawdown():.2%}

🤖 MODEL PERFORMANCE
"""
        for model_name, stats in self.get_model_performance().items():
            if stats.get('predictions_count', 0) > 0:
                report += f"├─ {model_name}:\n"
                report += f"│  ├─ Predictions: {stats['predictions_count']}\n"
                report += f"│  ├─ MAE: {stats['mae']:.4f}\n"
                report += f"│  └─ RMSE: {stats['rmse']:.4f}\n"
                
        report += f"""
💻 SYSTEM HEALTH
├─ Uptime: {datetime.utcnow() - self.start_time}
├─ Active Positions: {len([p for p in self.current_positions.values() if p.get('status') == 'open'])}
├─ Risk Violations: {len(self.risk_violations)}
└─ API Latency: {np.mean([l['latency_ms'] for l in self.api_latencies]) if self.api_latencies else 0:.2f}ms

════════════════════════════════════════════════════════════════
"""
        return report
        
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
            
        current_value = self.equity_curve[-1]
        peak_value = max(self.equity_curve)
        
        if peak_value > 0:
            return (peak_value - current_value) / peak_value
        return 0.0