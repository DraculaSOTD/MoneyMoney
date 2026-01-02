import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    SYSTEM_STATUS = "system_status"
    TRADE_EXECUTED = "trade_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_LIMIT = "risk_limit"
    LARGE_DRAWDOWN = "large_drawdown"
    MODEL_ERROR = "model_error"
    CONNECTION_LOST = "connection_lost"
    LOW_BALANCE = "low_balance"
    HIGH_PROFIT = "high_profit"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"

@dataclass
class Alert:
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class AlertChannel(ABC):
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        pass

class SlackChannel(AlertChannel):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def send_alert(self, alert: Alert) -> bool:
        try:
            await self._ensure_session()
            
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#990000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f"{alert.severity.value.upper()}: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Type",
                            "value": alert.type.value,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        }
                    ],
                    "footer": "Crypto ML Trading Bot",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Add metadata fields
            for key, value in alert.metadata.items():
                if key in ['symbol', 'price', 'quantity', 'pnl']:
                    payload["attachments"][0]["fields"].append({
                        "title": key.title(),
                        "value": str(value),
                        "short": True
                    })
                    
            async with self.session.post(self.webhook_url, json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
            
    async def close(self):
        if self.session:
            await self.session.close()

class TelegramChannel(AlertChannel):
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def send_alert(self, alert: Alert) -> bool:
        try:
            await self._ensure_session()
            
            emoji_map = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "🚨",
                AlertSeverity.CRITICAL: "🆘"
            }
            
            # Format message with markdown
            message = f"{emoji_map.get(alert.severity, '📢')} *{alert.severity.value.upper()}*\n\n"
            message += f"*{alert.title}*\n\n"
            message += f"{alert.message}\n\n"
            message += f"_Type: {alert.type.value}_\n"
            message += f"_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}_"
            
            # Add important metadata
            if alert.metadata:
                message += "\n\n📊 *Details:*\n"
                for key, value in alert.metadata.items():
                    if key in ['symbol', 'price', 'quantity', 'pnl', 'action']:
                        message += f"• {key.title()}: `{value}`\n"
                        
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with self.session.post(f"{self.api_url}/sendMessage", json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
            
    async def close(self):
        if self.session:
            await self.session.close()

class EmailChannel(AlertChannel):
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        
    async def send_alert(self, alert: Alert) -> bool:
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._send_email, alert)
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
            
    def _send_email(self, alert: Alert) -> bool:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            
            # Plain text version
            text = f"""
{alert.severity.value.upper()}: {alert.title}

{alert.message}

Type: {alert.type.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Details:
"""
            for key, value in alert.metadata.items():
                text += f"  {key}: {value}\n"
                
            # HTML version
            severity_colors = {
                AlertSeverity.INFO: "#4CAF50",
                AlertSeverity.WARNING: "#FF9800",
                AlertSeverity.ERROR: "#F44336",
                AlertSeverity.CRITICAL: "#B71C1C"
            }
            
            html = f"""
<html>
<body style="font-family: Arial, sans-serif;">
    <div style="background-color: {severity_colors.get(alert.severity, '#808080')}; color: white; padding: 10px;">
        <h2>{alert.severity.value.upper()}: {alert.title}</h2>
    </div>
    <div style="padding: 20px;">
        <p>{alert.message}</p>
        <hr>
        <p><strong>Type:</strong> {alert.type.value}</p>
        <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        
        <h3>Details:</h3>
        <table style="border-collapse: collapse; width: 100%;">
"""
            for key, value in alert.metadata.items():
                html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><strong>{key}</strong></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{value}</td>
            </tr>
"""
            html += """
        </table>
    </div>
</body>
</html>
"""
            
            part1 = MIMEText(text, 'plain')
            part2 = MIMEText(html, 'html')
            
            msg.attach(part1)
            msg.attach(part2)
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False

class DiscordChannel(AlertChannel):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def send_alert(self, alert: Alert) -> bool:
        try:
            await self._ensure_session()
            
            color_map = {
                AlertSeverity.INFO: 0x00ff00,
                AlertSeverity.WARNING: 0xffff00,
                AlertSeverity.ERROR: 0xff0000,
                AlertSeverity.CRITICAL: 0x8b0000
            }
            
            embed = {
                "title": f"{alert.severity.value.upper()}: {alert.title}",
                "description": alert.message,
                "color": color_map.get(alert.severity, 0x808080),
                "fields": [
                    {
                        "name": "Type",
                        "value": alert.type.value,
                        "inline": True
                    },
                    {
                        "name": "Time",
                        "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Crypto ML Trading Bot"
                },
                "timestamp": alert.timestamp.isoformat()
            }
            
            # Add metadata fields
            for key, value in alert.metadata.items():
                if key in ['symbol', 'price', 'quantity', 'pnl', 'action']:
                    embed["fields"].append({
                        "name": key.title(),
                        "value": str(value),
                        "inline": True
                    })
                    
            payload = {"embeds": [embed]}
            
            async with self.session.post(self.webhook_url, json=payload) as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
            
    async def close(self):
        if self.session:
            await self.session.close()

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.channels: List[AlertChannel] = []
        self.alert_rules: Dict[AlertType, Dict[str, Any]] = {}
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        self._setup_channels()
        self._setup_rules()
        
    def _setup_channels(self):
        alert_config = self.config.get('alerting', {})
        
        # Slack
        if alert_config.get('slack', {}).get('enabled', False):
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if webhook_url:
                self.channels.append(SlackChannel(webhook_url))
                logger.info("Slack alerting enabled")
                
        # Telegram
        if alert_config.get('telegram', {}).get('enabled', False):
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            if bot_token and chat_id:
                self.channels.append(TelegramChannel(bot_token, chat_id))
                logger.info("Telegram alerting enabled")
                
        # Email
        if alert_config.get('email', {}).get('enabled', False):
            email_config = alert_config['email']
            smtp_host = email_config.get('smtp_host', 'smtp.gmail.com')
            smtp_port = email_config.get('smtp_port', 587)
            username = os.getenv('EMAIL_USERNAME')
            password = os.getenv('EMAIL_PASSWORD')
            from_email = email_config.get('from_email', username)
            to_emails = email_config.get('to_emails', [])
            
            if username and password and to_emails:
                self.channels.append(EmailChannel(
                    smtp_host, smtp_port, username, password, from_email, to_emails
                ))
                logger.info("Email alerting enabled")
                
        # Discord
        if alert_config.get('discord', {}).get('enabled', False):
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            if webhook_url:
                self.channels.append(DiscordChannel(webhook_url))
                logger.info("Discord alerting enabled")
                
    def _setup_rules(self):
        rules_config = self.config.get('alerting', {}).get('rules', {})
        
        # Default rules
        default_rules = {
            AlertType.SYSTEM_STATUS: {
                'min_severity': AlertSeverity.ERROR,
                'rate_limit': timedelta(minutes=5)
            },
            AlertType.TRADE_EXECUTED: {
                'min_severity': AlertSeverity.INFO,
                'rate_limit': timedelta(seconds=30)
            },
            AlertType.RISK_LIMIT: {
                'min_severity': AlertSeverity.WARNING,
                'rate_limit': timedelta(minutes=1)
            },
            AlertType.LARGE_DRAWDOWN: {
                'min_severity': AlertSeverity.WARNING,
                'rate_limit': timedelta(minutes=10)
            },
            AlertType.CONNECTION_LOST: {
                'min_severity': AlertSeverity.ERROR,
                'rate_limit': timedelta(minutes=5)
            }
        }
        
        # Override with config
        for alert_type in AlertType:
            rule_config = rules_config.get(alert_type.value, {})
            self.alert_rules[alert_type] = {
                'min_severity': AlertSeverity(rule_config.get(
                    'min_severity', 
                    default_rules.get(alert_type, {}).get('min_severity', AlertSeverity.INFO).value
                )),
                'rate_limit': timedelta(seconds=rule_config.get(
                    'rate_limit_seconds',
                    default_rules.get(alert_type, {}).get('rate_limit', timedelta(minutes=5)).total_seconds()
                )),
                'last_sent': None
            }
            
    async def send_alert(self, alert: Alert) -> bool:
        # Check if alert should be sent based on rules
        rule = self.alert_rules.get(alert.type, {})
        
        # Check severity
        if alert.severity.value < rule.get('min_severity', AlertSeverity.INFO).value:
            return False
            
        # Check rate limit
        last_sent = rule.get('last_sent')
        if last_sent:
            time_since_last = datetime.utcnow() - last_sent
            if time_since_last < rule.get('rate_limit', timedelta(minutes=5)):
                logger.debug(f"Rate limit hit for {alert.type.value}")
                return False
                
        # Send to all channels
        results = await asyncio.gather(
            *[channel.send_alert(alert) for channel in self.channels],
            return_exceptions=True
        )
        
        success = any(r is True for r in results if not isinstance(r, Exception))
        
        if success:
            # Update last sent time
            rule['last_sent'] = datetime.utcnow()
            
            # Add to history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
                
            logger.info(f"Alert sent: {alert.type.value} - {alert.title}")
        else:
            logger.error(f"Failed to send alert: {alert.type.value} - {alert.title}")
            
        return success
        
    async def close(self):
        for channel in self.channels:
            if hasattr(channel, 'close'):
                await channel.close()
                
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [alert.to_dict() for alert in self.alert_history[-limit:]]
        
    def get_channel_status(self) -> Dict[str, bool]:
        return {
            channel.__class__.__name__: True
            for channel in self.channels
        }

class AlertBuilder:
    @staticmethod
    def system_started() -> Alert:
        return Alert(
            type=AlertType.SYSTEM_STATUS,
            severity=AlertSeverity.INFO,
            title="Trading System Started",
            message="The crypto ML trading system has been started successfully."
        )
        
    @staticmethod
    def system_stopped() -> Alert:
        return Alert(
            type=AlertType.SYSTEM_STATUS,
            severity=AlertSeverity.INFO,
            title="Trading System Stopped",
            message="The crypto ML trading system has been stopped."
        )
        
    @staticmethod
    def trade_executed(symbol: str, side: str, quantity: float, price: float) -> Alert:
        return Alert(
            type=AlertType.TRADE_EXECUTED,
            severity=AlertSeverity.INFO,
            title=f"Trade Executed: {symbol}",
            message=f"{side.upper()} {quantity} {symbol} @ ${price:.2f}",
            metadata={
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price
            }
        )
        
    @staticmethod
    def position_opened(symbol: str, side: str, quantity: float, entry_price: float) -> Alert:
        return Alert(
            type=AlertType.POSITION_OPENED,
            severity=AlertSeverity.INFO,
            title=f"Position Opened: {symbol}",
            message=f"Opened {side.upper()} position: {quantity} {symbol} @ ${entry_price:.2f}",
            metadata={
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price
            }
        )
        
    @staticmethod
    def position_closed(symbol: str, pnl: float, return_pct: float) -> Alert:
        severity = AlertSeverity.INFO if pnl >= 0 else AlertSeverity.WARNING
        return Alert(
            type=AlertType.POSITION_CLOSED,
            severity=severity,
            title=f"Position Closed: {symbol}",
            message=f"Closed position with P&L: ${pnl:.2f} ({return_pct:.2%})",
            metadata={
                'symbol': symbol,
                'pnl': pnl,
                'return_pct': return_pct
            }
        )
        
    @staticmethod
    def risk_limit_reached(limit_type: str, current_value: float, limit_value: float) -> Alert:
        return Alert(
            type=AlertType.RISK_LIMIT,
            severity=AlertSeverity.WARNING,
            title=f"Risk Limit Reached: {limit_type}",
            message=f"Current {limit_type}: {current_value:.2f} has reached limit: {limit_value:.2f}",
            metadata={
                'limit_type': limit_type,
                'current_value': current_value,
                'limit_value': limit_value
            }
        )
        
    @staticmethod
    def large_drawdown(current_drawdown: float, max_drawdown: float) -> Alert:
        severity = AlertSeverity.WARNING if current_drawdown < 0.15 else AlertSeverity.ERROR
        return Alert(
            type=AlertType.LARGE_DRAWDOWN,
            severity=severity,
            title="Large Drawdown Detected",
            message=f"Portfolio drawdown: {current_drawdown:.2%} (Max allowed: {max_drawdown:.2%})",
            metadata={
                'current_drawdown': current_drawdown,
                'max_drawdown': max_drawdown
            }
        )
        
    @staticmethod
    def connection_lost(exchange: str, error: str) -> Alert:
        return Alert(
            type=AlertType.CONNECTION_LOST,
            severity=AlertSeverity.ERROR,
            title=f"Connection Lost: {exchange}",
            message=f"Lost connection to {exchange}: {error}",
            metadata={
                'exchange': exchange,
                'error': error
            }
        )