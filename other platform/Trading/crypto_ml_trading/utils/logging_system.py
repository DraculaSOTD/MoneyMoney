"""
Production-Ready Logging System for Crypto ML Trading
Provides structured logging with multiple handlers, formatters, and monitoring capabilities.
"""

import logging
import logging.handlers
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import queue
import traceback


@dataclass
class LogRecord:
    """Structured log record for JSON logging."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        record = asdict(self)
        if self.extra_data:
            record.update(self.extra_data)
        return record


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        """
        Initialize JSON formatter.
        
        Args:
            include_extra: Include extra fields from log record
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create base log record
        log_record = LogRecord(
            timestamp=datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process
        )
        
        # Add extra data if available
        if self.include_extra and hasattr(record, '__dict__'):
            extra_data = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'stack_info', 'exc_info', 'exc_text']:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra_data[key] = value
                    except (TypeError, ValueError):
                        extra_data[key] = str(value)
            
            if extra_data:
                log_record.extra_data = extra_data
        
        # Handle exceptions
        if record.exc_info:
            log_record.extra_data = log_record.extra_data or {}
            log_record.extra_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_record.to_dict(), ensure_ascii=False)


class TradingLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with trading-specific context."""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """
        Initialize trading logger adapter.
        
        Args:
            logger: Base logger
            extra: Extra context fields
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message with extra context."""
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra.copy()
        return msg, kwargs
    
    def trade(self, symbol: str, action: str, quantity: float, 
              price: float, **kwargs):
        """Log trading activity."""
        extra = {
            'event_type': 'trade',
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price
        }
        extra.update(kwargs)
        self.info(f"Trade executed: {action} {quantity} {symbol} at {price}", 
                 extra=extra)
    
    def signal(self, symbol: str, signal_type: str, confidence: float,
               model: str, **kwargs):
        """Log trading signals."""
        extra = {
            'event_type': 'signal',
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'model': model
        }
        extra.update(kwargs)
        self.info(f"Signal generated: {signal_type} for {symbol} "
                 f"(confidence: {confidence:.2%}) by {model}", extra=extra)
    
    def performance(self, symbol: str, metric: str, value: float, 
                   period: str = None, **kwargs):
        """Log performance metrics."""
        extra = {
            'event_type': 'performance',
            'symbol': symbol,
            'metric': metric,
            'value': value
        }
        if period:
            extra['period'] = period
        extra.update(kwargs)
        self.info(f"Performance metric: {metric}={value} for {symbol}", 
                 extra=extra)
    
    def model_update(self, model: str, action: str, **kwargs):
        """Log model training/updating events."""
        extra = {
            'event_type': 'model_update',
            'model': model,
            'action': action
        }
        extra.update(kwargs)
        self.info(f"Model {action}: {model}", extra=extra)


class AsyncFileHandler(logging.Handler):
    """Asynchronous file handler for high-performance logging."""
    
    def __init__(self, filename: str, max_bytes: int = 10485760, 
                 backup_count: int = 5):
        """
        Initialize async file handler.
        
        Args:
            filename: Log file path
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize rotating file handler
        self.file_handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=max_bytes, backupCount=backup_count
        )
        
        # Initialize async queue and thread
        self.log_queue = queue.Queue()
        self.stop_logging = threading.Event()
        self.logging_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.logging_thread.start()
    
    def emit(self, record: logging.LogRecord):
        """Emit log record asynchronously."""
        if not self.stop_logging.is_set():
            try:
                self.log_queue.put_nowait(record)
            except queue.Full:
                # Drop log if queue is full
                pass
    
    def _log_worker(self):
        """Background worker thread for processing log records."""
        while not self.stop_logging.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                self.file_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Handle logging errors without causing recursion
                print(f"Logging error: {e}", file=sys.stderr)
    
    def close(self):
        """Close handler and stop background thread."""
        self.stop_logging.set()
        if self.logging_thread.is_alive():
            self.logging_thread.join(timeout=5.0)
        self.file_handler.close()
        super().close()


class LoggingSystem:
    """
    Centralized logging system for the trading application.
    
    Features:
    - Multiple log handlers (console, file, async file)
    - Structured JSON logging
    - Trading-specific log methods
    - Performance monitoring
    - Log rotation and management
    - Context managers for scoped logging
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize logging system.
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.loggers = {}
        self.handlers = {}
        self._setup_logging()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": {
                "console": {
                    "enabled": True,
                    "level": "INFO",
                    "format": "text"
                },
                "file": {
                    "enabled": True,
                    "level": "DEBUG",
                    "filename": "logs/trading_system.log",
                    "format": "json",
                    "max_bytes": 10485760,
                    "backup_count": 5,
                    "async": True
                }
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config["level"]))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup handlers
        handlers_config = self.config.get("handlers", {})
        
        # Console handler
        console_config = handlers_config.get("console", {})
        if console_config.get("enabled", True):
            self._setup_console_handler(console_config)
        
        # File handler
        file_config = handlers_config.get("file", {})
        if file_config.get("enabled", True):
            self._setup_file_handler(file_config)
    
    def _setup_console_handler(self, config: Dict[str, Any]):
        """Setup console logging handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, config.get("level", "INFO")))
        
        if config.get("format", "text") == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                self.config.get("format", 
                               "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        self.handlers['console'] = handler
    
    def _setup_file_handler(self, config: Dict[str, Any]):
        """Setup file logging handler."""
        filename = config.get("filename", "logs/trading_system.log")
        
        if config.get("async", False):
            handler = AsyncFileHandler(
                filename,
                max_bytes=config.get("max_bytes", 10485760),
                backup_count=config.get("backup_count", 5)
            )
        else:
            # Create directory if it doesn't exist
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            handler = logging.handlers.RotatingFileHandler(
                filename,
                maxBytes=config.get("max_bytes", 10485760),
                backupCount=config.get("backup_count", 5)
            )
        
        handler.setLevel(getattr(logging, config.get("level", "DEBUG")))
        
        if config.get("format", "json") == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                self.config.get("format",
                               "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        self.handlers['file'] = handler
    
    def get_logger(self, name: str, extra: Dict[str, Any] = None) -> TradingLoggerAdapter:
        """
        Get a logger instance with trading-specific functionality.
        
        Args:
            name: Logger name
            extra: Extra context fields
            
        Returns:
            TradingLoggerAdapter instance
        """
        if name not in self.loggers:
            base_logger = logging.getLogger(name)
            self.loggers[name] = TradingLoggerAdapter(base_logger, extra)
        
        return self.loggers[name]
    
    @contextmanager
    def scoped_context(self, **context):
        """
        Context manager for scoped logging with additional context.
        
        Args:
            **context: Additional context fields
            
        Yields:
            Logger with scoped context
        """
        logger_name = f"scoped_{int(time.time())}"
        logger = self.get_logger(logger_name, context)
        
        try:
            yield logger
        finally:
            # Clean up scoped logger
            if logger_name in self.loggers:
                del self.loggers[logger_name]
    
    def log_system_startup(self, version: str, environment: str):
        """Log system startup information."""
        logger = self.get_logger("system")
        logger.info("Trading system starting up", extra={
            'event_type': 'startup',
            'version': version,
            'environment': environment,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def log_system_shutdown(self):
        """Log system shutdown."""
        logger = self.get_logger("system")
        logger.info("Trading system shutting down", extra={
            'event_type': 'shutdown',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def close(self):
        """Close all handlers and cleanup resources."""
        for handler in self.handlers.values():
            handler.close()
        self.handlers.clear()
        self.loggers.clear()


# Global logging system instance
_logging_system = None


def setup_logging(config: Dict[str, Any] = None) -> LoggingSystem:
    """
    Setup global logging system.
    
    Args:
        config: Logging configuration
        
    Returns:
        LoggingSystem instance
    """
    global _logging_system
    _logging_system = LoggingSystem(config)
    return _logging_system


def get_logger(name: str, extra: Dict[str, Any] = None) -> TradingLoggerAdapter:
    """
    Get a logger instance from the global logging system.
    
    Args:
        name: Logger name
        extra: Extra context fields
        
    Returns:
        TradingLoggerAdapter instance
    """
    global _logging_system
    if _logging_system is None:
        _logging_system = LoggingSystem()
    return _logging_system.get_logger(name, extra)


# Example usage and testing
if __name__ == "__main__":
    # Test logging system
    config = {
        "level": "DEBUG",
        "handlers": {
            "console": {
                "enabled": True,
                "level": "INFO",
                "format": "text"
            },
            "file": {
                "enabled": True,
                "level": "DEBUG",
                "filename": "test_logs/trading_system.log",
                "format": "json",
                "async": True
            }
        }
    }
    
    # Setup logging system
    logging_system = setup_logging(config)
    
    # Test different loggers
    logger = get_logger("test", {"component": "test_module"})
    
    logger.info("This is a test message")
    logger.trade("BTCUSDT", "buy", 0.1, 50000.0, reason="breakout")
    logger.signal("ETHUSDT", "strong_buy", 0.85, "neural_ensemble")
    logger.performance("BTCUSDT", "sharpe_ratio", 1.5, "daily")
    logger.model_update("gru_attention", "training_completed", 
                       epochs=100, loss=0.001)
    
    # Test scoped logging
    with logging_system.scoped_context(trade_id="12345", symbol="BTCUSDT") as scoped_logger:
        scoped_logger.info("Starting trade analysis")
        scoped_logger.warning("Risk threshold exceeded", risk_level=0.15)
    
    # Test exception logging
    try:
        1 / 0
    except Exception as e:
        logger.exception("Test exception occurred")
    
    # Cleanup
    time.sleep(1)  # Allow async logging to complete
    logging_system.close()
    
    print("Logging system test completed!")