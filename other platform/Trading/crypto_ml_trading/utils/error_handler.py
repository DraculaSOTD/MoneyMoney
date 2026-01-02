"""
Robust Error Handling System for Crypto ML Trading
Provides comprehensive error handling, recovery, and monitoring capabilities.
"""

import sys
import traceback
import logging
import functools
import time
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import warnings
import signal
import atexit
from contextlib import contextmanager
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA = "data"
    MODEL = "model"
    TRADING = "trading"
    SYSTEM = "system"
    NETWORK = "network"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_id: str
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    error_message: str
    stack_trace: str
    function_name: str
    module_name: str
    additional_data: Dict[str, Any] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        return data


class TradingError(Exception):
    """Base exception for trading-specific errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.TRADING,
                 error_code: Optional[str] = None,
                 additional_data: Optional[Dict[str, Any]] = None):
        """
        Initialize trading error.
        
        Args:
            message: Error message
            severity: Error severity level
            category: Error category
            error_code: Optional error code
            additional_data: Additional context data
        """
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.error_code = error_code
        self.additional_data = additional_data or {}


class DataError(TradingError):
    """Data-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)


class ModelError(TradingError):
    """Model-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL, **kwargs)


class ValidationError(TradingError):
    """Validation errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class ConfigurationError(TradingError):
    """Configuration errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, blocking calls
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                # Check if circuit is OPEN
                if self.state == "OPEN":
                    if (time.time() - self.last_failure_time) > self.recovery_timeout:
                        self.state = "HALF_OPEN"
                    else:
                        raise TradingError(
                            f"Circuit breaker is OPEN for {func.__name__}",
                            severity=ErrorSeverity.HIGH
                        )
            
            try:
                result = func(*args, **kwargs)
                
                with self._lock:
                    if self.state == "HALF_OPEN":
                        # Success in HALF_OPEN state, close circuit
                        self.state = "CLOSED"
                        self.failure_count = 0
                
                return result
                
            except self.expected_exception as e:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                        logger.error(
                            f"Circuit breaker opened for {func.__name__} "
                            f"after {self.failure_count} failures"
                        )
                
                raise
        
        # Add circuit breaker control methods
        wrapper.reset = lambda: self._reset()
        wrapper.get_state = lambda: self.state
        
        return wrapper
    
    def _reset(self):
        """Reset circuit breaker."""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure_time = None


class ErrorHandler:
    """
    Comprehensive error handling system.
    
    Features:
    - Error tracking and classification
    - Automatic recovery strategies
    - Circuit breaker pattern
    - Error rate monitoring
    - Alert generation
    - Graceful degradation
    """
    
    def __init__(self, max_errors: int = 1000):
        """
        Initialize error handler.
        
        Args:
            max_errors: Maximum number of errors to track
        """
        self.max_errors = max_errors
        self.errors = []
        self.error_counts = {}
        self.recovery_strategies = {}
        self.alert_handlers = []
        self._lock = threading.Lock()
        self._error_queue = queue.Queue()
        self._stop_processing = threading.Event()
        
        # Start error processing thread
        self._processor_thread = threading.Thread(
            target=self._process_errors, daemon=True
        )
        self._processor_thread.start()
        
        # Register cleanup
        atexit.register(self.shutdown)
    
    def handle_error(self, func: Callable = None, *,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    max_retries: int = 3,
                    retry_delay: float = 1.0,
                    fallback: Optional[Callable] = None,
                    notify: bool = True) -> Callable:
        """
        Decorator for comprehensive error handling.
        
        Args:
            func: Function to wrap
            severity: Default error severity
            category: Error category
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            fallback: Fallback function if all retries fail
            notify: Send notifications for errors
            
        Returns:
            Decorated function
        """
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        return f(*args, **kwargs)
                        
                    except Exception as e:
                        last_error = e
                        
                        # Create error context
                        error_context = self._create_error_context(
                            e, f.__name__, f.__module__,
                            severity, category
                        )
                        
                        # Log error
                        self._log_error(error_context)
                        
                        # Attempt recovery
                        if attempt < max_retries - 1:
                            recovery_result = self._attempt_recovery(
                                error_context, f, args, kwargs
                            )
                            if recovery_result is not None:
                                return recovery_result
                            
                            # Wait before retry
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                
                # All retries failed
                if fallback:
                    logger.warning(f"Falling back for {f.__name__} after {max_retries} attempts")
                    try:
                        return fallback(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback failed: {fallback_error}")
                        raise last_error
                else:
                    raise last_error
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _create_error_context(self, error: Exception, function_name: str,
                            module_name: str, severity: ErrorSeverity,
                            category: ErrorCategory) -> ErrorContext:
        """Create error context from exception."""
        error_id = f"{int(time.time())}_{hash(str(error)) % 10000}"
        
        # Extract additional data from custom exceptions
        additional_data = {}
        if isinstance(error, TradingError):
            severity = error.severity
            category = error.category
            additional_data = error.additional_data
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            category=category,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            function_name=function_name,
            module_name=module_name,
            additional_data=additional_data
        )
    
    def _log_error(self, error_context: ErrorContext):
        """Log error and add to tracking."""
        # Log based on severity
        log_message = (
            f"Error in {error_context.function_name}: "
            f"{error_context.error_message} "
            f"[{error_context.severity.value}]"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Track error
        with self._lock:
            self.errors.append(error_context)
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)
            
            # Update error counts
            error_key = f"{error_context.category.value}:{error_context.error_type}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Queue for processing
        self._error_queue.put(error_context)
    
    def _attempt_recovery(self, error_context: ErrorContext, func: Callable,
                         args: tuple, kwargs: dict) -> Optional[Any]:
        """Attempt to recover from error."""
        recovery_key = f"{error_context.category.value}:{error_context.error_type}"
        
        if recovery_key in self.recovery_strategies:
            strategy = self.recovery_strategies[recovery_key]
            try:
                logger.info(f"Attempting recovery for {recovery_key}")
                result = strategy(error_context, func, args, kwargs)
                error_context.recovery_attempted = True
                error_context.recovery_successful = True
                return result
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
                error_context.recovery_attempted = True
                error_context.recovery_successful = False
        
        return None
    
    def register_recovery_strategy(self, category: ErrorCategory,
                                 error_type: Type[Exception],
                                 strategy: Callable):
        """
        Register recovery strategy for specific error type.
        
        Args:
            category: Error category
            error_type: Exception type
            strategy: Recovery function
        """
        key = f"{category.value}:{error_type.__name__}"
        self.recovery_strategies[key] = strategy
        logger.info(f"Registered recovery strategy for {key}")
    
    def register_alert_handler(self, handler: Callable[[ErrorContext], None]):
        """Register alert handler for critical errors."""
        self.alert_handlers.append(handler)
    
    def _process_errors(self):
        """Background thread for error processing."""
        while not self._stop_processing.is_set():
            try:
                error_context = self._error_queue.get(timeout=1.0)
                
                # Check if alert needed
                if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    for handler in self.alert_handlers:
                        try:
                            handler(error_context)
                        except Exception as e:
                            logger.error(f"Alert handler failed: {e}")
                
                # Check error rate
                self._check_error_rate()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing failed: {e}")
    
    def _check_error_rate(self):
        """Check error rate and trigger alerts if needed."""
        with self._lock:
            # Calculate error rate over last minute
            current_time = time.time()
            recent_errors = [
                e for e in self.errors
                if (current_time - float(e.timestamp.replace('T', ' ').split('.')[0])) < 60
            ]
            
            error_rate = len(recent_errors)
            
            # Alert if error rate is high
            if error_rate > 10:  # More than 10 errors per minute
                logger.critical(f"High error rate detected: {error_rate} errors/minute")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = len(self.errors)
            
            # Group by category
            by_category = {}
            by_severity = {}
            
            for error in self.errors:
                # Category stats
                cat = error.category.value
                by_category[cat] = by_category.get(cat, 0) + 1
                
                # Severity stats
                sev = error.severity.value
                by_severity[sev] = by_severity.get(sev, 0) + 1
            
            # Recovery stats
            recovery_attempts = sum(1 for e in self.errors if e.recovery_attempted)
            recovery_successes = sum(1 for e in self.errors if e.recovery_successful)
            
            return {
                'total_errors': total_errors,
                'by_category': by_category,
                'by_severity': by_severity,
                'error_types': dict(self.error_counts),
                'recovery_rate': recovery_successes / recovery_attempts if recovery_attempts > 0 else 0,
                'recent_errors': [e.to_dict() for e in self.errors[-10:]]
            }
    
    def shutdown(self):
        """Shutdown error handler."""
        self._stop_processing.set()
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5.0)


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _error_handler


# Decorator shortcuts
handle_error = _error_handler.handle_error
circuit_breaker = CircuitBreaker


# Context managers for error handling
@contextmanager
def error_context(operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    Context manager for error handling with context.
    
    Args:
        operation: Description of operation
        severity: Error severity if exception occurs
    """
    try:
        logger.debug(f"Starting operation: {operation}")
        yield
        logger.debug(f"Completed operation: {operation}")
    except Exception as e:
        logger.error(f"Failed operation: {operation} - {str(e)}")
        raise TradingError(
            f"Operation failed: {operation}",
            severity=severity,
            additional_data={'operation': operation, 'original_error': str(e)}
        )


@contextmanager
def suppress_and_log(*exceptions: Type[Exception]):
    """
    Context manager to suppress and log specific exceptions.
    
    Args:
        *exceptions: Exception types to suppress
    """
    try:
        yield
    except exceptions as e:
        logger.warning(f"Suppressed exception: {type(e).__name__} - {str(e)}")


# Recovery strategies
class RecoveryStrategies:
    """Common recovery strategies for different error types."""
    
    @staticmethod
    def retry_with_backoff(error_context: ErrorContext, func: Callable,
                          args: tuple, kwargs: dict) -> Any:
        """Retry with exponential backoff."""
        max_wait = 30  # Maximum wait time
        wait_time = min(2 ** (error_context.additional_data.get('attempt', 1)), max_wait)
        
        logger.info(f"Retrying after {wait_time} seconds")
        time.sleep(wait_time)
        
        return func(*args, **kwargs)
    
    @staticmethod
    def use_cached_value(error_context: ErrorContext, func: Callable,
                        args: tuple, kwargs: dict) -> Any:
        """Use cached value if available."""
        cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
        # This would integrate with your caching system
        logger.info("Using cached value for recovery")
        return None  # Placeholder
    
    @staticmethod
    def use_default_value(error_context: ErrorContext, func: Callable,
                         args: tuple, kwargs: dict) -> Any:
        """Return default value."""
        default = error_context.additional_data.get('default_value')
        logger.info(f"Using default value: {default}")
        return default


# Example usage and testing
if __name__ == "__main__":
    # Initialize error handler
    error_handler = get_error_handler()
    
    # Register recovery strategies
    error_handler.register_recovery_strategy(
        ErrorCategory.DATA,
        ValueError,
        RecoveryStrategies.use_default_value
    )
    
    # Test error handling decorator
    @handle_error(max_retries=3, retry_delay=0.5)
    def risky_calculation(x: int, y: int) -> float:
        """Test function that might fail."""
        if y == 0:
            raise ValueError("Division by zero")
        return x / y
    
    # Test circuit breaker
    @circuit_breaker(failure_threshold=3, recovery_timeout=5)
    def unreliable_service():
        """Test function with circuit breaker."""
        if np.random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Service unavailable")
        return "Success"
    
    # Test error handling
    print("Testing error handling...")
    try:
        result = risky_calculation(10, 0)
    except ValueError as e:
        print(f"Caught error: {e}")
    
    # Test circuit breaker
    print("\nTesting circuit breaker...")
    for i in range(10):
        try:
            result = unreliable_service()
            print(f"Attempt {i+1}: {result}")
        except Exception as e:
            print(f"Attempt {i+1}: {type(e).__name__} - {str(e)}")
        time.sleep(1)
    
    # Test context managers
    print("\nTesting context managers...")
    with error_context("test operation"):
        print("Performing operation...")
    
    with suppress_and_log(ValueError):
        raise ValueError("This will be suppressed")
    
    # Get error statistics
    print("\nError Statistics:")
    stats = error_handler.get_error_statistics()
    print(json.dumps(stats, indent=2))
    
    # Shutdown
    error_handler.shutdown()
    print("\nError handling test completed!")