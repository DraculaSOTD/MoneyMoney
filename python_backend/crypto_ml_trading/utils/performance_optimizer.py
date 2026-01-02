"""
Performance Optimization System for Crypto ML Trading
Provides profiling, caching, and optimization utilities for production performance.
"""

import numpy as np
import time
import functools
import threading
import multiprocessing
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import psutil
import cProfile
import pstats
import io
from dataclasses import dataclass
from collections import OrderedDict
import gc
import weakref
import pickle
import hashlib
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for profiling."""
    function_name: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    """
    
    def __init__(self, maxsize: int = 128):
        """
        Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items in cache
        """
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: Any, value: Any):
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update and move to end
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                if len(self.cache) > self.maxsize:
                    # Remove least recently used
                    self.cache.popitem(last=False)
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryPool:
    """
    Memory pool for efficient array allocation.
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32, 
                 pool_size: int = 10):
        """
        Initialize memory pool.
        
        Args:
            shape: Shape of arrays to pool
            dtype: Data type of arrays
            pool_size: Number of arrays to pre-allocate
        """
        self.shape = shape
        self.dtype = dtype
        self.pool_size = pool_size
        self.pool = []
        self.lock = threading.Lock()
        
        # Pre-allocate arrays
        for _ in range(pool_size):
            self.pool.append(np.empty(shape, dtype=dtype))
    
    def get(self) -> np.ndarray:
        """Get array from pool."""
        with self.lock:
            if self.pool:
                return self.pool.pop()
            else:
                # Allocate new array if pool is empty
                return np.empty(self.shape, dtype=self.dtype)
    
    def put(self, array: np.ndarray):
        """Return array to pool."""
        with self.lock:
            if len(self.pool) < self.pool_size:
                # Clear array content for security
                array.fill(0)
                self.pool.append(array)


class PerformanceOptimizer:
    """
    Performance optimization utilities for the trading system.
    
    Features:
    - Function profiling and timing
    - Memory usage monitoring
    - Caching with LRU eviction
    - Parallel execution utilities
    - Memory pooling
    - JIT compilation hints
    """
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.metrics = {}
        self.caches = {}
        self.memory_pools = {}
        self.profiler = None
        self._lock = threading.Lock()
    
    def profile(self, func: Callable = None, *, 
                enable_cprofile: bool = False) -> Callable:
        """
        Decorator for profiling function performance.
        
        Args:
            func: Function to profile
            enable_cprofile: Enable detailed cProfile analysis
            
        Returns:
            Decorated function
        """
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                func_name = f.__name__
                
                # Start timing
                start_time = time.time()
                start_cpu = time.process_time()
                
                # Get initial memory usage
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run with cProfile if enabled
                if enable_cprofile:
                    profiler = cProfile.Profile()
                    profiler.enable()
                
                try:
                    result = f(*args, **kwargs)
                finally:
                    # Stop timing
                    end_time = time.time()
                    end_cpu = time.process_time()
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    
                    if enable_cprofile:
                        profiler.disable()
                        # Save profile data
                        s = io.StringIO()
                        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                        ps.print_stats()
                        logger.debug(f"Profile for {func_name}:\n{s.getvalue()}")
                    
                    # Record metrics
                    execution_time = end_time - start_time
                    cpu_time = end_cpu - start_cpu
                    memory_delta = end_memory - start_memory
                    
                    with self._lock:
                        if func_name not in self.metrics:
                            self.metrics[func_name] = []
                        
                        self.metrics[func_name].append(PerformanceMetrics(
                            function_name=func_name,
                            execution_time=execution_time,
                            cpu_usage=cpu_time,
                            memory_usage=memory_delta
                        ))
                    
                    logger.debug(
                        f"{func_name} - Time: {execution_time:.4f}s, "
                        f"CPU: {cpu_time:.4f}s, Memory: {memory_delta:.2f}MB"
                    )
                
                return result
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def cache(self, maxsize: int = 128, ttl: Optional[float] = None) -> Callable:
        """
        Decorator for caching function results.
        
        Args:
            maxsize: Maximum cache size
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            Decorated function
        """
        def decorator(func):
            cache_key = f"{func.__module__}.{func.__name__}"
            
            # Create cache if not exists
            if cache_key not in self.caches:
                self.caches[cache_key] = LRUCache(maxsize)
            
            cache = self.caches[cache_key]
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key from arguments
                key = self._make_cache_key(args, kwargs)
                
                # Check cache
                cached = cache.get(key)
                if cached is not None:
                    if ttl is None or (time.time() - cached['time']) < ttl:
                        return cached['value']
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Store in cache
                cache.put(key, {
                    'value': result,
                    'time': time.time()
                })
                
                return result
            
            # Add cache control methods
            wrapper.cache_clear = cache.clear
            wrapper.cache_info = lambda: {
                'hits': cache.hits,
                'misses': cache.misses,
                'hit_rate': cache.hit_rate,
                'size': len(cache.cache)
            }
            
            return wrapper
        
        return decorator
    
    def _make_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Create cache key from function arguments."""
        # Serialize arguments
        key_data = pickle.dumps((args, sorted(kwargs.items())))
        # Create hash
        return hashlib.sha256(key_data).hexdigest()
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    max_workers: Optional[int] = None,
                    use_processes: bool = False) -> List[Any]:
        """
        Execute function in parallel across items.
        
        Args:
            func: Function to execute
            items: Items to process
            max_workers: Maximum number of workers
            use_processes: Use processes instead of threads
            
        Returns:
            List of results
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            results = list(executor.map(func, items))
        
        return results
    
    def batch_process(self, func: Callable, items: List[Any], 
                     batch_size: int = 32) -> List[Any]:
        """
        Process items in batches for better memory efficiency.
        
        Args:
            func: Function to process batch
            items: Items to process
            batch_size: Size of each batch
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = func(batch)
            results.extend(batch_results)
        
        return results
    
    def create_memory_pool(self, name: str, shape: Tuple[int, ...], 
                          dtype: np.dtype = np.float32, 
                          pool_size: int = 10):
        """
        Create memory pool for efficient array allocation.
        
        Args:
            name: Pool name
            shape: Array shape
            dtype: Data type
            pool_size: Number of arrays to pool
        """
        self.memory_pools[name] = MemoryPool(shape, dtype, pool_size)
    
    def get_array(self, pool_name: str) -> np.ndarray:
        """Get array from memory pool."""
        if pool_name not in self.memory_pools:
            raise ValueError(f"Memory pool '{pool_name}' not found")
        return self.memory_pools[pool_name].get()
    
    def return_array(self, pool_name: str, array: np.ndarray):
        """Return array to memory pool."""
        if pool_name not in self.memory_pools:
            raise ValueError(f"Memory pool '{pool_name}' not found")
        self.memory_pools[pool_name].put(array)
    
    def optimize_memory(self):
        """Optimize memory usage by running garbage collection."""
        # Clear caches
        for cache in self.caches.values():
            if cache.hit_rate < 0.1:  # Clear poorly performing caches
                cache.clear()
        
        # Run garbage collection
        gc.collect()
        
        # Log memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f}MB")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all tracked functions."""
        report = {}
        
        with self._lock:
            for func_name, metrics_list in self.metrics.items():
                if not metrics_list:
                    continue
                
                # Calculate statistics
                exec_times = [m.execution_time for m in metrics_list]
                cpu_times = [m.cpu_usage for m in metrics_list]
                memory_usages = [m.memory_usage for m in metrics_list]
                
                report[func_name] = {
                    'call_count': len(metrics_list),
                    'total_time': sum(exec_times),
                    'avg_time': np.mean(exec_times),
                    'min_time': min(exec_times),
                    'max_time': max(exec_times),
                    'avg_cpu_time': np.mean(cpu_times),
                    'avg_memory_delta': np.mean(memory_usages),
                    'total_memory_delta': sum(memory_usages)
                }
        
        # Add cache statistics
        cache_report = {}
        for cache_name, cache in self.caches.items():
            cache_report[cache_name] = cache.cache_info()
        
        report['caches'] = cache_report
        
        return report


# Global optimizer instance
_optimizer = PerformanceOptimizer()


def get_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    return _optimizer


# Decorator shortcuts
profile = _optimizer.profile
cache = _optimizer.cache


# Optimized numerical operations
class OptimizedOperations:
    """Optimized numerical operations for trading calculations."""
    
    @staticmethod
    @profile
    def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Optimized rolling mean calculation."""
        if len(data) < window:
            return np.full_like(data, np.nan)
        
        # Use cumsum for O(n) complexity
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window
    
    @staticmethod
    @profile
    def rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Optimized rolling standard deviation."""
        if len(data) < window:
            return np.full_like(data, np.nan)
        
        # Use Welford's online algorithm
        mean = OptimizedOperations.rolling_mean(data, window)
        
        # Pad mean to match data length
        mean_padded = np.concatenate([np.full(window-1, np.nan), mean])
        
        # Calculate rolling variance
        squared_diff = (data - mean_padded) ** 2
        variance = OptimizedOperations.rolling_mean(squared_diff, window)
        
        return np.sqrt(variance)
    
    @staticmethod
    @profile
    def matrix_multiply_optimized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized matrix multiplication using blocking."""
        # Use numpy's optimized BLAS routines
        return np.dot(A, B)
    
    @staticmethod
    @cache(maxsize=256)
    def calculate_returns(prices: np.ndarray, method: str = 'simple') -> np.ndarray:
        """Calculate returns with caching."""
        if method == 'simple':
            returns = np.diff(prices) / prices[:-1]
        elif method == 'log':
            returns = np.diff(np.log(prices))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return returns


# Example usage and testing
if __name__ == "__main__":
    # Test performance optimizer
    optimizer = get_optimizer()
    
    # Test profiling
    @profile
    def slow_function(n: int = 1000000):
        """Test function for profiling."""
        total = 0
        for i in range(n):
            total += i ** 2
        return total
    
    # Test caching
    @cache(maxsize=10)
    def expensive_calculation(x: int, y: int) -> int:
        """Test function for caching."""
        time.sleep(0.1)  # Simulate expensive operation
        return x * y
    
    # Run tests
    print("Testing profiling...")
    for _ in range(3):
        result = slow_function()
    
    print("\nTesting caching...")
    print(f"First call: {expensive_calculation(5, 10)}")
    print(f"Second call (cached): {expensive_calculation(5, 10)}")
    print(f"Cache info: {expensive_calculation.cache_info()}")
    
    # Test parallel processing
    print("\nTesting parallel processing...")
    items = list(range(10))
    results = optimizer.parallel_map(lambda x: x ** 2, items)
    print(f"Parallel results: {results}")
    
    # Test memory pool
    print("\nTesting memory pool...")
    optimizer.create_memory_pool('test_pool', shape=(1000, 100))
    arr1 = optimizer.get_array('test_pool')
    arr2 = optimizer.get_array('test_pool')
    print(f"Got arrays from pool: {arr1.shape}, {arr2.shape}")
    optimizer.return_array('test_pool', arr1)
    
    # Test optimized operations
    print("\nTesting optimized operations...")
    data = np.random.randn(10000)
    rolling_mean = OptimizedOperations.rolling_mean(data, 100)
    print(f"Calculated rolling mean: shape={rolling_mean.shape}")
    
    # Get performance report
    print("\nPerformance Report:")
    report = optimizer.get_performance_report()
    for func_name, stats in report.items():
        if func_name != 'caches':
            print(f"\n{func_name}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    print("\nPerformance optimization test completed!")