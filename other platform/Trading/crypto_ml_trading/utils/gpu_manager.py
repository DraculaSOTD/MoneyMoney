"""
GPU Management System for Crypto ML Trading.

Provides comprehensive GPU management including device selection, memory monitoring,
mixed precision training, and multi-GPU support.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
import logging
import psutil
import GPUtil
import os
from contextlib import contextmanager
import gc
import warnings

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    total_memory: float  # GB
    free_memory: float   # GB
    used_memory: float   # GB
    utilization: float   # Percentage
    temperature: float   # Celsius
    is_available: bool


class GPUManager:
    """
    Comprehensive GPU management for ML training and inference.
    
    Features:
    - Automatic GPU detection and selection
    - Memory monitoring and optimization
    - Mixed precision (AMP) support
    - Batch size auto-tuning
    - Multi-GPU device management
    - Automatic fallback to CPU
    """
    
    def __init__(self, 
                 prefer_gpu: bool = True,
                 gpu_ids: Optional[List[int]] = None,
                 memory_fraction: float = 0.9,
                 enable_mixed_precision: bool = True,
                 enable_cudnn_benchmark: bool = True):
        """
        Initialize GPU Manager.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU
            gpu_ids: Specific GPU IDs to use (None for automatic)
            memory_fraction: Maximum fraction of GPU memory to use
            enable_mixed_precision: Enable automatic mixed precision
            enable_cudnn_benchmark: Enable cuDNN auto-tuner
        """
        self.prefer_gpu = prefer_gpu
        self.gpu_ids = gpu_ids
        self.memory_fraction = memory_fraction
        self.enable_mixed_precision = enable_mixed_precision
        
        # Initialize CUDA settings
        if torch.cuda.is_available() and enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Detect available GPUs
        self.gpu_info = self._detect_gpus()
        
        # Select device
        self.device = self._select_device()
        
        # Initialize mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.enable_mixed_precision and self.is_gpu else None
        
        # Memory tracking
        self.memory_stats = {}
        
        logger.info(f"GPU Manager initialized with device: {self.device}")
        if self.is_gpu:
            logger.info(f"GPU Memory: {self.get_memory_info()}")
    
    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect all available GPUs."""
        gpu_info = []
        
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Running on CPU.")
            return gpu_info
        
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                info = GPUInfo(
                    index=i,
                    name=gpu.name,
                    total_memory=gpu.memoryTotal / 1024,  # Convert to GB
                    free_memory=gpu.memoryFree / 1024,
                    used_memory=gpu.memoryUsed / 1024,
                    utilization=gpu.load * 100,
                    temperature=gpu.temperature,
                    is_available=True
                )
                gpu_info.append(info)
                
        except Exception as e:
            logger.warning(f"Error detecting GPUs with GPUtil: {e}")
            # Fallback to PyTorch detection
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info = GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory=props.total_memory / (1024**3),
                    free_memory=0,  # Cannot determine without GPUtil
                    used_memory=0,
                    utilization=0,
                    temperature=0,
                    is_available=True
                )
                gpu_info.append(info)
        
        return gpu_info
    
    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if not self.prefer_gpu or not self.gpu_info:
            return torch.device('cpu')
        
        # Use specified GPU IDs if provided
        if self.gpu_ids:
            valid_ids = [id for id in self.gpu_ids if id < len(self.gpu_info)]
            if valid_ids:
                # Select GPU with most free memory
                best_gpu = max(valid_ids, key=lambda i: self.gpu_info[i].free_memory)
                return torch.device(f'cuda:{best_gpu}')
        
        # Otherwise, select GPU with most free memory
        best_gpu = max(range(len(self.gpu_info)), key=lambda i: self.gpu_info[i].free_memory)
        return torch.device(f'cuda:{best_gpu}')
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self.device.type == 'cuda'
    
    def to_device(self, data: Union[torch.Tensor, np.ndarray, List, Dict]) -> Any:
        """
        Move data to the selected device.
        
        Args:
            data: Data to move (tensor, array, list, or dict)
            
        Returns:
            Data on the selected device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
            return tensor.to(self.device)
        elif isinstance(data, list):
            return [self.to_device(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.to_device(value) for key, value in data.items()}
        else:
            return data
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array, handling device transfer.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Numpy array
        """
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    
    @contextmanager
    def autocast(self, enabled: bool = True):
        """
        Context manager for automatic mixed precision.
        
        Args:
            enabled: Whether to enable autocast
        """
        if self.is_gpu and self.enable_mixed_precision and enabled:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss
        """
        if self.scaler and self.is_gpu:
            return self.scaler.scale(loss)
        return loss
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step with gradient scaling.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if self.scaler and self.is_gpu:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def optimize_batch_size(self, model: torch.nn.Module, 
                          input_shape: Tuple[int, ...],
                          initial_batch_size: int = 32,
                          max_batch_size: int = 1024,
                          safety_factor: float = 0.9) -> int:
        """
        Find optimal batch size based on GPU memory.
        
        Args:
            model: PyTorch model
            input_shape: Shape of single input (without batch dimension)
            initial_batch_size: Starting batch size
            max_batch_size: Maximum batch size to try
            safety_factor: Memory safety factor
            
        Returns:
            Optimal batch size
        """
        if not self.is_gpu:
            return initial_batch_size
        
        model = model.to(self.device)
        model.eval()
        
        batch_size = initial_batch_size
        optimal_batch_size = initial_batch_size
        
        while batch_size <= max_batch_size:
            try:
                # Clear cache
                self.clear_cache()
                
                # Try forward pass
                dummy_input = torch.randn(batch_size, *input_shape, device=self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Check memory usage
                memory_used = self.get_memory_used()
                memory_total = self.get_memory_total()
                
                if memory_used / memory_total < self.memory_fraction * safety_factor:
                    optimal_batch_size = batch_size
                    batch_size *= 2
                else:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
            finally:
                self.clear_cache()
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        if not self.is_gpu:
            return {"total": 0, "used": 0, "free": 0, "percent": 0}
        
        gpu_id = self.device.index if self.device.index is not None else 0
        
        return {
            "total": torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3),
            "used": torch.cuda.memory_allocated(gpu_id) / (1024**3),
            "reserved": torch.cuda.memory_reserved(gpu_id) / (1024**3),
            "free": (torch.cuda.get_device_properties(gpu_id).total_memory - 
                    torch.cuda.memory_allocated(gpu_id)) / (1024**3),
            "percent": (torch.cuda.memory_allocated(gpu_id) / 
                       torch.cuda.get_device_properties(gpu_id).total_memory) * 100
        }
    
    def get_memory_used(self) -> float:
        """Get used GPU memory in GB."""
        if not self.is_gpu:
            return 0.0
        gpu_id = self.device.index if self.device.index is not None else 0
        return torch.cuda.memory_allocated(gpu_id) / (1024**3)
    
    def get_memory_total(self) -> float:
        """Get total GPU memory in GB."""
        if not self.is_gpu:
            return 0.0
        gpu_id = self.device.index if self.device.index is not None else 0
        return torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.is_gpu:
            torch.cuda.empty_cache()
            gc.collect()
    
    def synchronize(self):
        """Synchronize GPU operations."""
        if self.is_gpu:
            torch.cuda.synchronize(self.device)
    
    def set_memory_fraction(self, fraction: float):
        """
        Set maximum GPU memory fraction to use.
        
        Args:
            fraction: Fraction of GPU memory (0-1)
        """
        self.memory_fraction = max(0.1, min(1.0, fraction))
        
        if self.is_gpu:
            # This is a suggestion to PyTorch, not a hard limit
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
    
    def enable_gradient_checkpointing(self, model: torch.nn.Module):
        """
        Enable gradient checkpointing for memory-efficient training.
        
        Args:
            model: PyTorch model
        """
        def checkpoint_sequential(module):
            def custom_forward(*inputs):
                return torch.utils.checkpoint.checkpoint(
                    module._forward,
                    *inputs
                )
            module.forward = custom_forward
        
        # Apply to sequential modules
        for module in model.modules():
            if isinstance(module, torch.nn.Sequential):
                checkpoint_sequential(module)
    
    def profile_model(self, model: torch.nn.Module, 
                     input_shape: Tuple[int, ...],
                     batch_size: int = 1) -> Dict[str, Any]:
        """
        Profile model performance on GPU.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (without batch)
            batch_size: Batch size for profiling
            
        Returns:
            Profiling results
        """
        model = model.to(self.device)
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(batch_size, *input_shape, device=self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        self.synchronize()
        
        # Profile forward pass
        import time
        times = []
        memory_usage = []
        
        for _ in range(100):
            self.clear_cache()
            start_memory = self.get_memory_used()
            
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            self.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)
            memory_usage.append(self.get_memory_used() - start_memory)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "mean_memory": np.mean(memory_usage),
            "throughput": batch_size / np.mean(times),
            "device": str(self.device),
            "mixed_precision": self.enable_mixed_precision
        }
    
    @contextmanager
    def distributed_training(self, local_rank: int):
        """
        Context manager for distributed training setup.
        
        Args:
            local_rank: Local rank of the process
        """
        if not self.is_gpu:
            yield
            return
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f'cuda:{local_rank}')
        
        # Initialize process group
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
        
        try:
            yield
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    
    def get_multi_gpu_devices(self, num_gpus: Optional[int] = None) -> List[torch.device]:
        """
        Get list of GPU devices for multi-GPU training.
        
        Args:
            num_gpus: Number of GPUs to use (None for all)
            
        Returns:
            List of GPU devices
        """
        if not self.gpu_info:
            return [torch.device('cpu')]
        
        available_gpus = len(self.gpu_info)
        num_gpus = min(num_gpus or available_gpus, available_gpus)
        
        # Sort GPUs by free memory
        gpu_indices = sorted(
            range(available_gpus), 
            key=lambda i: self.gpu_info[i].free_memory,
            reverse=True
        )[:num_gpus]
        
        return [torch.device(f'cuda:{i}') for i in gpu_indices]
    
    def parallel_model(self, model: torch.nn.Module, 
                      device_ids: Optional[List[int]] = None) -> torch.nn.Module:
        """
        Create DataParallel model for multi-GPU training.
        
        Args:
            model: PyTorch model
            device_ids: GPU IDs to use
            
        Returns:
            DataParallel model
        """
        if not self.is_gpu or len(self.gpu_info) <= 1:
            return model.to(self.device)
        
        if device_ids is None:
            device_ids = list(range(len(self.gpu_info)))
        
        model = model.to(f'cuda:{device_ids[0]}')
        return torch.nn.DataParallel(model, device_ids=device_ids)
    
    def __repr__(self) -> str:
        """String representation of GPU Manager."""
        if self.is_gpu:
            memory_info = self.get_memory_info()
            return (f"GPUManager(device={self.device}, "
                   f"memory_used={memory_info['used']:.1f}GB/"
                   f"{memory_info['total']:.1f}GB)")
        else:
            return "GPUManager(device=cpu)"


# Singleton instance
_gpu_manager = None

def get_gpu_manager(**kwargs) -> GPUManager:
    """
    Get or create GPU manager singleton.
    
    Args:
        **kwargs: Arguments for GPUManager initialization
        
    Returns:
        GPU manager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(**kwargs)
    return _gpu_manager


# Utility functions
def auto_device() -> torch.device:
    """Get the best available device."""
    return get_gpu_manager().device


def to_device(data: Any) -> Any:
    """Move data to the best available device."""
    return get_gpu_manager().to_device(data)


def optimize_memory():
    """Optimize GPU memory usage."""
    manager = get_gpu_manager()
    manager.clear_cache()
    logger.info(f"GPU memory after optimization: {manager.get_memory_info()}")