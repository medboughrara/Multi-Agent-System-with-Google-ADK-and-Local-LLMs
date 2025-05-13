"""
System monitoring module for tracking hardware resources.
This module provides functions to monitor system resources like CPU, RAM, and GPU usage.
"""

import os
import platform
import psutil
import threading
import time
import logging
import atexit
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import GPU monitoring libraries in order of preference
GPU_HANDLER = None
GPU_AVAILABLE = False
CUDA_VERSION = None

try:
    import torch
    if torch.cuda.is_available():
        GPU_HANDLER = "torch"
        GPU_AVAILABLE = True
        CUDA_VERSION = torch.version.cuda
        logger.info(f"Using PyTorch for GPU monitoring. CUDA version: {CUDA_VERSION}")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except ImportError:
    logger.debug("PyTorch not available, trying next GPU monitoring option")

if not GPU_AVAILABLE:
    try:
        import GPUtil
        GPU_HANDLER = "gputil"
        GPU_AVAILABLE = True
        gpus = GPUtil.getGPUs()
        if gpus:
            logger.info(f"Using GPUtil for GPU monitoring. Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                logger.info(f"GPU Device: {gpu.name}, Memory: {gpu.memoryTotal/1024:.2f} GB")
    except ImportError:
        logger.debug("GPUtil not available, trying next GPU monitoring option")

if not GPU_AVAILABLE:
    try:
        import pynvml
        pynvml.nvmlInit()
        GPU_HANDLER = "nvml"
        GPU_AVAILABLE = True
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"Using NVML for GPU monitoring. Found {device_count} GPU(s)")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            logger.info(f"GPU Device: {name.decode()}, Memory: {info.total/1024**3:.2f} GB")
    except ImportError:
        logger.warning("No GPU monitoring libraries available. GPU monitoring will be disabled.")

class SystemMonitor:
    """
    Monitor system resources including CPU, RAM, and GPU (if available).
    """
    
    def __init__(self, interval: float = 1.0):
        """Initialize the system monitor."""
        self._interval = interval
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._history: Dict[str, List[float]] = {
            'cpu': [],
            'memory': [],
            'gpu_util': [],
            'gpu_memory': []
        }
        self._lock = threading.Lock()
        self.platform = platform.system().lower()
        
        # Initialize base metrics
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.cpu_count = psutil.cpu_count()
        self.gpu_info = self._get_gpu_info()
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self._monitor_thread:
            logger.info("System monitoring started")
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self._monitor_thread:
            logger.info("System monitoring stopped")
            self._stop_event.set()
            self._monitor_thread.join()
            self._monitor_thread = None
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            usage = self.get_current_usage()
            with self._lock:
                self._history['cpu'].append(usage['cpu'])
                self._history['memory'].append(usage['memory'])
                self._history['gpu_util'].append(usage.get('gpu_util', 0))
                self._history['gpu_memory'].append(usage.get('gpu_memory', 0))
            time.sleep(self._interval)
    
    def _get_gpu_info(self) -> Dict[str, float]:
        """Get GPU information based on available handler."""
        if not GPU_AVAILABLE:
            return {}
            
        if GPU_HANDLER == "torch":
            return self._get_gpu_info_torch()
        elif GPU_HANDLER == "gputil":
            return self._get_gpu_info_gputil()
        elif GPU_HANDLER == "nvml":
            return self._get_gpu_info_nvml()
        return {}
    
    def _get_gpu_info_torch(self) -> Dict[str, float]:
        """Get GPU information using PyTorch."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1024**3  # Convert to GB
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        
        return {
            'total_memory': total_memory,
            'used_memory': allocated,
            'reserved_memory': reserved,
            'free_memory': total_memory - allocated
        }
    
    def _get_gpu_info_gputil(self) -> Dict[str, float]:
        """Get GPU information using GPUtil."""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {}
            
            gpu = gpus[0]  # Use first GPU
            return {
                'total_memory': gpu.memoryTotal / 1024,  # Convert to GB
                'used_memory': gpu.memoryUsed / 1024,
                'free_memory': gpu.memoryFree / 1024,
                'gpu_util': gpu.load * 100
            }
        except Exception as e:
            logger.error(f"Error getting GPU info with GPUtil: {e}")
            return {}
    
    def _get_gpu_info_nvml(self) -> Dict[str, float]:
        """Get GPU information using NVML."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            return {
                'total_memory': info.total / 1024**3,
                'used_memory': info.used / 1024**3,
                'free_memory': info.free / 1024**3,
                'gpu_util': utilization.gpu
            }
        except Exception as e:
            logger.error(f"Error getting GPU info with NVML: {e}")
            return {}

    def get_current_usage(self) -> Dict[str, float]:
        """Get current system resource usage."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        usage = {
            'cpu': cpu_percent,
            'memory': memory_percent
        }
        
        if GPU_AVAILABLE:
            gpu_info = self._get_gpu_info()
            if gpu_info:
                usage.update({
                    'gpu_util': gpu_info.get('gpu_util', 0),
                    'gpu_memory': (gpu_info.get('used_memory', 0) / gpu_info.get('total_memory', 1)) * 100
                })
        
        return usage
    
    def get_resource_history(self) -> Dict[str, List[float]]:
        """Get resource usage history."""
        with self._lock:
            return {k: v.copy() for k, v in self._history.items()}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'platform': self.platform,
            'cpu_count': self.cpu_count,
            'total_memory_gb': self.total_memory,
            'gpu_available': GPU_AVAILABLE
        }
        
        if GPU_AVAILABLE:
            gpu_info = self._get_gpu_info()
            info.update({
                'gpu_handler': GPU_HANDLER,
                'cuda_version': CUDA_VERSION,
                'gpu_memory_gb': gpu_info.get('total_memory', 0)
            })
        
        return info
    
    def can_run_parallel(self, num_models: int, vram_per_model_gb: float = 4.0) -> bool:
        """Check if system can run models in parallel."""
        if not GPU_AVAILABLE:
            return False
            
        gpu_info = self._get_gpu_info()
        total_vram = gpu_info.get('total_memory', 0)
        free_vram = gpu_info.get('free_memory', 0)
        
        required_vram = num_models * vram_per_model_gb
        
        # Check if we have enough total and free VRAM
        if total_vram < required_vram:
            logger.warning(f"Insufficient GPU memory for parallel execution. Need {required_vram:.2f}GB, have {total_vram:.2f}GB")
            return False
            
        if free_vram < required_vram:
            logger.warning(f"Insufficient free GPU memory. Need {required_vram:.2f}GB, have {free_vram:.2f}GB free")
            return False
            
        return True

# Create a global instance for easy import
system_monitor = SystemMonitor()

# Start monitoring when the module is imported
system_monitor.start_monitoring()

def get_monitor() -> SystemMonitor:
    """Get the global SystemMonitor instance."""
    return system_monitor

# Ensure monitoring is stopped when the program exits
atexit.register(system_monitor.stop_monitoring)