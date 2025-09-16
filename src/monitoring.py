"""
System monitoring utilities for tracking GPU, CPU, and memory utilization during model training.
"""

import time
import threading
import psutil
import torch
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring will be limited")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available, using alternative GPU monitoring")


@dataclass
class SystemMetrics:
    """Container for system performance metrics"""
    timestamp: datetime
    gpu_utilization: List[float]  # Per-GPU utilization percentages
    gpu_memory_used: List[float]  # Per-GPU memory used in GB
    gpu_memory_total: List[float]  # Per-GPU total memory in GB
    gpu_temperature: List[float]  # Per-GPU temperature in Celsius
    cpu_utilization: float  # Overall CPU utilization percentage
    ram_used: float  # RAM used in GB
    ram_total: float  # Total RAM in GB
    tokens_per_second: Optional[float] = None  # Training speed metric


class SystemMonitor:
    """Real-time system monitoring for training benchmarks"""
    
    def __init__(self, sample_interval: float = 1.0):
        """
        Initialize the system monitor.
        
        Args:
            sample_interval: Time between samples in seconds
        """
        self.sample_interval = sample_interval
        self.monitoring = False
        self.metrics_history: List[SystemMetrics] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._tokens_callback: Optional[Callable[[], float]] = None
        
        # Initialize GPU monitoring
        self._init_gpu_monitoring()
        
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring libraries"""
        self.gpu_count = 0
        
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    self.use_pynvml = True
                    logging.info(f"Initialized pynvml for {self.gpu_count} GPUs")
                except Exception as e:
                    logging.warning(f"Failed to initialize pynvml: {e}")
                    self.use_pynvml = False
            else:
                self.use_pynvml = False
        else:
            logging.warning("CUDA not available, GPU monitoring disabled")
            
    def set_tokens_callback(self, callback: Callable[[], float]):
        """Set callback function to get current tokens per second"""
        self._tokens_callback = callback
        
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring:
            logging.warning("Monitoring already started")
            return
            
        self.monitoring = True
        self.metrics_history = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logging.info("Started system monitoring")
        
    def stop_monitoring(self):
        """Stop background monitoring thread"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logging.info("Stopped system monitoring")
        
    def _monitor_loop(self):
        """Main monitoring loop running in background thread"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
                
            time.sleep(self.sample_interval)
            
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        timestamp = datetime.now()
        
        # GPU metrics
        gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = self._get_gpu_metrics()
        
        # CPU metrics
        cpu_util = psutil.cpu_percent(interval=None)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        ram_used = memory.used / (1024**3)  # Convert to GB
        ram_total = memory.total / (1024**3)  # Convert to GB
        
        # Tokens per second (if callback is set)
        tokens_per_second = None
        if self._tokens_callback:
            try:
                tokens_per_second = self._tokens_callback()
            except Exception as e:
                logging.warning(f"Error getting tokens per second: {e}")
        
        return SystemMetrics(
            timestamp=timestamp,
            gpu_utilization=gpu_util,
            gpu_memory_used=gpu_mem_used,
            gpu_memory_total=gpu_mem_total,
            gpu_temperature=gpu_temp,
            cpu_utilization=cpu_util,
            ram_used=ram_used,
            ram_total=ram_total,
            tokens_per_second=tokens_per_second
        )
        
    def _get_gpu_metrics(self) -> tuple:
        """Get GPU utilization, memory, and temperature metrics"""
        if self.gpu_count == 0:
            return [], [], [], []
            
        gpu_util = []
        gpu_mem_used = []
        gpu_mem_total = []
        gpu_temp = []
        
        if self.use_pynvml:
            # Use pynvml for detailed GPU metrics
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util.append(util.gpu)
                    
                    # Memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem_used.append(mem_info.used / (1024**3))  # Convert to GB
                    gpu_mem_total.append(mem_info.total / (1024**3))  # Convert to GB
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temp.append(temp)
                    
                except Exception as e:
                    logging.warning(f"Error getting metrics for GPU {i}: {e}")
                    gpu_util.append(0.0)
                    gpu_mem_used.append(0.0)
                    gpu_mem_total.append(0.0)
                    gpu_temp.append(0.0)
                    
        elif GPUTIL_AVAILABLE:
            # Use GPUtil as fallback
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_util.append(gpu.load * 100)  # Convert to percentage
                    gpu_mem_used.append(gpu.memoryUsed / 1024)  # Convert MB to GB
                    gpu_mem_total.append(gpu.memoryTotal / 1024)  # Convert MB to GB
                    gpu_temp.append(gpu.temperature)
            except Exception as e:
                logging.warning(f"Error using GPUtil: {e}")
                
        else:
            # Use PyTorch as last resort (limited info)
            for i in range(self.gpu_count):
                try:
                    # PyTorch memory info
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    # Estimate utilization based on memory usage
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / (1024**3)
                    
                    gpu_util.append(0.0)  # Can't get utilization from PyTorch
                    gpu_mem_used.append(mem_reserved)
                    gpu_mem_total.append(total_memory)
                    gpu_temp.append(0.0)  # Can't get temperature from PyTorch
                    
                except Exception as e:
                    logging.warning(f"Error getting PyTorch GPU metrics for device {i}: {e}")
                    gpu_util.append(0.0)
                    gpu_mem_used.append(0.0)
                    gpu_mem_total.append(0.0)
                    gpu_temp.append(0.0)
        
        return gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp
        
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics without storing them"""
        return self._collect_metrics()
        
    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of collected metrics"""
        if not self.metrics_history:
            return {}
            
        gpu_util_avg = []
        gpu_mem_avg = []
        tokens_per_sec_samples = []
        cpu_samples = []
        ram_samples = []
        
        for metrics in self.metrics_history:
            cpu_samples.append(metrics.cpu_utilization)
            ram_samples.append(metrics.ram_used)
            
            if metrics.tokens_per_second is not None:
                tokens_per_sec_samples.append(metrics.tokens_per_second)
                
            if metrics.gpu_utilization:
                if not gpu_util_avg:
                    gpu_util_avg = [[] for _ in range(len(metrics.gpu_utilization))]
                    gpu_mem_avg = [[] for _ in range(len(metrics.gpu_memory_used))]
                    
                for i, util in enumerate(metrics.gpu_utilization):
                    gpu_util_avg[i].append(util)
                for i, mem in enumerate(metrics.gpu_memory_used):
                    gpu_mem_avg[i].append(mem)
        
        summary = {
            'duration_seconds': len(self.metrics_history) * self.sample_interval,
            'sample_count': len(self.metrics_history),
            'cpu_utilization_avg': sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            'cpu_utilization_max': max(cpu_samples) if cpu_samples else 0,
            'ram_used_avg_gb': sum(ram_samples) / len(ram_samples) if ram_samples else 0,
            'ram_used_max_gb': max(ram_samples) if ram_samples else 0,
        }
        
        if tokens_per_sec_samples:
            summary.update({
                'tokens_per_second_avg': sum(tokens_per_sec_samples) / len(tokens_per_sec_samples),
                'tokens_per_second_max': max(tokens_per_sec_samples),
                'tokens_per_second_min': min(tokens_per_sec_samples),
            })
            
        # GPU summaries
        for i, util_samples in enumerate(gpu_util_avg):
            if util_samples:
                summary[f'gpu_{i}_utilization_avg'] = sum(util_samples) / len(util_samples)
                summary[f'gpu_{i}_utilization_max'] = max(util_samples)
                
        for i, mem_samples in enumerate(gpu_mem_avg):
            if mem_samples:
                summary[f'gpu_{i}_memory_avg_gb'] = sum(mem_samples) / len(mem_samples)
                summary[f'gpu_{i}_memory_max_gb'] = max(mem_samples)
        
        return summary
        
    def clear_history(self):
        """Clear collected metrics history"""
        self.metrics_history = []
    
    def get_gpu_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get current GPU metrics in dictionary format for CLI compatibility"""
        gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = self._get_gpu_metrics()
        
        gpu_metrics = {}
        for i in range(len(gpu_util)):
            gpu_metrics[i] = {
                'utilization': gpu_util[i],
                'memory_used': gpu_mem_used[i] * 1024,  # Convert back to MB for compatibility
                'memory_total': gpu_mem_total[i] * 1024,  # Convert back to MB for compatibility
                'temperature': gpu_temp[i]
            }
        
        return gpu_metrics
    
    def get_cpu_usage(self) -> float:
        """Get current CPU utilization percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_metrics(self) -> Dict[str, float]:
        """Get current memory metrics"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'used': memory.used,
            'percent': memory.percent,
            'available': memory.available
        }
    
    def get_cpu_info(self) -> Dict[str, any]:
        """Get CPU information"""
        try:
            return {
                'count': psutil.cpu_count(),
                'brand': 'Unknown',  # psutil doesn't provide brand easily
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True)
            }
        except:
            return {
                'count': 'Unknown',
                'brand': 'Unknown',
                'physical_cores': 'Unknown',
                'logical_cores': 'Unknown'
            }


class TokensPerSecondTracker:
    """Helper class to track tokens processed per second during training"""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize tracker.
        
        Args:
            window_size: Number of recent measurements to average over
        """
        self.window_size = window_size
        self.token_counts = []
        self.timestamps = []
        
    def update(self, tokens_processed: int):
        """Update with number of tokens processed"""
        current_time = time.time()
        
        self.token_counts.append(tokens_processed)
        self.timestamps.append(current_time)
        
        # Keep only recent measurements
        if len(self.token_counts) > self.window_size:
            self.token_counts.pop(0)
            self.timestamps.pop(0)
            
    def get_tokens_per_second(self) -> float:
        """Get current tokens per second rate"""
        if len(self.token_counts) < 2:
            return 0.0
            
        # Calculate rate over the window
        total_tokens = sum(self.token_counts)
        time_span = self.timestamps[-1] - self.timestamps[0]
        
        if time_span <= 0:
            return 0.0
            
        return total_tokens / time_span
        
    def reset(self):
        """Reset the tracker"""
        self.token_counts = []
        self.timestamps = []