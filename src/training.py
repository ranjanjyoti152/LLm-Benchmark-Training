"""
Training pipeline with real-time metrics collection for LLM benchmarking.
Runs exactly 2 epochs while collecting performance metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import logging
import math
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
from collections import defaultdict
import os

from .monitoring import SystemMonitor
from .model_config import ModelLoader
from .dataset import BenchmarkDataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics collected during benchmarking."""
    timestamp: float
    epoch: int
    step: int
    loss: float
    tokens_per_second: float
    gpu_utilization: Dict[int, float]
    gpu_memory_used: Dict[int, float]
    gpu_memory_total: Dict[int, float]
    cpu_utilization: float
    ram_utilization: float
    ram_used_gb: float
    ram_total_gb: float
    batch_size: int
    sequence_length: int
    learning_rate: float


class BenchmarkTrainer:
    """
    Training pipeline that collects comprehensive metrics during LLM training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_size: str,
        output_dir: str = "./benchmark_results",
        monitoring_interval: float = 1.0,
        log_interval: int = 10
    ):
        """
        Initialize benchmark trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            model_size: Size of the model being trained
            output_dir: Directory to save results
            monitoring_interval: How often to collect system metrics (seconds)
            log_interval: How often to log progress (steps)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_size = model_size
        self.output_dir = output_dir
        self.monitoring_interval = monitoring_interval
        self.log_interval = log_interval
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring
        self.monitor = SystemMonitor()
        self.metrics_history: List[TrainingMetrics] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Training state
        self.total_tokens_processed = 0
        self.training_start_time = None
        self.current_epoch = 0
        self.current_step = 0
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_model_input_device(self):
        """Get the device where model expects input tensors."""
        try:
            # For multi-GPU models with device_map
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                # Find the device of the first layer (embedding layer)
                for name, device in self.model.hf_device_map.items():
                    if 'embed' in name.lower() or 'wte' in name.lower():
                        self.logger.info(f"🎯 Found input device from device_map: {device} (layer: {name})")
                        return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
                
                # Fallback: use the device of the first parameter
                first_param_device = next(self.model.parameters()).device
                self.logger.info(f"🎯 Using first parameter device: {first_param_device}")
                return first_param_device
            else:
                # Standard single device or DataParallel
                return self.device
        except Exception as e:
            self.logger.warning(f"Could not determine model input device: {e}, using default: {self.device}")
            return self.device
    
    def _prepare_model_for_training(self):
        """Prepare the model for stable training by checking and initializing parameters."""
        logger.info("🔧 Preparing model for stable training...")
        
        # Check for any NaN/inf parameters before training
        nan_params = []
        inf_params = []
        zero_params = []
        
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
            if torch.all(param == 0):
                zero_params.append(name)
        
        if nan_params:
            logger.warning(f"⚠️  NaN parameters detected: {nan_params}")
            # Reinitialize NaN parameters
            for name, param in self.model.named_parameters():
                if name in nan_params:
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
            logger.info("✅ NaN parameters reinitialized")
        
        if inf_params:
            logger.warning(f"⚠️  Inf parameters detected: {inf_params}")
            # Reinitialize Inf parameters
            for name, param in self.model.named_parameters():
                if name in inf_params:
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
            logger.info("✅ Inf parameters reinitialized")
        
        if zero_params:
            logger.info(f"ℹ️  Zero-initialized parameters found: {len(zero_params)} (this may be normal)")
        
        # Enable gradient checkpointing for large models to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                logger.info("✅ Enabled gradient checkpointing")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Set model to training mode with dropout for regularization
        self.model.train()
        
        # For distributed models, ensure all parameters are properly initialized
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            logger.info("✅ Model uses device_map, parameters distributed across GPUs")
        
        logger.info("✅ Model preparation completed")
        
    def _get_optimal_batch_size(self) -> int:
        """
        Determine optimal batch size based on model size and GPU memory.
        
        Returns:
            Optimal batch size for the current model
        """
        if not torch.cuda.is_available():
            return 2  # Conservative for CPU
        
        # Get model parameter count
        param_count = sum(p.numel() for p in self.model.parameters())
        param_count_b = param_count / 1e9  # Convert to billions
        
        # Get available GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Calculate available memory per GPU (accounting for model memory)
        if hasattr(self.model, 'module'):  # DataParallel
            # Model is replicated across GPUs
            model_memory_gb = param_count * 4 / (1024**3)  # FP32 assumption
            available_per_gpu = gpu_memory_gb - model_memory_gb
        else:
            # Single GPU or model parallel
            total_gpu_memory = gpu_memory_gb * torch.cuda.device_count()
            model_memory_gb = param_count * 4 / (1024**3)
            available_per_gpu = (total_gpu_memory - model_memory_gb) / torch.cuda.device_count()
        
        # Conservative memory allocation for training (leave 20% buffer)
        usable_memory = available_per_gpu * 0.8
        
        # Adaptive batch size based on model size and memory
        if param_count_b < 1.0:  # < 1B parameters
            batch_size = min(16, max(1, int(usable_memory / 2)))
        elif param_count_b < 3.0:  # 1-3B parameters  
            batch_size = min(8, max(1, int(usable_memory / 4)))
        elif param_count_b < 10.0:  # 3-10B parameters
            batch_size = min(4, max(1, int(usable_memory / 6)))
        elif param_count_b < 20.0:  # 10-20B parameters
            batch_size = min(2, max(1, int(usable_memory / 8)))
        else:  # > 20B parameters
            batch_size = 1
        
        self.logger.info(f"Model: {param_count_b:.1f}B params, GPU memory: {gpu_memory_gb:.1f}GB, "
                   f"Available: {usable_memory:.1f}GB, Batch size: {batch_size}")
        
        return batch_size
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    gpu_metrics = self.monitor.get_gpu_metrics()
                    cpu_usage = self.monitor.get_cpu_usage()
                    memory_metrics = self.monitor.get_memory_metrics()
                    
                    # Store current metrics for access by training loop
                    self.current_metrics = {
                        'gpu_metrics': gpu_metrics,
                        'cpu_usage': cpu_usage,
                        'memory_metrics': memory_metrics,
                        'timestamp': time.time()
                    }
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.warning(f"Error in monitoring thread: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
    def _stop_monitoring(self):
        """Stop background monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _calculate_tokens_per_second(
        self, 
        batch_size: int, 
        sequence_length: int, 
        time_elapsed: float
    ) -> float:
        """Calculate tokens per second for current batch."""
        total_tokens = batch_size * sequence_length
        return total_tokens / time_elapsed if time_elapsed > 0 else 0.0
    
    def _collect_metrics(
        self,
        loss: float,
        batch_size: int,
        sequence_length: int,
        tokens_per_second: float,
        learning_rate: float
    ) -> TrainingMetrics:
        """Collect current training metrics."""
        current_time = time.time()
        
        # Get current system metrics
        metrics = getattr(self, 'current_metrics', {})
        gpu_metrics = metrics.get('gpu_metrics', {})
        memory_metrics = metrics.get('memory_metrics', {})
        
        # Extract GPU utilization and memory
        gpu_utilization = {}
        gpu_memory_used = {}
        gpu_memory_total = {}
        
        for gpu_id, gpu_info in gpu_metrics.items():
            gpu_utilization[gpu_id] = gpu_info.get('utilization', 0.0)
            gpu_memory_used[gpu_id] = gpu_info.get('memory_used', 0.0)
            gpu_memory_total[gpu_id] = gpu_info.get('memory_total', 0.0)
        
        return TrainingMetrics(
            timestamp=current_time,
            epoch=self.current_epoch,
            step=self.current_step,
            loss=loss,
            tokens_per_second=tokens_per_second,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            cpu_utilization=metrics.get('cpu_usage', 0.0),
            ram_utilization=memory_metrics.get('percent', 0.0),
            ram_used_gb=memory_metrics.get('used', 0.0) / (1024**3),
            ram_total_gb=memory_metrics.get('total', 0.0) / (1024**3),
            batch_size=batch_size,
            sequence_length=sequence_length,
            learning_rate=learning_rate
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0
    ) -> Dict[str, Any]:
        """
        Train for one epoch while collecting metrics.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            epoch: Current epoch number
            
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            self.current_step = step
            step_start_time = time.time()
            
            # Move tensors to the correct device for the model
            model_input_device = self._get_model_input_device()
            input_ids = batch["input_ids"].to(model_input_device)
            labels = batch["labels"].to(model_input_device)
            
            batch_size, sequence_length = input_ids.shape
            
            # Forward pass with error handling
            optimizer.zero_grad()
            
            try:
                outputs = self.model(input_ids, labels=labels)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM during forward pass at step {step}, clearing cache and continuing")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # Handle different output types (simple loss vs object with loss)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, torch.Tensor):
                loss = outputs  # Direct loss return for DataParallel compatibility
            else:
                # Fallback for other formats
                loss = outputs.get('loss', outputs)
            
            # Handle DataParallel output (average if it's a vector)
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Critical: Check for NaN/inf loss before backward pass
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"⚠️  NaN/Inf loss detected at step {step}: {loss.item()}")
                logger.warning("Skipping backward pass and continuing with next batch")
                continue
            
            # Check for reasonable loss values
            loss_value = loss.item()
            if loss_value > 100.0:  # Unreasonably high loss
                logger.warning(f"⚠️  Very high loss detected at step {step}: {loss_value:.4f}")
                logger.warning("Reducing learning rate temporarily")
                # Temporarily reduce learning rate for this step
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            
            # Backward pass with gradient scaling for stability
            try:
                # For very small losses, scale them up to prevent underflow
                if loss_value < 1e-6:
                    scaled_loss = loss * 1e6
                    scale_factor = 1e6
                    logger.debug(f"Scaling very small loss {loss_value:.2e} by {scale_factor}")
                elif loss_value < 1e-3:
                    scaled_loss = loss * 1e3
                    scale_factor = 1e3
                else:
                    scaled_loss = loss
                    scale_factor = 1.0
                
                scaled_loss.backward()
                
                # Scale gradients back down if we scaled the loss
                if scale_factor > 1.0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data *= (1.0 / scale_factor)
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM during backward pass at step {step}, clearing cache and continuing")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # Synchronize gradients for distributed models
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                try:
                    # Ensure all gradient computations are complete
                    torch.cuda.synchronize()
                except Exception as e:
                    self.logger.warning(f"Error synchronizing CUDA: {e}")
            
            # Critical: Check gradients for NaN/inf before clipping
            nan_grads = False
            inf_grads = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_grads = True
                        logger.warning(f"NaN gradient detected in {name}")
                    if torch.isinf(param.grad).any():
                        inf_grads = True
                        logger.warning(f"Inf gradient detected in {name}")
            
            if nan_grads or inf_grads:
                logger.warning("⚠️  NaN/Inf gradients detected, skipping optimizer step")
                optimizer.zero_grad()
                continue
            
            # Aggressive gradient clipping for stability
            max_grad_norm = 0.5  # More conservative than 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            
            # Check if gradient norm is too high
            if grad_norm > max_grad_norm * 2:
                logger.warning(f"⚠️  High gradient norm detected: {grad_norm:.4f}, applying extra clipping")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm * 0.5)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Clear gradients and cache to prevent memory buildup
            optimizer.zero_grad()
            
            # Memory management for large models
            if torch.cuda.is_available():
                # Check GPU memory usage after every few steps
                if step % 5 == 0:  # Check every 5 steps
                    try:
                        for gpu_id in range(torch.cuda.device_count()):
                            memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                            memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)   # GB
                            memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
                            
                            # If memory usage is very high, clean up carefully
                            if memory_allocated > memory_total * 0.9:  # If >90% usage
                                self.logger.warning(f"High memory usage on GPU {gpu_id}: {memory_allocated:.1f}GB/{memory_total:.1f}GB")
                                # Only clear cache on this specific GPU if possible
                                with torch.cuda.device(gpu_id):
                                    torch.cuda.empty_cache()
                                break
                    except Exception as e:
                        self.logger.warning(f"Error checking GPU memory: {e}")
                
                # Less frequent and safer memory cleanup for distributed models
                if step % 20 == 0 and not hasattr(self.model, 'hf_device_map'):  # Only for non-distributed models
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        self.logger.warning(f"Error during memory cleanup: {e}")
            
            # Calculate timing and tokens
            step_time = time.time() - step_start_time
            tokens_this_batch = batch_size * sequence_length
            tokens_per_second = self._calculate_tokens_per_second(
                batch_size, sequence_length, step_time
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Collect metrics
            metrics = self._collect_metrics(
                loss=loss.item(),
                batch_size=batch_size,
                sequence_length=sequence_length,
                tokens_per_second=tokens_per_second,
                learning_rate=current_lr
            )
            
            self.metrics_history.append(metrics)
            
            # Update totals
            total_loss += loss.item()
            total_tokens += tokens_this_batch
            num_batches += 1
            self.total_tokens_processed += tokens_this_batch
            
            # Logging
            if step % self.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                elapsed_time = time.time() - epoch_start_time
                overall_tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
                
                logger.info(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Tokens/sec: {tokens_per_second:.1f} | "
                    f"Overall Tokens/sec: {overall_tokens_per_sec:.1f} | "
                    f"LR: {current_lr:.2e}"
                )
                
                # Log GPU metrics if available
                if hasattr(self, 'current_metrics'):
                    gpu_metrics = self.current_metrics.get('gpu_metrics', {})
                    for gpu_id, gpu_info in gpu_metrics.items():
                        logger.info(
                            f"GPU {gpu_id}: {gpu_info.get('utilization', 0):.1f}% | "
                            f"Memory: {gpu_info.get('memory_used', 0):.1f}/"
                            f"{gpu_info.get('memory_total', 0):.1f} MB"
                        )
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0.0
        
        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "epoch_time": epoch_time,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "num_batches": num_batches
        }
    
    def benchmark_training(
        self,
        num_samples: int = 1000,
        custom_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Run complete training benchmark for exactly 2 epochs.
        
        Args:
            num_samples: Number of samples for synthetic data (if no custom dataloader)
            custom_dataloader: Custom dataloader to use instead of synthetic data
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting benchmark training for {self.model_size} model")
        
        # Prepare model for stable training
        self._prepare_model_for_training()
        
        # Start monitoring
        self._start_monitoring()
        self.training_start_time = time.time()
        
        try:
            # Determine optimal batch size based on model size and available memory
            optimal_batch_size = self._get_optimal_batch_size()
            
            # Create dataloader if not provided
            if custom_dataloader is None:
                logger.info("Creating synthetic dataset for benchmarking...")
                dataloader = BenchmarkDataLoader.create_synthetic_dataloader(
                    tokenizer=self.tokenizer,
                    batch_size=optimal_batch_size,
                    num_samples=num_samples,
                    max_length=512,
                    num_workers=0  # Avoid multiprocessing issues
                )
            else:
                dataloader = custom_dataloader
            
            # Setup optimizer - use memory efficient settings for large models
            param_count = sum(p.numel() for p in self.model.parameters())
            param_count_b = param_count / 1e9
            
            logger.info(f"Model has {param_count_b:.1f}B parameters, configuring optimizer accordingly")
            
            if param_count > 1e9:  # For models > 1B params
                # Ultra-conservative settings for large models
                base_lr = 5e-6  # Much lower base learning rate
                if param_count_b > 10:  # For very large models (>10B)
                    base_lr = 1e-6
                elif param_count_b > 30:  # For extremely large models (>30B)
                    base_lr = 5e-7
                
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=base_lr,
                    weight_decay=0.01,
                    eps=1e-8,
                    betas=(0.9, 0.999),  # More stable betas
                    amsgrad=True  # Use AMSGrad variant for stability
                )
                logger.info(f"🔧 Using conservative AdamW for {param_count_b:.1f}B model with LR={base_lr:.1e}")
            else:
                # Standard settings for smaller models
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=1e-5,  # Reduced from 2e-5
                    weight_decay=0.01,
                    eps=1e-8,
                    betas=(0.9, 0.999)
                )
                logger.info(f"🔧 Using standard AdamW for {param_count_b:.1f}B model")
            
            # Setup learning rate scheduler with warmup
            total_steps = len(dataloader) * 2  # 2 epochs
            warmup_steps = min(100, total_steps // 10)  # 10% warmup or max 100 steps
            
            # Create a custom scheduler with warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return step / warmup_steps
                else:
                    # Cosine annealing after warmup
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + math.cos(progress * math.pi))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            logger.info(f"🔧 Configured scheduler with {warmup_steps} warmup steps out of {total_steps} total steps")
            
            # Train for exactly 2 epochs
            epoch_results = []
            
            for epoch in range(2):
                logger.info(f"Starting epoch {epoch + 1}/2")
                
                epoch_result = self.train_epoch(
                    dataloader=dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch
                )
                
                epoch_results.append(epoch_result)
                
                logger.info(
                    f"Epoch {epoch + 1} completed | "
                    f"Avg Loss: {epoch_result['avg_loss']:.4f} | "
                    f"Tokens/sec: {epoch_result['avg_tokens_per_second']:.1f} | "
                    f"Time: {epoch_result['epoch_time']:.1f}s"
                )
            
            total_training_time = time.time() - self.training_start_time
            
            # Calculate overall statistics
            total_tokens = sum(r['total_tokens'] for r in epoch_results)
            avg_tokens_per_second = total_tokens / total_training_time
            final_loss = epoch_results[-1]['avg_loss']
            
            results = {
                "model_size": self.model_size,
                "total_training_time": total_training_time,
                "total_tokens_processed": total_tokens,
                "avg_tokens_per_second": avg_tokens_per_second,
                "final_loss": final_loss,
                "epoch_results": epoch_results,
                "metrics_history": self.metrics_history,
                "num_epochs": 2,
                "benchmark_completed": True
            }
            
            logger.info(f"Benchmark completed for {self.model_size} model")
            logger.info(f"Total time: {total_training_time:.1f}s")
            logger.info(f"Total tokens: {total_tokens:,}")
            logger.info(f"Average tokens/sec: {avg_tokens_per_second:.1f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during benchmark training: {e}")
            raise
        finally:
            # Stop monitoring
            self._stop_monitoring()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics from collected metrics."""
        if not self.metrics_history:
            return {}
        
        # Extract metrics for analysis
        losses = [m.loss for m in self.metrics_history]
        tokens_per_sec = [m.tokens_per_second for m in self.metrics_history]
        cpu_utilizations = [m.cpu_utilization for m in self.metrics_history]
        ram_utilizations = [m.ram_utilization for m in self.metrics_history]
        
        # GPU metrics (average across all GPUs)
        gpu_utils = []
        gpu_memory_usage = []
        
        for m in self.metrics_history:
            if m.gpu_utilization:
                avg_gpu_util = sum(m.gpu_utilization.values()) / len(m.gpu_utilization)
                gpu_utils.append(avg_gpu_util)
            
            if m.gpu_memory_used and m.gpu_memory_total:
                total_used = sum(m.gpu_memory_used.values())
                total_available = sum(m.gpu_memory_total.values())
                gpu_memory_usage.append((total_used / total_available) * 100 if total_available > 0 else 0)
        
        return {
            "loss": {
                "final": losses[-1] if losses else 0,
                "min": min(losses) if losses else 0,
                "max": max(losses) if losses else 0,
                "avg": sum(losses) / len(losses) if losses else 0
            },
            "tokens_per_second": {
                "avg": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
                "min": min(tokens_per_sec) if tokens_per_sec else 0,
                "max": max(tokens_per_sec) if tokens_per_sec else 0
            },
            "cpu_utilization": {
                "avg": sum(cpu_utilizations) / len(cpu_utilizations) if cpu_utilizations else 0,
                "min": min(cpu_utilizations) if cpu_utilizations else 0,
                "max": max(cpu_utilizations) if cpu_utilizations else 0
            },
            "ram_utilization": {
                "avg": sum(ram_utilizations) / len(ram_utilizations) if ram_utilizations else 0,
                "min": min(ram_utilizations) if ram_utilizations else 0,
                "max": max(ram_utilizations) if ram_utilizations else 0
            },
            "gpu_utilization": {
                "avg": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                "min": min(gpu_utils) if gpu_utils else 0,
                "max": max(gpu_utils) if gpu_utils else 0
            },
            "gpu_memory_utilization": {
                "avg": sum(gpu_memory_usage) / len(gpu_memory_usage) if gpu_memory_usage else 0,
                "min": min(gpu_memory_usage) if gpu_memory_usage else 0,
                "max": max(gpu_memory_usage) if gpu_memory_usage else 0
            }
        }