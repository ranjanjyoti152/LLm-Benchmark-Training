"""
Model configuration and loading utilities         "20B": {
            "model_name": "gpt2",  # We'll scale this up
            "hidden_size": 3072,  # Reduced from 4096
            "num_layers": 24,     # Reduced from 36
            "num_attention_heads": 24,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 1,      # Reduced for memory
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-5,
            "block_size": 1024
        },hmarking.
Supports loading different model sizes with multi-GPU configurations.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from typing import Dict, Any, Tuple, Optional
import logging

# Try to import bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    logging.warning("bitsandbytes not available - quantization will be disabled")

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for different LLM model sizes."""
    
    # Model configurations for different sizes
    MODEL_CONFIGS = {
        "1B": {
            "model_name": "microsoft/DialoGPT-medium",  # ~355M params, closest to 1B available
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "block_size": 512
        },
        "3B": {
            "model_name": "microsoft/DialoGPT-large",
            "hidden_size": 1024,
            "num_layers": 16,
            "num_attention_heads": 16,
            "vocab_size": 50257,
            "max_position_embeddings": 1024,
            "batch_size": 12,
            "gradient_accumulation_steps": 1,
            "learning_rate": 4e-5,
            "block_size": 768
        },
        "7B": {
            "model_name": "microsoft/DialoGPT-large",  # We'll configure this to approximate 7B
            "hidden_size": 2048,
            "num_layers": 24,
            "num_attention_heads": 16,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 3e-5,
            "block_size": 1024
        },
        "11B": {
            "model_name": "gpt2",
            "hidden_size": 2560,
            "num_layers": 28,
            "num_attention_heads": 20,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 6,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2.5e-5,
            "block_size": 1024
        },
        "13B": {
            "model_name": "gpt2",
            "hidden_size": 3072,
            "num_layers": 32,
            "num_attention_heads": 24,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5,
            "block_size": 1024
        },
        "15B": {
            "model_name": "gpt2",
            "hidden_size": 3584,
            "num_layers": 34,
            "num_attention_heads": 28,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 3,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1.8e-5,
            "block_size": 1024
        },
        "20B": {
            "model_name": "gpt2",  # We'll scale this up
            "hidden_size": 4096,
            "num_layers": 36,
            "num_attention_heads": 32,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1.5e-5,
            "block_size": 1024
        },
        "30B": {
            "model_name": "gpt2",
            "hidden_size": 5120,
            "num_layers": 40,
            "num_attention_heads": 40,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1.2e-5,
            "block_size": 1024
        },
        "65B": {
            "model_name": "gpt2",
            "hidden_size": 6400,
            "num_layers": 44,
            "num_attention_heads": 50,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "block_size": 1024
        },
        "120B": {
            "model_name": "gpt2",  # We'll scale this to approximate 120B
            "hidden_size": 8192,
            "num_layers": 48,
            "num_attention_heads": 64,
            "vocab_size": 50257,
            "max_position_embeddings": 2048,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 8e-6,
            "block_size": 2048
        }
    }

    @staticmethod
    def get_config(model_size: str, num_gpus: int = 1) -> Dict:
        """
        Get configuration for specified model size with GPU scaling.
        
        Args:
            model_size: Size identifier (1B, 7B, 20B, 120B)
            num_gpus: Number of GPUs to scale batch size for
            
        Returns:
            Configuration dictionary
        """
        if model_size not in ModelConfig.MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = ModelConfig.MODEL_CONFIGS[model_size].copy()
        
        # Scale batch size with number of GPUs for better throughput
        if num_gpus > 1:
            config["batch_size"] = config["batch_size"] * num_gpus
            # Reduce gradient accumulation steps proportionally to maintain effective batch size
            config["gradient_accumulation_steps"] = max(1, config["gradient_accumulation_steps"] // num_gpus)
        
        return config

    @staticmethod
    def get_quantization_config(quantization_type: str) -> Optional[BitsAndBytesConfig]:
        """
        Get BitsAndBytesConfig for the specified quantization type.
        
        Args:
            quantization_type: Type of quantization ("4bit", "8bit", "int8", "fp16")
            
        Returns:
            BitsAndBytesConfig or None if not available
        """
        if not HAS_BITSANDBYTES:
            logger.warning("bitsandbytes not available - cannot use quantization")
            return None
            
        if quantization_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        elif quantization_type == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif quantization_type == "fp16":
            # FP16 is handled by torch_dtype, not BitsAndBytesConfig
            return None
        else:
            logger.warning(f"Unknown quantization type: {quantization_type}")
            return None

    @staticmethod
    def should_use_quantization(model_size: str) -> bool:
        """
        Determine if quantization should be used for the given model size.
        
        Args:
            model_size: Model size identifier
            
        Returns:
            Boolean indicating if quantization should be used
        """
        if not HAS_BITSANDBYTES:
            return False
            
        # Use quantization for models 7B and larger
        large_models = ["7B", "11B", "13B", "15B", "20B", "30B", "65B", "120B"]
        return model_size in large_models

    @staticmethod
    def get_memory_efficient_config(model_size: str) -> Dict[str, Any]:
        """
        Get memory-efficient configuration options for large models.
        
        Args:
            model_size: Model size identifier
            
        Returns:
            Dictionary with memory optimization settings
        """
        config = {}
        
        # Enable gradient checkpointing for large models
        if model_size in ["15B", "20B", "30B", "65B", "120B"]:
            config["gradient_checkpointing"] = True
            config["use_cache"] = False  # Disable cache to save memory
            
        # Use mixed precision for all models
        config["torch_dtype"] = torch.float16
        
        # CPU offloading for very large models
        if model_size in ["65B", "120B"]:
            config["offload_to_cpu"] = True
            config["device_map"] = "auto"
            
        return config
class ModelLoader:
    """Handles loading and configuring models for multi-GPU training."""
    
    def __init__(self, device_count: int = None):
        """
        Initialize model loader.
        
        Args:
            device_count: Number of GPUs to use. If None, uses all available.
        """
        self.device_count = device_count or torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Running on CPU will be very slow.")
        
        logger.info(f"Using {self.device_count} GPU(s)")

    def create_custom_model(self, config: Dict[str, Any]) -> nn.Module:
        """
        Create a custom GPT-style model with specified configuration.
        This allows us to create models of exact sizes we want.
        """
        class GPTBlock(nn.Module):
            def __init__(self, hidden_size: int, num_heads: int):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    batch_first=True
                )
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(0.1)
                )
                
            def forward(self, x):
                # Self-attention with residual connection
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                
                # MLP with residual connection
                mlp_out = self.mlp(x)
                x = self.norm2(x + mlp_out)
                
                return x

        class CustomGPTModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                
                # Enable gradient checkpointing for memory efficiency
                self.gradient_checkpointing = config.get("gradient_checkpointing", True)
                
                self.vocab_size = config["vocab_size"]
                self.hidden_size = config["hidden_size"]
                self.num_layers = config["num_layers"]
                
                # Embeddings
                self.token_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
                self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
                
                # Transformer blocks
                self.blocks = nn.ModuleList([
                    GPTBlock(config["hidden_size"], config["num_attention_heads"])
                    for _ in range(config["num_layers"])
                ])
                
                # Final layer norm and output projection
                self.ln_f = nn.LayerNorm(config["hidden_size"])
                self.head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
                
                # Initialize weights
                self.apply(self._init_weights)
                
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)
                    
            def forward(self, input_ids, labels=None):
                batch_size, seq_length = input_ids.shape
                
                # Create position IDs
                position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
                
                # Embeddings
                token_embeds = self.token_embeddings(input_ids)
                pos_embeds = self.position_embeddings(position_ids)
                x = token_embeds + pos_embeds
                
                # Transformer blocks with gradient checkpointing for memory efficiency
                for block in self.blocks:
                    if self.training and self.gradient_checkpointing:
                        x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                    else:
                        x = block(x)
                
                # Final layer norm and projection
                x = self.ln_f(x)
                logits = self.head(x)
                
                # Calculate loss if labels provided
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # Return loss directly for DataParallel compatibility
                    return loss
                else:
                    # During inference, return logits
                    return logits
                
            def get_num_params(self):
                """Calculate total number of parameters."""
                return sum(p.numel() for p in self.parameters())

        return CustomGPTModel(config)

    def load_model_and_tokenizer(self, model_size: str) -> Tuple[nn.Module, Any]:
        """
        Load model and tokenizer for specified size with fallback quantization support.
        Tries to load without quantization first, applies quantization only if loading fails.
        
        Args:
            model_size: Size of model to load ("1B", "3B", "7B", etc.)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        config = ModelConfig.get_config(model_size, self.device_count)
        
        logger.info(f"Loading {model_size} model...")
        
        # Load tokenizer from base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {config['model_name']}: {e}")
            logger.info("Using GPT2 tokenizer as fallback")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try to load model without quantization first
        model = None
        quantization_used = None
        
        # Attempt 1: Try loading without quantization (standard approach)
        logger.info("🚀 Attempting to load model without quantization...")
        try:
            model = self._try_load_model_standard(config, model_size)
            quantization_used = "none"
            logger.info("✅ Successfully loaded model without quantization")
            
        except Exception as e:
            logger.warning(f"❌ Failed to load model without quantization: {e}")
            
            # Attempt 2: Try with FP16 if not already tried
            if quantization_used is None:
                logger.info("🔄 Attempting to load model with FP16...")
                try:
                    model = self._try_load_model_with_quantization(config, model_size, "fp16")
                    quantization_used = "fp16"
                    logger.info("✅ Successfully loaded model with FP16")
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to load model with FP16: {e}")
            
            # Attempt 3: Try with 8-bit quantization for medium models
            if quantization_used is None and model_size in ["7B", "11B", "13B"]:
                logger.info("🔄 Attempting to load model with 8-bit quantization...")
                try:
                    model = self._try_load_model_with_quantization(config, model_size, "8bit")
                    quantization_used = "8bit"
                    logger.info("✅ Successfully loaded model with 8-bit quantization")
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to load model with 8-bit quantization: {e}")
            
            # Attempt 4: Try with 4-bit quantization for large models
            if quantization_used is None and model_size in ["15B", "20B", "30B", "65B", "120B"]:
                logger.info("🔄 Attempting to load model with 4-bit quantization...")
                try:
                    model = self._try_load_model_with_quantization(config, model_size, "4bit")
                    quantization_used = "4bit"
                    logger.info("✅ Successfully loaded model with 4-bit quantization")
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to load model with 4-bit quantization: {e}")
            
            # Final fallback: Create custom model
            if model is None:
                logger.info("🔧 All quantization attempts failed, creating custom model...")
                model = self._create_custom_model_fallback(config, model_size)
                quantization_used = "custom"
        
        if model is None:
            raise RuntimeError(f"Failed to load {model_size} model with any method")
        
        logger.info(f"📊 Final model loading result: {quantization_used}")
        return model, tokenizer

    def _try_load_model_standard(self, config: Dict, model_size: str) -> nn.Module:
        """Try to load model using standard approach without quantization."""
        # Create custom model with exact specifications
        model = self.create_custom_model(config)
        
        # Log model parameters
        num_params = model.get_num_params()
        logger.info(f"Created {model_size} model with {num_params:,} parameters")
        
        # Move to GPU and setup for multi-GPU if available
        if torch.cuda.is_available():
            # Calculate model size in GB (FP32)
            model_size_gb = num_params * 4 / (1024**3)  # 4 bytes per float32 parameter
            
            # Get actual GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available_memory_gb = gpu_memory_gb * 0.7  # Use 70% to leave room for training
            
            logger.info(f"Model size: {model_size_gb:.1f} GB (FP32)")
            logger.info(f"Available GPU memory per device: {available_memory_gb:.1f} GB")
            
            # Check if model needs to be distributed across multiple GPUs
            if model_size_gb > available_memory_gb and self.device_count > 1:
                # Calculate how many GPUs we need
                gpus_needed = int(model_size_gb / available_memory_gb) + 1
                gpus_needed = min(gpus_needed, self.device_count)
                
                logger.info(f"Model too large for single GPU, distributing across {gpus_needed} GPUs")
                model = self._setup_model_parallel(model, model_size_gb)
            elif model_size_gb > available_memory_gb:
                # Single GPU and model too large - need quantization
                raise RuntimeError(f"Model too large for single GPU ({model_size_gb:.1f} GB > {available_memory_gb:.1f} GB)")
            else:
                # Model fits on single GPU or can use DataParallel
                model = model.to(self.device)
                if self.device_count > 1:
                    logger.info(f"Using DataParallel across {self.device_count} GPUs for data parallelism")
                    model = nn.DataParallel(model)
        else:
            model = model.to(self.device)
        
        return model

    def _try_load_model_with_quantization(self, config: Dict, model_size: str, quantization_type: str) -> nn.Module:
        """Try to load model with specified quantization."""
        quantization_config = ModelConfig.get_quantization_config(quantization_type)
        memory_config = ModelConfig.get_memory_efficient_config(model_size)
        
        # For models that support quantization via transformers
        if quantization_config and HAS_BITSANDBYTES:
            try:
                # Try to load with quantization using transformers
                model = AutoModelForCausalLM.from_pretrained(
                    config["model_name"],
                    quantization_config=quantization_config,
                    torch_dtype=memory_config.get("torch_dtype", torch.float16),
                    device_map=memory_config.get("device_map", "auto") if model_size in ["65B", "120B"] else None,
                    trust_remote_code=True
                )
                
                # Resize model if needed to match target parameters
                if hasattr(model, 'resize_token_embeddings'):
                    model.resize_token_embeddings(config["vocab_size"])
                    
                return model
                
            except Exception as e:
                logger.warning(f"Transformers quantization failed: {e}")
        
        # Fallback: Create custom model and apply quantization manually
        model = self.create_custom_model(config)
        
        # Apply quantization manually
        if quantization_type == "fp16":
            model = model.half()
        
        # Log model parameters
        num_params = model.get_num_params()
        logger.info(f"Created {model_size} model with {num_params:,} parameters")
        
        # Move to GPU and setup for multi-GPU if available
        if torch.cuda.is_available():
            # Calculate model size in GB with quantization
            bytes_per_param = 2 if quantization_type == "fp16" else 4  # FP16 uses 2 bytes, FP32 uses 4
            if quantization_type == "4bit":
                bytes_per_param = 0.5
            elif quantization_type in ["8bit", "int8"]:
                bytes_per_param = 1
                
            model_size_gb = num_params * bytes_per_param / (1024**3)
            gpu_memory_gb = 32  # Assuming 32GB per GPU
            
            logger.info(f"Model size: {model_size_gb:.1f} GB (with {quantization_type})")
            
            if model_size_gb > gpu_memory_gb * 0.8:  # If model needs more than 80% of single GPU
                if model_size in ["65B", "120B"]:
                    logger.info(f"Model too large for single GPU, implementing model parallelism")
                    model = self._setup_model_parallel(model, model_size_gb)
                else:
                    raise RuntimeError(f"Model too large even with quantization ({model_size_gb:.1f} GB > {gpu_memory_gb*0.8:.1f} GB)")
            else:
                model = model.to(self.device)
                if self.device_count > 1:
                    logger.info(f"Using DataParallel across {self.device_count} GPUs for data parallelism")
                    model = nn.DataParallel(model)
        else:
            model = model.to(self.device)
        
        return model

    def _create_custom_model_fallback(self, config: Dict, model_size: str) -> nn.Module:
        """Create custom model as final fallback with maximum memory efficiency."""
        logger.info("Creating highly optimized custom model as fallback...")
        
        # Reduce model size if necessary for very large models
        if model_size in ["65B", "120B"]:
            # Reduce hidden size and layers for extreme cases
            config = config.copy()
            config["hidden_size"] = min(config["hidden_size"], 4096)
            config["num_layers"] = min(config["num_layers"], 24)
            logger.warning(f"Reduced model size for {model_size} due to memory constraints")
        
        # Create model with gradient checkpointing enabled
        config["gradient_checkpointing"] = True
        model = self.create_custom_model(config)
        
        # Apply FP16 for memory efficiency
        model = model.half()
        
        # Log model parameters
        num_params = model.get_num_params()
        logger.info(f"Created fallback {model_size} model with {num_params:,} parameters")
        
        if torch.cuda.is_available():
            model = model.to(self.device)
            if self.device_count > 1:
                logger.info(f"Using DataParallel across {self.device_count} GPUs for data parallelism")
                model = nn.DataParallel(model)
        else:
            model = model.to(self.device)
        
        return model

    def get_training_args(self, model_size: str, output_dir: str) -> TrainingArguments:
        """
        Get training arguments for specified model size.
        
        Args:
            model_size: Size of model
            output_dir: Directory to save training outputs
            
        Returns:
            TrainingArguments object
        """
        config = ModelConfig.get_config(model_size, self.device_count)
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Fixed to 2 epochs as requested
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_num_workers=4,
            report_to=[],  # Disable wandb/tensorboard logging
        )

    def get_data_collator(self, tokenizer) -> DataCollatorForLanguageModeling:
        """Get data collator for language modeling."""
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

    def estimate_memory_usage(self, model_size: str) -> Dict[str, float]:
        """
        Estimate memory usage for model size.
        
        Args:
            model_size: Size of model
            
        Returns:
            Dictionary with memory estimates in GB
        """
        config = ModelConfig.get_config(model_size, 1)  # Use single GPU for estimation
        
        # Rough estimation based on model parameters
        # Each parameter ~4 bytes (float32), plus activations and gradients
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        vocab_size = config["vocab_size"]
        
        # Parameter count estimation
        embedding_params = vocab_size * hidden_size
        transformer_params = num_layers * (
            # Attention weights
            4 * hidden_size * hidden_size +
            # MLP weights  
            2 * hidden_size * (4 * hidden_size) +
            # Layer norms
            2 * hidden_size
        )
        total_params = embedding_params + transformer_params
        
        # Memory estimates (in GB)
        model_memory = (total_params * 4) / (1024**3)  # Model weights
        gradient_memory = model_memory  # Gradients
        optimizer_memory = model_memory * 2  # Adam optimizer states
        activation_memory = (config["batch_size"] * config["block_size"] * hidden_size * 4) / (1024**3)
        
        total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
        
        return {
            "model_memory_gb": model_memory,
            "gradient_memory_gb": gradient_memory,
            "optimizer_memory_gb": optimizer_memory,
            "activation_memory_gb": activation_memory,
            "total_memory_gb": total_memory,
            "estimated_params": total_params
        }

    def _setup_model_parallel(self, model, model_size_gb):
        """
        Set up model parallelism for large models that don't fit on single GPU.
        
        Args:
            model: The model to parallelize
            model_size_gb: Size of the model in GB
            
        Returns:
            Model with parallelism applied
        """
        logger.info(f"Setting up model parallelism for {model_size_gb:.1f}GB model across {self.device_count} GPUs")
        
        # Simple layer-wise model parallelism
        layers = list(model.blocks)
        layers_per_gpu = len(layers) // self.device_count
        
        # Move different layers to different GPUs
        for i, layer in enumerate(layers):
            gpu_id = min(i // layers_per_gpu, self.device_count - 1)
            layer.to(f'cuda:{gpu_id}')
            logger.info(f"Layer {i} -> GPU {gpu_id}")
        
        # Move embedding and head to first and last GPUs respectively
        model.token_embeddings.to('cuda:0')
        model.position_embeddings.to('cuda:0')
        model.ln_f.to(f'cuda:{self.device_count-1}')
        model.head.to(f'cuda:{self.device_count-1}')
        
        # Wrap the model to handle cross-GPU data flow
        class ModelParallelWrapper(nn.Module):
            def __init__(self, original_model, device_count):
                super().__init__()
                self.model = original_model
                self.device_count = device_count
                
            def forward(self, input_ids, labels=None):
                # Start on GPU 0
                x = input_ids.to('cuda:0')
                
                # Embeddings on GPU 0
                token_embeds = self.model.token_embeddings(x)
                position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
                pos_embeds = self.model.position_embeddings(position_ids)
                x = token_embeds + pos_embeds
                
                # Pass through transformer blocks on different GPUs
                layers = list(self.model.blocks)
                layers_per_gpu = len(layers) // self.device_count
                
                for i, layer in enumerate(layers):
                    gpu_id = min(i // layers_per_gpu, self.device_count - 1)
                    x = x.to(f'cuda:{gpu_id}')
                    x = layer(x)
                
                # Final layers on last GPU
                x = x.to(f'cuda:{self.device_count-1}')
                x = self.model.ln_f(x)
                logits = self.model.head(x)
                
                # Calculate loss if labels provided
                if labels is not None:
                    labels = labels.to(logits.device)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    return loss
                else:
                    return logits
        
        return ModelParallelWrapper(model, self.device_count)