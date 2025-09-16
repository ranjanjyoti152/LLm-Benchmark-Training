#!/usr/bin/env python3
"""
Test script to validate device mapping and tensor placement fixes.
"""

import sys
import os
import torch
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
import model_config
import training
import dataset

from model_config import ModelLoader
from training import BenchmarkTrainer
from dataset import create_synthetic_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_device_mapping():
    """Test that models are properly distributed and tensors are on correct devices."""
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping device mapping test")
        return
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPUs")
    
    if gpu_count < 2:
        logger.warning("Need at least 2 GPUs for meaningful test")
        return
    
    # Test with a model large enough to require distribution
    model_size = "7B"  # Should be distributed across GPUs
    
    logger.info(f"Testing device mapping for {model_size} model")
    
    # Load model with device mapping
    model_loader = ModelLoader(num_gpus=gpu_count)
    model, tokenizer, quantization = model_loader.load_model(model_size)
    
    if model is None:
        logger.error("Failed to load model")
        return False
    
    # Check if model has device map
    if hasattr(model, 'hf_device_map'):
        logger.info("✅ Model has device_map")
        logger.info(f"Device map: {model.hf_device_map}")
        
        # Check device distribution
        devices_used = set()
        for name, device in model.hf_device_map.items():
            devices_used.add(device)
            logger.info(f"  {name} -> {device}")
        
        logger.info(f"Model distributed across {len(devices_used)} devices: {sorted(devices_used)}")
        
        if len(devices_used) > 1:
            logger.info("✅ Model is properly distributed across multiple GPUs")
        else:
            logger.warning("⚠️ Model is only on one GPU despite multiple available")
    else:
        logger.warning("⚠️ Model does not have device_map")
    
    # Test trainer device detection
    trainer = BenchmarkTrainer(
        model=model,
        tokenizer=tokenizer,
        model_size=model_size,
        output_dir="./test_results"
    )
    
    input_device = trainer._get_model_input_device()
    logger.info(f"Trainer detected input device: {input_device}")
    
    # Test tensor placement
    logger.info("Testing tensor placement...")
    
    # Create a small dataset
    dataset = create_synthetic_dataset(num_samples=2, max_length=128, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Get a batch and test device placement
    try:
        batch = next(iter(dataloader))
        logger.info(f"Original batch device: {batch['input_ids'].device}")
        
        # Test device placement
        model_input_device = trainer._get_model_input_device()
        input_ids = batch["input_ids"].to(model_input_device)
        logger.info(f"Moved to model input device: {input_ids.device}")
        
        # Test a forward pass
        logger.info("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
            logger.info("✅ Forward pass successful!")
            
            if hasattr(outputs, 'logits'):
                logger.info(f"Output logits shape: {outputs.logits.shape}")
                logger.info(f"Output logits device: {outputs.logits.device}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        return False
    
    finally:
        # Cleanup
        del model
        torch.cuda.empty_cache()

def test_parameter_counting():
    """Test that parameter counting works correctly for distributed models."""
    
    logger.info("Testing parameter counting...")
    
    # Test with different model sizes
    for model_size in ["1B", "7B"]:
        logger.info(f"\nTesting {model_size} model:")
        
        model_loader = ModelLoader(num_gpus=torch.cuda.device_count())
        model, tokenizer, quantization = model_loader.load_model(model_size)
        
        if model is None:
            logger.warning(f"Could not load {model_size} model")
            continue
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_count_b = param_count / 1e9
        
        logger.info(f"Parameter count: {param_count:,} ({param_count_b:.2f}B)")
        
        # Expected ranges (rough estimates)
        expected_ranges = {
            "1B": (0.5, 1.5),
            "7B": (6.0, 8.0),
        }
        
        if model_size in expected_ranges:
            min_expected, max_expected = expected_ranges[model_size]
            if min_expected <= param_count_b <= max_expected:
                logger.info(f"✅ Parameter count in expected range: {min_expected}B-{max_expected}B")
            else:
                logger.warning(f"⚠️ Parameter count outside expected range: {min_expected}B-{max_expected}B")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("🧪 Testing device mapping and tensor placement fixes...")
    
    try:
        # Test device mapping
        success = test_device_mapping()
        
        # Test parameter counting
        test_parameter_counting()
        
        if success:
            logger.info("✅ All tests passed!")
        else:
            logger.error("❌ Some tests failed")
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise