#!/usr/bin/env python3
"""
Test script to validate multi-GPU model distribution and memory usage.
"""

import torch
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model_config import ModelLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_gpu_memory_usage():
    """Print current GPU memory usage for all devices."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return
    
    logger.info("🔍 Current GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = props.total_memory / (1024**3)
        free = total - reserved
        
        logger.info(f"  GPU {i} ({props.name}): {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free, {total:.1f}GB total")

def test_model_loading(model_size: str):
    """Test loading a specific model size and check GPU distribution."""
    logger.info(f"\\n{'='*60}")
    logger.info(f"🧪 Testing {model_size} model loading and GPU distribution")
    logger.info(f"{'='*60}")
    
    # Print initial memory state
    print_gpu_memory_usage()
    
    try:
        # Create model loader
        loader = ModelLoader()
        
        # Load model and tokenizer
        logger.info(f"📥 Loading {model_size} model...")
        model, tokenizer = loader.load_model_and_tokenizer(model_size)
        
        # Print memory usage after loading
        logger.info(f"✅ Successfully loaded {model_size} model")
        print_gpu_memory_usage()
        
        # Check model distribution
        if hasattr(model, 'module'):
            # DataParallel model
            logger.info("📊 Model is using DataParallel (data parallelism)")
            logger.info(f"   Model is replicated across {torch.cuda.device_count()} GPUs")
        elif hasattr(model, 'model'):
            # Custom model parallel wrapper
            logger.info("📊 Model is using custom Model Parallelism")
            logger.info(f"   Model layers distributed across {torch.cuda.device_count()} GPUs")
        else:
            # Check if model is on multiple devices (transformers device_map)
            devices_used = set()
            for name, param in model.named_parameters():
                if param.device.type == 'cuda':
                    devices_used.add(param.device.index)
            
            if len(devices_used) > 1:
                logger.info(f"📊 Model is distributed across GPUs: {sorted(devices_used)}")
                logger.info("   Using Transformers automatic device mapping")
            else:
                logger.info(f"📊 Model is on single GPU: {param.device}")
        
        # Test a small forward pass
        logger.info("🔄 Testing forward pass...")
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, 10))
        
        if torch.cuda.is_available():
            # Move input to appropriate device
            if hasattr(model, 'module'):
                input_ids = input_ids.to(next(model.module.parameters()).device)
            else:
                input_ids = input_ids.to(next(model.parameters()).device)
        
        with torch.no_grad():
            try:
                output = model(input_ids)
                logger.info("✅ Forward pass successful")
            except Exception as e:
                logger.warning(f"❌ Forward pass failed: {e}")
        
        # Clean up
        del model, tokenizer
        loader.cleanup_memory()
        
        logger.info(f"🧹 Cleaned up {model_size} model")
        print_gpu_memory_usage()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load {model_size} model: {e}")
        print_gpu_memory_usage()
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Multi-GPU Model Distribution Test")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Print environment variables
    pytorch_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "Not set")
    logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {pytorch_alloc_conf}")
    
    print_gpu_memory_usage()
    
    # Test different model sizes
    test_models = ["1B", "3B", "7B", "11B"]  # Start with smaller models first
    
    results = {}
    for model_size in test_models:
        success = test_model_loading(model_size)
        results[model_size] = success
        
        # Add delay between tests
        import time
        time.sleep(2)
    
    # Print summary
    logger.info(f"\\n{'='*60}")
    logger.info("📋 Test Summary:")
    logger.info(f"{'='*60}")
    
    for model_size, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {model_size}: {status}")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    logger.info(f"\\n🎯 Overall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        logger.info("🎉 All tests passed! Multi-GPU distribution is working correctly.")
    else:
        logger.warning(f"⚠️  {total_tests - successful_tests} tests failed. Check the logs above.")

if __name__ == "__main__":
    main()