#!/usr/bin/env python3
"""
Test script for enhanced memory management and multi-GPU utilization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.model_config import ModelLoader
from src.training import BenchmarkTrainer

def test_memory_management():
    """Test the enhanced memory management features."""
    print("🧪 Testing Enhanced Memory Management")
    print("=" * 50)
    
    # Check GPU memory before starting
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {total_memory:.1f} GB total memory")
        print()
    
    # Test model loading with memory awareness
    test_sizes = ["7B", "11B"]  # Medium-large models that might cause OOM
    
    loader = ModelLoader()
    
    for size in test_sizes:
        print(f"🔍 Testing {size} model memory management:")
        print("-" * 40)
        
        try:
            # Test model loading
            model, tokenizer = loader.load_model_and_tokenizer(size)
            
            # Check model memory usage
            if hasattr(model, 'get_num_params'):
                num_params = model.get_num_params()
            elif hasattr(model, 'module') and hasattr(model.module, 'get_num_params'):
                num_params = model.module.get_num_params()
            else:
                num_params = sum(p.numel() for p in model.parameters())
            
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {num_params:,} ({num_params/1e9:.1f}B)")
            print(f"   Model type: {type(model).__name__}")
            
            # Test trainer memory management
            trainer = BenchmarkTrainer(
                model=model,
                tokenizer=tokenizer,
                model_size=size,
                output_dir=f"./test_memory_{size}",
                monitoring_interval=1.0,
                log_interval=5
            )
            
            # Test optimal batch size calculation
            optimal_batch_size = trainer._get_optimal_batch_size()
            print(f"   Optimal batch size: {optimal_batch_size}")
            
            # Check GPU memory after loading
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"   GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved / {total:.1f}GB total")
            
            # Clean up
            del model, tokenizer, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ {size} memory test completed successfully\n")
            
        except Exception as e:
            print(f"❌ {size} memory test failed: {e}")
            
            # Clean up on failure
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'trainer' in locals():
                del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print()

def test_gpu_utilization():
    """Test multi-GPU utilization."""
    print("🌐 Testing Multi-GPU Utilization")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping multi-GPU test")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / (1024**3):.1f} GB")
    
    print("\n✅ Multi-GPU detection completed")

def main():
    """Run all memory management tests."""
    print("🚀 Enhanced Memory Management Test Suite")
    print("=" * 60)
    print()
    
    test_gpu_utilization()
    print()
    test_memory_management()
    
    print("=" * 60)
    print("✅ All memory management tests completed!")

if __name__ == "__main__":
    main()