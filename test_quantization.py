#!/usr/bin/env python3
"""
Test script for quantization and new model configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_config import ModelConfig
import torch

def test_model_configs():
    """Test all model configurations."""
    print("🔍 Testing Model Configurations")
    print("=" * 50)
    
    all_sizes = ["1B", "3B", "7B", "11B", "13B", "15B", "20B", "30B", "65B", "120B"]
    
    for size in all_sizes:
        try:
            config = ModelConfig.get_config(size)
            quantization = config.get("quantization", "none")
            
            # Calculate approximate parameters
            hidden_size = config["hidden_size"]
            num_layers = config["num_layers"]
            vocab_size = config["vocab_size"]
            
            # Rough calculation: embedding + layers + output
            # Each transformer layer has roughly 12 * hidden_size^2 parameters
            approx_params = (
                vocab_size * hidden_size +  # Input embedding
                num_layers * 12 * hidden_size * hidden_size +  # Transformer layers
                hidden_size * vocab_size  # Output layer
            )
            
            approx_params_b = approx_params / 1e9
            
            print(f"📊 {size:>4} Model:")
            print(f"   • Hidden Size: {hidden_size:,}")
            print(f"   • Layers: {num_layers}")
            print(f"   • Approx Params: {approx_params_b:.1f}B")
            print(f"   • Quantization: {quantization}")
            print(f"   • Batch Size: {config['batch_size']}")
            print(f"   • Learning Rate: {config['learning_rate']}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing {size}: {e}")
            print()

def test_quantization_configs():
    """Test quantization configurations."""
    print("🎛️  Testing Quantization Configurations")
    print("=" * 50)
    
    quantization_types = ["fp16", "8bit", "4bit", "int8"]
    
    for quant_type in quantization_types:
        try:
            config = ModelConfig.get_quantization_config(quant_type)
            
            if config is None:
                print(f"📋 {quant_type:>6}: Standard PyTorch (no BitsAndBytesConfig needed)")
            else:
                print(f"⚙️  {quant_type:>6}: BitsAndBytesConfig configured")
                if hasattr(config, 'load_in_4bit'):
                    print(f"            4-bit loading: {config.load_in_4bit}")
                if hasattr(config, 'load_in_8bit'):
                    print(f"            8-bit loading: {config.load_in_8bit}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing {quant_type}: {e}")
            print()

def test_memory_efficiency():
    """Test memory efficiency configurations."""
    print("💾 Testing Memory Efficiency Configurations")
    print("=" * 50)
    
    large_models = ["15B", "20B", "30B", "65B", "120B"]
    
    for size in large_models:
        try:
            memory_config = ModelConfig.get_memory_efficient_config(size)
            
            print(f"🧠 {size:>4} Memory Optimizations:")
            for key, value in memory_config.items():
                print(f"   • {key}: {value}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing {size}: {e}")
            print()

def main():
    """Run all tests."""
    print("🚀 Quantization and Model Configuration Test Suite")
    print("=" * 60)
    print()
    
    # Check if CUDA is available
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
    print()
    
    # Check bitsandbytes availability
    try:
        import bitsandbytes as bnb
        print("✅ bitsandbytes available - quantization enabled")
    except ImportError:
        print("⚠️  bitsandbytes not available - quantization disabled")
    print()
    
    # Run tests
    test_model_configs()
    test_quantization_configs()
    test_memory_efficiency()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()