#!/usr/bin/env python3
"""
Test script for fallback quantization strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_config import ModelLoader
import torch

def test_fallback_quantization():
    """Test the fallback quantization strategy."""
    print("🧪 Testing Fallback Quantization Strategy")
    print("=" * 50)
    
    # Test with different model sizes
    test_sizes = ["1B", "3B", "7B"]  # Start with smaller models for testing
    
    loader = ModelLoader()
    
    for size in test_sizes:
        print(f"\n🔍 Testing {size} model:")
        print("-" * 30)
        
        try:
            model, tokenizer = loader.load_model_and_tokenizer(size)
            
            # Check model properties
            if hasattr(model, 'get_num_params'):
                num_params = model.get_num_params()
            elif hasattr(model, 'module') and hasattr(model.module, 'get_num_params'):
                num_params = model.module.get_num_params()
            else:
                num_params = sum(p.numel() for p in model.parameters())
            
            print(f"✅ Successfully loaded {size} model")
            print(f"   Parameters: {num_params:,}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Device: {next(model.parameters()).device}")
            print(f"   Dtype: {next(model.parameters()).dtype}")
            
            # Clean up GPU memory
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed to load {size} model: {e}")
            
    print("\n" + "=" * 50)
    print("✅ Fallback quantization test completed!")

if __name__ == "__main__":
    test_fallback_quantization()