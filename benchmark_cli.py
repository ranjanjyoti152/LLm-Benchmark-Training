#!/usr/bin/env python3
"""
Command-line interface for the LLM Multi-GPU Benchmarking Tool.
Provides comprehensive options for customizing benchmark runs.
"""

import argparse
import sys
import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.benchmark import BenchmarkOrchestrator
from src.monitoring import SystemMonitor


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, console: bool = True):
    """Setup logging configuration."""
    handlers = []
    
    if console:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_models(model_sizes: List[str]) -> List[str]:
    """Validate model sizes."""
    valid_sizes = ["1B", "3B", "7B", "11B", "13B", "15B", "20B", "30B", "65B", "120B"]
    validated = []
    
    for size in model_sizes:
        if size in valid_sizes:
            validated.append(size)
        else:
            print(f"Warning: Invalid model size '{size}'. Valid sizes: {valid_sizes}")
    
    return validated


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and provide recommendations."""
    monitor = SystemMonitor()
    
    requirements = {
        "cuda_available": False,
        "gpu_count": 0,
        "total_memory_gb": 0,
        "recommendations": []
    }
    
    try:
        import torch
        requirements["cuda_available"] = torch.cuda.is_available()
        requirements["gpu_count"] = torch.cuda.device_count()
        
        if requirements["cuda_available"]:
            gpu_metrics = monitor.get_gpu_metrics()
            total_gpu_memory = sum(
                gpu_info.get("memory_total", 0) 
                for gpu_info in gpu_metrics.values()
            ) / 1024  # Convert to GB
            requirements["total_gpu_memory_gb"] = total_gpu_memory
        
        # Memory requirements
        memory_info = monitor.get_memory_metrics()
        requirements["total_memory_gb"] = memory_info.get("total", 0) / (1024**3)
        
        # Generate recommendations
        if not requirements["cuda_available"]:
            requirements["recommendations"].append("⚠️  CUDA not available. Benchmarks will run very slowly on CPU.")
        
        if requirements["gpu_count"] == 0:
            requirements["recommendations"].append("⚠️  No GPUs detected. Multi-GPU benchmarking not possible.")
        elif requirements["gpu_count"] == 1:
            requirements["recommendations"].append("ℹ️  Single GPU detected. Multi-GPU features will be disabled.")
        else:
            requirements["recommendations"].append(f"✅ {requirements['gpu_count']} GPUs detected. Multi-GPU benchmarking available.")
        
        if requirements.get("total_gpu_memory_gb", 0) < 8:
            requirements["recommendations"].append("⚠️  Low GPU memory. Large models (20B, 120B) may fail.")
        
        if requirements["total_memory_gb"] < 16:
            requirements["recommendations"].append("⚠️  Low system memory. Consider reducing batch sizes.")
            
    except Exception as e:
        requirements["error"] = str(e)
        requirements["recommendations"].append(f"❌ Error checking system: {e}")
    
    return requirements


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Multi-GPU Benchmarking Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark with default settings
  python benchmark_cli.py run

  # Benchmark specific models with custom settings
  python benchmark_cli.py run --models 1B 7B --samples 2000 --output ./my_results

  # Check system requirements
  python benchmark_cli.py check-system

  # Generate sample configuration file
  python benchmark_cli.py generate-config --output my_config.yaml

  # Run with custom configuration
  python benchmark_cli.py run --config my_config.yaml

  # Quick benchmark (fewer samples, faster)
  python benchmark_cli.py run --quick

  # Verbose output with debug logging
  python benchmark_cli.py run --verbose --log-level DEBUG
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run benchmark command
    run_parser = subparsers.add_parser('run', help='Run benchmark suite')
    run_parser.add_argument('--config', '-c', type=str, default='configs/config.yaml',
                           help='Configuration file path (default: configs/config.yaml)')
    run_parser.add_argument('--models', '-m', nargs='+', default=None,
                           help='Model sizes to benchmark (1B 7B 20B 120B)')
    run_parser.add_argument('--samples', '-s', type=int, default=None,
                           help='Number of training samples per model')
    run_parser.add_argument('--output', '-o', type=str, default=None,
                           help='Output directory for results')
    run_parser.add_argument('--epochs', '-e', type=int, default=2,
                           help='Number of training epochs (default: 2)')
    run_parser.add_argument('--quick', action='store_true',
                           help='Quick benchmark with fewer samples (100)')
    run_parser.add_argument('--skip-on-error', action='store_true', default=True,
                           help='Skip failed models and continue (default: True)')
    run_parser.add_argument('--no-excel', action='store_true',
                           help='Skip Excel export (save JSON only)')
    run_parser.add_argument('--individual-files', action='store_true', default=True,
                           help='Save individual Excel files per model')
    run_parser.add_argument('--no-ollama', action='store_true',
                           help='Disable Ollama AI analysis')
    run_parser.add_argument('--ollama-url', default='http://92.168.100.67:11434',
                           help='Ollama server URL (default: http://92.168.100.67:11434)')
    run_parser.add_argument('--ollama-model', default='gpt-oss-20b',
                           help='Ollama model name (default: gpt-oss-20b)')
    
    # System check command
    check_parser = subparsers.add_parser('check-system', help='Check system requirements')
    check_parser.add_argument('--detailed', action='store_true',
                             help='Show detailed system information')
    
    # Generate config command
    config_parser = subparsers.add_parser('generate-config', help='Generate sample configuration file')
    config_parser.add_argument('--output', '-o', type=str, default='config_sample.yaml',
                              help='Output path for configuration file')
    config_parser.add_argument('--overwrite', action='store_true',
                              help='Overwrite existing file')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available model configurations')
    
    # Estimate command
    estimate_parser = subparsers.add_parser('estimate', help='Estimate benchmark requirements')
    estimate_parser.add_argument('--models', '-m', nargs='+', default=['1B', '7B', '20B', '120B'],
                                help='Model sizes to estimate')
    estimate_parser.add_argument('--samples', '-s', type=int, default=1000,
                                help='Number of training samples')
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    parser.add_argument('--no-console-log', action='store_true',
                       help='Disable console logging')
    
    return parser


def cmd_run(args) -> int:
    """Run benchmark command."""
    print("🚀 Starting LLM Multi-GPU Benchmark Suite")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Override config with command line arguments
    if args.models:
        args.models = validate_models(args.models)
        if not args.models:
            print("❌ No valid model sizes specified")
            return 1
    
    model_sizes = args.models or config.get('models', {}).get('sizes', ['1B', '7B', '20B', '120B'])
    num_samples = args.samples or (100 if args.quick else config.get('training', {}).get('num_samples', 1000))
    output_dir = args.output or config.get('output', {}).get('directory', './benchmark_results')
    
    print(f"📋 Configuration:")
    print(f"   Models: {model_sizes}")
    print(f"   Samples per model: {num_samples}")
    print(f"   Output directory: {output_dir}")
    print(f"   Epochs: {args.epochs}")
    print()
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    requirements = check_system_requirements()
    
    for rec in requirements.get("recommendations", []):
        print(f"   {rec}")
    print()
    
    if not requirements.get("cuda_available", False):
        response = input("⚠️  CUDA not available. Continue with CPU? (y/N): ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            return 0
    
    # Create and run orchestrator
    try:
        orchestrator = BenchmarkOrchestrator(
            output_dir=output_dir,
            model_sizes=model_sizes,
            num_samples=num_samples,
            enable_ollama=not args.no_ollama,
            ollama_url=args.ollama_url
        )
        
        print("🎯 Starting benchmark execution...")
        results = orchestrator.run_benchmark_suite()
        
        if not args.no_excel:
            print("📊 Exporting results to Excel...")
            excel_files = orchestrator.export_all_results()
            print(f"   Created {len(excel_files)} Excel files")
        
        print()
        print("✅ Benchmark completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📈 Successfully benchmarked: {results.get('successful_benchmarks', 0)}/{results.get('models_attempted', 0)} models")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_check_system(args) -> int:
    """Check system requirements command."""
    print("🔍 System Requirements Check")
    print("=" * 40)
    
    requirements = check_system_requirements()
    
    print(f"🖥️  CUDA Available: {'✅' if requirements.get('cuda_available') else '❌'}")
    print(f"🎮 GPU Count: {requirements.get('gpu_count', 0)}")
    print(f"💾 Total System Memory: {requirements.get('total_memory_gb', 0):.1f} GB")
    
    if requirements.get('total_gpu_memory_gb'):
        print(f"🎮 Total GPU Memory: {requirements.get('total_gpu_memory_gb', 0):.1f} GB")
    
    print()
    print("📋 Recommendations:")
    for rec in requirements.get("recommendations", []):
        print(f"   {rec}")
    
    if args.detailed:
        print()
        print("🔧 Detailed System Information:")
        
        try:
            monitor = SystemMonitor()
            
            # CPU info
            cpu_info = monitor.get_cpu_info()
            print(f"   CPU: {cpu_info.get('brand', 'Unknown')}")
            print(f"   CPU Cores: {cpu_info.get('count', 'Unknown')}")
            
            # GPU info
            if requirements.get('cuda_available'):
                import torch
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            
        except Exception as e:
            print(f"   Error getting detailed info: {e}")
    
    return 0


def cmd_generate_config(args) -> int:
    """Generate configuration file command."""
    output_path = Path(args.output)
    
    if output_path.exists() and not args.overwrite:
        print(f"❌ File {output_path} already exists. Use --overwrite to replace it.")
        return 1
    
    # Read the default config
    default_config_path = Path(__file__).parent / "configs" / "config.yaml"
    
    try:
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config_content = f.read()
        else:
            # Fallback default config
            config_content = """# Configuration file for LLM benchmarking tool

# Model configurations
models:
  sizes: ["1B", "7B", "20B", "120B"]
  
# Training parameters
training:
  epochs: 2
  num_samples: 1000
  
# Output settings
output:
  directory: "./benchmark_results"
  save_individual_results: true

# System monitoring
monitoring:
  interval: 1.0
  log_interval: 10
"""
        
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Configuration file generated: {output_path}")
        print("📝 Edit this file to customize your benchmark settings.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating config file: {e}")
        return 1


def cmd_list_models(args) -> int:
    """List available models command."""
    print("📋 Available Model Configurations")
    print("=" * 40)
    
    from src.model_config import ModelConfig
    
    for size, config in ModelConfig.MODEL_CONFIGS.items():
        print(f"🤖 {size} Model:")
        print(f"   Hidden Size: {config['hidden_size']}")
        print(f"   Layers: {config['num_layers']}")
        print(f"   Attention Heads: {config['num_attention_heads']}")
        print(f"   Default Batch Size: {config['batch_size']}")
        print()
    
    return 0


def cmd_estimate(args) -> int:
    """Estimate benchmark requirements command."""
    print("📊 Benchmark Requirements Estimation")
    print("=" * 45)
    
    from src.model_config import ModelLoader
    
    model_loader = ModelLoader()
    total_time = 0
    max_memory = 0
    
    print(f"{'Model':<8} {'Est. Time':<12} {'Memory (GB)':<12} {'Parameters'}")
    print("-" * 50)
    
    for model_size in args.models:
        try:
            estimates = model_loader.estimate_memory_usage(model_size)
            
            # Rough time estimate (very approximate)
            time_per_sample = {
                "1B": 0.1, "7B": 0.5, "20B": 1.5, "120B": 5.0
            }.get(model_size, 1.0)
            
            estimated_minutes = (args.samples * time_per_sample * 2) / 60  # 2 epochs
            memory_gb = estimates["total_memory_gb"]
            params = estimates["estimated_params"]
            
            total_time += estimated_minutes
            max_memory = max(max_memory, memory_gb)
            
            print(f"{model_size:<8} {estimated_minutes:<12.1f} {memory_gb:<12.1f} {params:>12,}")
            
        except Exception as e:
            print(f"{model_size:<8} {'ERROR':<12} {'ERROR':<12} {str(e)[:20]}")
    
    print("-" * 50)
    print(f"{'Total':<8} {total_time:<12.1f} {max_memory:<12.1f} {'(peak memory)'}")
    print()
    print(f"📝 Notes:")
    print(f"   • Estimates are approximate and may vary significantly")
    print(f"   • Actual performance depends on hardware and system load")
    print(f"   • Large models may require significant GPU memory")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        console=not args.no_console_log
    )
    
    # Route to appropriate command
    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'check-system':
        return cmd_check_system(args)
    elif args.command == 'generate-config':
        return cmd_generate_config(args)
    elif args.command == 'list-models':
        return cmd_list_models(args)
    elif args.command == 'estimate':
        return cmd_estimate(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())