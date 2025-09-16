# LLM Multi-GPU Benchmarking Tool 🚀

A comprehensive benchmarking tool for testing Large Language Model (LLM) performance on multi-GPU systems. This tool trains models of different sizes (1B, 7B, 20B, 120B parameters) for exactly 2 epochs while collecting detailed performance metrics.

## ✨ Features

- **Multi-Model Support**: Benchmark 1B, 7B, 20B, and 120B parameter models
- **Real-time Monitoring**: Track GPU utilization, CPU usage, RAM consumption, and tokens/second
- **Multi-GPU Support**: Automatic detection and utilization of multiple GPUs
- **Comprehensive Metrics**: Collect detailed performance data during training
- **Excel Export**: Generate structured Excel files with charts and analysis
- **Configurable**: Extensive configuration options via YAML files
- **CLI Interface**: User-friendly command-line interface with multiple options

## 🎯 Key Metrics Collected

- **Tokens per second** - Training throughput
- **GPU utilization** - Per-GPU usage percentages
- **GPU memory usage** - Memory consumption per GPU
- **CPU utilization** - System CPU usage
- **RAM utilization** - System memory usage
- **Training loss** - Model training progress
- **System information** - Hardware specifications and environment

## 🚀 Quick Start

### Option 1: Interactive Script
```bash
./run_benchmark.sh
```

### Option 2: Direct CLI
```bash
# Full benchmark with all models
python benchmark_cli.py run

# Quick benchmark (100 samples)
python benchmark_cli.py run --quick

# Custom benchmark
python benchmark_cli.py run --models 1B 7B --samples 2000 --output ./my_results
```

### Option 3: Python Import
```python
from src.benchmark import BenchmarkOrchestrator

orchestrator = BenchmarkOrchestrator(
    model_sizes=["1B", "7B"],
    num_samples=1000,
    output_dir="./results"
)

results = orchestrator.run_benchmark_suite()
```

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU(s) (recommended)
- 16GB+ RAM (32GB+ recommended for large models)
- 50GB+ free disk space

### Memory Requirements by Model Size
- **1B model**: ~4GB GPU memory
- **7B model**: ~14GB GPU memory  
- **20B model**: ~40GB GPU memory
- **120B model**: ~240GB GPU memory

**Note**: Large models may require multiple GPUs or model parallelism.

## 🛠️ Installation

### 1. Clone or Download
```bash
git clone <repository_url>
cd Benchmark
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python benchmark_cli.py check-system
```

## 🎮 Usage Guide

### Command Line Interface

#### Check System Requirements
```bash
python benchmark_cli.py check-system --detailed
```

#### Run Benchmarks
```bash
# Full benchmark (all models, 1000 samples each)
python benchmark_cli.py run

# Quick benchmark (100 samples)
python benchmark_cli.py run --quick

# Specific models only
python benchmark_cli.py run --models 1B 7B

# Custom configuration
python benchmark_cli.py run --config my_config.yaml

# More samples for detailed analysis
python benchmark_cli.py run --samples 5000
```

#### Generate Configuration
```bash
python benchmark_cli.py generate-config --output my_config.yaml
```

#### Estimate Requirements
```bash
python benchmark_cli.py estimate --models 1B 7B --samples 2000
```

#### List Available Models
```bash
python benchmark_cli.py list-models
```

### Configuration Files

Create custom configurations in YAML format:

```yaml
# config.yaml
models:
  sizes: ["1B", "7B"]

training:
  epochs: 2
  num_samples: 1000
  batch_sizes:
    "1B": 8
    "7B": 4

output:
  directory: "./my_results"
  save_individual_results: true

monitoring:
  interval: 1.0
  log_interval: 10
```

## 📊 Output Files

The tool generates several types of output files:

### Excel Files
- `benchmark_comparison_YYYYMMDD_HHMMSS.xlsx` - Comparison across all models
- `benchmark_[MODEL_SIZE]_YYYYMMDD_HHMMSS.xlsx` - Individual model results
- `system_info_YYYYMMDD_HHMMSS.xlsx` - System specifications

### JSON Files
- `complete_benchmark_results.json` - Raw results data
- `system_info.json` - System information
- `benchmark_estimates.json` - Pre-run estimates

### Log Files
- `benchmark.log` - Detailed execution logs

### Excel File Contents
Each Excel file contains:
- **Summary Sheet**: Key performance metrics and statistics
- **Raw Metrics Sheet**: Timestamped data for every training step
- **Charts**: Tokens/second, GPU utilization, loss curves, etc.

## 🔧 Advanced Configuration

### Custom Model Configurations
```yaml
models:
  custom_models:
    "custom_3B":
      hidden_size: 2048
      num_layers: 32
      num_attention_heads: 32
      vocab_size: 50257
      max_position_embeddings: 2048
```

### Performance Tuning
```yaml
performance:
  skip_on_error: true
  memory_cleanup: true
  compile_model: false  # Experimental: PyTorch 2.0 compilation

system:
  mixed_precision: true
  gradient_checkpointing: false
```

### Dataset Options
```yaml
dataset:
  type: "synthetic"  # or "files", "huggingface"
  max_length: 512
  text_files: ["data/text1.txt", "data/text2.txt"]  # for type: "files"
  huggingface_dataset: "wikitext-2"  # for type: "huggingface"
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch sizes
python benchmark_cli.py run --models 1B 7B  # Skip larger models

# Or edit config.yaml:
training:
  batch_sizes:
    "7B": 2
    "20B": 1
```

#### Slow Performance on CPU
The tool is designed for GPU usage. CPU-only execution will be very slow.
```bash
# Check GPU availability
python benchmark_cli.py check-system
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Permission Errors
```bash
# Make scripts executable
chmod +x run_benchmark.sh benchmark_cli.py
```

### GPU Memory Optimization Tips

1. **Reduce batch sizes** for large models
2. **Enable gradient checkpointing** in config
3. **Use mixed precision training** (enabled by default)
4. **Close other GPU applications** before running
5. **Run models sequentially** rather than in parallel

## 📈 Understanding Results

### Key Metrics Explained

- **Tokens/Second**: Higher is better - indicates training throughput
- **GPU Utilization**: Should be close to 100% for optimal performance
- **GPU Memory Utilization**: Indicates how much GPU memory is being used
- **CPU Utilization**: High values may indicate data loading bottlenecks
- **Training Loss**: Should decrease over time (though only 2 epochs)

### Performance Analysis

The tool generates comparative charts showing:
- Throughput comparison across model sizes
- Resource utilization patterns
- Scaling efficiency with model size
- Hardware bottleneck identification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information about your problem

## 🔮 Roadmap

Future enhancements planned:
- [ ] Support for more model architectures (T5, BERT, etc.)
- [ ] Distributed training across multiple nodes
- [ ] Integration with MLflow for experiment tracking
- [ ] Web-based dashboard for real-time monitoring
- [ ] Support for custom datasets and tasks
- [ ] Automated performance regression detection
- [ ] Integration with cloud platforms (AWS, GCP, Azure)

---

**Happy Benchmarking! 🎯**