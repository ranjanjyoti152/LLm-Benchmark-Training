#!/bin/bash
# Simple run script for the LLM benchmarking tool

echo "🚀 LLM Multi-GPU Benchmarking Tool"
echo "=================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "⚠️  No virtual environment detected."
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    # Activate existing environment
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

# Check if dependencies are installed
if ! python -c "import torch" 2>/dev/null; then
    echo "📦 Installing missing dependencies..."
    pip install -r requirements.txt
fi

echo "🔍 Checking system requirements..."
python benchmark_cli.py check-system

echo
echo "Choose an option:"
echo "1) Run full benchmark (all models)"
echo "2) Run quick benchmark (100 samples)"
echo "3) Check system requirements"
echo "4) Generate configuration file"
echo "5) Custom run (specify options)"
echo "6) Exit"
echo

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "🎯 Running full benchmark..."
        python benchmark_cli.py run
        ;;
    2)
        echo "⚡ Running quick benchmark..."
        python benchmark_cli.py run --quick
        ;;
    3)
        echo "🔍 System requirements check..."
        python benchmark_cli.py check-system --detailed
        ;;
    4)
        echo "📝 Generating configuration file..."
        python benchmark_cli.py generate-config --output my_config.yaml
        ;;
    5)
        echo "🎮 Custom benchmark options:"
        echo "Available models: 1B, 7B, 20B, 120B"
        read -p "Enter model sizes (space-separated): " models
        read -p "Enter number of samples (default 1000): " samples
        read -p "Enter output directory (default ./benchmark_results): " output
        
        # Set defaults
        samples=${samples:-1000}
        output=${output:-./benchmark_results}
        
        echo "🎯 Running custom benchmark..."
        python benchmark_cli.py run --models $models --samples $samples --output "$output"
        ;;
    6)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please try again."
        exit 1
        ;;
esac

echo
echo "✅ Done! Check the output directory for results."