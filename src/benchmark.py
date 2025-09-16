"""
Main benchmarking orchestrator that coordinates running benchmarks across all model sizes.
"""

import os
import sys
import time
import logging
import json
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
import traceback

from .monitoring import SystemMonitor
from .model_config import ModelLoader, ModelConfig
from .training import BenchmarkTrainer
from .excel_export import ExcelExporter
from .ollama_analysis import OllamaAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """
    Main orchestrator for running comprehensive LLM benchmarks.
    """
    
    def __init__(self, output_dir: str = "./benchmark_results", model_sizes: Optional[List[str]] = None, 
                 num_samples: int = 1000, enable_ollama: bool = True, ollama_url: str = "http://92.168.100.67:11434"):
        """
        Initialize the benchmark orchestrator.
        
        Args:
            output_dir: Directory to save results
            model_sizes: List of model sizes to benchmark
            num_samples: Number of training samples per model
            enable_ollama: Whether to enable Ollama AI analysis
            ollama_url: URL of Ollama server
        """
        self.output_dir = output_dir
        self.model_sizes = model_sizes or ["1B", "7B", "20B", "120B"]
        self.num_samples = num_samples
        self.enable_ollama = enable_ollama
        self.ollama_url = ollama_url
        
        # Initialize components
        self.system_monitor = SystemMonitor()
        self.model_loader = ModelLoader()
        self.excel_exporter = ExcelExporter()
        
        # Initialize Ollama analyzer if enabled
        self.ollama_analyzer = None
        if self.enable_ollama:
            try:
                self.ollama_analyzer = OllamaAnalyzer(ollama_url)
                if self.ollama_analyzer.test_connection():
                    logger.info("✅ Ollama analyzer connected successfully")
                else:
                    logger.warning("⚠️ Ollama connection failed, will use basic analysis")
                    self.ollama_analyzer = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Ollama analyzer: {e}")
                self.ollama_analyzer = None
        
        # Results storage
        self.benchmark_results = {}
        self.system_info = {}
        
        # Default attributes for backward compatibility
        self.skip_on_error = True
        self.save_individual_results = True
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized BenchmarkOrchestrator")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model sizes to benchmark: {self.model_sizes}")
        logger.info(f"Training samples per model: {num_samples}")
        
        if self.ollama_analyzer:
            logger.info(f"🤖 AI analysis enabled via Ollama at {ollama_url}")
        else:
            logger.info("📊 Using standard analysis only")
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information."""
        logger.info("Collecting system information...")
        
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # GPU information
        if torch.cuda.is_available():
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info[f"gpu_{i}"] = {
                    "name": gpu_props.name,
                    "total_memory": gpu_props.total_memory,
                    "major": gpu_props.major,
                    "minor": gpu_props.minor,
                    "multi_processor_count": gpu_props.multi_processor_count
                }
            system_info["gpu_info"] = gpu_info
        
        # System metrics
        try:
            cpu_info = self.system_monitor.get_cpu_info()
            memory_info = self.system_monitor.get_memory_metrics()
            
            system_info.update({
                "cpu_info": cpu_info,
                "memory_info": memory_info
            })
        except Exception as e:
            logger.warning(f"Could not collect system metrics: {e}")
        
        self.system_info = system_info
        return system_info
    
    def estimate_benchmark_requirements(self) -> Dict[str, Any]:
        """Estimate time and resource requirements for the benchmark."""
        logger.info("Estimating benchmark requirements...")
        
        estimates = {}
        total_estimated_time = 0
        total_estimated_memory = 0
        
        for model_size in self.model_sizes:
            try:
                memory_estimate = self.model_loader.estimate_memory_usage(model_size)
                
                # Rough time estimation (very approximate)
                # Based on typical training speeds for different model sizes
                time_per_sample = {
                    "1B": 0.1,   # seconds per sample
                    "7B": 0.5,
                    "20B": 1.5,
                    "120B": 5.0
                }.get(model_size, 1.0)
                
                estimated_time = (self.num_samples * time_per_sample * 2) / 60  # 2 epochs, convert to minutes
                
                estimates[model_size] = {
                    "estimated_time_minutes": estimated_time,
                    "estimated_memory_gb": memory_estimate["total_memory_gb"],
                    "estimated_params": memory_estimate["estimated_params"]
                }
                
                total_estimated_time += estimated_time
                total_estimated_memory = max(total_estimated_memory, memory_estimate["total_memory_gb"])
                
            except Exception as e:
                logger.warning(f"Could not estimate requirements for {model_size}: {e}")
        
        estimates["total"] = {
            "total_estimated_time_minutes": total_estimated_time,
            "peak_memory_requirement_gb": total_estimated_memory,
            "total_models": len(self.model_sizes)
        }
        
        logger.info(f"Total estimated time: {total_estimated_time:.1f} minutes")
        logger.info(f"Peak memory requirement: {total_estimated_memory:.1f} GB")
        
        return estimates
    
    def benchmark_single_model(self, model_size: str) -> Dict[str, Any]:
        """
        Benchmark a single model size.
        
        Args:
            model_size: Size of model to benchmark
            
        Returns:
            Benchmark results for the model
        """
        # Clean up previous model to free memory
        if hasattr(self, '_current_model'):
            del self._current_model
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"Starting benchmark for {model_size} model...")
        
        try:
            # Load model and tokenizer
            model, tokenizer = self.model_loader.load_model_and_tokenizer(model_size)
            self._current_model = model  # Keep reference for cleanup
            
            logger.info(f"Successfully loaded {model_size} model")
            
            # Create trainer
            trainer = BenchmarkTrainer(
                model=model,
                tokenizer=tokenizer,
                model_size=model_size,
                output_dir=os.path.join(self.output_dir, f"{model_size}_training"),
                monitoring_interval=1.0,
                log_interval=10
            )
            
            # Run benchmark
            results = trainer.benchmark_training(num_samples=self.num_samples)
            
            # Add metrics summary
            results["metrics_summary"] = trainer.get_metrics_summary()
            
            # Clean up model to free memory
            del model, trainer
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as cache_error:
                    logger.warning(f"Error during CUDA cache cleanup: {cache_error}")
            
            logger.info(f"Completed benchmark for {model_size} model")
            return results
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during {model_size} benchmark: {e}")
            
            # Clean up current model
            if hasattr(self, '_current_model'):
                del self._current_model
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.warning("Could not clear CUDA cache during OOM recovery")
            gc.collect()
            
            # Try with quantization as fallback
            logger.info(f"🔄 Retrying {model_size} with forced quantization...")
            return self._retry_with_quantization(model_size)
            
        except (torch.AcceleratorError, RuntimeError) as e:
            if "CUDA" in str(e) or "illegal memory access" in str(e):
                logger.error(f"CUDA error during {model_size} benchmark: {e}")
                
                # Clean up and try to recover
                if hasattr(self, '_current_model'):
                    del self._current_model
                if 'model' in locals():
                    del model
                if 'trainer' in locals():
                    del trainer
                
                # Reset CUDA context
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    logger.warning("Could not reset CUDA context")
                
                gc.collect()
                logger.error(f"✗ Critical error benchmarking {model_size}: {e}")
                logger.info("Continuing with next model...")
                return None
            else:
                # Re-raise non-CUDA errors
                raise e
                
        except Exception as e:
            logger.error(f"Error benchmarking {model_size} model: {e}")
            logger.error(traceback.format_exc())
            
            # Clean up on any error
            if hasattr(self, '_current_model'):
                del self._current_model
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "model_size": model_size,
                "benchmark_completed": False,
                "error": str(e),
                "error_traceback": traceback.format_exc()
            }
    
    def _retry_with_quantization(self, model_size: str) -> Dict[str, Any]:
        """
        Retry benchmarking with forced quantization when OOM occurs.
        
        Args:
            model_size: Size of model to benchmark
            
        Returns:
            Benchmark results for the model
        """
        from src.model_config import ModelConfig
        
        # Try different quantization methods in order of preference
        quantization_methods = ["fp16", "8bit", "4bit"]
        
        for quant_method in quantization_methods:
            logger.info(f"🔧 Attempting {model_size} with {quant_method} quantization...")
            
            try:
                # Force quantization by temporarily modifying the model config
                config = ModelConfig.get_config(model_size, self.model_loader.device_count)
                
                # Create a new model loader instance for quantized loading
                from src.model_config import ModelLoader
                quantized_loader = ModelLoader(self.model_loader.device_count)
                
                # Force load with specific quantization
                model = quantized_loader._try_load_model_with_quantization(config, model_size, quant_method)
                
                # Load tokenizer
                from transformers import AutoTokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception:
                    logger.info("Using GPT2 tokenizer as fallback")
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    tokenizer.pad_token = tokenizer.eos_token
                
                logger.info(f"✅ Successfully loaded {model_size} with {quant_method}")
                
                # Create trainer with reduced batch size for quantized models
                trainer = BenchmarkTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    model_size=model_size,
                    output_dir=os.path.join(self.output_dir, f"{model_size}_training_quantized"),
                    monitoring_interval=1.0,
                    log_interval=10
                )
                
                # Reduce the number of samples for quantized retry
                reduced_samples = max(10, self.num_samples // 4)  # Use 1/4 samples or minimum 10
                logger.info(f"Running with reduced samples: {reduced_samples}")
                
                # Run benchmark
                results = trainer.benchmark_training(num_samples=reduced_samples)
                results["quantization_used"] = quant_method
                results["reduced_samples"] = True
                results["original_samples"] = self.num_samples
                
                # Add metrics summary
                results["metrics_summary"] = trainer.get_metrics_summary()
                
                # Clean up
                del model, trainer, quantized_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Completed quantized benchmark for {model_size} model with {quant_method}")
                return results
                
            except Exception as e:
                logger.warning(f"❌ {quant_method} quantization failed for {model_size}: {e}")
                
                # Clean up on failure
                if 'model' in locals():
                    del model
                if 'trainer' in locals():
                    del trainer
                if 'quantized_loader' in locals():
                    del quantized_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                continue  # Try next quantization method
        
        # If all quantization methods failed
        logger.error(f"All quantization methods failed for {model_size}")
        return {
            "model_size": model_size,
            "benchmark_completed": False,
            "error": "All quantization methods failed due to memory constraints",
            "quantization_attempted": quantization_methods
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark across all specified model sizes.
        
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info("Starting full benchmark suite...")
        start_time = time.time()
        
        # Collect system information
        self.collect_system_info()
        
        # Estimate requirements
        estimates = self.estimate_benchmark_requirements()
        
        # Save system info and estimates
        with open(os.path.join(self.output_dir, "system_info.json"), 'w') as f:
            json.dump(self.system_info, f, indent=2, default=str)
        
        with open(os.path.join(self.output_dir, "benchmark_estimates.json"), 'w') as f:
            json.dump(estimates, f, indent=2, default=str)
        
        # Run benchmarks for each model size
        successful_benchmarks = 0
        failed_benchmarks = 0
        
        for i, model_size in enumerate(self.model_sizes, 1):
            logger.info(f"Benchmarking model {i}/{len(self.model_sizes)}: {model_size}")
            
            try:
                result = self.benchmark_single_model(model_size)
                self.benchmark_results[model_size] = result
                
                if result.get("benchmark_completed", False):
                    successful_benchmarks += 1
                    logger.info(f"✓ {model_size} benchmark completed successfully")
                    
                    # Save individual results if requested
                    if self.save_individual_results:
                        try:
                            self.excel_exporter.export_single_model_results(model_size, result)
                        except Exception as e:
                            logger.warning(f"Could not save individual results for {model_size}: {e}")
                else:
                    failed_benchmarks += 1
                    logger.error(f"✗ {model_size} benchmark failed")
                
            except Exception as e:
                failed_benchmarks += 1
                logger.error(f"✗ Critical error benchmarking {model_size}: {e}")
                
                if not self.skip_on_error:
                    logger.error("Stopping benchmark due to error (skip_on_error=False)")
                    break
                else:
                    logger.info("Continuing with next model...")
                    self.benchmark_results[model_size] = {
                        "model_size": model_size,
                        "benchmark_completed": False,
                        "error": str(e)
                    }
        
        total_time = time.time() - start_time
        
        # Create final summary
        summary = {
            "benchmark_start_time": datetime.fromtimestamp(start_time).isoformat(),
            "benchmark_end_time": datetime.now().isoformat(),
            "total_duration_seconds": total_time,
            "total_duration_minutes": total_time / 60,
            "models_attempted": len(self.model_sizes),
            "successful_benchmarks": successful_benchmarks,
            "failed_benchmarks": failed_benchmarks,
            "success_rate": successful_benchmarks / len(self.model_sizes) * 100,
            "system_info": self.system_info,
            "estimates": estimates,
            "benchmark_results": self.benchmark_results
        }
        
        # Save complete results
        results_file = os.path.join(self.output_dir, "complete_benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Benchmark suite completed in {total_time/60:.1f} minutes")
        logger.info(f"Successful: {successful_benchmarks}/{len(self.model_sizes)} models")
        
        return summary
    
    def _create_enhanced_reports(self, summary: Dict[str, Any]) -> List[str]:
        """
        Create enhanced reports with AI analysis if available.
        
        Args:
            summary: Benchmark summary data
            
        Returns:
            List of created file paths
        """
        excel_files = []
        
        # First create standard Excel reports
        standard_files = self.export_all_results()
        excel_files.extend(standard_files)
        
        # If Ollama is available, create AI-enhanced report
        if self.ollama_analyzer:
            try:
                logger.info("🤖 Generating AI-enhanced analysis report...")
                
                # Load complete results from JSON
                json_file = os.path.join(self.output_dir, "complete_benchmark_results.json")
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        complete_results = json.load(f)
                    
                    # Get AI analysis
                    ai_analysis = self.ollama_analyzer.analyze_benchmark_results(complete_results)
                    
                    # Create enhanced Excel report
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    enhanced_excel_path = os.path.join(
                        self.output_dir, 
                        f"ai_enhanced_benchmark_report_{timestamp}.xlsx"
                    )
                    
                    self.ollama_analyzer.create_enhanced_excel_report(
                        complete_results, ai_analysis, enhanced_excel_path
                    )
                    
                    excel_files.append(enhanced_excel_path)
                    
                    # Save AI analysis as JSON for reference
                    ai_analysis_path = os.path.join(
                        self.output_dir,
                        f"ai_analysis_{timestamp}.json"
                    )
                    with open(ai_analysis_path, 'w') as f:
                        json.dump(ai_analysis, f, indent=2)
                    
                    logger.info(f"✅ AI-enhanced report created: {enhanced_excel_path}")
                else:
                    logger.warning("⚠️ Complete results JSON not found, skipping AI analysis")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create AI-enhanced report: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.info("ℹ️ Ollama analyzer not available, using standard reports only")
        
        return excel_files

    def export_all_results(self) -> List[str]:
        """
        Export all results to Excel files.
        
        Returns:
            List of created file paths
        """
        logger.info("Exporting all results to Excel...")
        
        created_files = []
        
        try:
            # Export comparison results
            if self.benchmark_results:
                comparison_file = self.excel_exporter.export_comparison_results(self.benchmark_results)
                created_files.append(comparison_file)
            
            # Export system info
            if self.system_info:
                system_file = self.excel_exporter.export_system_info(self.system_info)
                created_files.append(system_file)
            
            # Export individual results if not already done
            if not self.save_individual_results:
                for model_size, results in self.benchmark_results.items():
                    if results.get("benchmark_completed", False):
                        individual_file = self.excel_exporter.export_single_model_results(model_size, results)
                        created_files.append(individual_file)
            
            logger.info(f"Created {len(created_files)} Excel files")
            return created_files
            
        except Exception as e:
            logger.error(f"Error exporting results to Excel: {e}")
            return created_files
    
    def run_benchmark_suite(self) -> Dict[str, Any]:
        """
        Run the complete benchmark suite with all exports.
        
        Returns:
            Complete benchmark summary
        """
        logger.info("="*80)
        logger.info("STARTING LLM MULTI-GPU BENCHMARK SUITE")
        logger.info("="*80)
        
        try:
            # Run benchmarks
            summary = self.run_full_benchmark()
            
            # Export results and create enhanced reports with AI analysis
            excel_files = self._create_enhanced_reports(summary)
            summary["excel_files"] = excel_files
            
            logger.info("="*80)
            logger.info("BENCHMARK SUITE COMPLETED")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"Excel files created: {len(excel_files)}")
            logger.info("="*80)
            
            return summary
            
        except Exception as e:
            logger.error(f"Critical error in benchmark suite: {e}")
            logger.error(traceback.format_exc())
            raise


def main():
    """Main entry point for running benchmarks."""
    # Parse command line arguments (basic implementation)
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Multi-GPU Benchmark Tool")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["1B", "7B", "20B", "120B"], 
                       help="Model sizes to benchmark")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--skip-on-error", action="store_true", default=True,
                       help="Skip failed models and continue")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(
        output_dir=args.output_dir,
        model_sizes=args.models,
        num_samples=args.samples,
        skip_on_error=args.skip_on_error
    )
    
    # Run benchmark suite
    try:
        results = orchestrator.run_benchmark_suite()
        logger.info("Benchmark completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())