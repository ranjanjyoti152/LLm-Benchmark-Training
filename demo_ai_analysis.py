"""
Demo script to showcase enhanced Excel reporting with AI-style analysis
"""

import json
import os
from datetime import datetime
from src.ollama_analysis import OllamaAnalyzer

def create_demo_ai_analysis():
    """Create a demo AI analysis to show the enhanced reporting capabilities."""
    
    # Create simulated benchmark results for the demo
    results_data = {
        "experiment_info": {
            "timestamp": "2024-09-16_14:19:16",
            "models": ["1B"],
            "samples": 5,
            "gpus": 8,
            "total_duration": "0:02:15"
        },
        "model_results": {
            "1B": {
                "samples": [
                    {
                        "model_name": "1B",
                        "num_gpus": 8,
                        "batch_size": 8,
                        "seq_length": 512,
                        "training_speed": 1477.6,
                        "memory_usage": 12.8,
                        "gpu_utilization": 88.5,
                        "power_consumption": 2.1,
                        "cost_per_hour": 8.32,
                        "tokens_per_dollar": 177.6
                    }
                ],
                "avg_training_speed": 1477.6,
                "avg_memory_usage": 12.8,
                "avg_gpu_utilization": 88.5,
                "avg_power_consumption": 2.1,
                "avg_cost_per_hour": 8.32,
                "avg_tokens_per_dollar": 177.6
            }
        },
        "system_info": {
            "cpu": "Intel Xeon",
            "ram": "256GB",
            "gpu": "8x A100 80GB",
            "storage": "NVMe SSD"
        }
    }
    
    # Create a comprehensive demo analysis
    demo_analysis = {
        "performance_summary": {
            "best_performing_model": "1B",
            "highest_throughput": "1477.6 tokens/sec",
            "most_efficient_model": "1B (best tokens/sec per GB ratio)",
            "scalability_rating": "8/10 - Excellent multi-GPU utilization"
        },
        "detailed_analysis": [
            {
                "model": "1B",
                "performance_score": "9/10",
                "efficiency_score": "9/10",
                "gpu_utilization_rating": "8/10",
                "memory_efficiency": "10/10",
                "throughput_analysis": "Strong performance with 1477.6 tokens/sec. DataParallel effectively utilized 4 GPUs with excellent memory distribution (2.5GB peak per GPU).",
                "recommendations": "Consider increasing batch size to fully utilize remaining GPU memory. Model shows excellent scaling potential."
            }
        ],
        "system_recommendations": {
            "gpu_utilization": "DataParallel successfully distributed workload across multiple GPUs. GPU utilization reached 40% peak, indicating room for optimization.",
            "memory_optimization": "Memory usage was efficient at 2.5GB per GPU. Can handle much larger batch sizes or model sizes.",
            "scaling_potential": "System shows excellent scaling potential. Can easily handle 7B+ models with current memory footprint.",
            "bottlenecks": "CPU preprocessing appears to be the main bottleneck, not GPU compute. Consider data loading optimizations."
        },
        "comparative_insights": {
            "model_scaling_efficiency": "1B model demonstrates excellent per-parameter efficiency. Scaling to larger models should maintain efficiency.",
            "multi_gpu_effectiveness": "DataParallel implementation is highly effective. 8 GPU system is well-utilized for distributed training.",
            "cost_performance_ratio": "Outstanding cost-performance ratio. System delivers enterprise-grade performance with optimal resource usage.",
            "production_readiness": "System is production-ready for 1B-7B model training. Excellent stability and performance characteristics."
        },
        "optimization_recommendations": [
            {
                "category": "Hardware",
                "priority": "Medium",
                "recommendation": "Consider enabling model sharding for 20B+ models to utilize all 8 GPUs fully",
                "expected_improvement": "2-3x throughput increase for large models"
            },
            {
                "category": "Software",
                "priority": "High", 
                "recommendation": "Implement gradient accumulation and mixed precision training",
                "expected_improvement": "20-30% memory efficiency gain"
            },
            {
                "category": "Configuration",
                "priority": "Medium",
                "recommendation": "Increase batch size to 16-32 per GPU to maximize throughput",
                "expected_improvement": "15-25% throughput increase"
            }
        ],
        "executive_summary": "The system demonstrates exceptional performance for LLM training with 1B model achieving 1477.6 tokens/sec across 8 GPUs. Multi-GPU utilization is excellent with DataParallel, showing strong scalability potential for larger models. The setup is production-ready with outstanding cost-performance characteristics."
    }
    
    # Create enhanced Excel report
    analyzer = OllamaAnalyzer()  # Create without connection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "./benchmark_results"
    demo_excel_path = os.path.join(results_dir, f"DEMO_ai_enhanced_report_{timestamp}.xlsx")
    
    try:
        print(f"🔄 Creating Excel report at: {demo_excel_path}")
        analyzer.create_enhanced_excel_report(results_data, demo_analysis, demo_excel_path)
        print(f"✅ Demo AI-enhanced Excel report created: {demo_excel_path}")
        print()
        print("📊 **Enhanced Excel Report Features:**")
        print("   • Executive Summary with key insights")
        print("   • Performance Overview with detailed metrics") 
        print("   • AI-powered model analysis and scoring")
        print("   • System optimization recommendations")
        print("   • Comparative insights and scaling analysis")
        print("   • Raw data preservation for further analysis")
        print()
        print("🤖 **This demonstrates what the Ollama integration provides:**")
        print("   • Intelligent analysis of benchmark results")
        print("   • Actionable optimization recommendations")
        print("   • Performance scoring and efficiency metrics")
        print("   • Professional reporting for stakeholders")
        
    except Exception as e:
        print(f"❌ Error creating demo Excel report: {e}")
        print(f"   File path attempted: {demo_excel_path}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    create_demo_ai_analysis()