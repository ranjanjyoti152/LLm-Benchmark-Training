"""
Ollama integration for AI-powered benchmark analysis and reporting.
"""

import json
import logging
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaAnalyzer:
    """
    Integrates with Ollama server to analyze benchmark results using AI.
    """
    
    def __init__(self, ollama_url: str = "http://92.168.100.67:11434", model: str = "gpt-oss:20b"):
        """
        Initialize Ollama analyzer.
        
        Args:
            ollama_url: URL of Ollama server
            model: Model name to use for analysis
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minute timeout for AI analysis
        
    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m.get('name', '') for m in models]
                logger.info(f"Connected to Ollama. Available models: {available_models}")
                return True
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to Ollama server at {self.ollama_url}")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to Ollama server at {self.ollama_url}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return False
    
    def analyze_benchmark_results(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send benchmark results to Ollama for AI analysis.
        
        Args:
            results_data: Complete benchmark results from JSON
            
        Returns:
            Structured analysis from AI
        """
        logger.info(f"Sending benchmark results to Ollama for analysis...")
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(results_data)
        
        try:
            # Send to Ollama
            response = self._query_ollama(prompt)
            
            # Parse AI response into structured format
            analysis = self._parse_ai_response(response)
            
            logger.info("Successfully received AI analysis from Ollama")
            return analysis
            
        except Exception as e:
            logger.error(f"Error during Ollama analysis: {e}")
            return self._create_fallback_analysis(results_data)
    
    def _create_analysis_prompt(self, results_data: Dict[str, Any]) -> str:
        """Create a detailed prompt for AI analysis."""
        
        # Extract key metrics
        system_info = results_data.get('system_info', {})
        model_results = results_data.get('model_results', {})
        
        prompt = f"""
Please analyze the following LLM multi-GPU benchmark results and provide a structured analysis in JSON format.

SYSTEM CONFIGURATION:
- GPUs: {system_info.get('gpu_count', 'Unknown')} x {system_info.get('gpu_name', 'Unknown')}
- Total GPU Memory: {system_info.get('total_gpu_memory_gb', 'Unknown')} GB
- CPU: {system_info.get('cpu_info', 'Unknown')}
- RAM: {system_info.get('total_memory_gb', 'Unknown')} GB

BENCHMARK RESULTS:
{json.dumps(model_results, indent=2)}

Please provide your analysis in the following JSON structure:

{{
    "performance_summary": {{
        "best_performing_model": "model_name",
        "highest_throughput": "X tokens/sec",
        "most_efficient_model": "model_name (best tokens/sec per GB)",
        "scalability_rating": "1-10 scale"
    }},
    "detailed_analysis": [
        {{
            "model": "1B",
            "performance_score": "1-10",
            "efficiency_score": "1-10", 
            "gpu_utilization_rating": "1-10",
            "memory_efficiency": "1-10",
            "throughput_analysis": "detailed analysis",
            "recommendations": "optimization suggestions"
        }}
    ],
    "system_recommendations": {{
        "gpu_utilization": "analysis of how well GPUs were used",
        "memory_optimization": "suggestions for memory usage",
        "scaling_potential": "how well this setup can scale",
        "bottlenecks": "identified performance bottlenecks"
    }},
    "comparative_insights": {{
        "model_scaling_efficiency": "how performance scales with model size",
        "multi_gpu_effectiveness": "effectiveness of multi-GPU setup", 
        "cost_performance_ratio": "analysis of cost vs performance",
        "production_readiness": "assessment for production use"
    }},
    "optimization_recommendations": [
        {{
            "category": "Hardware/Software/Configuration",
            "priority": "High/Medium/Low",
            "recommendation": "specific recommendation",
            "expected_improvement": "estimated improvement"
        }}
    ],
    "executive_summary": "2-3 sentence high-level summary of results and key insights"
}}

Focus on practical insights, performance optimization, and actionable recommendations. Be specific with numbers and comparisons.
"""
        return prompt
    
    def _query_ollama(self, prompt: str) -> str:
        """Send query to Ollama and get response."""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for analytical responses
                "top_p": 0.9,
                "num_predict": 4000  # Allow longer responses
            }
        }
        
        response = self.session.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            raise Exception(f"Ollama request failed: {response.status_code} - {response.text}")
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured format."""
        
        try:
            # Try to extract JSON from response
            # Look for JSON content between ```json and ``` or just find JSON object
            start_markers = ['```json', '{']
            end_markers = ['```', '}']
            
            json_start = -1
            json_end = -1
            
            # Find JSON start
            for marker in start_markers:
                pos = response.find(marker)
                if pos != -1:
                    json_start = pos + len(marker) if marker == '```json' else pos
                    break
            
            if json_start == -1:
                raise ValueError("No JSON found in response")
            
            # Find matching closing brace
            brace_count = 0
            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end == -1:
                raise ValueError("No complete JSON found in response")
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
            
        except Exception as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            return self._extract_structured_analysis(response)
    
    def _extract_structured_analysis(self, response: str) -> Dict[str, Any]:
        """Extract structured analysis from free-form AI response."""
        
        # Basic structure with AI insights extracted as text
        return {
            "performance_summary": {
                "ai_analysis": response[:500] + "..." if len(response) > 500 else response
            },
            "detailed_analysis": [],
            "system_recommendations": {
                "full_analysis": response
            },
            "executive_summary": response[:200] + "..." if len(response) > 200 else response
        }
    
    def _create_fallback_analysis(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic analysis if AI analysis fails."""
        
        model_results = results_data.get('model_results', {})
        
        # Calculate basic metrics
        best_model = ""
        highest_throughput = 0
        
        for model, data in model_results.items():
            throughput = data.get('average_tokens_per_sec', 0)
            if throughput > highest_throughput:
                highest_throughput = throughput
                best_model = model
        
        return {
            "performance_summary": {
                "best_performing_model": best_model,
                "highest_throughput": f"{highest_throughput:.1f} tokens/sec",
                "analysis_status": "Basic analysis (AI analysis failed)"
            },
            "detailed_analysis": [
                {
                    "model": model,
                    "throughput": data.get('average_tokens_per_sec', 0),
                    "total_time": data.get('total_training_time', 0),
                    "parameters": data.get('model_parameters', 0)
                }
                for model, data in model_results.items()
            ],
            "executive_summary": f"Benchmark completed. Best performing model: {best_model} with {highest_throughput:.1f} tokens/sec"
        }
    
    def create_enhanced_excel_report(self, results_data: Dict[str, Any], ai_analysis: Dict[str, Any], output_path: str):
        """
        Create comprehensive Excel report combining benchmark data and AI analysis.
        
        Args:
            results_data: Original benchmark results
            ai_analysis: AI analysis results
            output_path: Path for output Excel file
        """
        logger.info(f"Creating enhanced Excel report: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # 1. Executive Summary Sheet
            self._create_executive_summary_sheet(writer, results_data, ai_analysis)
            
            # 2. Performance Overview Sheet
            self._create_performance_overview_sheet(writer, results_data, ai_analysis)
            
            # 3. Detailed Model Analysis Sheet
            self._create_detailed_analysis_sheet(writer, results_data, ai_analysis)
            
            # 4. System Information Sheet
            self._create_system_info_sheet(writer, results_data)
            
            # 5. AI Recommendations Sheet
            self._create_recommendations_sheet(writer, ai_analysis)
            
            # 6. Raw Data Sheet
            self._create_raw_data_sheet(writer, results_data)
        
        logger.info(f"Enhanced Excel report created: {output_path}")
    
    def _create_executive_summary_sheet(self, writer, results_data: Dict, ai_analysis: Dict):
        """Create executive summary sheet."""
        
        summary_data = []
        perf_summary = ai_analysis.get('performance_summary', {})
        
        summary_data.append(['Metric', 'Value'])
        summary_data.append(['Best Performing Model', perf_summary.get('best_performing_model', 'N/A')])
        summary_data.append(['Highest Throughput', perf_summary.get('highest_throughput', 'N/A')])
        summary_data.append(['Most Efficient Model', perf_summary.get('most_efficient_model', 'N/A')])
        summary_data.append(['Scalability Rating', perf_summary.get('scalability_rating', 'N/A')])
        summary_data.append(['', ''])
        summary_data.append(['Executive Summary', ''])
        summary_data.append(['', ai_analysis.get('executive_summary', 'No summary available')])
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Executive Summary', index=False, header=False)
    
    def _create_performance_overview_sheet(self, writer, results_data: Dict, ai_analysis: Dict):
        """Create performance overview sheet."""
        
        model_results = results_data.get('model_results', {})
        perf_data = []
        
        for model, data in model_results.items():
            perf_data.append({
                'Model': model,
                'Parameters': f"{data.get('model_parameters', 0):,}",
                'Tokens/Sec': f"{data.get('average_tokens_per_sec', 0):.1f}",
                'Total Time (s)': f"{data.get('total_training_time', 0):.1f}",
                'Peak Memory (GB)': f"{data.get('peak_memory_gb', 0):.1f}",
                'GPU Utilization (%)': f"{data.get('avg_gpu_utilization', 0):.1f}",
                'Status': data.get('status', 'Unknown')
            })
        
        df = pd.DataFrame(perf_data)
        df.to_excel(writer, sheet_name='Performance Overview', index=False)
    
    def _create_detailed_analysis_sheet(self, writer, results_data: Dict, ai_analysis: Dict):
        """Create detailed analysis sheet."""
        
        detailed_analysis = ai_analysis.get('detailed_analysis', [])
        
        if detailed_analysis:
            df = pd.DataFrame(detailed_analysis)
        else:
            # Fallback to basic analysis
            model_results = results_data.get('model_results', {})
            analysis_data = []
            
            for model, data in model_results.items():
                analysis_data.append({
                    'model': model,
                    'throughput_tokens_per_sec': data.get('average_tokens_per_sec', 0),
                    'total_time_seconds': data.get('total_training_time', 0),
                    'parameters': data.get('model_parameters', 0),
                    'peak_memory_gb': data.get('peak_memory_gb', 0),
                    'status': data.get('status', 'Unknown')
                })
            
            df = pd.DataFrame(analysis_data)
        
        df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
    
    def _create_system_info_sheet(self, writer, results_data: Dict):
        """Create system information sheet."""
        
        system_info = results_data.get('system_info', {})
        
        sys_data = []
        for key, value in system_info.items():
            sys_data.append([key.replace('_', ' ').title(), str(value)])
        
        df = pd.DataFrame(sys_data, columns=['Component', 'Details'])
        df.to_excel(writer, sheet_name='System Information', index=False)
    
    def _create_recommendations_sheet(self, writer, ai_analysis: Dict):
        """Create recommendations sheet."""
        
        recommendations = ai_analysis.get('optimization_recommendations', [])
        system_rec = ai_analysis.get('system_recommendations', {})
        
        # Optimization recommendations
        if recommendations:
            df_opt = pd.DataFrame(recommendations)
            df_opt.to_excel(writer, sheet_name='Recommendations', index=False, startrow=0)
        
        # System recommendations
        if system_rec:
            sys_rec_data = []
            for key, value in system_rec.items():
                sys_rec_data.append([key.replace('_', ' ').title(), str(value)])
            
            df_sys = pd.DataFrame(sys_rec_data, columns=['Category', 'Recommendation'])
            start_row = len(recommendations) + 3 if recommendations else 0
            df_sys.to_excel(writer, sheet_name='Recommendations', index=False, startrow=start_row)
    
    def _create_raw_data_sheet(self, writer, results_data: Dict):
        """Create raw data sheet with complete results."""
        
        # Convert nested dict to flat structure for Excel
        flattened_data = self._flatten_dict(results_data)
        
        raw_data = []
        for key, value in flattened_data.items():
            raw_data.append([key, str(value)])
        
        df = pd.DataFrame(raw_data, columns=['Key', 'Value'])
        df.to_excel(writer, sheet_name='Raw Data', index=False)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)