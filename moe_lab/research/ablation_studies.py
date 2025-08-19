"""Ablation study framework for systematic MoE research."""

import json
import logging
import itertools
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch for testing
    class MockTensor:
        def __init__(self, *args, **kwargs):
            pass
        def item(self):
            return 0.0
    
    class MockModule:
        def __init__(self, *args, **kwargs):
            pass
    
    torch = type('torch', (), {
        'Tensor': MockTensor,
        'no_grad': lambda: None,
        'cuda': type('cuda', (), {'is_available': lambda: False})()
    })()
    nn = type('nn', (), {'Module': MockModule})()


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    
    # Base configuration
    base_config: Dict[str, Any]
    
    # Parameters to ablate
    ablation_params: Dict[str, List[Any]]
    
    # Study metadata
    study_name: str
    study_description: str = ""
    
    # Experimental settings
    num_runs_per_config: int = 3
    evaluation_metrics: List[str] = None
    
    # Output settings
    output_dir: str = "./ablation_results"
    save_intermediate: bool = True
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["perplexity", "throughput", "memory_usage"]


@dataclass
class AblationResult:
    """Results from a single ablation configuration."""
    
    config: Dict[str, Any]
    metrics: Dict[str, float]
    std_metrics: Dict[str, float]  # Standard deviations
    individual_runs: List[Dict[str, float]]
    runtime_info: Dict[str, Any]
    
    
class AblationStudy:
    """Generic ablation study framework."""
    
    def __init__(
        self,
        config: AblationConfig,
        model_factory: callable,
        trainer_factory: callable,
        evaluator: callable,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.evaluator = evaluator
        
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[AblationResult] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for ablation study."""
        logger = logging.getLogger(f"ablation_{self.config.study_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler(self.output_dir / "ablation_study.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def generate_configurations(self) -> List[Dict[str, Any]]:
        """Generate all configuration combinations for ablation."""
        
        # Get parameter names and values
        param_names = list(self.config.ablation_params.keys())
        param_values = list(self.config.ablation_params.values())
        
        # Generate all combinations
        configurations = []
        for combination in itertools.product(*param_values):
            config = self.config.base_config.copy()
            
            # Apply ablation parameters
            for param_name, value in zip(param_names, combination):
                # Support nested parameter paths (e.g., "model.hidden_size")
                self._set_nested_param(config, param_name, value)
            
            configurations.append(config)
        
        self.logger.info(f"Generated {len(configurations)} configurations for ablation study")
        return configurations
    
    def _set_nested_param(self, config: Dict[str, Any], param_path: str, value: Any):
        """Set a nested parameter using dot notation."""
        keys = param_path.split('.')
        current = config
        
        # Navigate to the parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def run_study(self) -> Dict[str, Any]:
        """Run complete ablation study."""
        self.logger.info(f"Starting ablation study: {self.config.study_name}")
        
        configurations = self.generate_configurations()
        
        for i, config in enumerate(configurations):
            self.logger.info(f"Running configuration {i+1}/{len(configurations)}")
            
            try:
                result = self._run_configuration(config, i)
                self.results.append(result)
                
                if self.config.save_intermediate:
                    self._save_intermediate_result(result, i)
                    
            except Exception as e:
                self.logger.error(f"Configuration {i} failed: {e}")
                # Continue with next configuration
                continue
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Generate final report
        final_report = self._generate_report(analysis)
        
        self.logger.info("Ablation study completed")
        return final_report
    
    def _run_configuration(self, config: Dict[str, Any], config_idx: int) -> AblationResult:
        """Run a single configuration multiple times."""
        self.logger.info(f"Running configuration {config_idx} with {self.config.num_runs_per_config} runs")
        
        individual_runs = []
        
        for run_idx in range(self.config.num_runs_per_config):
            try:
                # Create model and trainer
                model = self.model_factory(config)
                trainer = self.trainer_factory(model, config)
                
                # Train model (simplified for robustness)
                if hasattr(trainer, 'train'):
                    trainer.train()
                
                # Evaluate model
                metrics = self.evaluator(model, config)
                individual_runs.append(metrics)
                
            except Exception as e:
                self.logger.warning(f"Run {run_idx} of config {config_idx} failed: {e}")
                # Create dummy metrics to maintain structure
                dummy_metrics = {metric: float('inf') for metric in self.config.evaluation_metrics}
                individual_runs.append(dummy_metrics)
        
        # Aggregate results
        aggregated_metrics = {}
        std_metrics = {}
        
        for metric in self.config.evaluation_metrics:
            values = [run.get(metric, float('inf')) for run in individual_runs]
            valid_values = [v for v in values if v != float('inf')]
            
            if valid_values:
                aggregated_metrics[metric] = np.mean(valid_values)
                std_metrics[metric] = np.std(valid_values)
            else:
                aggregated_metrics[metric] = float('inf')
                std_metrics[metric] = 0.0
        
        return AblationResult(
            config=config,
            metrics=aggregated_metrics,
            std_metrics=std_metrics,
            individual_runs=individual_runs,
            runtime_info={
                "config_idx": config_idx,
                "successful_runs": len([r for r in individual_runs if all(v != float('inf') for v in r.values())]),
                "total_runs": len(individual_runs)
            }
        )
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze ablation study results."""
        if not self.results:
            return {"error": "No successful results to analyze"}
        
        analysis = {
            "parameter_effects": {},
            "best_configurations": {},
            "sensitivity_analysis": {},
            "statistical_significance": {}
        }
        
        # Analyze parameter effects
        for param in self.config.ablation_params.keys():
            analysis["parameter_effects"][param] = self._analyze_parameter_effect(param)
        
        # Find best configurations for each metric
        for metric in self.config.evaluation_metrics:
            analysis["best_configurations"][metric] = self._find_best_config(metric)
        
        # Sensitivity analysis
        analysis["sensitivity_analysis"] = self._compute_sensitivity()
        
        return analysis
    
    def _analyze_parameter_effect(self, param_name: str) -> Dict[str, Any]:
        """Analyze the effect of a specific parameter."""
        param_effects = {}
        
        # Group results by parameter value
        param_groups = {}
        for result in self.results:
            param_value = self._get_nested_param(result.config, param_name)
            param_value_str = str(param_value)
            
            if param_value_str not in param_groups:
                param_groups[param_value_str] = []
            param_groups[param_value_str].append(result)
        
        # Analyze effect on each metric
        for metric in self.config.evaluation_metrics:
            metric_by_param = {}
            
            for param_value, results in param_groups.items():
                values = [r.metrics.get(metric, float('inf')) for r in results]
                valid_values = [v for v in values if v != float('inf')]
                
                if valid_values:
                    metric_by_param[param_value] = {
                        "mean": np.mean(valid_values),
                        "std": np.std(valid_values),
                        "count": len(valid_values)
                    }
            
            param_effects[metric] = metric_by_param
        
        return param_effects
    
    def _get_nested_param(self, config: Dict[str, Any], param_path: str) -> Any:
        """Get a nested parameter value using dot notation."""
        keys = param_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _find_best_config(self, metric: str) -> Dict[str, Any]:
        """Find the best configuration for a specific metric."""
        valid_results = [r for r in self.results if r.metrics.get(metric, float('inf')) != float('inf')]
        
        if not valid_results:
            return {"error": f"No valid results for metric {metric}"}
        
        # For most metrics, lower is better (perplexity, memory_usage)
        # For throughput, higher is better
        if metric in ["throughput", "accuracy", "f1_score"]:
            best_result = max(valid_results, key=lambda r: r.metrics[metric])
        else:
            best_result = min(valid_results, key=lambda r: r.metrics[metric])
        
        return {
            "config": best_result.config,
            "metric_value": best_result.metrics[metric],
            "std": best_result.std_metrics[metric],
            "runtime_info": best_result.runtime_info
        }
    
    def _compute_sensitivity(self) -> Dict[str, Any]:
        """Compute parameter sensitivity analysis."""
        sensitivity = {}
        
        for param in self.config.ablation_params.keys():
            param_sensitivity = {}
            
            for metric in self.config.evaluation_metrics:
                # Calculate variance in metric across parameter values
                param_values = []
                metric_values = []
                
                for result in self.results:
                    param_val = self._get_nested_param(result.config, param)
                    metric_val = result.metrics.get(metric, float('inf'))
                    
                    if metric_val != float('inf'):
                        param_values.append(param_val)
                        metric_values.append(metric_val)
                
                if len(metric_values) > 1:
                    # Simple sensitivity measure: coefficient of variation
                    cv = np.std(metric_values) / np.mean(metric_values) if np.mean(metric_values) > 0 else 0
                    param_sensitivity[metric] = cv
                else:
                    param_sensitivity[metric] = 0.0
            
            sensitivity[param] = param_sensitivity
        
        return sensitivity
    
    def _save_intermediate_result(self, result: AblationResult, config_idx: int):
        """Save intermediate result."""
        output_file = self.output_dir / f"config_{config_idx}_result.json"
        
        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    def _generate_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final ablation study report."""
        report = {
            "study_info": {
                "name": self.config.study_name,
                "description": self.config.study_description,
                "total_configurations": len(self.results),
                "parameters_studied": list(self.config.ablation_params.keys()),
                "metrics_evaluated": self.config.evaluation_metrics
            },
            "results": [asdict(result) for result in self.results],
            "analysis": analysis,
            "recommendations": self._generate_recommendations(analysis)
        }
        
        # Save final report
        output_file = self.output_dir / f"{self.config.study_name}_ablation_report.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Final report saved to {output_file}")
        return report
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis."""
        recommendations = {
            "best_overall_config": None,
            "parameter_recommendations": {},
            "trade_off_analysis": {}
        }
        
        # Find best overall configuration (multi-objective)
        if self.results:
            # Simple scoring: normalize metrics and sum
            scores = []
            for result in self.results:
                score = 0
                valid_metrics = 0
                
                for metric in self.config.evaluation_metrics:
                    value = result.metrics.get(metric, float('inf'))
                    if value != float('inf'):
                        # Normalize by best value for this metric
                        all_values = [r.metrics.get(metric, float('inf')) for r in self.results if r.metrics.get(metric, float('inf')) != float('inf')]
                        if all_values:
                            if metric in ["throughput", "accuracy"]:
                                normalized = value / max(all_values)  # Higher is better
                            else:
                                normalized = min(all_values) / value  # Lower is better
                            score += normalized
                            valid_metrics += 1
                
                if valid_metrics > 0:
                    scores.append((score / valid_metrics, result))
                else:
                    scores.append((0, result))
            
            if scores:
                best_score, best_result = max(scores, key=lambda x: x[0])
                recommendations["best_overall_config"] = {
                    "config": best_result.config,
                    "score": best_score,
                    "metrics": best_result.metrics
                }
        
        # Parameter recommendations
        if "parameter_effects" in analysis:
            for param, effects in analysis["parameter_effects"].items():
                param_recommendations = {}
                
                for metric, param_effects in effects.items():
                    if param_effects:
                        # Find best parameter value for this metric
                        if metric in ["throughput", "accuracy"]:
                            best_param = max(param_effects.items(), key=lambda x: x[1]["mean"])
                        else:
                            best_param = min(param_effects.items(), key=lambda x: x[1]["mean"])
                        
                        param_recommendations[metric] = {
                            "best_value": best_param[0],
                            "metric_value": best_param[1]["mean"]
                        }
                
                recommendations["parameter_recommendations"][param] = param_recommendations
        
        return recommendations


class RouterAblation(AblationStudy):
    """Specialized ablation study for router algorithms."""
    
    def __init__(self, config: AblationConfig, **kwargs):
        # Add router-specific metrics
        router_metrics = ["load_variance", "entropy", "expert_utilization_gini"]
        if config.evaluation_metrics:
            config.evaluation_metrics.extend(router_metrics)
        else:
            config.evaluation_metrics = ["perplexity", "throughput"] + router_metrics
        
        super().__init__(config, **kwargs)
    
    def _analyze_routing_quality(self) -> Dict[str, Any]:
        """Analyze routing-specific quality metrics."""
        routing_analysis = {
            "load_balancing_quality": {},
            "routing_diversity": {},
            "expert_utilization": {}
        }
        
        for result in self.results:
            config_id = str(result.runtime_info.get("config_idx", "unknown"))
            
            # Load balancing quality (lower variance is better)
            load_var = result.metrics.get("load_variance", float('inf'))
            if load_var != float('inf'):
                routing_analysis["load_balancing_quality"][config_id] = 1.0 / (1.0 + load_var)
            
            # Routing diversity (higher entropy is better)
            entropy = result.metrics.get("entropy", 0.0)
            routing_analysis["routing_diversity"][config_id] = entropy
            
            # Expert utilization equality (lower Gini is better)
            gini = result.metrics.get("expert_utilization_gini", float('inf'))
            if gini != float('inf'):
                routing_analysis["expert_utilization"][config_id] = 1.0 - gini
        
        return routing_analysis


class ExpertAblation(AblationStudy):
    """Specialized ablation study for expert architectures."""
    
    def __init__(self, config: AblationConfig, **kwargs):
        # Add expert-specific metrics
        expert_metrics = ["expert_specialization", "parameter_efficiency"]
        if config.evaluation_metrics:
            config.evaluation_metrics.extend(expert_metrics)
        else:
            config.evaluation_metrics = ["perplexity", "throughput"] + expert_metrics
        
        super().__init__(config, **kwargs)


class ArchitectureAblation(AblationStudy):
    """Specialized ablation study for overall MoE architectures."""
    
    def __init__(self, config: AblationConfig, **kwargs):
        # Add architecture-specific metrics
        arch_metrics = ["memory_efficiency", "compute_efficiency", "scaling_factor"]
        if config.evaluation_metrics:
            config.evaluation_metrics.extend(arch_metrics)
        else:
            config.evaluation_metrics = ["perplexity", "throughput"] + arch_metrics
        
        super().__init__(config, **kwargs)
    
    def _analyze_scaling_properties(self) -> Dict[str, Any]:
        """Analyze how architectures scale with different parameters."""
        scaling_analysis = {}
        
        # Analyze parameter count vs performance
        param_counts = []
        performances = []
        
        for result in self.results:
            # Extract parameter count from config or metrics
            param_count = result.config.get("total_parameters", 0)
            if param_count == 0:
                # Estimate from model config
                hidden_size = result.config.get("hidden_size", 768)
                num_experts = result.config.get("num_experts", 8)
                num_layers = result.config.get("num_layers", 12)
                param_count = hidden_size * hidden_size * num_experts * num_layers  # Rough estimate
            
            performance = result.metrics.get("perplexity", float('inf'))
            
            if performance != float('inf'):
                param_counts.append(param_count)
                performances.append(performance)
        
        if len(param_counts) > 1:
            # Simple scaling analysis
            scaling_analysis["parameter_efficiency"] = {
                "correlation": np.corrcoef(param_counts, performances)[0, 1] if len(param_counts) > 1 else 0,
                "best_efficiency": min(zip(performances, param_counts), key=lambda x: x[0] * x[1]) if param_counts else None
            }
        
        return scaling_analysis