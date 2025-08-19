"""Experimental framework for reproducible MoE research."""

import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import scipy.stats as stats

from ..models.moe_model import MoEModel
from ..training.trainer import MoETrainer


@dataclass
class ExperimentConfig:
    """Configuration for MoE experiments."""
    
    # Model configuration
    model_config: Dict[str, Any]
    
    # Training configuration  
    training_config: Dict[str, Any]
    
    # Dataset configuration
    dataset_config: Dict[str, Any]
    
    # Experiment metadata
    experiment_name: str
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evaluation configuration
    eval_metrics: List[str] = None
    baseline_models: List[str] = None
    
    # Statistical configuration
    num_runs: int = 3
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ["perplexity", "throughput", "memory_usage", "expert_utilization"]
        if self.baseline_models is None:
            self.baseline_models = ["dense", "switch", "mixtral"]


@dataclass 
class ExperimentResult:
    """Results from a single experimental run."""
    
    # Performance metrics
    metrics: Dict[str, float]
    
    # Training history
    training_history: Dict[str, List[float]]
    
    # Model statistics
    model_stats: Dict[str, Any]
    
    # Runtime information
    runtime_info: Dict[str, Any]
    
    # Routing analysis
    routing_analysis: Optional[Dict[str, Any]] = None


class ExperimentRunner:
    """Automated experiment runner with statistical validation."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "./experiment_results",
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Set random seeds for reproducibility
        self._set_seeds(config.seed)
        
        # Initialize results storage
        self.results: List[ExperimentResult] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup experiment logger."""
        logger = logging.getLogger(f"experiment_{self.config.experiment_name}")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "experiment.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete experiment with multiple runs."""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        experiment_start_time = time.time()
        
        # Run multiple trials
        for run_idx in range(self.config.num_runs):
            self.logger.info(f"Starting run {run_idx + 1}/{self.config.num_runs}")
            
            # Set seed for this run
            run_seed = self.config.seed + run_idx
            self._set_seeds(run_seed)
            
            # Run single trial
            result = self._run_single_trial(run_idx, run_seed)
            self.results.append(result)
            
            # Save intermediate results
            self._save_intermediate_results(run_idx)
            
        experiment_duration = time.time() - experiment_start_time
        
        # Analyze results across runs
        aggregated_results = self._aggregate_results()
        
        # Statistical validation
        statistical_analysis = self._perform_statistical_analysis()
        
        # Compile final report
        final_report = {
            "config": asdict(self.config),
            "individual_results": [asdict(result) for result in self.results],
            "aggregated_results": aggregated_results,
            "statistical_analysis": statistical_analysis,
            "experiment_metadata": {
                "total_duration": experiment_duration,
                "num_runs": len(self.results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Save final report
        self._save_final_report(final_report)
        
        self.logger.info(f"Experiment completed in {experiment_duration:.2f} seconds")
        
        return final_report
    
    def _run_single_trial(self, run_idx: int, seed: int) -> ExperimentResult:
        """Run a single experimental trial."""
        trial_start_time = time.time()
        
        # Create model
        model = self._create_model()
        
        # Create trainer  
        trainer = self._create_trainer(model)
        
        # Load dataset
        train_loader, eval_loader = self._load_datasets()
        
        # Training
        self.logger.info(f"Training model for run {run_idx}")
        training_result = trainer.train(
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            **self.config.training_config
        )
        
        # Evaluation
        self.logger.info(f"Evaluating model for run {run_idx}")
        eval_metrics = self._evaluate_model(model, eval_loader)
        
        # Model analysis
        model_stats = self._analyze_model(model)
        
        # Routing analysis (if MoE model)
        routing_analysis = None
        if hasattr(model, 'moe_layers'):
            routing_analysis = self._analyze_routing(model, eval_loader)
        
        trial_duration = time.time() - trial_start_time
        
        return ExperimentResult(
            metrics=eval_metrics,
            training_history=training_result.history if hasattr(training_result, 'history') else {},
            model_stats=model_stats,
            runtime_info={
                "trial_duration": trial_duration,
                "seed": seed,
                "device": str(model.device) if hasattr(model, 'device') else self.config.device
            },
            routing_analysis=routing_analysis
        )
    
    def _create_model(self) -> nn.Module:
        """Create model from configuration."""
        model_config = self.config.model_config.copy()
        model_type = model_config.pop("type", "MoEModel")
        
        if model_type == "MoEModel":
            model = MoEModel(**model_config)
        else:
            # Support for other model types
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model.to(self.config.device)
    
    def _create_trainer(self, model: nn.Module) -> MoETrainer:
        """Create trainer from configuration."""
        return MoETrainer(model=model)
    
    def _load_datasets(self) -> Tuple[DataLoader, DataLoader]:
        """Load training and evaluation datasets."""
        # This would be implemented based on dataset_config
        # For now, return dummy loaders
        from torch.utils.data import TensorDataset
        
        # Dummy data for demonstration
        vocab_size = self.config.model_config.get("vocab_size", 32000)
        seq_length = 512
        batch_size = self.config.training_config.get("batch_size", 16)
        
        train_data = torch.randint(0, vocab_size, (1000, seq_length))
        eval_data = torch.randint(0, vocab_size, (200, seq_length))
        
        train_dataset = TensorDataset(train_data)
        eval_dataset = TensorDataset(eval_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, eval_loader
    
    def _evaluate_model(self, model: nn.Module, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        model.eval()
        metrics = {}
        
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch[0].to(self.config.device)
                
                # Forward pass
                if hasattr(model, 'forward'):
                    outputs = model(input_ids)
                    if hasattr(outputs, 'last_hidden_state'):
                        # MoE model output
                        logits = model.lm_head(outputs.last_hidden_state)
                        loss = nn.CrossEntropyLoss()(
                            logits.view(-1, logits.size(-1)),
                            input_ids[:, 1:].contiguous().view(-1)
                        )
                    else:
                        # Standard model output
                        loss = outputs
                else:
                    loss = torch.tensor(0.0)
                
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.numel()
        
        eval_time = time.time() - start_time
        
        # Calculate metrics
        metrics["perplexity"] = torch.exp(torch.tensor(total_loss / len(eval_loader))).item()
        metrics["tokens_per_second"] = total_tokens / eval_time
        metrics["memory_usage"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        
        return metrics
    
    def _analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture and parameters."""
        stats = {}
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats["total_parameters"] = total_params
        stats["trainable_parameters"] = trainable_params
        stats["model_size_mb"] = total_params * 4 / 1e6  # Assuming float32
        
        # MoE-specific stats
        if hasattr(model, 'moe_layers'):
            stats["num_moe_layers"] = len(model.moe_layers)
            stats["num_experts"] = model.num_experts
            stats["experts_per_token"] = model.experts_per_token
            
            # Calculate active parameters
            active_params = 0
            for name, module in model.named_modules():
                if hasattr(module, 'experts'):
                    # Count parameters for experts_per_token experts
                    expert_params = sum(p.numel() for p in module.experts.experts[0].parameters())
                    active_params += expert_params * model.experts_per_token
                elif 'moe' not in name.lower():
                    # Count all non-MoE parameters
                    active_params += sum(p.numel() for p in module.parameters())
                    
            stats["active_parameters"] = active_params
            stats["sparsity_ratio"] = 1 - (active_params / total_params)
        
        return stats
    
    def _analyze_routing(self, model: nn.Module, eval_loader: DataLoader) -> Dict[str, Any]:
        """Analyze routing patterns in MoE model."""
        model.eval()
        routing_stats = {
            "expert_utilization": [],
            "load_variance": [],
            "entropy": []
        }
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch[0].to(self.config.device)
                outputs = model(input_ids, return_routing_info=True)
                
                if hasattr(outputs, 'routing_info') and outputs.routing_info is not None:
                    routing_info = outputs.routing_info
                    routing_stats["load_variance"].append(routing_info.load_variance)
                    routing_stats["entropy"].append(routing_info.entropy)
                    
                    # Expert utilization
                    if routing_info.selected_experts is not None:
                        expert_counts = torch.bincount(
                            routing_info.selected_experts.flatten(),
                            minlength=model.num_experts
                        ).float()
                        utilization = expert_counts / expert_counts.sum()
                        routing_stats["expert_utilization"].append(utilization.cpu().numpy())
        
        # Aggregate routing statistics
        analysis = {}
        if routing_stats["load_variance"]:
            analysis["avg_load_variance"] = np.mean(routing_stats["load_variance"])
            analysis["avg_entropy"] = np.mean(routing_stats["entropy"])
            
            # Expert utilization statistics
            all_utilization = np.stack(routing_stats["expert_utilization"])
            analysis["expert_utilization_mean"] = all_utilization.mean(axis=0).tolist()
            analysis["expert_utilization_std"] = all_utilization.std(axis=0).tolist()
            analysis["utilization_gini"] = self._calculate_gini_coefficient(all_utilization.mean(axis=0))
        
        return analysis
    
    def _calculate_gini_coefficient(self, utilization: np.ndarray) -> float:
        """Calculate Gini coefficient for expert utilization inequality."""
        if len(utilization) == 0:
            return 0.0
            
        sorted_util = np.sort(utilization)
        n = len(sorted_util)
        index = np.arange(1, n + 1)
        gini = 2 * np.sum(index * sorted_util) / (n * np.sum(sorted_util)) - (n + 1) / n
        return gini
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        if not self.results:
            return {}
            
        aggregated = {}
        
        # Aggregate metrics
        all_metrics = [result.metrics for result in self.results]
        metric_keys = all_metrics[0].keys()
        
        for key in metric_keys:
            values = [metrics[key] for metrics in all_metrics]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values
            }
        
        # Aggregate model statistics
        all_model_stats = [result.model_stats for result in self.results]
        model_stat_keys = all_model_stats[0].keys()
        
        aggregated["model_stats"] = {}
        for key in model_stat_keys:
            values = [stats[key] for stats in all_model_stats]
            if isinstance(values[0], (int, float)):
                aggregated["model_stats"][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
            else:
                aggregated["model_stats"][key] = values[0]  # Take first value for non-numeric
        
        return aggregated
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical tests on results."""
        if len(self.results) < 2:
            return {"note": "Statistical analysis requires at least 2 runs"}
            
        analysis = {}
        
        # Extract metric values
        metrics = {}
        for result in self.results:
            for key, value in result.metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Statistical tests for each metric
        for metric_name, values in metrics.items():
            if len(values) >= 3:  # Need at least 3 for meaningful statistics
                # Normality test
                _, normality_p = stats.shapiro(values)
                
                # Confidence interval
                confidence_interval = stats.t.interval(
                    self.config.confidence_level,
                    len(values) - 1,
                    loc=np.mean(values),
                    scale=stats.sem(values)
                )
                
                analysis[metric_name] = {
                    "normality_test_p_value": normality_p,
                    "is_normal": normality_p > 0.05,
                    "confidence_interval": confidence_interval,
                    "coefficient_of_variation": np.std(values) / np.mean(values)
                }
        
        return analysis
    
    def _save_intermediate_results(self, run_idx: int):
        """Save intermediate results after each run."""
        output_file = self.output_dir / f"run_{run_idx}_results.json"
        
        result_dict = asdict(self.results[run_idx])
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
    
    def _save_final_report(self, report: Dict[str, Any]):
        """Save final experiment report."""
        output_file = self.output_dir / f"{self.config.experiment_name}_final_report.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Final report saved to {output_file}")


class StatisticalValidator:
    """Statistical validation utilities for MoE experiments."""
    
    @staticmethod
    def compare_experiments(
        experiment_results_a: List[Dict[str, float]],
        experiment_results_b: List[Dict[str, float]], 
        metric: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compare two experiments statistically."""
        
        values_a = [result[metric] for result in experiment_results_a]
        values_b = [result[metric] for result in experiment_results_b]
        
        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) + 
                             (len(values_b) - 1) * np.var(values_b, ddof=1)) / 
                             (len(values_a) + len(values_b) - 2))
        cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
        
        return {
            "metric": metric,
            "t_test": {
                "statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < alpha
            },
            "effect_size": {
                "cohens_d": cohens_d,
                "magnitude": StatisticalValidator._interpret_cohens_d(cohens_d)
            },
            "mann_whitney": {
                "statistic": u_stat,
                "p_value": u_p_value,
                "significant": u_p_value < alpha
            },
            "summary": {
                "experiment_a_mean": np.mean(values_a),
                "experiment_b_mean": np.mean(values_b),
                "difference": np.mean(values_a) - np.mean(values_b),
                "relative_improvement": (np.mean(values_a) - np.mean(values_b)) / np.mean(values_b) * 100
            }
        }
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


class ResultsAnalyzer:
    """Analyze and visualize experimental results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
    
    def load_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment results from file."""
        results_file = self.results_dir / f"{experiment_name}_final_report.json"
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def compare_experiments(
        self,
        experiment_names: List[str],
        metrics: List[str],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple experiments."""
        
        # Load all experiment results
        all_results = {}
        for name in experiment_names:
            all_results[name] = self.load_experiment_results(name)
        
        # Compare each pair of experiments
        comparisons = {}
        for i, exp_a in enumerate(experiment_names):
            for j, exp_b in enumerate(experiment_names[i+1:], i+1):
                comparison_key = f"{exp_a}_vs_{exp_b}"
                comparisons[comparison_key] = {}
                
                for metric in metrics:
                    # Extract metric values
                    results_a = all_results[exp_a]["individual_results"]
                    results_b = all_results[exp_b]["individual_results"]
                    
                    comparison = StatisticalValidator.compare_experiments(
                        [r["metrics"] for r in results_a],
                        [r["metrics"] for r in results_b],
                        metric
                    )
                    
                    comparisons[comparison_key][metric] = comparison
        
        comparison_report = {
            "experiments": experiment_names,
            "metrics": metrics,
            "comparisons": comparisons,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison_report, f, indent=2, default=str)
        
        return comparison_report