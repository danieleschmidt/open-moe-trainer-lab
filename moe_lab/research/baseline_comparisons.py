"""Baseline model implementations for comparative studies."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..models.moe_model import MoEModel, MoEOutput
from ..models.router import RoutingInfo


@dataclass
class PerformanceMetrics:
    """Standard performance metrics for model comparison."""
    
    perplexity: float
    throughput: float  # tokens per second
    memory_usage: float  # GB
    flops: float  # FLOPs per token
    latency: float  # seconds per batch
    parameters: int  # total parameters
    active_parameters: Optional[int] = None  # for sparse models


class DenseBaseline(nn.Module):
    """Dense transformer baseline for comparison."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> MoEOutput:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            
        token_embeddings = self.embed_tokens(input_ids)
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = token_embeddings + position_embeddings
        
        # Forward through layers
        for layer in self.layers:
            if attention_mask is not None:
                src_key_padding_mask = (attention_mask == 0)
                hidden_states = layer(hidden_states, src_key_padding_mask=src_key_padding_mask)
            else:
                hidden_states = layer(hidden_states)
        
        # Final norm
        hidden_states = self.ln_f(hidden_states)
        
        # Create dummy routing info for compatibility
        dummy_routing_info = RoutingInfo(
            expert_weights=None,
            selected_experts=None,
            router_logits=None,
            load_variance=0.0,
            entropy=0.0
        )
        
        return MoEOutput(
            last_hidden_state=hidden_states,
            routing_info=dummy_routing_info,
            load_balancing_loss=None,
            router_z_loss=None,
            expert_weights=None
        )


class SwitchBaseline(MoEModel):
    """Switch Transformer baseline implementation."""
    
    def __init__(self, **kwargs):
        # Override defaults for Switch Transformer
        switch_defaults = {
            "experts_per_token": 1,  # Switch uses single expert
            "router_type": "switch",
            "aux_loss_coef": 0.01,
            "z_loss_coef": 0.001
        }
        switch_defaults.update(kwargs)
        super().__init__(**switch_defaults)


class MixtralBaseline(MoEModel):
    """Mixtral-style baseline implementation."""
    
    def __init__(self, **kwargs):
        # Override defaults for Mixtral
        mixtral_defaults = {
            "num_experts": 8,
            "experts_per_token": 2,
            "router_type": "top_k",
            "aux_loss_coef": 0.01,
            "z_loss_coef": 0.001
        }
        mixtral_defaults.update(kwargs)
        super().__init__(**mixtral_defaults)


class PerformanceBenchmark:
    """Benchmark suite for comparing model performance."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    def benchmark_model(
        self,
        model: nn.Module,
        dataloader,
        num_batches: int = 100,
        warmup_batches: int = 10
    ) -> PerformanceMetrics:
        """Comprehensive performance benchmark."""
        
        model.eval()
        model = model.to(self.device)
        
        # Warmup
        self._warmup(model, dataloader, warmup_batches)
        
        # Benchmark metrics
        perplexity = self._measure_perplexity(model, dataloader, num_batches)
        throughput = self._measure_throughput(model, dataloader, num_batches)
        memory_usage = self._measure_memory_usage(model, dataloader)
        flops = self._estimate_flops(model, next(iter(dataloader))[0])
        latency = self._measure_latency(model, dataloader, num_batches)
        
        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        active_params = self._count_active_parameters(model)
        
        return PerformanceMetrics(
            perplexity=perplexity,
            throughput=throughput,
            memory_usage=memory_usage,
            flops=flops,
            latency=latency,
            parameters=total_params,
            active_parameters=active_params
        )
    
    def _warmup(self, model: nn.Module, dataloader, num_batches: int):
        """Warmup runs to stabilize performance measurements."""
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                input_ids = batch[0].to(self.device)
                _ = model(input_ids)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    def _measure_perplexity(self, model: nn.Module, dataloader, num_batches: int) -> float:
        """Measure model perplexity."""
        total_loss = 0.0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                input_ids = batch[0].to(self.device)
                
                outputs = model(input_ids)
                logits = model.lm_head(outputs.last_hidden_state)
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                total_tokens += shift_labels.numel()
        
        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()
    
    def _measure_throughput(self, model: nn.Module, dataloader, num_batches: int) -> float:
        """Measure inference throughput (tokens/second)."""
        total_tokens = 0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                input_ids = batch[0].to(self.device)
                _ = model(input_ids)
                
                total_tokens += input_ids.numel()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        duration = end_time - start_time
        
        return total_tokens / duration
    
    def _measure_memory_usage(self, model: nn.Module, dataloader) -> float:
        """Measure peak memory usage during inference."""
        if not torch.cuda.is_available():
            return 0.0
            
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            input_ids = batch[0].to(self.device)
            _ = model(input_ids)
            torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
        return peak_memory
    
    def _estimate_flops(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Estimate FLOPs per token."""
        # Simplified FLOP counting - in practice would use profiling tools
        
        total_params = sum(p.numel() for p in model.parameters())
        batch_size, seq_len = sample_input.shape
        
        # Rough estimate: 2 * params * sequence_length for forward pass
        # This is a simplified approximation
        flops_per_token = 2 * total_params
        
        return flops_per_token
    
    def _measure_latency(self, model: nn.Module, dataloader, num_batches: int) -> float:
        """Measure average batch latency."""
        latencies = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                input_ids = batch[0].to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(input_ids)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latencies.append(end_time - start_time)
        
        return np.mean(latencies)
    
    def _count_active_parameters(self, model: nn.Module) -> Optional[int]:
        """Count active parameters for sparse models."""
        if hasattr(model, 'experts_per_token'):
            # MoE model - count active parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Estimate based on routing
            if hasattr(model, 'moe_layers') and model.moe_layers:
                moe_params = 0
                non_moe_params = 0
                
                for name, module in model.named_modules():
                    if hasattr(module, 'experts'):
                        # Count expert parameters
                        expert_params = sum(p.numel() for p in module.experts.parameters())
                        moe_params += expert_params
                    elif 'moe' not in name.lower():
                        # Count non-MoE parameters
                        non_moe_params += sum(p.numel() for p in module.parameters())
                
                # Active parameters = non-MoE + (experts_per_token / num_experts) * MoE
                if hasattr(model, 'num_experts'):
                    expert_utilization = model.experts_per_token / model.num_experts
                    active_moe_params = moe_params * expert_utilization
                    return int(non_moe_params + active_moe_params)
            
            return total_params
        else:
            # Dense model - all parameters are active
            return sum(p.numel() for p in model.parameters())


class BaselineComparison:
    """Compare experimental models against established baselines."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.benchmark = PerformanceBenchmark(device)
        
    def run_comparison(
        self,
        experimental_model: nn.Module,
        baseline_models: Dict[str, nn.Module],
        dataloader,
        num_batches: int = 100
    ) -> Dict[str, Any]:
        """Run comprehensive comparison between experimental and baseline models."""
        
        results = {}
        
        # Benchmark experimental model
        print("Benchmarking experimental model...")
        results["experimental"] = self.benchmark.benchmark_model(
            experimental_model, dataloader, num_batches
        )
        
        # Benchmark baseline models
        for name, model in baseline_models.items():
            print(f"Benchmarking {name} baseline...")
            results[name] = self.benchmark.benchmark_model(
                model, dataloader, num_batches
            )
        
        # Generate comparison analysis
        analysis = self._analyze_results(results)
        
        return {
            "results": results,
            "analysis": analysis,
            "benchmark_config": {
                "num_batches": num_batches,
                "device": self.device
            }
        }
    
    def _analyze_results(self, results: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze and compare benchmark results."""
        
        experimental_result = results["experimental"]
        baseline_results = {k: v for k, v in results.items() if k != "experimental"}
        
        analysis = {
            "improvements": {},
            "summary": {},
            "rankings": {}
        }
        
        # Calculate improvements over each baseline
        for baseline_name, baseline_result in baseline_results.items():
            improvements = {}
            
            # Perplexity (lower is better)
            perplexity_improvement = (baseline_result.perplexity - experimental_result.perplexity) / baseline_result.perplexity * 100
            improvements["perplexity"] = perplexity_improvement
            
            # Throughput (higher is better)  
            throughput_improvement = (experimental_result.throughput - baseline_result.throughput) / baseline_result.throughput * 100
            improvements["throughput"] = throughput_improvement
            
            # Memory efficiency (lower usage is better)
            memory_improvement = (baseline_result.memory_usage - experimental_result.memory_usage) / baseline_result.memory_usage * 100
            improvements["memory_efficiency"] = memory_improvement
            
            # Parameter efficiency
            if experimental_result.active_parameters and baseline_result.active_parameters:
                param_efficiency = (baseline_result.active_parameters - experimental_result.active_parameters) / baseline_result.active_parameters * 100
                improvements["parameter_efficiency"] = param_efficiency
            
            analysis["improvements"][baseline_name] = improvements
        
        # Overall summary
        all_models = list(results.keys())
        metrics = ["perplexity", "throughput", "memory_usage", "parameters"]
        
        for metric in metrics:
            values = [(name, getattr(results[name], metric)) for name in all_models]
            
            if metric in ["perplexity", "memory_usage", "parameters"]:
                # Lower is better
                ranked = sorted(values, key=lambda x: x[1])
            else:
                # Higher is better  
                ranked = sorted(values, key=lambda x: x[1], reverse=True)
            
            analysis["rankings"][metric] = [name for name, _ in ranked]
        
        # Summary statistics
        analysis["summary"] = {
            "experimental_rank_avg": np.mean([
                analysis["rankings"][metric].index("experimental") + 1 
                for metric in metrics
            ]),
            "best_metrics": [
                metric for metric in metrics 
                if analysis["rankings"][metric][0] == "experimental"
            ],
            "total_models_compared": len(all_models)
        }
        
        return analysis


class RouterComparison:
    """Compare different routing algorithms."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    def compare_routers(
        self,
        base_model_config: Dict[str, Any],
        router_configs: Dict[str, Dict[str, Any]],
        dataloader,
        num_batches: int = 50
    ) -> Dict[str, Any]:
        """Compare different routing algorithms on the same base architecture."""
        
        results = {}
        
        for router_name, router_config in router_configs.items():
            print(f"Testing {router_name} router...")
            
            # Create model with specific router
            model_config = base_model_config.copy()
            model_config.update(router_config)
            
            model = MoEModel(**model_config).to(self.device)
            
            # Benchmark routing-specific metrics
            routing_metrics = self._measure_routing_metrics(model, dataloader, num_batches)
            
            # Standard performance metrics
            benchmark = PerformanceBenchmark(self.device)
            performance_metrics = benchmark.benchmark_model(model, dataloader, num_batches)
            
            results[router_name] = {
                "routing_metrics": routing_metrics,
                "performance_metrics": performance_metrics
            }
        
        # Analyze routing comparison
        analysis = self._analyze_routing_results(results)
        
        return {
            "results": results,
            "analysis": analysis,
            "base_config": base_model_config
        }
    
    def _measure_routing_metrics(
        self,
        model: nn.Module,
        dataloader,
        num_batches: int
    ) -> Dict[str, float]:
        """Measure routing-specific metrics."""
        
        model.eval()
        
        routing_stats = {
            "load_variances": [],
            "entropies": [],
            "utilizations": []
        }
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                input_ids = batch[0].to(self.device)
                outputs = model(input_ids, return_routing_info=True)
                
                if hasattr(outputs, 'routing_info') and outputs.routing_info:
                    routing_info = outputs.routing_info
                    routing_stats["load_variances"].append(routing_info.load_variance)
                    routing_stats["entropies"].append(routing_info.entropy)
                    
                    # Expert utilization
                    if routing_info.selected_experts is not None:
                        expert_counts = torch.bincount(
                            routing_info.selected_experts.flatten(),
                            minlength=model.num_experts
                        ).float()
                        utilization = expert_counts / expert_counts.sum()
                        routing_stats["utilizations"].append(utilization.cpu().numpy())
        
        # Aggregate metrics
        metrics = {}
        if routing_stats["load_variances"]:
            metrics["avg_load_variance"] = np.mean(routing_stats["load_variances"])
            metrics["avg_entropy"] = np.mean(routing_stats["entropies"])
            
            # Utilization analysis
            all_utilizations = np.stack(routing_stats["utilizations"])
            metrics["utilization_gini"] = self._calculate_gini(all_utilizations.mean(axis=0))
            metrics["utilization_std"] = all_utilizations.std()
            
            # Load balancing quality (lower variance is better)
            metrics["load_balancing_score"] = 1.0 / (1.0 + metrics["avg_load_variance"])
        
        return metrics
    
    def _calculate_gini(self, utilization: np.ndarray) -> float:
        """Calculate Gini coefficient for expert utilization."""
        if len(utilization) == 0:
            return 0.0
            
        sorted_util = np.sort(utilization)
        n = len(sorted_util)
        index = np.arange(1, n + 1)
        gini = 2 * np.sum(index * sorted_util) / (n * np.sum(sorted_util)) - (n + 1) / n
        return gini
    
    def _analyze_routing_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze routing comparison results."""
        
        analysis = {
            "best_routers": {},
            "rankings": {},
            "trade_offs": {}
        }
        
        # Extract metrics for comparison
        router_names = list(results.keys())
        
        routing_metrics = [
            "avg_load_variance",
            "avg_entropy", 
            "utilization_gini",
            "load_balancing_score"
        ]
        
        performance_metrics = [
            "perplexity",
            "throughput",
            "memory_usage"
        ]
        
        # Rank routers for each metric
        for metric in routing_metrics + performance_metrics:
            values = []
            for router_name in router_names:
                if metric in routing_metrics:
                    value = results[router_name]["routing_metrics"].get(metric, float('inf'))
                else:
                    value = getattr(results[router_name]["performance_metrics"], metric, float('inf'))
                values.append((router_name, value))
            
            # Sort based on metric (some are lower-is-better, others higher-is-better)
            if metric in ["avg_load_variance", "utilization_gini", "perplexity", "memory_usage"]:
                ranked = sorted(values, key=lambda x: x[1])  # Lower is better
            else:
                ranked = sorted(values, key=lambda x: x[1], reverse=True)  # Higher is better
            
            analysis["rankings"][metric] = [name for name, _ in ranked]
            analysis["best_routers"][metric] = ranked[0][0]
        
        # Identify trade-offs
        for router_name in router_names:
            routing_rank = np.mean([
                analysis["rankings"][metric].index(router_name) 
                for metric in routing_metrics if metric in analysis["rankings"]
            ])
            
            performance_rank = np.mean([
                analysis["rankings"][metric].index(router_name)
                for metric in performance_metrics if metric in analysis["rankings"]
            ])
            
            analysis["trade_offs"][router_name] = {
                "routing_rank": routing_rank,
                "performance_rank": performance_rank,
                "overall_rank": (routing_rank + performance_rank) / 2
            }
        
        return analysis