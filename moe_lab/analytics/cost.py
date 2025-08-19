"""Cost analysis for MoE models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

from ..models import MoEModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CostReport:
    """Cost analysis report for MoE model."""
    
    flops_per_token: float
    memory_bandwidth_gb: float
    throughput: float
    compute_reduction: float
    memory_usage_mb: float
    inference_latency_ms: float
    
    
@dataclass
class HardwareProfile:
    """Hardware configuration profile."""
    
    name: str
    compute_tflops: float  # Peak compute in TFLOPS
    memory_bandwidth_gbps: float  # Memory bandwidth in GB/s
    memory_capacity_gb: float  # Total memory in GB
    

class MoECostAnalyzer:
    """Analyzer for MoE model computational costs and efficiency."""
    
    HARDWARE_PROFILES = {
        'a100_80gb': HardwareProfile(
            name='A100 80GB',
            compute_tflops=312.0,  # FP16 tensor ops
            memory_bandwidth_gbps=2000.0,
            memory_capacity_gb=80.0
        ),
        'v100_32gb': HardwareProfile(
            name='V100 32GB', 
            compute_tflops=125.0,
            memory_bandwidth_gbps=900.0,
            memory_capacity_gb=32.0
        ),
        'h100_80gb': HardwareProfile(
            name='H100 80GB',
            compute_tflops=989.0,  # FP16 tensor ops
            memory_bandwidth_gbps=3350.0,
            memory_capacity_gb=80.0
        ),
        'cpu_server': HardwareProfile(
            name='CPU Server',
            compute_tflops=2.0,  # Rough estimate
            memory_bandwidth_gbps=200.0,
            memory_capacity_gb=256.0
        )
    }
    
    def __init__(
        self, 
        model: MoEModel,
        hardware_profile: str = "a100_80gb"
    ):
        self.model = model
        
        if hardware_profile not in self.HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware profile: {hardware_profile}")
            
        self.hardware = self.HARDWARE_PROFILES[hardware_profile]
        
        # Model statistics
        self.model_stats = self._compute_model_stats()
        
    def _compute_model_stats(self) -> Dict[str, Any]:
        """Compute model parameter and architecture statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Count expert parameters
        expert_params = 0
        router_params = 0
        other_params = 0
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            if 'expert' in name.lower():
                expert_params += param_count
            elif 'router' in name.lower():
                router_params += param_count
            else:
                other_params += param_count
                
        # Active parameters (assuming top-k routing)
        active_experts_per_layer = self.model.experts_per_token
        num_moe_layers = len(self.model.moe_layers)
        
        # Estimate active parameters during inference
        if num_moe_layers > 0:
            params_per_expert = expert_params // (self.model.num_experts * num_moe_layers)
            active_expert_params = params_per_expert * active_experts_per_layer * num_moe_layers
        else:
            active_expert_params = 0
            
        active_params = other_params + router_params + active_expert_params
        
        return {
            'total_parameters': total_params,
            'expert_parameters': expert_params,
            'router_parameters': router_params,
            'other_parameters': other_params,
            'active_parameters': active_params,
            'sparsity_ratio': 1.0 - (active_params / total_params) if total_params > 0 else 0.0,
            'num_experts': self.model.num_experts,
            'experts_per_token': self.model.experts_per_token,
            'num_moe_layers': num_moe_layers
        }
        
    def analyze(
        self, 
        batch_size: int = 1,
        sequence_length: int = 512,
        precision: str = "fp16"
    ) -> CostReport:
        """Analyze computational costs for given input configuration."""
        
        # Compute FLOPs
        flops_per_token = self._compute_flops_per_token(sequence_length, precision)
        total_flops = flops_per_token * batch_size * sequence_length
        
        # Memory analysis
        memory_usage = self._compute_memory_usage(batch_size, sequence_length, precision)
        memory_bandwidth_required = self._compute_memory_bandwidth(
            batch_size, sequence_length, precision
        )
        
        # Throughput estimation
        compute_throughput = self._estimate_compute_throughput(total_flops)
        memory_throughput = self._estimate_memory_throughput(memory_bandwidth_required)
        
        # Overall throughput is limited by the bottleneck
        throughput = min(compute_throughput, memory_throughput)
        
        # Latency estimation
        inference_latency = self._estimate_latency(batch_size, sequence_length, precision)
        
        # Compute reduction vs dense model
        compute_reduction = self._compute_sparsity_benefit()
        
        return CostReport(
            flops_per_token=flops_per_token,
            memory_bandwidth_gb=memory_bandwidth_required,
            throughput=throughput,
            compute_reduction=compute_reduction,
            memory_usage_mb=memory_usage,
            inference_latency_ms=inference_latency
        )
        
    def _compute_flops_per_token(self, sequence_length: int, precision: str) -> float:
        """Compute FLOPs per token for MoE model."""
        # Base transformer FLOPs (attention + non-MoE layers)
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers
        vocab_size = getattr(self.model, 'vocab_size', 32000)
        
        # Attention FLOPs per layer
        # QKV projection: 3 * seq_len * hidden_size^2
        # Attention computation: 2 * seq_len^2 * hidden_size  
        # Output projection: seq_len * hidden_size^2
        attention_flops_per_layer = (
            3 * sequence_length * hidden_size * hidden_size +  # QKV
            2 * sequence_length * sequence_length * hidden_size +  # Attention
            sequence_length * hidden_size * hidden_size  # Output projection
        )
        
        total_attention_flops = attention_flops_per_layer * num_layers
        
        # MoE layer FLOPs
        moe_flops = 0
        if self.model_stats['num_moe_layers'] > 0:
            # Router FLOPs: hidden_size * num_experts
            router_flops_per_layer = hidden_size * self.model.num_experts
            
            # Expert FLOPs (only for active experts)
            expert_hidden_size = hidden_size * 4  # Typical FFN expansion
            expert_flops_per_active = (
                2 * hidden_size * expert_hidden_size  # Two linear layers
            )
            
            active_expert_flops_per_layer = (
                expert_flops_per_active * self.model.experts_per_token
            )
            
            moe_flops_per_layer = router_flops_per_layer + active_expert_flops_per_layer
            moe_flops = moe_flops_per_layer * self.model_stats['num_moe_layers']
            
        # Non-MoE FFN layers
        non_moe_layers = num_layers - self.model_stats['num_moe_layers']
        ffn_flops_per_layer = 2 * sequence_length * hidden_size * (hidden_size * 4)
        non_moe_flops = ffn_flops_per_layer * non_moe_layers
        
        # Embedding and output projection
        embedding_flops = sequence_length * hidden_size * vocab_size
        
        total_flops = (
            total_attention_flops + 
            moe_flops + 
            non_moe_flops + 
            embedding_flops
        )
        
        return total_flops / sequence_length  # FLOPs per token
        
    def _compute_memory_usage(
        self, 
        batch_size: int, 
        sequence_length: int, 
        precision: str
    ) -> float:
        """Compute memory usage in MB."""
        bytes_per_param = 2 if precision == "fp16" else 4  # fp32
        
        # Model parameters (only active ones are loaded for MoE)
        model_memory = self.model_stats['active_parameters'] * bytes_per_param
        
        # Activations memory
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers
        
        # Rough estimate of activation memory
        activation_memory = (
            batch_size * sequence_length * hidden_size * num_layers * bytes_per_param * 4
        )  # Factor of 4 for intermediate activations
        
        # KV cache for generation
        kv_cache_memory = (
            2 * batch_size * sequence_length * hidden_size * num_layers * bytes_per_param
        )
        
        total_memory = model_memory + activation_memory + kv_cache_memory
        
        return total_memory / (1024 * 1024)  # Convert to MB
        
    def _compute_memory_bandwidth(
        self, 
        batch_size: int, 
        sequence_length: int, 
        precision: str
    ) -> float:
        """Compute memory bandwidth requirements in GB/s."""
        bytes_per_param = 2 if precision == "fp16" else 4
        
        # Weight loading for active parameters
        weight_bandwidth = self.model_stats['active_parameters'] * bytes_per_param
        
        # Activation bandwidth (rough estimate)
        hidden_size = self.model.hidden_size
        activation_bandwidth = (
            batch_size * sequence_length * hidden_size * 
            self.model.num_layers * bytes_per_param * 2  # Read + write
        )
        
        total_bandwidth = weight_bandwidth + activation_bandwidth
        
        return total_bandwidth / (1024**3)  # Convert to GB
        
    def _estimate_compute_throughput(self, total_flops: float) -> float:
        """Estimate throughput based on compute constraints."""
        # Convert TFLOPS to FLOPS
        peak_flops = self.hardware.compute_tflops * 1e12
        
        # Assume 50% efficiency for real workloads
        effective_flops = peak_flops * 0.5
        
        # Tokens per second based on compute
        return effective_flops / total_flops
        
    def _estimate_memory_throughput(self, memory_bandwidth_gb: float) -> float:
        """Estimate throughput based on memory constraints."""
        # Assume 80% memory bandwidth utilization
        effective_bandwidth = self.hardware.memory_bandwidth_gbps * 0.8
        
        # Simple throughput estimation
        if memory_bandwidth_gb > 0:
            return effective_bandwidth / memory_bandwidth_gb
        else:
            return float('inf')
            
    def _estimate_latency(
        self, 
        batch_size: int, 
        sequence_length: int, 
        precision: str
    ) -> float:
        """Estimate inference latency in milliseconds."""
        # Simple latency model based on compute and memory
        flops_per_token = self._compute_flops_per_token(sequence_length, precision)
        total_flops = flops_per_token * batch_size * sequence_length
        
        # Compute time
        effective_flops = self.hardware.compute_tflops * 1e12 * 0.5
        compute_time = total_flops / effective_flops
        
        # Memory access time
        memory_bandwidth = self._compute_memory_bandwidth(batch_size, sequence_length, precision)
        memory_time = memory_bandwidth / (self.hardware.memory_bandwidth_gbps * 0.8)
        
        # Total latency (dominated by the slower of compute or memory)
        latency_seconds = max(compute_time, memory_time)
        
        return latency_seconds * 1000  # Convert to ms
        
    def _compute_sparsity_benefit(self) -> float:
        """Compute computational benefit from sparsity."""
        return self.model_stats['sparsity_ratio']
        
    def compare_with_dense(
        self, 
        hidden_size: int,
        num_layers: int,
        batch_size: int = 1,
        sequence_length: int = 512
    ) -> Dict[str, float]:
        """Compare MoE model with equivalent dense model."""
        
        # Dense model FLOPs estimation
        dense_flops_per_token = self._compute_dense_flops(
            hidden_size, num_layers, sequence_length
        )
        
        # MoE model FLOPs
        moe_flops_per_token = self._compute_flops_per_token(sequence_length, "fp16")
        
        # Compute reduction
        compute_reduction = 1.0 - (moe_flops_per_token / dense_flops_per_token)
        
        # Parameter comparison
        dense_params = self._estimate_dense_parameters(hidden_size, num_layers)
        param_reduction = 1.0 - (self.model_stats['active_parameters'] / dense_params)
        
        return {
            'compute_reduction': compute_reduction,
            'parameter_reduction': param_reduction,
            'dense_flops_per_token': dense_flops_per_token,
            'moe_flops_per_token': moe_flops_per_token,
            'dense_parameters': dense_params,
            'moe_active_parameters': self.model_stats['active_parameters']
        }
        
    def _compute_dense_flops(
        self, 
        hidden_size: int, 
        num_layers: int, 
        sequence_length: int
    ) -> float:
        """Compute FLOPs per token for equivalent dense model."""
        vocab_size = 32000  # Default vocabulary size
        
        # Attention FLOPs per layer (same as MoE)
        attention_flops_per_layer = (
            3 * sequence_length * hidden_size * hidden_size +
            2 * sequence_length * sequence_length * hidden_size +
            sequence_length * hidden_size * hidden_size
        )
        
        # FFN FLOPs per layer
        ffn_hidden_size = hidden_size * 4
        ffn_flops_per_layer = 2 * sequence_length * hidden_size * ffn_hidden_size
        
        # Total per layer
        total_flops_per_layer = attention_flops_per_layer + ffn_flops_per_layer
        
        # All layers + embeddings
        total_flops = (
            total_flops_per_layer * num_layers +
            sequence_length * hidden_size * vocab_size
        )
        
        return total_flops / sequence_length
        
    def _estimate_dense_parameters(self, hidden_size: int, num_layers: int) -> int:
        """Estimate parameters for equivalent dense model."""
        vocab_size = 32000
        
        # Embedding parameters
        embedding_params = vocab_size * hidden_size * 2  # Input + output embeddings
        
        # Per layer parameters
        # Attention: 4 * hidden_size^2 (QKV + output projection)
        # FFN: 2 * hidden_size * (hidden_size * 4)
        # Layer norms: 2 * hidden_size (attention + FFN)
        params_per_layer = (
            4 * hidden_size * hidden_size +  # Attention
            2 * hidden_size * (hidden_size * 4) +  # FFN
            2 * hidden_size  # Layer norms
        )
        
        total_params = embedding_params + params_per_layer * num_layers
        
        return total_params
        
    def profile_batch_sizes(
        self, 
        batch_sizes: List[int],
        sequence_length: int = 512
    ) -> Dict[int, CostReport]:
        """Profile costs across different batch sizes."""
        results = {}
        
        for batch_size in batch_sizes:
            try:
                cost_report = self.analyze(
                    batch_size=batch_size,
                    sequence_length=sequence_length
                )
                results[batch_size] = cost_report
            except Exception as e:
                logger.warning(f"Error profiling batch size {batch_size}: {e}")
                
        return results
        
    def generate_cost_report(self, save_to: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive cost analysis report."""
        
        # Test different configurations
        test_configs = [
            {'batch_size': 1, 'sequence_length': 512},
            {'batch_size': 8, 'sequence_length': 1024},
            {'batch_size': 32, 'sequence_length': 2048}
        ]
        
        results = {}
        for i, config in enumerate(test_configs):
            config_name = f"config_{i+1}_b{config['batch_size']}_s{config['sequence_length']}"
            results[config_name] = self.analyze(**config)
            
        # Dense model comparison
        dense_comparison = self.compare_with_dense(
            hidden_size=self.model.hidden_size,
            num_layers=self.model.num_layers
        )
        
        report = {
            'model_statistics': self.model_stats,
            'hardware_profile': {
                'name': self.hardware.name,
                'compute_tflops': self.hardware.compute_tflops,
                'memory_bandwidth_gbps': self.hardware.memory_bandwidth_gbps,
                'memory_capacity_gb': self.hardware.memory_capacity_gb
            },
            'cost_analysis': results,
            'dense_model_comparison': dense_comparison
        }
        
        if save_to:
            import json
            with open(save_to, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Cost analysis report saved to {save_to}")
            
        return report