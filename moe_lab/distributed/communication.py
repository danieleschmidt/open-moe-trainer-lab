"""Optimized communication patterns for distributed MoE training."""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class CommunicationStats:
    """Statistics for communication operations."""
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_ops: int = 0
    average_latency_ms: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    compression_ratio: float = 1.0


class ExpertCommunicator:
    """
    Optimized communication for expert-parallel MoE training.
    
    Handles efficient routing of tokens to experts across different devices/nodes
    with support for overlapping computation and communication.
    """
    
    def __init__(self, expert_parallel_size: int, expert_parallel_rank: int, 
                 enable_compression: bool = False, overlap_comm: bool = True):
        self.expert_parallel_size = expert_parallel_size
        self.expert_parallel_rank = expert_parallel_rank
        self.enable_compression = enable_compression
        self.overlap_comm = overlap_comm
        
        # Communication statistics
        self.stats = CommunicationStats()
        
        # Setup process group
        self.expert_group = None
        self._setup_process_groups()
        
        # Compression codec
        self.compression_codec = None
        if enable_compression:
            self._setup_compression()
        
        # Async communication executor
        self.comm_executor = ThreadPoolExecutor(max_workers=4) if overlap_comm else None
        
        logger.info(f"Initialized ExpertCommunicator: rank={expert_parallel_rank}, "
                   f"size={expert_parallel_size}, compression={enable_compression}")
    
    def _setup_process_groups(self):
        """Setup process groups for expert parallelism."""
        if not dist.is_initialized():
            logger.warning("Distributed not initialized, skipping process group setup")
            return
        
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
        
        # Create expert parallel groups
        expert_groups = []
        for i in range(0, world_size, self.expert_parallel_size):
            group_ranks = list(range(i, min(i + self.expert_parallel_size, world_size)))
            group = dist.new_group(ranks=group_ranks)
            expert_groups.append(group)
            
            # Set our group if we belong to it
            if world_rank in group_ranks:
                self.expert_group = group
        
        logger.info(f"Setup expert parallel groups: {len(expert_groups)} groups")
    
    def _setup_compression(self):
        """Setup gradient/activation compression."""
        try:
            # Simple quantization-based compression
            class QuantizationCodec:
                def __init__(self, bits: int = 8):
                    self.bits = bits
                    self.scale_factor = (2 ** bits - 1)
                
                def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    """Compress tensor using quantization."""
                    # Compute min/max for quantization
                    min_val = tensor.min()
                    max_val = tensor.max()
                    
                    # Quantize
                    scale = (max_val - min_val) / self.scale_factor
                    quantized = ((tensor - min_val) / scale).round().byte()
                    
                    return quantized, min_val, scale
                
                def decompress(self, quantized: torch.Tensor, min_val: torch.Tensor, 
                             scale: torch.Tensor) -> torch.Tensor:
                    """Decompress quantized tensor."""
                    return quantized.float() * scale + min_val
            
            self.compression_codec = QuantizationCodec()
            logger.info("Setup quantization compression codec")
            
        except Exception as e:
            logger.warning(f"Failed to setup compression: {e}")
            self.enable_compression = False
    
    def all_to_all_expert_dispatch(self, tokens: torch.Tensor, expert_assignments: torch.Tensor,
                                 num_experts_per_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient all-to-all dispatch of tokens to experts across ranks.
        
        Args:
            tokens: Input tokens [batch_size, seq_len, hidden_size]
            expert_assignments: Expert assignments for each token [batch_size, seq_len, top_k]
            num_experts_per_rank: Number of experts per rank
            
        Returns:
            Tuple of (dispatched_tokens, routing_info)
        """
        if not dist.is_initialized() or self.expert_group is None:
            # Fallback for single-process training
            return tokens, expert_assignments
        
        batch_size, seq_len, hidden_size = tokens.shape
        total_tokens = batch_size * seq_len
        
        # Flatten tokens and assignments
        flat_tokens = tokens.view(total_tokens, hidden_size)
        flat_assignments = expert_assignments.view(total_tokens, -1)
        
        # Prepare send buffers for each rank
        send_buffers = []
        send_counts = []
        
        for rank in range(self.expert_parallel_size):
            # Find tokens assigned to experts on this rank
            expert_start = rank * num_experts_per_rank
            expert_end = (rank + 1) * num_experts_per_rank
            
            # Create mask for tokens going to this rank
            rank_mask = ((flat_assignments >= expert_start) & 
                        (flat_assignments < expert_end)).any(dim=1)
            
            rank_tokens = flat_tokens[rank_mask]
            send_buffers.append(rank_tokens)
            send_counts.append(rank_tokens.shape[0])
        
        # All-to-all communication
        if self.overlap_comm and self.comm_executor:
            # Asynchronous communication
            future = self.comm_executor.submit(
                self._async_all_to_all, send_buffers, send_counts, hidden_size
            )
            received_tokens = future.result()
        else:
            # Synchronous communication
            received_tokens = self._sync_all_to_all(send_buffers, send_counts, hidden_size)
        
        # Update statistics
        total_sent = sum(buf.numel() * 4 for buf in send_buffers)  # 4 bytes per float32
        self.stats.total_bytes_sent += total_sent
        self.stats.total_ops += 1
        
        return received_tokens, expert_assignments
    
    def _sync_all_to_all(self, send_buffers: List[torch.Tensor], send_counts: List[int],
                        hidden_size: int) -> torch.Tensor:
        """Synchronous all-to-all communication."""
        # Gather send counts from all ranks
        all_send_counts = [torch.zeros(self.expert_parallel_size, dtype=torch.int32) 
                          for _ in range(self.expert_parallel_size)]
        
        send_count_tensor = torch.tensor(send_counts, dtype=torch.int32)
        dist.all_gather(all_send_counts, send_count_tensor, group=self.expert_group)
        
        # Calculate receive counts
        receive_counts = [counts[self.expert_parallel_rank].item() for counts in all_send_counts]
        total_receive = sum(receive_counts)
        
        # Prepare receive buffer
        receive_buffer = torch.zeros(total_receive, hidden_size, dtype=torch.float32, 
                                   device=send_buffers[0].device if send_buffers else torch.device('cpu'))
        
        # Perform all-to-all
        if total_receive > 0:
            # Concatenate send buffers
            if send_buffers:
                send_tensor = torch.cat(send_buffers, dim=0)
            else:
                send_tensor = torch.zeros(0, hidden_size, dtype=torch.float32)
            
            # All-to-all scatter
            dist.all_to_all_single(receive_buffer, send_tensor, 
                                 output_split_sizes=receive_counts,
                                 input_split_sizes=send_counts,
                                 group=self.expert_group)
        
        return receive_buffer
    
    def _async_all_to_all(self, send_buffers: List[torch.Tensor], send_counts: List[int],
                         hidden_size: int) -> torch.Tensor:
        """Asynchronous all-to-all communication with overlap."""
        # For now, use synchronous implementation
        # In practice, this would use non-blocking operations
        return self._sync_all_to_all(send_buffers, send_counts, hidden_size)
    
    def gather_expert_outputs(self, expert_outputs: torch.Tensor, 
                            original_token_indices: torch.Tensor) -> torch.Tensor:
        """
        Gather expert outputs back to original token positions.
        
        Args:
            expert_outputs: Outputs from local experts
            original_token_indices: Original indices of tokens
            
        Returns:
            Gathered outputs in original order
        """
        if not dist.is_initialized() or self.expert_group is None:
            return expert_outputs
        
        # All-gather outputs from all ranks
        gathered_outputs = [torch.zeros_like(expert_outputs) 
                           for _ in range(self.expert_parallel_size)]
        
        dist.all_gather(gathered_outputs, expert_outputs, group=self.expert_group)
        
        # Concatenate and reorder
        all_outputs = torch.cat(gathered_outputs, dim=0)
        
        # Reorder based on original indices
        if original_token_indices is not None:
            sorted_indices = torch.argsort(original_token_indices)
            all_outputs = all_outputs[sorted_indices]
        
        return all_outputs
    
    def broadcast_router_weights(self, router_weights: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """Broadcast router weights from source rank."""
        if not dist.is_initialized() or self.expert_group is None:
            return router_weights
        
        dist.broadcast(router_weights, src_rank, group=self.expert_group)
        return router_weights
    
    def reduce_load_balancing_loss(self, local_loss: torch.Tensor) -> torch.Tensor:
        """Reduce load balancing loss across expert parallel group."""
        if not dist.is_initialized() or self.expert_group is None:
            return local_loss
        
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM, group=self.expert_group)
        local_loss /= self.expert_parallel_size
        
        return local_loss
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            'total_bytes_sent': self.stats.total_bytes_sent,
            'total_bytes_received': self.stats.total_bytes_received,
            'total_ops': self.stats.total_ops,
            'average_latency_ms': self.stats.average_latency_ms,
            'peak_bandwidth_gbps': self.stats.peak_bandwidth_gbps,
            'compression_ratio': self.stats.compression_ratio
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.comm_executor:
            self.comm_executor.shutdown(wait=True)
        logger.info("Cleaned up ExpertCommunicator")


class AllToAllExpertDispatch:
    """
    Optimized all-to-all expert dispatch with batching and pipelining.
    
    This class implements advanced communication patterns for efficiently
    dispatching tokens to experts in a distributed setting.
    """
    
    def __init__(self, world_size: int, expert_parallel_size: int, 
                 batch_tokens: bool = True, pipeline_stages: int = 1):
        self.world_size = world_size
        self.expert_parallel_size = expert_parallel_size
        self.batch_tokens = batch_tokens
        self.pipeline_stages = pipeline_stages
        
        # Communication buffers
        self.send_buffers = {}
        self.recv_buffers = {}
        
        # Performance tracking
        self.dispatch_times = []
        self.bandwidth_utilization = []
        
        logger.info(f"Initialized AllToAllExpertDispatch with {pipeline_stages} pipeline stages")
    
    def dispatch_tokens(self, tokens: torch.Tensor, expert_ids: torch.Tensor,
                       expert_weights: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Dispatch tokens to appropriate expert ranks.
        
        Args:
            tokens: Input tokens [num_tokens, hidden_size]
            expert_ids: Expert IDs for each token [num_tokens, top_k]
            expert_weights: Expert weights for each token [num_tokens, top_k]
            
        Returns:
            Dictionary mapping expert_rank -> (expert_tokens, expert_weights)
        """
        if self.pipeline_stages > 1:
            return self._pipelined_dispatch(tokens, expert_ids, expert_weights)
        else:
            return self._single_stage_dispatch(tokens, expert_ids, expert_weights)
    
    def _single_stage_dispatch(self, tokens: torch.Tensor, expert_ids: torch.Tensor,
                              expert_weights: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Single-stage token dispatch."""
        import time
        start_time = time.time()
        
        dispatched_tokens = {}
        num_tokens, hidden_size = tokens.shape
        top_k = expert_ids.shape[1]
        
        # Group tokens by target rank
        for rank in range(self.expert_parallel_size):
            rank_tokens = []
            rank_weights = []
            
            for token_idx in range(num_tokens):
                for k in range(top_k):
                    expert_id = expert_ids[token_idx, k].item()
                    expert_rank = expert_id // (self.world_size // self.expert_parallel_size)
                    
                    if expert_rank == rank:
                        rank_tokens.append(tokens[token_idx])
                        rank_weights.append(expert_weights[token_idx, k])
            
            if rank_tokens:
                dispatched_tokens[rank] = (
                    torch.stack(rank_tokens),
                    torch.tensor(rank_weights, device=tokens.device)
                )
        
        # Track performance
        dispatch_time = time.time() - start_time
        self.dispatch_times.append(dispatch_time)
        
        return dispatched_tokens
    
    def _pipelined_dispatch(self, tokens: torch.Tensor, expert_ids: torch.Tensor,
                           expert_weights: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Pipelined token dispatch for better throughput."""
        # Split tokens into pipeline stages
        num_tokens = tokens.shape[0]
        tokens_per_stage = num_tokens // self.pipeline_stages
        
        dispatched_tokens = defaultdict(list)
        
        for stage in range(self.pipeline_stages):
            start_idx = stage * tokens_per_stage
            end_idx = (stage + 1) * tokens_per_stage if stage < self.pipeline_stages - 1 else num_tokens
            
            stage_tokens = tokens[start_idx:end_idx]
            stage_expert_ids = expert_ids[start_idx:end_idx]
            stage_expert_weights = expert_weights[start_idx:end_idx]
            
            stage_dispatch = self._single_stage_dispatch(stage_tokens, stage_expert_ids, stage_expert_weights)
            
            # Accumulate results
            for rank, (rank_tokens, rank_weights) in stage_dispatch.items():
                dispatched_tokens[rank].append((rank_tokens, rank_weights))
        
        # Concatenate results from all stages
        final_dispatch = {}
        for rank, stage_results in dispatched_tokens.items():
            all_tokens = torch.cat([result[0] for result in stage_results], dim=0)
            all_weights = torch.cat([result[1] for result in stage_results], dim=0)
            final_dispatch[rank] = (all_tokens, all_weights)
        
        return final_dispatch
    
    def estimate_communication_cost(self, tokens: torch.Tensor, expert_ids: torch.Tensor) -> Dict[str, float]:
        """Estimate communication cost for the dispatch operation."""
        num_tokens, hidden_size = tokens.shape
        bytes_per_token = hidden_size * 4  # float32
        
        # Count tokens going to each rank
        rank_token_counts = defaultdict(int)
        for token_idx in range(num_tokens):
            for k in range(expert_ids.shape[1]):
                expert_id = expert_ids[token_idx, k].item()
                expert_rank = expert_id // (self.world_size // self.expert_parallel_size)
                rank_token_counts[expert_rank] += 1
        
        total_bytes = sum(count * bytes_per_token for count in rank_token_counts.values())
        max_bytes_per_rank = max(rank_token_counts.values()) * bytes_per_token if rank_token_counts else 0
        
        return {
            'total_communication_bytes': total_bytes,
            'max_bytes_per_rank': max_bytes_per_rank,
            'load_imbalance_ratio': max_bytes_per_rank / (total_bytes / len(rank_token_counts)) if rank_token_counts else 1.0,
            'estimated_latency_ms': total_bytes / (1e9) * 1000,  # Assuming 1 GB/s bandwidth
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.dispatch_times:
            return {}
        
        return {
            'average_dispatch_time_ms': np.mean(self.dispatch_times) * 1000,
            'min_dispatch_time_ms': np.min(self.dispatch_times) * 1000,
            'max_dispatch_time_ms': np.max(self.dispatch_times) * 1000,
            'total_dispatches': len(self.dispatch_times),
            'average_bandwidth_utilization': np.mean(self.bandwidth_utilization) if self.bandwidth_utilization else 0.0
        }


class CommunicationProfiler:
    """Profile communication patterns and bottlenecks."""
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.bandwidth_usage = defaultdict(list)
        self.message_sizes = defaultdict(list)
    
    def profile_operation(self, operation_name: str):
        """Context manager to profile communication operations."""
        import time
        from contextlib import contextmanager
        
        @contextmanager
        def profiler():
            start_time = time.time()
            yield
            end_time = time.time()
            self.operation_times[operation_name].append(end_time - start_time)
        
        return profiler()
    
    def record_bandwidth(self, operation_name: str, bytes_transferred: int, duration: float):
        """Record bandwidth usage for an operation."""
        bandwidth_gbps = (bytes_transferred * 8) / (duration * 1e9)  # Convert to Gbps
        self.bandwidth_usage[operation_name].append(bandwidth_gbps)
        self.message_sizes[operation_name].append(bytes_transferred)
    
    def get_profile_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report."""
        report = {
            'operations': {},
            'summary': {
                'total_operations': sum(len(times) for times in self.operation_times.values()),
                'most_time_consuming': None,
                'highest_bandwidth': None
            }
        }
        
        for op_name, times in self.operation_times.items():
            if times:
                bandwidths = self.bandwidth_usage.get(op_name, [])
                sizes = self.message_sizes.get(op_name, [])
                
                report['operations'][op_name] = {
                    'count': len(times),
                    'avg_time_ms': np.mean(times) * 1000,
                    'min_time_ms': np.min(times) * 1000,
                    'max_time_ms': np.max(times) * 1000,
                    'total_time_ms': np.sum(times) * 1000,
                    'avg_bandwidth_gbps': np.mean(bandwidths) if bandwidths else 0,
                    'avg_message_size_mb': np.mean(sizes) / (1024 * 1024) if sizes else 0
                }
        
        # Find most time-consuming operation
        if report['operations']:
            max_time_op = max(report['operations'].keys(), 
                            key=lambda x: report['operations'][x]['total_time_ms'])
            report['summary']['most_time_consuming'] = max_time_op
            
            max_bandwidth_op = max(report['operations'].keys(),
                                 key=lambda x: report['operations'][x]['avg_bandwidth_gbps'])
            report['summary']['highest_bandwidth'] = max_bandwidth_op
        
        return report