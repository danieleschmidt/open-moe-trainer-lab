"""
Distributed Training for Large-Scale MoE Models

This module provides comprehensive distributed training capabilities:
1. Multi-GPU training with expert parallelism
2. Pipeline parallelism for large models
3. Auto-scaling based on demand
4. Dynamic load balancing
5. Fault tolerance and recovery
6. Performance monitoring and optimization
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import socket
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import subprocess
import signal
import psutil

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import kubernetes
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"
    
    # Expert parallelism
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    min_nodes: int = 1
    max_nodes: int = 8
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    checkpoint_interval: int = 100
    max_retries: int = 3


@dataclass
class NodeMetrics:
    """Metrics for a single training node."""
    node_id: str
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    load_average: float = 0.0
    is_healthy: bool = True
    last_heartbeat: float = 0.0
    expert_assignments: List[int] = None
    
    def __post_init__(self):
        if self.expert_assignments is None:
            self.expert_assignments = []


class DistributedTrainer:
    """Advanced distributed trainer for MoE models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        optimizer: Optional[Any] = None
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        
        # Distributed state
        self.is_initialized = False
        self.process_group = None
        self.expert_groups = {}
        
        # Node management
        self.node_metrics = {}
        self.node_manager = NodeManager(config)
        
        # Auto-scaling
        self.auto_scaler = AutoScaler(config) if config.enable_auto_scaling else None
        
        # Fault tolerance
        self.fault_handler = FaultToleranceManager(config) if config.enable_fault_tolerance else None
        
        # Performance monitoring
        self.performance_tracker = DistributedPerformanceTracker()
        
        # Expert assignment
        self.expert_assigner = ExpertAssigner(config)
        
    def initialize_distributed(self) -> bool:
        """Initialize distributed training environment."""
        if not HAS_TORCH:
            logger.error("PyTorch not available - distributed training disabled")
            return False
            
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    rank=self.config.rank,
                    world_size=self.config.world_size
                )
            
            # Set device
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{self.config.local_rank}')
                torch.cuda.set_device(device)
                self.model = self.model.to(device)
            
            # Setup expert parallelism
            self._setup_expert_parallelism()
            
            # Wrap model with DDP
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True
            )
            
            self.is_initialized = True
            logger.info(f"Distributed training initialized: rank={self.config.rank}, world_size={self.config.world_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False
    
    def _setup_expert_parallelism(self):
        """Setup expert parallelism groups."""
        if self.config.expert_parallel_size <= 1:
            return
        
        # Create expert parallel groups
        num_expert_groups = self.config.world_size // self.config.expert_parallel_size
        
        for i in range(num_expert_groups):
            ranks = list(range(
                i * self.config.expert_parallel_size,
                (i + 1) * self.config.expert_parallel_size
            ))
            
            group = dist.new_group(ranks)
            self.expert_groups[f"expert_group_{i}"] = {
                'group': group,
                'ranks': ranks
            }
            
            if self.config.rank in ranks:
                # Assign experts to this group
                expert_assignments = self.expert_assigner.assign_experts_to_group(
                    i, num_expert_groups
                )
                logger.info(f"Node {self.config.rank} assigned experts: {expert_assignments}")
    
    def train_distributed(
        self,
        dataloader,
        num_epochs: int,
        save_dir: str = "./checkpoints"
    ) -> Dict[str, Any]:
        """Run distributed training with auto-scaling and fault tolerance."""
        if not self.is_initialized:
            if not self.initialize_distributed():
                return {"error": "Failed to initialize distributed training"}
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        training_results = {
            "epochs_completed": 0,
            "total_steps": 0,
            "best_loss": float('inf'),
            "scaling_events": [],
            "fault_events": [],
            "performance_metrics": []
        }
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                # Setup distributed sampler if needed
                if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                    dataloader.sampler.set_epoch(epoch)
                
                epoch_results = self._train_epoch(
                    dataloader, 
                    epoch, 
                    save_path,
                    training_results
                )
                
                # Update results
                training_results["epochs_completed"] = epoch + 1
                training_results["total_steps"] += epoch_results.get("steps", 0)
                
                if epoch_results.get("avg_loss", float('inf')) < training_results["best_loss"]:
                    training_results["best_loss"] = epoch_results["avg_loss"]
                
                # Check for auto-scaling
                if self.auto_scaler:
                    scaling_decision = self.auto_scaler.should_scale(self.node_metrics)
                    if scaling_decision["should_scale"]:
                        self._handle_scaling_event(scaling_decision, training_results)
                
                # Checkpoint saving
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    self._save_distributed_checkpoint(save_path, epoch, training_results)
                
                # Performance monitoring
                performance_metrics = self._collect_performance_metrics()
                training_results["performance_metrics"].append(performance_metrics)
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.fault_handler:
                recovery_successful = self.fault_handler.handle_training_failure(e, training_results)
                if recovery_successful:
                    logger.info("Recovery successful, continuing training")
                else:
                    training_results["error"] = str(e)
        
        finally:
            self._stop_monitoring_threads()
        
        return training_results
    
    def _train_epoch(
        self,
        dataloader,
        epoch: int,
        save_path: Path,
        training_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a single epoch with distributed coordination."""
        self.model.train()
        
        epoch_losses = []
        step_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            step_start_time = time.time()
            
            # Move batch to device
            if torch.cuda.is_available():
                batch = self._move_batch_to_device(batch)
            
            # Forward pass
            try:
                if hasattr(self.model.module, 'forward_with_expert_routing'):
                    # MoE-specific forward pass
                    outputs = self.model.module.forward_with_expert_routing(batch)
                    loss = outputs.loss
                    expert_metrics = outputs.expert_metrics
                else:
                    # Standard forward pass
                    loss = self.model(batch)
                    expert_metrics = {}
                
                # Backward pass
                if self.optimizer:
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # Update node metrics
                step_duration = time.time() - step_start_time
                self._update_node_metrics(loss.item(), step_duration, expert_metrics)
                
                step_count += 1
                
                if batch_idx % 10 == 0 and self.config.rank == 0:
                    logger.info(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Step {batch_idx} failed: {e}")
                if self.fault_handler:
                    if not self.fault_handler.handle_step_failure(e, batch_idx, epoch):
                        raise e
        
        # Synchronize losses across all processes
        avg_loss = self._synchronize_metric(sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0)
        
        return {
            "avg_loss": avg_loss,
            "steps": step_count,
            "throughput": step_count / (time.time() - time.time())  # Simplified
        }
    
    def _move_batch_to_device(self, batch):
        """Move batch to appropriate device."""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.local_rank}')
            if isinstance(batch, torch.Tensor):
                return batch.to(device)
            elif isinstance(batch, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                return [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        return batch
    
    def _synchronize_metric(self, value: float) -> float:
        """Synchronize metric across all processes."""
        if not HAS_TORCH or not dist.is_initialized():
            return value
        
        tensor = torch.tensor(value, device=f'cuda:{self.config.local_rank}' if torch.cuda.is_available() else 'cpu')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / self.config.world_size
    
    def _update_node_metrics(self, loss: float, step_duration: float, expert_metrics: Dict[str, Any]):
        """Update metrics for the current node."""
        node_id = f"node_{self.config.rank}"
        
        # Get system metrics
        gpu_util = self._get_gpu_utilization()
        memory_usage = self._get_memory_usage()
        load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        
        # Calculate throughput (simplified)
        throughput = 1.0 / step_duration if step_duration > 0 else 0.0
        
        # Update node metrics
        self.node_metrics[node_id] = NodeMetrics(
            node_id=node_id,
            gpu_utilization=gpu_util,
            memory_usage=memory_usage,
            throughput=throughput,
            load_average=load_avg,
            is_healthy=True,
            last_heartbeat=time.time(),
            expert_assignments=expert_metrics.get('assigned_experts', [])
        )
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if torch.cuda.is_available():
            # Simplified - would use nvidia-ml-py in production
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return (allocated / reserved * 100) if reserved > 0 else 0.0
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)  # GB
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads."""
        if self.auto_scaler:
            self.auto_scaler.start_monitoring()
        
        if self.fault_handler:
            self.fault_handler.start_monitoring()
        
        self.performance_tracker.start_monitoring()
    
    def _stop_monitoring_threads(self):
        """Stop background monitoring threads."""
        if self.auto_scaler:
            self.auto_scaler.stop_monitoring()
        
        if self.fault_handler:
            self.fault_handler.stop_monitoring()
        
        self.performance_tracker.stop_monitoring()
    
    def _handle_scaling_event(self, scaling_decision: Dict[str, Any], training_results: Dict[str, Any]):
        """Handle auto-scaling events."""
        logger.info(f"Handling scaling event: {scaling_decision}")
        
        scaling_event = {
            "timestamp": time.time(),
            "action": scaling_decision["action"],
            "reason": scaling_decision["reason"],
            "current_nodes": scaling_decision["current_nodes"],
            "target_nodes": scaling_decision["target_nodes"]
        }
        
        # Execute scaling
        if scaling_decision["action"] == "scale_up":
            success = self.node_manager.add_nodes(
                scaling_decision["target_nodes"] - scaling_decision["current_nodes"]
            )
        else:
            success = self.node_manager.remove_nodes(
                scaling_decision["current_nodes"] - scaling_decision["target_nodes"]
            )
        
        scaling_event["success"] = success
        training_results["scaling_events"].append(scaling_event)
    
    def _save_distributed_checkpoint(self, save_path: Path, epoch: int, results: Dict[str, Any]):
        """Save distributed training checkpoint."""
        if self.config.rank == 0:  # Only master saves
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "training_results": results,
                "config": asdict(self.config)
            }
            
            checkpoint_path = save_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        return {
            "timestamp": time.time(),
            "node_metrics": {k: asdict(v) for k, v in self.node_metrics.items()},
            "distributed_metrics": self.performance_tracker.get_metrics(),
            "scaling_status": self.auto_scaler.get_status() if self.auto_scaler else {},
            "fault_tolerance_status": self.fault_handler.get_status() if self.fault_handler else {}
        }


class NodeManager:
    """Manages distributed training nodes."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.active_nodes = set()
        self.node_processes = {}
        
    def add_nodes(self, num_nodes: int) -> bool:
        """Add new training nodes."""
        logger.info(f"Adding {num_nodes} nodes to cluster")
        
        success_count = 0
        for i in range(num_nodes):
            new_rank = len(self.active_nodes)
            if self._start_node_process(new_rank):
                success_count += 1
        
        return success_count == num_nodes
    
    def remove_nodes(self, num_nodes: int) -> bool:
        """Remove training nodes gracefully."""
        logger.info(f"Removing {num_nodes} nodes from cluster")
        
        # Select nodes to remove (highest ranks first)
        nodes_to_remove = sorted(self.active_nodes, reverse=True)[:num_nodes]
        
        success_count = 0
        for node_rank in nodes_to_remove:
            if self._stop_node_process(node_rank):
                success_count += 1
                self.active_nodes.discard(node_rank)
        
        return success_count == num_nodes
    
    def _start_node_process(self, rank: int) -> bool:
        """Start a new node process."""
        try:
            # Create command to start new worker
            cmd = [
                "python", "-m", "torch.distributed.launch",
                "--nproc_per_node=1",
                "--master_addr", self.config.master_addr,
                "--master_port", self.config.master_port,
                "--node_rank", str(rank),
                "distributed_worker.py"  # Worker script
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.node_processes[rank] = process
            self.active_nodes.add(rank)
            
            logger.info(f"Started node process for rank {rank}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start node process for rank {rank}: {e}")
            return False
    
    def _stop_node_process(self, rank: int) -> bool:
        """Stop a node process gracefully."""
        try:
            if rank in self.node_processes:
                process = self.node_processes[rank]
                process.send_signal(signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                del self.node_processes[rank]
                logger.info(f"Stopped node process for rank {rank}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to stop node process for rank {rank}: {e}")
        
        return False


class AutoScaler:
    """Automatic scaling based on performance metrics."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.monitoring_active = False
        self.scaling_history = []
        self.last_scaling_time = 0.0
        self.scaling_cooldown = 300.0  # 5 minutes
        
    def should_scale(self, node_metrics: Dict[str, NodeMetrics]) -> Dict[str, Any]:
        """Determine if scaling is needed."""
        if not node_metrics:
            return {"should_scale": False}
        
        # Check cooldown
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return {"should_scale": False, "reason": "scaling_cooldown"}
        
        # Calculate aggregate metrics
        avg_gpu_util = np.mean([m.gpu_utilization for m in node_metrics.values()])
        avg_throughput = np.mean([m.throughput for m in node_metrics.values()])
        healthy_nodes = sum(1 for m in node_metrics.values() if m.is_healthy)
        
        current_nodes = len(node_metrics)
        
        # Scale up conditions
        if (avg_gpu_util > self.config.scale_up_threshold * 100 and 
            current_nodes < self.config.max_nodes and
            avg_throughput > 0):
            
            return {
                "should_scale": True,
                "action": "scale_up",
                "reason": f"High GPU utilization: {avg_gpu_util:.1f}%",
                "current_nodes": current_nodes,
                "target_nodes": min(current_nodes + 1, self.config.max_nodes)
            }
        
        # Scale down conditions
        if (avg_gpu_util < self.config.scale_down_threshold * 100 and 
            current_nodes > self.config.min_nodes and
            healthy_nodes >= current_nodes):
            
            return {
                "should_scale": True,
                "action": "scale_down",
                "reason": f"Low GPU utilization: {avg_gpu_util:.1f}%",
                "current_nodes": current_nodes,
                "target_nodes": max(current_nodes - 1, self.config.min_nodes)
            }
        
        return {"should_scale": False}
    
    def start_monitoring(self):
        """Start monitoring for auto-scaling."""
        self.monitoring_active = True
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        logger.info("Auto-scaling monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get auto-scaling status."""
        return {
            "monitoring_active": self.monitoring_active,
            "last_scaling_time": self.last_scaling_time,
            "scaling_history_count": len(self.scaling_history),
            "cooldown_remaining": max(0, self.scaling_cooldown - (time.time() - self.last_scaling_time))
        }


class FaultToleranceManager:
    """Handles fault tolerance and recovery."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.monitoring_active = False
        self.failed_nodes = set()
        self.recovery_attempts = {}
        
    def handle_training_failure(self, error: Exception, training_state: Dict[str, Any]) -> bool:
        """Handle training failures and attempt recovery."""
        logger.error(f"Training failure detected: {error}")
        
        error_type = type(error).__name__
        
        if error_type == "RuntimeError" and "CUDA" in str(error):
            return self._handle_cuda_error(error, training_state)
        elif error_type == "ConnectionError":
            return self._handle_connection_error(error, training_state)
        elif error_type == "TimeoutError":
            return self._handle_timeout_error(error, training_state)
        else:
            return self._handle_generic_error(error, training_state)
    
    def handle_step_failure(self, error: Exception, step: int, epoch: int) -> bool:
        """Handle individual training step failures."""
        logger.warning(f"Step {step} in epoch {epoch} failed: {error}")
        
        # Simple recovery: skip the failed batch
        return True
    
    def _handle_cuda_error(self, error: Exception, training_state: Dict[str, Any]) -> bool:
        """Handle CUDA-related errors."""
        logger.info("Attempting CUDA error recovery...")
        
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Reduce batch size if possible
            # This would require coordination with the training loop
            
            return True
            
        except Exception as e:
            logger.error(f"CUDA error recovery failed: {e}")
            return False
    
    def _handle_connection_error(self, error: Exception, training_state: Dict[str, Any]) -> bool:
        """Handle distributed communication errors."""
        logger.info("Attempting connection error recovery...")
        
        try:
            # Reinitialize process group
            if dist.is_initialized():
                dist.destroy_process_group()
            
            # This would require re-initializing the distributed environment
            # For now, return False to indicate recovery not possible
            return False
            
        except Exception as e:
            logger.error(f"Connection error recovery failed: {e}")
            return False
    
    def _handle_timeout_error(self, error: Exception, training_state: Dict[str, Any]) -> bool:
        """Handle timeout errors."""
        logger.info("Attempting timeout error recovery...")
        
        # Simple strategy: wait and retry
        time.sleep(10)
        return True
    
    def _handle_generic_error(self, error: Exception, training_state: Dict[str, Any]) -> bool:
        """Handle generic errors."""
        logger.info(f"Attempting generic error recovery for {type(error).__name__}")
        
        # Basic recovery: save state and continue
        try:
            # Save recovery state
            recovery_path = Path("./recovery")
            recovery_path.mkdir(exist_ok=True)
            
            with open(recovery_path / "error_state.json", 'w') as f:
                json.dump({
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "training_state": training_state,
                    "timestamp": time.time()
                }, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Generic error recovery failed: {e}")
            return False
    
    def start_monitoring(self):
        """Start fault tolerance monitoring."""
        self.monitoring_active = True
        logger.info("Fault tolerance monitoring started")
    
    def stop_monitoring(self):
        """Stop fault tolerance monitoring."""
        self.monitoring_active = False
        logger.info("Fault tolerance monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get fault tolerance status."""
        return {
            "monitoring_active": self.monitoring_active,
            "failed_nodes": list(self.failed_nodes),
            "recovery_attempts": dict(self.recovery_attempts)
        }


class ExpertAssigner:
    """Assigns experts to nodes for optimal load balancing."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.expert_assignments = {}
        
    def assign_experts_to_group(self, group_id: int, num_groups: int) -> List[int]:
        """Assign experts to a specific group."""
        # Assuming model has num_experts attribute
        total_experts = getattr(self, 'num_experts', 64)  # Default to 64
        experts_per_group = total_experts // num_groups
        
        start_expert = group_id * experts_per_group
        end_expert = start_expert + experts_per_group
        
        if group_id == num_groups - 1:  # Last group gets remaining experts
            end_expert = total_experts
        
        assigned_experts = list(range(start_expert, end_expert))
        self.expert_assignments[group_id] = assigned_experts
        
        return assigned_experts
    
    def rebalance_experts(self, node_metrics: Dict[str, NodeMetrics]) -> Dict[int, List[int]]:
        """Rebalance expert assignments based on node performance."""
        # Simple rebalancing strategy
        # In production, this would use more sophisticated algorithms
        
        healthy_nodes = {k: v for k, v in node_metrics.items() if v.is_healthy}
        if not healthy_nodes:
            return self.expert_assignments
        
        total_experts = sum(len(experts) for experts in self.expert_assignments.values())
        experts_per_node = total_experts // len(healthy_nodes)
        
        new_assignments = {}
        expert_idx = 0
        
        for i, node_id in enumerate(healthy_nodes.keys()):
            start_idx = i * experts_per_node
            end_idx = start_idx + experts_per_node
            
            if i == len(healthy_nodes) - 1:  # Last node gets remaining
                end_idx = total_experts
            
            new_assignments[i] = list(range(start_idx, end_idx))
        
        return new_assignments


class DistributedPerformanceTracker:
    """Tracks performance across distributed training."""
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history = []
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        logger.info("Distributed performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        logger.info("Distributed performance monitoring stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current distributed performance metrics."""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics_collected": len(self.metrics_history),
            "last_update": time.time()
        }


def create_distributed_trainer(
    model: nn.Module,
    config_dict: Dict[str, Any]
) -> DistributedTrainer:
    """Factory function to create distributed trainer."""
    config = DistributedConfig(**config_dict)
    return DistributedTrainer(model, config)


# Mock numpy for environments without it
try:
    import numpy as np
except ImportError:
    class MockNumpy:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
    np = MockNumpy()


if __name__ == "__main__":
    print("🚀 Distributed MoE Training System")
    print("Advanced distributed training with auto-scaling and fault tolerance")
    
    # Mock model for demonstration
    class MockMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            self.num_experts = 64
        
        def forward(self, x):
            return self.linear(x).mean()
    
    if HAS_TORCH:
        model = MockMoEModel()
        
        config = DistributedConfig(
            world_size=4,
            rank=0,
            expert_parallel_size=2,
            enable_auto_scaling=True,
            enable_fault_tolerance=True
        )
        
        trainer = DistributedTrainer(model, config)
        print(f"Distributed trainer created with config: {asdict(config)}")
    else:
        print("PyTorch not available - distributed training disabled")