"""Auto-scaling and resource optimization for MoE models."""

import torch
import torch.nn as nn
import psutil
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import threading
import queue
from contextlib import contextmanager

from ..models import MoEModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    throughput: float
    latency: float
    timestamp: float


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    
    action: str  # "scale_up", "scale_down", "no_change"
    target_batch_size: int
    target_num_experts: int
    confidence: float
    reasoning: str


class ResourceMonitor:
    """Real-time resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.monitoring_thread = None
        self.is_monitoring = False
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Resource monitoring stopped")
        
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                if not self.metrics_queue.full():
                    self.metrics_queue.put(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and system memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_utilization = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Try to get GPU utilization (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = gpu_util.gpu
            except:
                gpu_utilization = 0.0
                
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_info.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            throughput=0.0,  # To be filled by caller
            latency=0.0,     # To be filled by caller
            timestamp=time.time()
        )
        
    def get_latest_metrics(self) -> Optional[ResourceMetrics]:
        """Get latest resource metrics."""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
            
    def get_metrics_history(self, count: int = 10) -> List[ResourceMetrics]:
        """Get recent metrics history."""
        metrics = []
        temp_metrics = []
        
        # Drain queue and keep recent metrics
        while not self.metrics_queue.empty() and len(temp_metrics) < count:
            try:
                temp_metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
                
        # Return most recent metrics
        return temp_metrics[-count:] if temp_metrics else []


class AdaptiveExpertPool(nn.Module):
    """Expert pool that can dynamically adjust number of active experts."""
    
    def __init__(
        self,
        base_num_experts: int,
        max_experts: int,
        hidden_size: int,
        expert_hidden_size: int,
        activation: str = "gelu"
    ):
        super().__init__()
        self.base_num_experts = base_num_experts
        self.max_experts = max_experts
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        
        # Create maximum number of experts
        self.all_experts = nn.ModuleList()
        for i in range(max_experts):
            expert = self._create_expert(activation)
            self.all_experts.append(expert)
            
        # Track active experts
        self.num_active_experts = base_num_experts
        self.expert_usage_stats = torch.zeros(max_experts)
        
    def _create_expert(self, activation: str) -> nn.Module:
        """Create a single expert."""
        layers = [
            nn.Linear(self.hidden_size, self.expert_hidden_size),
        ]
        
        if activation == "gelu":
            layers.append(nn.GELU())
        elif activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "swish":
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(self.expert_hidden_size, self.hidden_size))
        
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Forward through specific expert."""
        if expert_idx >= self.num_active_experts:
            # Use a random active expert as fallback
            expert_idx = expert_idx % self.num_active_experts
            
        return self.all_experts[expert_idx](x)
        
    def scale_experts(self, new_num_experts: int):
        """Scale number of active experts."""
        new_num_experts = max(1, min(new_num_experts, self.max_experts))
        
        if new_num_experts != self.num_active_experts:
            logger.info(f"Scaling experts from {self.num_active_experts} to {new_num_experts}")
            self.num_active_experts = new_num_experts
            
            # Reset usage stats for new configuration
            self.expert_usage_stats = torch.zeros(self.max_experts)
            
    def update_usage_stats(self, expert_selections: torch.Tensor):
        """Update expert usage statistics."""
        for expert_idx in expert_selections.flatten():
            if expert_idx >= 0 and expert_idx < self.num_active_experts:
                self.expert_usage_stats[expert_idx] += 1
                
    def get_usage_distribution(self) -> torch.Tensor:
        """Get expert usage distribution."""
        if self.expert_usage_stats.sum() > 0:
            return self.expert_usage_stats[:self.num_active_experts] / self.expert_usage_stats[:self.num_active_experts].sum()
        else:
            return torch.ones(self.num_active_experts) / self.num_active_experts


class AutoScaler:
    """Automatic scaling controller for MoE models."""
    
    def __init__(
        self,
        model: MoEModel,
        min_batch_size: int = 4,
        max_batch_size: int = 128,
        min_experts: int = 2,
        max_experts: int = 32,
        scaling_interval: float = 30.0,  # seconds
        utilization_target: float = 0.8,
        latency_target: float = 100.0  # ms
    ):
        self.model = model
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.scaling_interval = scaling_interval
        self.utilization_target = utilization_target
        self.latency_target = latency_target
        
        # Current configuration
        self.current_batch_size = min_batch_size
        self.current_num_experts = getattr(model, 'num_experts', min_experts)
        
        # Monitoring
        self.resource_monitor = ResourceMonitor()
        self.performance_history = []
        
        # Scaling state
        self.last_scaling_time = time.time()
        self.scaling_cooldown = 60.0  # seconds
        
    def start(self):
        """Start auto-scaling."""
        self.resource_monitor.start_monitoring()
        logger.info("Auto-scaler started")
        
    def stop(self):
        """Stop auto-scaling."""
        self.resource_monitor.stop_monitoring()
        logger.info("Auto-scaler stopped")
        
    def should_scale(self) -> bool:
        """Check if scaling decision should be made."""
        current_time = time.time()
        return (current_time - self.last_scaling_time) >= self.scaling_interval
        
    def make_scaling_decision(
        self, 
        current_throughput: float,
        current_latency: float
    ) -> ScalingDecision:
        """Make scaling decision based on current metrics."""
        
        # Get recent resource metrics
        metrics_history = self.resource_monitor.get_metrics_history(count=10)
        
        if not metrics_history:
            return ScalingDecision(
                action="no_change",
                target_batch_size=self.current_batch_size,
                target_num_experts=self.current_num_experts,
                confidence=0.0,
                reasoning="No metrics available"
            )
            
        # Compute average metrics
        avg_cpu = sum(m.cpu_percent for m in metrics_history) / len(metrics_history)
        avg_memory = sum(m.memory_percent for m in metrics_history) / len(metrics_history)
        avg_gpu_memory = sum(m.gpu_memory_used / m.gpu_memory_total for m in metrics_history if m.gpu_memory_total > 0) / len(metrics_history)
        avg_gpu_util = sum(m.gpu_utilization for m in metrics_history) / len(metrics_history)
        
        # Decision logic
        action = "no_change"
        target_batch_size = self.current_batch_size
        target_num_experts = self.current_num_experts
        confidence = 0.5
        reasoning = "No significant change needed"
        
        # Scale up conditions
        if (avg_gpu_util < 50 and avg_gpu_memory < 0.7 and current_latency < self.latency_target):
            if self.current_batch_size < self.max_batch_size:
                action = "scale_up"
                target_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.5))
                confidence = 0.8
                reasoning = "Low GPU utilization, can increase batch size"
            elif self.current_num_experts < self.max_experts:
                action = "scale_up" 
                target_num_experts = min(self.max_experts, self.current_num_experts + 2)
                confidence = 0.7
                reasoning = "Low GPU utilization, can add more experts"
                
        # Scale down conditions
        elif (avg_gpu_memory > 0.9 or current_latency > self.latency_target * 1.5):
            if self.current_batch_size > self.min_batch_size:
                action = "scale_down"
                target_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
                confidence = 0.9
                reasoning = "High memory usage or latency, reducing batch size"
            elif self.current_num_experts > self.min_experts:
                action = "scale_down"
                target_num_experts = max(self.min_experts, self.current_num_experts - 1)
                confidence = 0.8
                reasoning = "High resource usage, reducing experts"
                
        # Memory pressure override
        if avg_gpu_memory > 0.95:
            action = "scale_down"
            target_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.5))
            confidence = 1.0
            reasoning = "Critical memory pressure, emergency scale down"
            
        return ScalingDecision(
            action=action,
            target_batch_size=target_batch_size,
            target_num_experts=target_num_experts,
            confidence=confidence,
            reasoning=reasoning
        )
        
    def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply scaling decision."""
        if decision.action == "no_change":
            return True
            
        try:
            # Update batch size
            if decision.target_batch_size != self.current_batch_size:
                self.current_batch_size = decision.target_batch_size
                logger.info(f"Scaled batch size to {self.current_batch_size}")
                
            # Update number of experts (if model supports it)
            if (decision.target_num_experts != self.current_num_experts and 
                hasattr(self.model, 'scale_experts')):
                self.model.scale_experts(decision.target_num_experts)
                self.current_num_experts = decision.target_num_experts
                logger.info(f"Scaled experts to {self.current_num_experts}")
                
            self.last_scaling_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply scaling decision: {e}")
            return False
            
    def get_current_config(self) -> Dict[str, Any]:
        """Get current scaling configuration."""
        return {
            'batch_size': self.current_batch_size,
            'num_experts': self.current_num_experts,
            'last_scaling_time': self.last_scaling_time
        }


class LoadBalancer:
    """Load balancer for distributed MoE inference."""
    
    def __init__(self, expert_endpoints: List[str]):
        self.expert_endpoints = expert_endpoints
        self.expert_loads = {endpoint: 0.0 for endpoint in expert_endpoints}
        self.expert_response_times = {endpoint: [] for endpoint in expert_endpoints}
        
    def select_expert_endpoint(self, expert_idx: int) -> str:
        """Select best endpoint for expert."""
        # Simple round-robin for now
        # In practice, would consider load, latency, etc.
        return self.expert_endpoints[expert_idx % len(self.expert_endpoints)]
        
    def update_load(self, endpoint: str, load: float):
        """Update load for endpoint."""
        self.expert_loads[endpoint] = load
        
    def update_response_time(self, endpoint: str, response_time: float):
        """Update response time for endpoint."""
        self.expert_response_times[endpoint].append(response_time)
        
        # Keep only recent measurements
        max_history = 100
        if len(self.expert_response_times[endpoint]) > max_history:
            self.expert_response_times[endpoint] = self.expert_response_times[endpoint][-max_history:]
            
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        import numpy as np
        
        stats = {}
        for endpoint in self.expert_endpoints:
            response_times = self.expert_response_times[endpoint]
            if response_times:
                stats[endpoint] = {
                    'current_load': self.expert_loads[endpoint],
                    'avg_response_time': np.mean(response_times),
                    'p95_response_time': np.percentile(response_times, 95),
                    'num_requests': len(response_times)
                }
            else:
                stats[endpoint] = {
                    'current_load': self.expert_loads[endpoint],
                    'avg_response_time': 0.0,
                    'p95_response_time': 0.0,
                    'num_requests': 0
                }
                
        return stats


class AutoScalingMoEModel(MoEModel):
    """MoE model with auto-scaling capabilities."""
    
    def __init__(self, *args, **kwargs):
        # Extract auto-scaling specific parameters
        self.enable_auto_scaling = kwargs.pop('enable_auto_scaling', False)
        self.max_experts = kwargs.pop('max_experts', kwargs.get('num_experts', 8) * 2)
        
        super().__init__(*args, **kwargs)
        
        # Setup auto-scaling
        if self.enable_auto_scaling:
            self.auto_scaler = AutoScaler(
                model=self,
                max_experts=self.max_experts
            )
            
            # Replace expert pools with adaptive ones
            self._setup_adaptive_experts()
            
    def _setup_adaptive_experts(self):
        """Setup adaptive expert pools."""
        for i, layer in enumerate(self.layers):
            if i in self.moe_layers and hasattr(layer, 'experts'):
                # Replace with adaptive expert pool
                adaptive_pool = AdaptiveExpertPool(
                    base_num_experts=self.num_experts,
                    max_experts=self.max_experts,
                    hidden_size=self.hidden_size,
                    expert_hidden_size=self.hidden_size * 4
                )
                layer.experts = adaptive_pool
                
    def scale_experts(self, new_num_experts: int):
        """Scale number of active experts."""
        for i, layer in enumerate(self.layers):
            if i in self.moe_layers and hasattr(layer.experts, 'scale_experts'):
                layer.experts.scale_experts(new_num_experts)
                
        self.num_experts = new_num_experts
        
    def start_auto_scaling(self):
        """Start auto-scaling."""
        if hasattr(self, 'auto_scaler'):
            self.auto_scaler.start()
            
    def stop_auto_scaling(self):
        """Stop auto-scaling."""
        if hasattr(self, 'auto_scaler'):
            self.auto_scaler.stop()
            
    @contextmanager
    def auto_scaling_context(self):
        """Context manager for auto-scaling."""
        if hasattr(self, 'auto_scaler'):
            self.start_auto_scaling()
            
        try:
            yield
        finally:
            if hasattr(self, 'auto_scaler'):
                self.stop_auto_scaling()