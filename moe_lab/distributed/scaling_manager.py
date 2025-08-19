"""Advanced distributed scaling and resource management for MoE systems."""

import time
import threading
import multiprocessing
import socket
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import numpy as np

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch for testing
    torch = type('torch', (), {
        'cuda': type('cuda', (), {
            'is_available': lambda: False,
            'device_count': lambda: 0
        })()
    })()
    dist = type('dist', (), {
        'init_process_group': lambda *args, **kwargs: None,
        'get_rank': lambda: 0,
        'get_world_size': lambda: 1,
        'barrier': lambda: None,
        'all_gather': lambda *args: None,
        'all_reduce': lambda *args: None
    })()


class ScalingStrategy(Enum):
    """Scaling strategies for distributed MoE."""
    HORIZONTAL = "horizontal"  # Add more nodes
    VERTICAL = "vertical"      # Increase resources per node
    EXPERT_PARALLEL = "expert_parallel"  # Distribute experts across nodes
    PIPELINE_PARALLEL = "pipeline_parallel"  # Pipeline model layers


@dataclass
class NodeStatus:
    """Status of a compute node."""
    
    node_id: str
    rank: int
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    network_bandwidth: float
    active_experts: List[int]
    current_load: float
    last_heartbeat: float
    is_healthy: bool


@dataclass
class ScalingDecision:
    """Scaling decision with justification."""
    
    strategy: ScalingStrategy
    action: str  # "scale_up", "scale_down", "rebalance"
    target_nodes: List[str]
    expected_improvement: float
    confidence: float
    reasoning: str
    timestamp: float


class DistributedLoadBalancer:
    """Intelligent load balancer for distributed MoE inference."""
    
    def __init__(
        self,
        rebalancing_threshold: float = 0.8,
        load_smoothing_factor: float = 0.1,
        expert_migration_cost: float = 0.05
    ):
        self.rebalancing_threshold = rebalancing_threshold
        self.load_smoothing_factor = load_smoothing_factor
        self.expert_migration_cost = expert_migration_cost
        
        # State tracking
        self.node_loads = defaultdict(float)
        self.expert_placement = defaultdict(list)  # node_id -> [expert_ids]
        self.routing_history = deque(maxlen=1000)
        self.load_history = defaultdict(lambda: deque(maxlen=100))
        
        self.balancer_lock = threading.Lock()
        
    def update_node_load(self, node_id: str, current_load: float):
        """Update current load for a node."""
        
        with self.balancer_lock:
            # Apply exponential smoothing
            old_load = self.node_loads.get(node_id, current_load)
            self.node_loads[node_id] = (
                old_load * (1 - self.load_smoothing_factor) +
                current_load * self.load_smoothing_factor
            )
            
            self.load_history[node_id].append({
                'load': current_load,
                'timestamp': time.time()
            })
    
    def route_request(self, expert_id: int, request_size: float = 1.0) -> str:
        """Route request to optimal node."""
        
        with self.balancer_lock:
            # Find nodes hosting this expert
            candidate_nodes = [
                node_id for node_id, experts in self.expert_placement.items()
                if expert_id in experts
            ]
            
            if not candidate_nodes:
                # Expert not placed, assign to least loaded node
                candidate_nodes = list(self.node_loads.keys())
            
            if not candidate_nodes:
                return "node_0"  # Default fallback
            
            # Select node with lowest load
            best_node = min(candidate_nodes, key=lambda n: self.node_loads.get(n, 0))
            
            # Update routing history
            self.routing_history.append({
                'expert_id': expert_id,
                'node_id': best_node,
                'request_size': request_size,
                'timestamp': time.time()
            })
            
            return best_node
    
    def check_rebalancing_needed(self) -> bool:
        """Check if load rebalancing is needed."""
        
        if len(self.node_loads) < 2:
            return False
        
        loads = list(self.node_loads.values())
        max_load = max(loads)
        min_load = min(loads)
        
        # Check if load imbalance exceeds threshold
        load_imbalance = (max_load - min_load) / max(max_load, 1e-6)
        
        return load_imbalance > self.rebalancing_threshold
    
    def suggest_expert_migration(self) -> List[Tuple[int, str, str]]:
        """Suggest expert migrations to balance load."""
        
        if not self.check_rebalancing_needed():
            return []
        
        migrations = []
        
        # Find overloaded and underloaded nodes
        loads = [(node_id, load) for node_id, load in self.node_loads.items()]
        loads.sort(key=lambda x: x[1], reverse=True)
        
        overloaded_nodes = [node for node, load in loads[:len(loads)//2]]
        underloaded_nodes = [node for node, load in loads[len(loads)//2:]]
        
        # Suggest migrations from overloaded to underloaded nodes
        for overloaded_node in overloaded_nodes:
            if not overloaded_nodes or not underloaded_nodes:
                break
                
            experts = self.expert_placement.get(overloaded_node, [])
            if experts:
                # Migrate least recently used expert
                expert_to_migrate = experts[0]  # Simplified selection
                target_node = underloaded_nodes[0]
                
                migrations.append((expert_to_migrate, overloaded_node, target_node))
                
                # Remove from overloaded, might move to next target
                if len(migrations) >= 3:  # Limit migrations per round
                    break
        
        return migrations
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        
        with self.balancer_lock:
            if not self.node_loads:
                return {}
            
            loads = list(self.node_loads.values())
            
            return {
                'num_nodes': len(self.node_loads),
                'avg_load': np.mean(loads),
                'max_load': max(loads),
                'min_load': min(loads),
                'load_variance': np.var(loads),
                'load_imbalance_ratio': (max(loads) - min(loads)) / max(loads, 1e-6),
                'rebalancing_needed': self.check_rebalancing_needed(),
                'total_requests_routed': len(self.routing_history),
                'experts_placed': sum(len(experts) for experts in self.expert_placement.values())
            }


class AutoScaler:
    """Automatic scaling manager for MoE clusters."""
    
    def __init__(
        self,
        min_nodes: int = 1,
        max_nodes: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scaling_cooldown: float = 300.0  # 5 minutes
    ):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_cooldown = scaling_cooldown
        
        # Scaling state
        self.current_nodes = set()
        self.last_scaling_time = 0.0
        self.scaling_history = deque(maxlen=100)
        self.pending_nodes = set()
        
        # Performance tracking
        self.cluster_metrics = deque(maxlen=50)
        
        self.scaler_lock = threading.Lock()
        
    def update_cluster_metrics(self, metrics: Dict[str, float]):
        """Update cluster-wide performance metrics."""
        
        with self.scaler_lock:
            metrics_with_timestamp = metrics.copy()
            metrics_with_timestamp['timestamp'] = time.time()
            self.cluster_metrics.append(metrics_with_timestamp)
    
    def evaluate_scaling_need(self) -> Optional[ScalingDecision]:
        """Evaluate if cluster scaling is needed."""
        
        if not self.cluster_metrics:
            return None
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return None
        
        # Get recent metrics
        recent_metrics = list(self.cluster_metrics)[-10:]
        
        # Calculate average metrics
        avg_cpu_util = np.mean([m.get('cpu_utilization', 0) for m in recent_metrics])
        avg_memory_util = np.mean([m.get('memory_utilization', 0) for m in recent_metrics])
        avg_throughput = np.mean([m.get('throughput', 0) for m in recent_metrics])
        avg_latency = np.mean([m.get('latency', 0) for m in recent_metrics])
        
        # Determine scaling need
        overall_utilization = max(avg_cpu_util, avg_memory_util)
        
        if overall_utilization > self.scale_up_threshold and len(self.current_nodes) < self.max_nodes:
            return ScalingDecision(
                strategy=ScalingStrategy.HORIZONTAL,
                action="scale_up",
                target_nodes=self._suggest_new_nodes(1),
                expected_improvement=0.2,
                confidence=0.8,
                reasoning=f"High utilization ({overall_utilization:.2f}) exceeds threshold ({self.scale_up_threshold})",
                timestamp=current_time
            )
        
        elif overall_utilization < self.scale_down_threshold and len(self.current_nodes) > self.min_nodes:
            return ScalingDecision(
                strategy=ScalingStrategy.HORIZONTAL,
                action="scale_down",
                target_nodes=self._suggest_nodes_to_remove(1),
                expected_improvement=0.1,
                confidence=0.7,
                reasoning=f"Low utilization ({overall_utilization:.2f}) below threshold ({self.scale_down_threshold})",
                timestamp=current_time
            )
        
        return None
    
    def _suggest_new_nodes(self, count: int) -> List[str]:
        """Suggest new node IDs for scaling up."""
        new_nodes = []
        for i in range(count):
            node_id = f"node_{len(self.current_nodes) + len(self.pending_nodes) + i}"
            new_nodes.append(node_id)
        return new_nodes
    
    def _suggest_nodes_to_remove(self, count: int) -> List[str]:
        """Suggest nodes to remove for scaling down."""
        # Remove least utilized nodes (simplified)
        if len(self.current_nodes) <= count:
            return []
        
        # Get nodes sorted by utilization (would need actual metrics)
        nodes_to_remove = list(self.current_nodes)[:count]
        return nodes_to_remove
    
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision."""
        
        with self.scaler_lock:
            success = False
            
            try:
                if decision.action == "scale_up":
                    success = self._scale_up(decision.target_nodes)
                elif decision.action == "scale_down":
                    success = self._scale_down(decision.target_nodes)
                
                if success:
                    self.last_scaling_time = time.time()
                    self.scaling_history.append(asdict(decision))
                
                return success
                
            except Exception as e:
                print(f"Scaling execution failed: {e}")
                return False
    
    def _scale_up(self, new_nodes: List[str]) -> bool:
        """Scale up by adding new nodes."""
        
        for node_id in new_nodes:
            # Simulate node provisioning
            self.current_nodes.add(node_id)
            self.pending_nodes.add(node_id)
            
            # In real implementation, would:
            # 1. Provision new compute instance
            # 2. Install and configure MoE runtime
            # 3. Join distributed training group
            # 4. Load balance experts to new node
        
        print(f"Scaled up: Added {len(new_nodes)} nodes")
        return True
    
    def _scale_down(self, nodes_to_remove: List[str]) -> bool:
        """Scale down by removing nodes."""
        
        for node_id in nodes_to_remove:
            if node_id in self.current_nodes:
                # Simulate graceful node removal
                self.current_nodes.remove(node_id)
                
                # In real implementation, would:
                # 1. Migrate experts to other nodes
                # 2. Drain pending requests
                # 3. Leave distributed training group
                # 4. Terminate compute instance
        
        print(f"Scaled down: Removed {len(nodes_to_remove)} nodes")
        return True
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        
        with self.scaler_lock:
            recent_decisions = list(self.scaling_history)[-10:]
            
            scale_up_count = sum(1 for d in self.scaling_history if d['action'] == 'scale_up')
            scale_down_count = sum(1 for d in self.scaling_history if d['action'] == 'scale_down')
            
            return {
                'current_nodes_count': len(self.current_nodes),
                'min_nodes': self.min_nodes,
                'max_nodes': self.max_nodes,
                'pending_nodes_count': len(self.pending_nodes),
                'total_scaling_events': len(self.scaling_history),
                'scale_up_events': scale_up_count,
                'scale_down_events': scale_down_count,
                'last_scaling_time': self.last_scaling_time,
                'recent_decisions': recent_decisions,
                'scaling_cooldown_remaining': max(0, 
                    self.scaling_cooldown - (time.time() - self.last_scaling_time)
                )
            }


class DistributedResourceManager:
    """Comprehensive resource management for distributed MoE systems."""
    
    def __init__(
        self,
        node_discovery_port: int = 9999,
        heartbeat_interval: float = 30.0,
        node_timeout: float = 120.0
    ):
        self.node_discovery_port = node_discovery_port
        self.heartbeat_interval = heartbeat_interval
        self.node_timeout = node_timeout
        
        # Resource tracking
        self.nodes: Dict[str, NodeStatus] = {}
        self.resource_history = deque(maxlen=1000)
        self.expert_assignments: Dict[int, List[str]] = defaultdict(list)
        
        # Components
        self.load_balancer = DistributedLoadBalancer()
        self.autoscaler = AutoScaler()
        
        # Coordination
        self.manager_lock = threading.Lock()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
    def register_node(self, node_id: str, capabilities: Dict[str, Any]) -> bool:
        """Register a new compute node."""
        
        with self.manager_lock:
            node_status = NodeStatus(
                node_id=node_id,
                rank=capabilities.get('rank', 0),
                cpu_percent=0.0,
                memory_percent=0.0,
                gpu_memory_percent=0.0,
                network_bandwidth=capabilities.get('network_bandwidth', 1000.0),
                active_experts=[],
                current_load=0.0,
                last_heartbeat=time.time(),
                is_healthy=True
            )
            
            self.nodes[node_id] = node_status
            self.autoscaler.current_nodes.add(node_id)
            
            print(f"Registered node: {node_id}")
            return True
    
    def update_node_status(
        self,
        node_id: str,
        cpu_percent: float,
        memory_percent: float,
        gpu_memory_percent: float = 0.0,
        active_experts: Optional[List[int]] = None
    ):
        """Update status for a registered node."""
        
        with self.manager_lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            node.cpu_percent = cpu_percent
            node.memory_percent = memory_percent
            node.gpu_memory_percent = gpu_memory_percent
            node.active_experts = active_experts or []
            node.current_load = max(cpu_percent, memory_percent, gpu_memory_percent) / 100.0
            node.last_heartbeat = time.time()
            node.is_healthy = True
            
            # Update load balancer
            self.load_balancer.update_node_load(node_id, node.current_load)
            self.load_balancer.expert_placement[node_id] = active_experts or []
            
            return True
    
    def allocate_expert(self, expert_id: int, preferred_nodes: Optional[List[str]] = None) -> str:
        """Allocate an expert to an optimal node."""
        
        with self.manager_lock:
            # Use load balancer to find optimal node
            target_node = self.load_balancer.route_request(expert_id)
            
            # Update expert assignments
            if target_node not in self.expert_assignments[expert_id]:
                self.expert_assignments[expert_id].append(target_node)
            
            return target_node
    
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        print("Started distributed resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Stopped distributed resource monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Check node health
                self._check_node_health(current_time)
                
                # Collect cluster metrics
                cluster_metrics = self._collect_cluster_metrics()
                
                # Update autoscaler
                self.autoscaler.update_cluster_metrics(cluster_metrics)
                
                # Check for scaling needs
                scaling_decision = self.autoscaler.evaluate_scaling_need()
                if scaling_decision:
                    print(f"Scaling decision: {scaling_decision.action} - {scaling_decision.reasoning}")
                    self.autoscaler.execute_scaling_decision(scaling_decision)
                
                # Check for rebalancing needs
                if self.load_balancer.check_rebalancing_needed():
                    migrations = self.load_balancer.suggest_expert_migration()
                    if migrations:
                        print(f"Suggested {len(migrations)} expert migrations for load balancing")
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _check_node_health(self, current_time: float):
        """Check health of all registered nodes."""
        
        with self.manager_lock:
            unhealthy_nodes = []
            
            for node_id, node in self.nodes.items():
                time_since_heartbeat = current_time - node.last_heartbeat
                
                if time_since_heartbeat > self.node_timeout:
                    node.is_healthy = False
                    unhealthy_nodes.append(node_id)
            
            # Handle unhealthy nodes
            for node_id in unhealthy_nodes:
                self._handle_unhealthy_node(node_id)
    
    def _handle_unhealthy_node(self, node_id: str):
        """Handle an unhealthy node."""
        
        print(f"Node {node_id} is unhealthy - redistributing experts")
        
        # Redistribute experts from unhealthy node
        node = self.nodes[node_id]
        experts_to_redistribute = node.active_experts.copy()
        
        for expert_id in experts_to_redistribute:
            # Find new node for expert
            new_node = self.allocate_expert(expert_id)
            print(f"Redistributed expert {expert_id} from {node_id} to {new_node}")
        
        # Remove from autoscaler
        self.autoscaler.current_nodes.discard(node_id)
    
    def _collect_cluster_metrics(self) -> Dict[str, float]:
        """Collect cluster-wide performance metrics."""
        
        with self.manager_lock:
            if not self.nodes:
                return {}
            
            healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]
            
            if not healthy_nodes:
                return {}
            
            # Aggregate metrics
            total_cpu = sum(node.cpu_percent for node in healthy_nodes)
            total_memory = sum(node.memory_percent for node in healthy_nodes)
            total_gpu_memory = sum(node.gpu_memory_percent for node in healthy_nodes)
            
            avg_cpu = total_cpu / len(healthy_nodes)
            avg_memory = total_memory / len(healthy_nodes)
            avg_gpu_memory = total_gpu_memory / len(healthy_nodes)
            
            # Estimate throughput and latency (simplified)
            throughput = len(healthy_nodes) * 100.0  # tokens/sec per node
            latency = max(avg_cpu, avg_memory) / 100.0 * 0.1  # Rough estimate
            
            metrics = {
                'cpu_utilization': avg_cpu / 100.0,
                'memory_utilization': avg_memory / 100.0,
                'gpu_memory_utilization': avg_gpu_memory / 100.0,
                'throughput': throughput,
                'latency': latency,
                'healthy_nodes': len(healthy_nodes),
                'total_nodes': len(self.nodes)
            }
            
            # Store in history
            self.resource_history.append({
                'metrics': metrics,
                'timestamp': time.time()
            })
            
            return metrics
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        
        with self.manager_lock:
            # Node summary
            healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
            total_experts = sum(len(node.active_experts) for node in self.nodes.values())
            
            # Component stats
            load_balancer_stats = self.load_balancer.get_load_balancing_stats()
            autoscaler_stats = self.autoscaler.get_scaling_stats()
            
            return {
                'cluster_summary': {
                    'total_nodes': len(self.nodes),
                    'healthy_nodes': healthy_nodes,
                    'total_experts': total_experts,
                    'monitoring_active': self.monitoring_active
                },
                'nodes': {
                    node_id: asdict(node) 
                    for node_id, node in self.nodes.items()
                },
                'load_balancer': load_balancer_stats,
                'autoscaler': autoscaler_stats,
                'expert_assignments': dict(self.expert_assignments),
                'recent_metrics': list(self.resource_history)[-5:] if self.resource_history else []
            }
    
    def simulate_distributed_training(self, num_steps: int = 10) -> Dict[str, Any]:
        """Simulate distributed training workload for testing."""
        
        print(f"Simulating {num_steps} training steps...")
        
        # Register some test nodes
        test_nodes = ['node_0', 'node_1', 'node_2']
        for i, node_id in enumerate(test_nodes):
            self.register_node(node_id, {
                'rank': i,
                'network_bandwidth': 1000.0
            })
        
        # Simulate training steps
        for step in range(num_steps):
            # Simulate varying workload
            base_load = 0.5 + 0.3 * np.sin(step * 0.5)  # Oscillating load
            
            for i, node_id in enumerate(test_nodes):
                # Add some noise and variation per node
                node_load = base_load + 0.1 * np.random.random() + 0.1 * i
                cpu_percent = min(95, max(10, node_load * 100))
                memory_percent = min(90, max(20, (node_load + 0.1) * 100))
                
                # Assign some experts
                active_experts = [i * 3 + j for j in range(3)]  # 3 experts per node
                
                self.update_node_status(
                    node_id,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    gpu_memory_percent=cpu_percent * 0.8,
                    active_experts=active_experts
                )
            
            time.sleep(0.1)  # Brief simulation delay
        
        return self.get_cluster_status()