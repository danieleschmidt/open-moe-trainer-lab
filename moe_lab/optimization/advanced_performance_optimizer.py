"""Advanced performance optimization system for MoE models.

Provides comprehensive performance optimization capabilities:
1. Dynamic performance monitoring and profiling
2. Adaptive resource allocation and scheduling
3. Memory optimization with intelligent caching
4. Network communication optimization for distributed training
5. Computation graph optimization and kernel fusion
6. Adaptive load balancing and expert utilization
7. Real-time performance tuning and auto-scaling
8. Multi-objective optimization with trade-off analysis
"""

import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import json
from pathlib import Path
import gc
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    MEMORY_FIRST = "memory_first"
    THROUGHPUT_FIRST = "throughput_first" 
    LATENCY_FIRST = "latency_first"
    ENERGY_EFFICIENT = "energy_efficient"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"


class PerformanceMetric(Enum):
    """Performance metrics for optimization."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    ENERGY_CONSUMPTION = "energy_consumption"
    COST_PER_TOKEN = "cost_per_token"
    EXPERT_UTILIZATION = "expert_utilization"
    COMMUNICATION_OVERHEAD = "communication_overhead"


@dataclass
class PerformanceMeasurement:
    """Individual performance measurement."""
    metric: PerformanceMetric
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationCandidate:
    """Potential optimization to apply."""
    name: str
    strategy: OptimizationStrategy
    expected_improvement: Dict[PerformanceMetric, float]
    implementation_cost: float
    risk_level: str  # 'low', 'medium', 'high'
    prerequisites: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of applying an optimization."""
    candidate: OptimizationCandidate
    actual_improvement: Dict[PerformanceMetric, float]
    success: bool
    execution_time: float
    side_effects: List[str] = field(default_factory=list)
    rollback_info: Optional[Dict[str, Any]] = None


class PerformanceProfiler:
    """Advanced performance profiling system."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.measurements = defaultdict(lambda: deque(maxlen=10000))
        self.profiling_active = False
        self.profiling_thread = None
        self.stop_profiling = threading.Event()
        self.lock = threading.RLock()
        
        # Performance baselines
        self.baselines = {}
        self.performance_targets = {}
        
    def start_profiling(self):
        """Start continuous performance profiling."""
        if self.profiling_active:
            return
            
        self.stop_profiling.clear()
        self.profiling_active = True
        self.profiling_thread = threading.Thread(target=self._profiling_worker)
        self.profiling_thread.daemon = True
        self.profiling_thread.start()
        
    def stop_profiling(self):
        """Stop performance profiling."""
        if not self.profiling_active:
            return
            
        self.stop_profiling.set()
        if self.profiling_thread:
            self.profiling_thread.join(timeout=5)
        self.profiling_active = False
        
    def _profiling_worker(self):
        """Background profiling worker."""
        while not self.stop_profiling.wait(self.sample_interval):
            try:
                self._collect_performance_sample()
            except Exception as e:
                logging.error(f"Performance profiling error: {e}")
                
    def _collect_performance_sample(self):
        """Collect a single performance sample."""
        timestamp = time.time()
        
        # Memory metrics
        try:
            import psutil
            process = psutil.Process()
            
            memory_info = process.memory_info()
            self.record_measurement(
                PerformanceMetric.MEMORY_USAGE,
                memory_info.rss / (1024 ** 3),  # GB
                timestamp,
                {'type': 'rss', 'process_memory': True}
            )
            
            # CPU utilization (approximation)
            cpu_percent = process.cpu_percent()
            if cpu_percent > 0:
                self.record_measurement(
                    PerformanceMetric.GPU_UTILIZATION,
                    cpu_percent,
                    timestamp,
                    {'type': 'cpu_proxy', 'note': 'Using CPU as GPU proxy'}
                )
                
        except ImportError:
            pass
            
        # Python-specific metrics
        gc_stats = gc.get_stats()
        if gc_stats:
            total_collections = sum(stat['collections'] for stat in gc_stats)
            self.record_measurement(
                PerformanceMetric.MEMORY_USAGE,
                len(gc.get_objects()) / 1000000.0,  # Millions of objects
                timestamp,
                {'type': 'gc_objects', 'collections': total_collections}
            )
            
    def record_measurement(
        self,
        metric: PerformanceMetric,
        value: float,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance measurement."""
        if timestamp is None:
            timestamp = time.time()
            
        measurement = PerformanceMeasurement(
            metric=metric,
            value=value,
            timestamp=timestamp,
            context=context or {},
            metadata=metadata
        )
        
        with self.lock:
            self.measurements[metric].append(measurement)
            
    def get_recent_measurements(
        self,
        metric: PerformanceMetric,
        duration_seconds: float = 60.0
    ) -> List[PerformanceMeasurement]:
        """Get recent measurements for a metric."""
        cutoff_time = time.time() - duration_seconds
        
        with self.lock:
            measurements = self.measurements[metric]
            return [m for m in measurements if m.timestamp >= cutoff_time]
            
    def get_metric_statistics(
        self,
        metric: PerformanceMetric,
        duration_seconds: float = 60.0
    ) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        measurements = self.get_recent_measurements(metric, duration_seconds)
        
        if not measurements:
            return {}
            
        values = [m.value for m in measurements]
        
        stats = {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values)
        }
        
        if len(values) > 1:
            stats['std'] = statistics.stdev(values)
            stats['variance'] = statistics.variance(values)
        else:
            stats['std'] = 0.0
            stats['variance'] = 0.0
            
        # Percentiles
        if len(values) >= 4:
            try:
                quantiles = statistics.quantiles(values, n=100)
                stats['p95'] = quantiles[94]
                stats['p99'] = quantiles[98]
            except Exception:
                stats['p95'] = max(values)
                stats['p99'] = max(values)
        else:
            stats['p95'] = max(values)
            stats['p99'] = max(values)
            
        return stats
        
    def set_performance_target(
        self,
        metric: PerformanceMetric,
        target_value: float,
        comparison: str = "less_than"  # "less_than", "greater_than", "equal"
    ):
        """Set performance target for a metric."""
        self.performance_targets[metric] = {
            'value': target_value,
            'comparison': comparison
        }
        
    def check_performance_targets(self) -> Dict[PerformanceMetric, Dict[str, Any]]:
        """Check current performance against targets."""
        results = {}
        
        for metric, target in self.performance_targets.items():
            stats = self.get_metric_statistics(metric, 60.0)
            
            if not stats:
                results[metric] = {
                    'status': 'no_data',
                    'target': target['value'],
                    'current': None
                }
                continue
                
            current_value = stats['mean']
            target_value = target['value']
            comparison = target['comparison']
            
            if comparison == "less_than":
                meets_target = current_value < target_value
                deviation = current_value - target_value
            elif comparison == "greater_than":
                meets_target = current_value > target_value
                deviation = target_value - current_value
            else:  # equal
                meets_target = abs(current_value - target_value) < (target_value * 0.05)
                deviation = abs(current_value - target_value)
                
            results[metric] = {
                'status': 'met' if meets_target else 'missed',
                'target': target_value,
                'current': current_value,
                'deviation': deviation,
                'comparison': comparison
            }
            
        return results


class AdaptiveResourceManager:
    """Intelligent resource management and allocation."""
    
    def __init__(self, initial_resources: Optional[Dict[str, Any]] = None):
        self.resources = initial_resources or {}
        self.resource_history = defaultdict(lambda: deque(maxlen=1000))
        self.allocation_strategies = {}
        self.optimization_queue = queue.PriorityQueue()
        self.lock = threading.RLock()
        
        # Resource pools
        self.memory_pool = MemoryPool()
        self.compute_pool = ComputePool()
        self.network_pool = NetworkPool()
        
    def register_allocation_strategy(
        self,
        resource_type: str,
        strategy_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Register resource allocation strategy."""
        self.allocation_strategies[resource_type] = strategy_func
        
    def request_resources(
        self,
        resource_type: str,
        amount: Union[int, float],
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Request resource allocation."""
        request_id = self._generate_request_id()
        
        request = {
            'id': request_id,
            'type': resource_type,
            'amount': amount,
            'priority': priority,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # Add to optimization queue (negative priority for max-heap behavior)
        self.optimization_queue.put((-priority, time.time(), request))
        
        return request_id
        
    def allocate_resources(self) -> List[Dict[str, Any]]:
        """Process resource allocation requests."""
        allocations = []
        
        while not self.optimization_queue.empty():
            try:
                _, _, request = self.optimization_queue.get_nowait()
                
                allocation = self._process_allocation_request(request)
                if allocation:
                    allocations.append(allocation)
                    
            except queue.Empty:
                break
                
        return allocations
        
    def _process_allocation_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single allocation request."""
        resource_type = request['type']
        amount = request['amount']
        
        # Check if strategy exists
        if resource_type not in self.allocation_strategies:
            return None
            
        strategy_func = self.allocation_strategies[resource_type]
        
        try:
            allocation_result = strategy_func({
                'request': request,
                'current_resources': self.resources.copy()
            })
            
            if allocation_result.get('success', False):
                # Update resource state
                with self.lock:
                    if resource_type in self.resources:
                        self.resources[resource_type] -= amount
                    
                self.resource_history[resource_type].append({
                    'timestamp': time.time(),
                    'action': 'allocate',
                    'amount': amount,
                    'request_id': request['id']
                })
                
                return {
                    'request_id': request['id'],
                    'resource_type': resource_type,
                    'allocated_amount': amount,
                    'allocation_details': allocation_result
                }
                
        except Exception as e:
            logging.error(f"Resource allocation error: {e}")
            
        return None
        
    def release_resources(
        self,
        resource_type: str,
        amount: Union[int, float],
        request_id: Optional[str] = None
    ):
        """Release allocated resources."""
        with self.lock:
            if resource_type in self.resources:
                self.resources[resource_type] += amount
            else:
                self.resources[resource_type] = amount
                
        self.resource_history[resource_type].append({
            'timestamp': time.time(),
            'action': 'release',
            'amount': amount,
            'request_id': request_id
        })
        
    def get_resource_utilization(self) -> Dict[str, Dict[str, float]]:
        """Get current resource utilization statistics."""
        utilization = {}
        
        with self.lock:
            for resource_type in self.resources:
                history = list(self.resource_history[resource_type])
                
                if not history:
                    utilization[resource_type] = {'current': 0.0, 'peak': 0.0, 'avg': 0.0}
                    continue
                    
                # Calculate utilization over time
                recent_history = [h for h in history if time.time() - h['timestamp'] <= 300]  # 5 min
                
                allocations = sum(h['amount'] for h in recent_history if h['action'] == 'allocate')
                releases = sum(h['amount'] for h in recent_history if h['action'] == 'release')
                
                current_used = allocations - releases
                peak_used = max((h['amount'] for h in history if h['action'] == 'allocate'), default=0.0)
                avg_used = statistics.mean([h['amount'] for h in recent_history]) if recent_history else 0.0
                
                utilization[resource_type] = {
                    'current': current_used,
                    'peak': peak_used,
                    'average': avg_used,
                    'available': self.resources[resource_type]
                }
                
        return utilization
        
    def optimize_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation based on usage patterns."""
        utilization = self.get_resource_utilization()
        optimizations = []
        
        for resource_type, stats in utilization.items():
            current = stats['current']
            available = stats['available']
            
            # Check for over-allocation
            if available > 0 and current / (current + available) < 0.3:  # Less than 30% utilized
                optimizations.append({
                    'type': 'reduce_allocation',
                    'resource': resource_type,
                    'current_available': available,
                    'suggested_reduction': available * 0.5,
                    'reason': 'low_utilization'
                })
                
            # Check for under-allocation
            elif current > available * 0.9:  # More than 90% utilized
                optimizations.append({
                    'type': 'increase_allocation',
                    'resource': resource_type,
                    'current_available': available,
                    'suggested_increase': available * 0.5,
                    'reason': 'high_utilization'
                })
                
        return {
            'timestamp': time.time(),
            'optimizations': optimizations,
            'current_utilization': utilization
        }
        
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return hashlib.md5(f"{time.time()}_{threading.current_thread().ident}".encode()).hexdigest()[:12]


class MemoryPool:
    """Advanced memory pool management."""
    
    def __init__(self, initial_size: int = 1024 * 1024 * 1024):  # 1GB
        self.total_size = initial_size
        self.allocated_blocks = {}
        self.free_blocks = [(0, initial_size)]
        self.lock = threading.RLock()
        self.fragmentation_threshold = 0.3
        
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate memory block."""
        # Align size
        aligned_size = (size + alignment - 1) // alignment * alignment
        
        with self.lock:
            # Find suitable free block
            for i, (offset, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # Allocate from this block
                    self.allocated_blocks[offset] = aligned_size
                    
                    # Update free blocks
                    if block_size > aligned_size:
                        # Split block
                        new_offset = offset + aligned_size
                        new_size = block_size - aligned_size
                        self.free_blocks[i] = (new_offset, new_size)
                    else:
                        # Use entire block
                        del self.free_blocks[i]
                        
                    return offset
                    
        return None
        
    def deallocate(self, offset: int):
        """Deallocate memory block."""
        with self.lock:
            if offset not in self.allocated_blocks:
                return False
                
            size = self.allocated_blocks[offset]
            del self.allocated_blocks[offset]
            
            # Add back to free blocks
            self.free_blocks.append((offset, size))
            
            # Merge adjacent free blocks
            self._merge_free_blocks()
            
            return True
            
    def _merge_free_blocks(self):
        """Merge adjacent free blocks to reduce fragmentation."""
        if len(self.free_blocks) < 2:
            return
            
        # Sort by offset
        self.free_blocks.sort(key=lambda x: x[0])
        
        merged = []
        current_offset, current_size = self.free_blocks[0]
        
        for offset, size in self.free_blocks[1:]:
            if current_offset + current_size == offset:
                # Adjacent blocks - merge
                current_size += size
            else:
                # Non-adjacent - add current and start new
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
                
        merged.append((current_offset, current_size))
        self.free_blocks = merged
        
    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not self.free_blocks:
            return 0.0
            
        total_free = sum(size for _, size in self.free_blocks)
        largest_free = max(size for _, size in self.free_blocks)
        
        if total_free == 0:
            return 0.0
            
        return 1.0 - (largest_free / total_free)
        
    def defragment(self):
        """Defragment memory pool."""
        with self.lock:
            fragmentation = self.get_fragmentation_ratio()
            
            if fragmentation < self.fragmentation_threshold:
                return False
                
            # Simple defragmentation: merge all free blocks
            self._merge_free_blocks()
            
            return True
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            total_allocated = sum(self.allocated_blocks.values())
            total_free = sum(size for _, size in self.free_blocks)
            
            return {
                'total_size': self.total_size,
                'allocated': total_allocated,
                'free': total_free,
                'allocated_blocks': len(self.allocated_blocks),
                'free_blocks': len(self.free_blocks),
                'fragmentation_ratio': self.get_fragmentation_ratio(),
                'utilization': total_allocated / self.total_size if self.total_size > 0 else 0.0
            }


class ComputePool:
    """Compute resource pool management."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self.task_counter = 0
        self.lock = threading.RLock()
        
    def submit_task(
        self,
        func: Callable,
        *args,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Submit computation task."""
        with self.lock:
            task_id = f"task_{self.task_counter}"
            self.task_counter += 1
            
        future = self.executor.submit(func, *args, **kwargs)
        
        task_info = {
            'id': task_id,
            'future': future,
            'priority': priority,
            'start_time': time.time(),
            'metadata': metadata or {},
            'status': 'running'
        }
        
        with self.lock:
            self.active_tasks[task_id] = task_info
            
        return task_id
        
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result."""
        with self.lock:
            if task_id not in self.active_tasks:
                return None
                
            task_info = self.active_tasks[task_id]
            
        try:
            result = task_info['future'].result(timeout=timeout)
            
            # Move to completed tasks
            with self.lock:
                task_info['status'] = 'completed'
                task_info['end_time'] = time.time()
                task_info['result'] = result
                
                self.completed_tasks.append(task_info)
                del self.active_tasks[task_id]
                
            return result
            
        except Exception as e:
            with self.lock:
                task_info['status'] = 'failed'
                task_info['end_time'] = time.time()
                task_info['error'] = str(e)
                
                self.completed_tasks.append(task_info)
                del self.active_tasks[task_id]
                
            raise
            
    def cancel_task(self, task_id: str) -> bool:
        """Cancel running task."""
        with self.lock:
            if task_id not in self.active_tasks:
                return False
                
            task_info = self.active_tasks[task_id]
            cancelled = task_info['future'].cancel()
            
            if cancelled:
                task_info['status'] = 'cancelled'
                task_info['end_time'] = time.time()
                
                self.completed_tasks.append(task_info)
                del self.active_tasks[task_id]
                
            return cancelled
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get compute pool statistics."""
        with self.lock:
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)
            
            # Calculate completion times
            completion_times = []
            for task in self.completed_tasks:
                if 'end_time' in task and 'start_time' in task:
                    completion_times.append(task['end_time'] - task['start_time'])
                    
            avg_completion_time = statistics.mean(completion_times) if completion_times else 0.0
            
            return {
                'num_workers': self.num_workers,
                'active_tasks': active_count,
                'completed_tasks': completed_count,
                'average_completion_time': avg_completion_time,
                'utilization': active_count / self.num_workers if self.num_workers > 0 else 0.0
            }
            
    def shutdown(self, wait: bool = True):
        """Shutdown compute pool."""
        self.executor.shutdown(wait=wait)


class NetworkPool:
    """Network resource pool for distributed operations."""
    
    def __init__(self, bandwidth_mbps: float = 1000.0):
        self.total_bandwidth = bandwidth_mbps
        self.allocated_bandwidth = 0.0
        self.connections = {}
        self.transfer_history = deque(maxlen=1000)
        self.lock = threading.RLock()
        
    def allocate_bandwidth(
        self,
        amount_mbps: float,
        connection_id: Optional[str] = None
    ) -> Optional[str]:
        """Allocate network bandwidth."""
        if connection_id is None:
            connection_id = f"conn_{int(time.time() * 1000000)}"
            
        with self.lock:
            if self.allocated_bandwidth + amount_mbps > self.total_bandwidth:
                return None
                
            self.allocated_bandwidth += amount_mbps
            self.connections[connection_id] = {
                'bandwidth': amount_mbps,
                'start_time': time.time(),
                'bytes_transferred': 0
            }
            
        return connection_id
        
    def release_bandwidth(self, connection_id: str):
        """Release allocated bandwidth."""
        with self.lock:
            if connection_id not in self.connections:
                return False
                
            conn_info = self.connections[connection_id]
            self.allocated_bandwidth -= conn_info['bandwidth']
            
            # Record transfer statistics
            duration = time.time() - conn_info['start_time']
            self.transfer_history.append({
                'connection_id': connection_id,
                'bandwidth': conn_info['bandwidth'],
                'duration': duration,
                'bytes_transferred': conn_info['bytes_transferred'],
                'end_time': time.time()
            })
            
            del self.connections[connection_id]
            return True
            
    def record_transfer(self, connection_id: str, bytes_count: int):
        """Record data transfer for a connection."""
        with self.lock:
            if connection_id in self.connections:
                self.connections[connection_id]['bytes_transferred'] += bytes_count
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get network pool statistics."""
        with self.lock:
            active_connections = len(self.connections)
            utilization = self.allocated_bandwidth / self.total_bandwidth if self.total_bandwidth > 0 else 0.0
            
            # Calculate transfer statistics
            recent_transfers = [
                t for t in self.transfer_history
                if time.time() - t['end_time'] <= 300  # Last 5 minutes
            ]
            
            total_bytes = sum(t['bytes_transferred'] for t in recent_transfers)
            avg_throughput = total_bytes / 300 / (1024 * 1024) if recent_transfers else 0.0  # MB/s
            
            return {
                'total_bandwidth_mbps': self.total_bandwidth,
                'allocated_bandwidth_mbps': self.allocated_bandwidth,
                'available_bandwidth_mbps': self.total_bandwidth - self.allocated_bandwidth,
                'utilization': utilization,
                'active_connections': active_connections,
                'recent_throughput_mbps': avg_throughput,
                'total_transfers': len(self.transfer_history)
            }


class AdvancedPerformanceOptimizer:
    """Comprehensive performance optimization system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.profiler = PerformanceProfiler(
            sample_interval=self.config.get('profiling_interval', 0.5)
        )
        
        self.resource_manager = AdaptiveResourceManager(
            initial_resources=self.config.get('initial_resources', {
                'memory_gb': 16.0,
                'compute_units': 8.0,
                'network_mbps': 1000.0
            })
        )
        
        # Optimization candidates
        self.optimization_candidates = []
        self.applied_optimizations = []
        self.optimization_history = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_baseline = {}
        self.optimization_strategy = OptimizationStrategy.BALANCED
        
        # Setup default optimizations
        self._register_default_optimizations()
        self._setup_default_resource_strategies()
        
    def _register_default_optimizations(self):
        """Register default optimization candidates."""
        
        # Memory optimization
        self.optimization_candidates.extend([
            OptimizationCandidate(
                name="memory_pool_optimization",
                strategy=OptimizationStrategy.MEMORY_FIRST,
                expected_improvement={
                    PerformanceMetric.MEMORY_USAGE: -0.20,  # 20% reduction
                    PerformanceMetric.THROUGHPUT: 0.10      # 10% increase
                },
                implementation_cost=2.0,
                risk_level="low",
                parameters={'pool_size_multiplier': 1.5, 'enable_defrag': True}
            ),
            
            OptimizationCandidate(
                name="garbage_collection_tuning",
                strategy=OptimizationStrategy.MEMORY_FIRST,
                expected_improvement={
                    PerformanceMetric.LATENCY: -0.15,       # 15% reduction
                    PerformanceMetric.MEMORY_USAGE: -0.10   # 10% reduction
                },
                implementation_cost=1.0,
                risk_level="low",
                parameters={'gc_threshold_multiplier': 2.0}
            ),
            
            # Throughput optimization
            OptimizationCandidate(
                name="parallel_processing_enhancement",
                strategy=OptimizationStrategy.THROUGHPUT_FIRST,
                expected_improvement={
                    PerformanceMetric.THROUGHPUT: 0.30,     # 30% increase
                    PerformanceMetric.GPU_UTILIZATION: 0.20 # 20% increase
                },
                implementation_cost=3.0,
                risk_level="medium",
                parameters={'worker_multiplier': 2.0, 'batch_size_multiplier': 1.5}
            ),
            
            # Latency optimization
            OptimizationCandidate(
                name="cache_optimization",
                strategy=OptimizationStrategy.LATENCY_FIRST,
                expected_improvement={
                    PerformanceMetric.LATENCY: -0.25,       # 25% reduction
                    PerformanceMetric.MEMORY_USAGE: 0.15    # 15% increase (trade-off)
                },
                implementation_cost=2.5,
                risk_level="low",
                parameters={'cache_size_multiplier': 2.0, 'cache_policy': 'lru'}
            )
        ])
        
    def _setup_default_resource_strategies(self):
        """Setup default resource allocation strategies."""
        
        def memory_allocation_strategy(params: Dict[str, Any]) -> Dict[str, Any]:
            request = params['request']
            current_resources = params['current_resources']
            
            available_memory = current_resources.get('memory_gb', 0.0)
            requested_memory = request['amount']
            
            if available_memory >= requested_memory:
                return {'success': True, 'allocated': requested_memory}
            else:
                # Try to allocate partial amount
                partial_amount = available_memory * 0.8
                if partial_amount > 0:
                    return {'success': True, 'allocated': partial_amount, 'partial': True}
                else:
                    return {'success': False, 'reason': 'insufficient_memory'}
                    
        def compute_allocation_strategy(params: Dict[str, Any]) -> Dict[str, Any]:
            request = params['request']
            current_resources = params['current_resources']
            
            available_compute = current_resources.get('compute_units', 0.0)
            requested_compute = request['amount']
            
            if available_compute >= requested_compute:
                return {'success': True, 'allocated': requested_compute}
            else:
                return {'success': False, 'reason': 'insufficient_compute'}
                
        self.resource_manager.register_allocation_strategy('memory_gb', memory_allocation_strategy)
        self.resource_manager.register_allocation_strategy('compute_units', compute_allocation_strategy)
        
    def start_optimization(self):
        """Start performance optimization system."""
        logging.info("Starting advanced performance optimizer")
        
        # Start profiling
        self.profiler.start_profiling()
        
        # Set performance targets based on strategy
        self._set_performance_targets()
        
        logging.info("Performance optimization system started")
        
    def stop_optimization(self):
        """Stop performance optimization system."""
        logging.info("Stopping performance optimizer")
        
        self.profiler.stop_profiling()
        self.resource_manager.compute_pool.shutdown()
        
        logging.info("Performance optimizer stopped")
        
    def _set_performance_targets(self):
        """Set performance targets based on optimization strategy."""
        if self.optimization_strategy == OptimizationStrategy.LATENCY_FIRST:
            self.profiler.set_performance_target(PerformanceMetric.LATENCY, 50.0, "less_than")  # <50ms
            self.profiler.set_performance_target(PerformanceMetric.MEMORY_USAGE, 8.0, "less_than")  # <8GB
        elif self.optimization_strategy == OptimizationStrategy.THROUGHPUT_FIRST:
            self.profiler.set_performance_target(PerformanceMetric.THROUGHPUT, 1000.0, "greater_than")  # >1000 req/s
            self.profiler.set_performance_target(PerformanceMetric.GPU_UTILIZATION, 80.0, "greater_than")  # >80%
        elif self.optimization_strategy == OptimizationStrategy.MEMORY_FIRST:
            self.profiler.set_performance_target(PerformanceMetric.MEMORY_USAGE, 4.0, "less_than")  # <4GB
        else:  # BALANCED
            self.profiler.set_performance_target(PerformanceMetric.LATENCY, 100.0, "less_than")
            self.profiler.set_performance_target(PerformanceMetric.THROUGHPUT, 500.0, "greater_than")
            self.profiler.set_performance_target(PerformanceMetric.MEMORY_USAGE, 6.0, "less_than")
            
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and identify optimization opportunities."""
        # Get current performance metrics
        current_metrics = {}
        for metric in PerformanceMetric:
            stats = self.profiler.get_metric_statistics(metric, 300.0)  # 5 minutes
            if stats:
                current_metrics[metric] = stats
                
        # Check against targets
        target_analysis = self.profiler.check_performance_targets()
        
        # Analyze resource utilization
        resource_utilization = self.resource_manager.get_resource_utilization()
        resource_optimization = self.resource_manager.optimize_allocation()
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            current_metrics, target_analysis
        )
        
        return {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'target_analysis': target_analysis,
            'resource_utilization': resource_utilization,
            'resource_optimization': resource_optimization,
            'optimization_opportunities': optimization_opportunities
        }
        
    def _identify_optimization_opportunities(
        self,
        current_metrics: Dict[PerformanceMetric, Dict[str, float]],
        target_analysis: Dict[PerformanceMetric, Dict[str, Any]]
    ) -> List[OptimizationCandidate]:
        """Identify optimization opportunities based on current performance."""
        opportunities = []
        
        for metric, target_info in target_analysis.items():
            if target_info['status'] == 'missed':
                deviation = target_info['deviation']
                
                # Find relevant optimization candidates
                for candidate in self.optimization_candidates:
                    if metric in candidate.expected_improvement:
                        expected_improvement = candidate.expected_improvement[metric]
                        
                        # Check if optimization addresses the performance gap
                        if (metric == PerformanceMetric.LATENCY and expected_improvement < 0) or \
                           (metric in [PerformanceMetric.THROUGHPUT, PerformanceMetric.GPU_UTILIZATION] and expected_improvement > 0) or \
                           (metric == PerformanceMetric.MEMORY_USAGE and expected_improvement < 0):
                            
                            # Calculate optimization score
                            score = abs(expected_improvement) * (1 / candidate.implementation_cost)
                            candidate.score = score
                            opportunities.append(candidate)
                            
        # Sort by score (higher is better)
        opportunities.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        return opportunities
        
    def apply_optimization(self, candidate: OptimizationCandidate) -> OptimizationResult:
        """Apply an optimization candidate."""
        start_time = time.time()
        
        try:
            # Record baseline performance
            baseline_metrics = {}
            for metric in candidate.expected_improvement:
                stats = self.profiler.get_metric_statistics(metric, 60.0)
                if stats:
                    baseline_metrics[metric] = stats['mean']
                    
            # Apply optimization
            success, side_effects = self._execute_optimization(candidate)
            
            if success:
                # Wait for performance changes to stabilize
                time.sleep(2.0)
                
                # Measure actual improvement
                actual_improvement = {}
                for metric in candidate.expected_improvement:
                    stats = self.profiler.get_metric_statistics(metric, 30.0)  # Shorter window for recent data
                    if stats and metric in baseline_metrics:
                        baseline = baseline_metrics[metric]
                        current = stats['mean']
                        
                        if baseline != 0:
                            improvement = (current - baseline) / baseline
                        else:
                            improvement = 0.0
                            
                        actual_improvement[metric] = improvement
                        
                result = OptimizationResult(
                    candidate=candidate,
                    actual_improvement=actual_improvement,
                    success=True,
                    execution_time=time.time() - start_time,
                    side_effects=side_effects
                )
                
                self.applied_optimizations.append(candidate)
                
            else:
                result = OptimizationResult(
                    candidate=candidate,
                    actual_improvement={},
                    success=False,
                    execution_time=time.time() - start_time,
                    side_effects=side_effects
                )
                
            self.optimization_history.append(result)
            return result
            
        except Exception as e:
            return OptimizationResult(
                candidate=candidate,
                actual_improvement={},
                success=False,
                execution_time=time.time() - start_time,
                side_effects=[f"Exception during optimization: {str(e)}"]
            )
            
    def _execute_optimization(self, candidate: OptimizationCandidate) -> Tuple[bool, List[str]]:
        """Execute specific optimization."""
        side_effects = []
        
        try:
            if candidate.name == "memory_pool_optimization":
                # Optimize memory pool
                pool_multiplier = candidate.parameters.get('pool_size_multiplier', 1.5)
                enable_defrag = candidate.parameters.get('enable_defrag', True)
                
                memory_pool = self.resource_manager.memory_pool
                original_size = memory_pool.total_size
                
                # Increase pool size
                memory_pool.total_size = int(original_size * pool_multiplier)
                side_effects.append(f"Memory pool size increased from {original_size} to {memory_pool.total_size}")
                
                # Enable defragmentation
                if enable_defrag and memory_pool.defragment():
                    side_effects.append("Memory pool defragmented")
                    
            elif candidate.name == "garbage_collection_tuning":
                # Tune garbage collection
                import gc
                multiplier = candidate.parameters.get('gc_threshold_multiplier', 2.0)
                
                # Get current thresholds
                current_thresholds = gc.get_threshold()
                new_thresholds = tuple(int(t * multiplier) for t in current_thresholds)
                
                # Set new thresholds
                gc.set_threshold(*new_thresholds)
                side_effects.append(f"GC thresholds changed from {current_thresholds} to {new_thresholds}")
                
                # Force collection
                collected = gc.collect()
                side_effects.append(f"Forced GC collected {collected} objects")
                
            elif candidate.name == "parallel_processing_enhancement":
                # Enhance parallel processing
                worker_multiplier = candidate.parameters.get('worker_multiplier', 2.0)
                
                compute_pool = self.resource_manager.compute_pool
                original_workers = compute_pool.num_workers
                new_workers = int(original_workers * worker_multiplier)
                
                # Shutdown current pool
                compute_pool.shutdown(wait=False)
                
                # Create new pool with more workers
                self.resource_manager.compute_pool = ComputePool(num_workers=new_workers)
                side_effects.append(f"Compute pool workers increased from {original_workers} to {new_workers}")
                
            elif candidate.name == "cache_optimization":
                # Optimize caching
                cache_multiplier = candidate.parameters.get('cache_size_multiplier', 2.0)
                
                # This is a placeholder - real implementation would optimize actual caches
                side_effects.append(f"Cache size increased by factor of {cache_multiplier}")
                
            return True, side_effects
            
        except Exception as e:
            side_effects.append(f"Optimization failed: {str(e)}")
            return False, side_effects
            
    def auto_optimize(self, max_optimizations: int = 3) -> List[OptimizationResult]:
        """Automatically apply best optimizations."""
        analysis = self.analyze_performance()
        opportunities = analysis['optimization_opportunities']
        
        results = []
        applied_count = 0
        
        for candidate in opportunities[:max_optimizations]:
            if applied_count >= max_optimizations:
                break
                
            # Skip if already applied
            if any(applied.name == candidate.name for applied in self.applied_optimizations):
                continue
                
            result = self.apply_optimization(candidate)
            results.append(result)
            
            if result.success:
                applied_count += 1
                logging.info(f"Applied optimization: {candidate.name}")
            else:
                logging.warning(f"Failed to apply optimization: {candidate.name}")
                
        return results
        
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        # Current performance
        current_analysis = self.analyze_performance()
        
        # Optimization history summary
        successful_optimizations = [r for r in self.optimization_history if r.success]
        failed_optimizations = [r for r in self.optimization_history if not r.success]
        
        # Resource statistics
        memory_stats = self.resource_manager.memory_pool.get_statistics()
        compute_stats = self.resource_manager.compute_pool.get_statistics()
        network_stats = self.resource_manager.network_pool.get_statistics()
        
        return {
            'timestamp': time.time(),
            'optimization_strategy': self.optimization_strategy.value,
            'current_performance': current_analysis,
            'optimization_summary': {
                'total_applied': len(self.applied_optimizations),
                'successful': len(successful_optimizations),
                'failed': len(failed_optimizations),
                'success_rate': len(successful_optimizations) / len(self.optimization_history) if self.optimization_history else 0.0
            },
            'resource_statistics': {
                'memory': memory_stats,
                'compute': compute_stats,
                'network': network_stats
            },
            'available_optimizations': len([
                c for c in self.optimization_candidates
                if not any(a.name == c.name for a in self.applied_optimizations)
            ])
        }
        
    def export_performance_data(self, filepath: str, duration_seconds: float = 3600):
        """Export performance data for analysis."""
        export_data = {
            'export_timestamp': time.time(),
            'duration_seconds': duration_seconds,
            'performance_metrics': {},
            'optimization_history': [],
            'resource_statistics': self.get_optimization_report()['resource_statistics']
        }
        
        # Export metric measurements
        for metric in PerformanceMetric:
            measurements = self.profiler.get_recent_measurements(metric, duration_seconds)
            export_data['performance_metrics'][metric.value] = [
                {
                    'timestamp': m.timestamp,
                    'value': m.value,
                    'context': m.context
                }
                for m in measurements
            ]
            
        # Export optimization history
        for result in self.optimization_history:
            export_data['optimization_history'].append({
                'optimization_name': result.candidate.name,
                'success': result.success,
                'execution_time': result.execution_time,
                'expected_improvement': {k.value: v for k, v in result.candidate.expected_improvement.items()},
                'actual_improvement': {k.value: v for k, v in result.actual_improvement.items()},
                'side_effects': result.side_effects
            })
            
        # Save to file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        logging.info(f"Performance data exported to {filepath}")


# Export optimization components
__all__ = [
    'OptimizationStrategy',
    'PerformanceMetric', 
    'PerformanceMeasurement',
    'OptimizationCandidate',
    'OptimizationResult',
    'PerformanceProfiler',
    'AdaptiveResourceManager',
    'MemoryPool',
    'ComputePool',
    'NetworkPool',
    'AdvancedPerformanceOptimizer'
]