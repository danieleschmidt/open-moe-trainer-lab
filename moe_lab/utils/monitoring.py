"""Comprehensive monitoring and metrics collection for MoE training."""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    temperature_cpu: float = 0.0
    temperature_gpu: float = 0.0


@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    timestamp: float
    epoch: int
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    load_balancing_loss: float = 0.0
    router_z_loss: float = 0.0
    expert_load_variance: float = 0.0
    router_entropy: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0


@dataclass
class ExpertMetrics:
    """Expert utilization metrics."""
    timestamp: float
    expert_id: int
    layer_id: int
    utilization_rate: float
    avg_routing_weight: float
    num_tokens_processed: int
    compute_time_ms: float
    memory_usage_mb: float


@dataclass
class ModelMetrics:
    """Model-specific metrics."""
    timestamp: float
    num_parameters: int
    num_active_parameters: int
    model_size_mb: float
    forward_time_ms: float
    backward_time_ms: float
    activation_memory_mb: float


class MetricsCollector:
    """Thread-safe metrics collection with configurable retention."""
    
    def __init__(self, max_history: int = 10000, collection_interval: float = 1.0):
        self.max_history = max_history
        self.collection_interval = collection_interval
        
        # Metrics storage
        self._system_metrics: deque = deque(maxlen=max_history)
        self._training_metrics: deque = deque(maxlen=max_history)
        self._expert_metrics: defaultdict = defaultdict(lambda: deque(maxlen=max_history))
        self._model_metrics: deque = deque(maxlen=max_history)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Collection control
        self._collecting = False
        self._collection_thread = None
        
        # Callbacks
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.info(f"Initialized MetricsCollector with {max_history} max history")
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback function for real-time metrics processing."""
        self._callbacks.append(callback)
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if self._collecting:
            logger.warning("Metrics collection already running")
            return
        
        self._collecting = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Started automatic metrics collection")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Stopped automatic metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while self._collecting:
            try:
                self.collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def collect_system_metrics(self):
        """Collect current system resource metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics (if available)
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            gpu_utilization = 0.0
            temperature_gpu = 0.0
            
            if torch.cuda.is_available():
                try:
                    gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    # GPU utilization (if nvidia-ml-py is available)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utilization = gpu_util.gpu
                        
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        temperature_gpu = temp
                    except ImportError:
                        pass  # pynvml not available
                except Exception:
                    pass  # GPU metrics not available
            
            # CPU temperature (if available)
            temperature_cpu = 0.0
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    temperature_cpu = temps['coretemp'][0].current
            except Exception:
                pass  # Temperature not available
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                gpu_memory_used_gb=gpu_memory_used,
                gpu_memory_total_gb=gpu_memory_total,
                gpu_utilization=gpu_utilization,
                temperature_cpu=temperature_cpu,
                temperature_gpu=temperature_gpu
            )
            
            with self._lock:
                self._system_metrics.append(metrics)
            
            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback({'type': 'system', 'data': asdict(metrics)})
                except Exception as e:
                    logger.error(f"Error in metrics callback: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def record_training_metrics(self, **kwargs):
        """Record training metrics."""
        metrics = TrainingMetrics(
            timestamp=time.time(),
            **kwargs
        )
        
        with self._lock:
            self._training_metrics.append(metrics)
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback({'type': 'training', 'data': asdict(metrics)})
            except Exception as e:
                logger.error(f"Error in training metrics callback: {e}")
    
    def record_expert_metrics(self, expert_id: int, layer_id: int, **kwargs):
        """Record expert-specific metrics."""
        metrics = ExpertMetrics(
            timestamp=time.time(),
            expert_id=expert_id,
            layer_id=layer_id,
            **kwargs
        )
        
        key = (expert_id, layer_id)
        with self._lock:
            self._expert_metrics[key].append(metrics)
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback({'type': 'expert', 'data': asdict(metrics)})
            except Exception as e:
                logger.error(f"Error in expert metrics callback: {e}")
    
    def record_model_metrics(self, **kwargs):
        """Record model-specific metrics."""
        metrics = ModelMetrics(
            timestamp=time.time(),
            **kwargs
        )
        
        with self._lock:
            self._model_metrics.append(metrics)
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback({'type': 'model', 'data': asdict(metrics)})
            except Exception as e:
                logger.error(f"Error in model metrics callback: {e}")
    
    def get_recent_metrics(self, metric_type: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics of specified type."""
        with self._lock:
            if metric_type == 'system':
                metrics = list(self._system_metrics)[-count:]
                return [asdict(m) for m in metrics]
            elif metric_type == 'training':
                metrics = list(self._training_metrics)[-count:]
                return [asdict(m) for m in metrics]
            elif metric_type == 'model':
                metrics = list(self._model_metrics)[-count:]
                return [asdict(m) for m in metrics]
            elif metric_type == 'expert':
                all_expert_metrics = []
                for expert_metrics in self._expert_metrics.values():
                    all_expert_metrics.extend([asdict(m) for m in list(expert_metrics)[-count:]])
                return sorted(all_expert_metrics, key=lambda x: x['timestamp'])[-count:]
            else:
                return []
    
    def get_summary_stats(self, metric_type: str, time_window: float = 300.0) -> Dict[str, Any]:
        """Get summary statistics for metrics within time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            if metric_type == 'system':
                recent_metrics = [m for m in self._system_metrics if m.timestamp > cutoff_time]
                if not recent_metrics:
                    return {}
                
                return {
                    'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
                    'max_cpu_percent': np.max([m.cpu_percent for m in recent_metrics]),
                    'avg_memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
                    'max_memory_percent': np.max([m.memory_percent for m in recent_metrics]),
                    'avg_gpu_utilization': np.mean([m.gpu_utilization for m in recent_metrics]),
                    'max_gpu_memory_used': np.max([m.gpu_memory_used_gb for m in recent_metrics]),
                    'sample_count': len(recent_metrics)
                }
            
            elif metric_type == 'training':
                recent_metrics = [m for m in self._training_metrics if m.timestamp > cutoff_time]
                if not recent_metrics:
                    return {}
                
                return {
                    'avg_loss': np.mean([m.loss for m in recent_metrics]),
                    'min_loss': np.min([m.loss for m in recent_metrics]),
                    'avg_tokens_per_second': np.mean([m.tokens_per_second for m in recent_metrics]),
                    'avg_grad_norm': np.mean([m.grad_norm for m in recent_metrics]),
                    'avg_expert_load_variance': np.mean([m.expert_load_variance for m in recent_metrics]),
                    'avg_router_entropy': np.mean([m.router_entropy for m in recent_metrics]),
                    'sample_count': len(recent_metrics)
                }
            
            elif metric_type == 'expert':
                all_recent_metrics = []
                for expert_metrics in self._expert_metrics.values():
                    all_recent_metrics.extend([m for m in expert_metrics if m.timestamp > cutoff_time])
                
                if not all_recent_metrics:
                    return {}
                
                # Group by expert
                expert_stats = defaultdict(list)
                for m in all_recent_metrics:
                    expert_stats[m.expert_id].append(m.utilization_rate)
                
                utilization_by_expert = {eid: np.mean(rates) for eid, rates in expert_stats.items()}
                
                return {
                    'expert_utilization_mean': np.mean(list(utilization_by_expert.values())),
                    'expert_utilization_std': np.std(list(utilization_by_expert.values())),
                    'most_utilized_expert': max(utilization_by_expert.keys(), key=utilization_by_expert.get),
                    'least_utilized_expert': min(utilization_by_expert.keys(), key=utilization_by_expert.get),
                    'num_active_experts': len(utilization_by_expert),
                    'sample_count': len(all_recent_metrics)
                }
            
        return {}
    
    def export_metrics(self, output_path: str, time_window: Optional[float] = None):
        """Export collected metrics to file."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        export_data = {
            'metadata': {
                'export_timestamp': current_time,
                'time_window': time_window,
                'max_history': self.max_history
            },
            'system_metrics': [],
            'training_metrics': [],
            'model_metrics': [],
            'expert_metrics': []
        }
        
        with self._lock:
            # Export system metrics
            for m in self._system_metrics:
                if m.timestamp > cutoff_time:
                    export_data['system_metrics'].append(asdict(m))
            
            # Export training metrics
            for m in self._training_metrics:
                if m.timestamp > cutoff_time:
                    export_data['training_metrics'].append(asdict(m))
            
            # Export model metrics
            for m in self._model_metrics:
                if m.timestamp > cutoff_time:
                    export_data['model_metrics'].append(asdict(m))
            
            # Export expert metrics
            for expert_metrics in self._expert_metrics.values():
                for m in expert_metrics:
                    if m.timestamp > cutoff_time:
                        export_data['expert_metrics'].append(asdict(m))
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported metrics to {output_path}")
    
    def clear_metrics(self, metric_type: Optional[str] = None):
        """Clear stored metrics."""
        with self._lock:
            if metric_type is None or metric_type == 'system':
                self._system_metrics.clear()
            if metric_type is None or metric_type == 'training':
                self._training_metrics.clear()
            if metric_type is None or metric_type == 'model':
                self._model_metrics.clear()
            if metric_type is None or metric_type == 'expert':
                self._expert_metrics.clear()
        
        logger.info(f"Cleared {metric_type or 'all'} metrics")


class PerformanceMonitor:
    """High-level performance monitoring with alerting."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts_enabled = True
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'gpu_memory_percent': 95.0,
            'temperature_cpu': 85.0,
            'temperature_gpu': 85.0,
            'loss_spike_factor': 2.0,
            'grad_norm_threshold': 10.0
        }
        
        # Alert state tracking
        self._alert_state = defaultdict(bool)
        self._last_alert_time = defaultdict(float)
        self._alert_cooldown = 300.0  # 5 minutes
        
        logger.info("Initialized PerformanceMonitor")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system health and return status."""
        health_status = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'alerts': [],
            'warnings': [],
            'system_ok': True,
            'training_ok': True
        }
        
        # Get recent metrics
        system_stats = self.metrics_collector.get_summary_stats('system', time_window=60.0)
        training_stats = self.metrics_collector.get_summary_stats('training', time_window=300.0)
        
        # Check system resources
        if system_stats:
            if system_stats.get('max_cpu_percent', 0) > self.alert_thresholds['cpu_percent']:
                health_status['alerts'].append(f"High CPU usage: {system_stats['max_cpu_percent']:.1f}%")
                health_status['system_ok'] = False
            
            if system_stats.get('max_memory_percent', 0) > self.alert_thresholds['memory_percent']:
                health_status['alerts'].append(f"High memory usage: {system_stats['max_memory_percent']:.1f}%")
                health_status['system_ok'] = False
            
            gpu_memory_percent = 0
            if system_stats.get('max_gpu_memory_used', 0) > 0:
                recent_system = self.metrics_collector.get_recent_metrics('system', count=1)
                if recent_system:
                    gpu_total = recent_system[0]['gpu_memory_total_gb']
                    if gpu_total > 0:
                        gpu_memory_percent = (system_stats['max_gpu_memory_used'] / gpu_total) * 100
            
            if gpu_memory_percent > self.alert_thresholds['gpu_memory_percent']:
                health_status['alerts'].append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
                health_status['system_ok'] = False
        
        # Check training metrics
        if training_stats:
            if training_stats.get('avg_grad_norm', 0) > self.alert_thresholds['grad_norm_threshold']:
                health_status['alerts'].append(f"High gradient norm: {training_stats['avg_grad_norm']:.2f}")
                health_status['training_ok'] = False
            
            # Check for loss spikes
            recent_training = self.metrics_collector.get_recent_metrics('training', count=10)
            if len(recent_training) >= 5:
                recent_losses = [m['loss'] for m in recent_training[-5:]]
                older_losses = [m['loss'] for m in recent_training[-10:-5]]
                
                if older_losses and recent_losses:
                    recent_avg = np.mean(recent_losses)
                    older_avg = np.mean(older_losses)
                    
                    if recent_avg > older_avg * self.alert_thresholds['loss_spike_factor']:
                        health_status['alerts'].append(f"Loss spike detected: {recent_avg:.4f} vs {older_avg:.4f}")
                        health_status['training_ok'] = False
        
        # Overall status
        if health_status['alerts']:
            health_status['overall_status'] = 'critical' if not (health_status['system_ok'] and health_status['training_ok']) else 'warning'
        
        return health_status
    
    def generate_performance_report(self, time_window: float = 3600.0) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'time_window_hours': time_window / 3600.0,
            'system_performance': self.metrics_collector.get_summary_stats('system', time_window),
            'training_performance': self.metrics_collector.get_summary_stats('training', time_window),
            'expert_utilization': self.metrics_collector.get_summary_stats('expert', time_window),
            'health_status': self.check_system_health(),
            'recommendations': []
        }
        
        # Add recommendations based on metrics
        if report['system_performance']:
            sys_perf = report['system_performance']
            
            if sys_perf.get('avg_cpu_percent', 0) > 80:
                report['recommendations'].append("Consider reducing batch size or increasing gradient accumulation to reduce CPU load")
            
            if sys_perf.get('avg_memory_percent', 0) > 80:
                report['recommendations'].append("High memory usage detected - consider model sharding or reducing model size")
            
            if sys_perf.get('avg_gpu_utilization', 0) < 50:
                report['recommendations'].append("Low GPU utilization - consider increasing batch size or optimizing data loading")
        
        if report['training_performance']:
            train_perf = report['training_performance']
            
            if train_perf.get('avg_expert_load_variance', 0) > 0.5:
                report['recommendations'].append("High expert load variance - consider adjusting load balancing loss coefficient")
            
            if train_perf.get('avg_router_entropy', 0) < 1.0:
                report['recommendations'].append("Low router entropy - experts may not be sufficiently diverse")
        
        return report


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def setup_monitoring(max_history: int = 10000, collection_interval: float = 1.0, auto_start: bool = True) -> MetricsCollector:
    """Setup global monitoring system."""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(max_history, collection_interval)
    
    if auto_start:
        _global_metrics_collector.start_collection()
    
    logger.info("Setup monitoring system")
    return _global_metrics_collector


def cleanup_monitoring():
    """Cleanup global monitoring system."""
    global _global_metrics_collector
    if _global_metrics_collector:
        _global_metrics_collector.stop_collection()
        _global_metrics_collector = None
    logger.info("Cleaned up monitoring system")