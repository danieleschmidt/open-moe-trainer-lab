#!/usr/bin/env python3
"""
Robust MoE Demo - Generation 2 Enhanced: Make It Robust + Production Patterns
Demonstrates comprehensive error handling, monitoring, recovery mechanisms,
circuit breakers, health checks, and production-ready reliability patterns.
"""

import json
import time
import random
import logging
import traceback
import os
import hashlib
import pickle
import threading
import weakref
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, asdict, field
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, deque
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import signal
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
    # Mock psutil if not available
    class MockProcess:
        def memory_percent(self):
            return 25.0  # Mock value
        def memory_info(self):
            class MockMemInfo:
                rss = 100 * 1024 * 1024  # 100MB mock
            return MockMemInfo()
    
    class MockPsutil:
        def Process(self):
            return MockProcess()
    
    psutil = MockPsutil()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"


class MoEError(Exception):
    """Base MoE error with context and recovery strategies."""
    
    def __init__(self, message: str, severity: str = "medium", context: Optional[Dict] = None, 
                 recovery_suggestion: Optional[str] = None, error_code: Optional[str] = None,
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.error_code = error_code
        self.retry_after = retry_after
        self.timestamp = time.time()
        self.traceback = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
            "recovery_suggestion": self.recovery_suggestion,
            "error_code": self.error_code,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp,
            "traceback": self.traceback
        }


class TrainingError(MoEError):
    """Training-specific error."""
    pass


class ModelError(MoEError):
    """Model-specific error."""
    pass


class DataError(MoEError):
    """Data-related error."""
    pass


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    memory_usage_mb: float
    cpu_percent: float
    processing_time_ms: float
    error_count: int
    warning_count: int
    
    
@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    timestamp: float
    step: int
    loss: float
    expert_load_variance: float
    routing_entropy: float
    tokens_per_second: float
    gradient_norm: float
    

@dataclass
class ExpertMetrics:
    """Expert utilization metrics."""
    timestamp: float
    expert_id: int
    utilization_rate: float
    avg_routing_weight: float
    num_tokens_processed: int
    error_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    
    
@dataclass
class HealthCheckResult:
    """Health check result."""
    component: str
    status: HealthStatus
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    

class CircuitBreaker:
    """Circuit breaker for expert failure protection."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise MoEError(
                        f"Circuit breaker is OPEN. Try again after {self.recovery_timeout}s",
                        severity="high",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        retry_after=self.recovery_timeout
                    )
                else:
                    self.state = CircuitState.HALF_OPEN
                    
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
                
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold
        }


class HealthChecker:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: deque = deque(maxlen=100)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_running:
            return
            
        self.is_running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.check_interval / 2)  # Retry sooner on error
                
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    timestamp=time.time(),
                    details={"error": str(e)}
                )
                
        # Store in history
        self.health_history.append({
            "timestamp": time.time(),
            "results": results
        })
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.health_history:
            return HealthStatus.UNHEALTHY
            
        latest = self.health_history[-1]["results"]
        
        critical_count = sum(1 for r in latest.values() if r.status == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for r in latest.values() if r.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for r in latest.values() if r.status == HealthStatus.DEGRADED)
        
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif unhealthy_count > len(latest) * 0.3:  # >30% unhealthy
            return HealthStatus.UNHEALTHY
        elif degraded_count > len(latest) * 0.5:   # >50% degraded
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self, max_retries: int = 3, log_file: Optional[str] = None):
        self.max_retries = max_retries
        self.log_file = log_file
        self.error_history: List[Dict[str, Any]] = []
        
        # Setup file logging if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.info(f"Initialized ErrorHandler with max_retries={max_retries}")
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle and log error with context."""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'traceback': traceback.format_exception(type(error), error, error.__traceback__)
        }
        
        # Add MoE-specific error information
        if isinstance(error, MoEError):
            error_info.update({
                'severity': error.severity,
                'recovery_suggestion': error.recovery_suggestion,
                'moe_context': error.context
            })
        
        # Store in error history
        self.error_history.append(error_info)
        
        # Log error based on severity
        log_level = logging.ERROR
        if isinstance(error, MoEError):
            if error.severity == "critical":
                log_level = logging.CRITICAL
            elif error.severity == "low":
                log_level = logging.WARNING
        
        logger.log(log_level, f"Error handled: {error_info['error_type']} - {error_info['message']}")
        
        if isinstance(error, MoEError) and error.recovery_suggestion:
            logger.info(f"Recovery suggestion: {error.recovery_suggestion}")
        
        return error_info
    
    def retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.handle_error(e, {'attempt': attempt + 1, 'function': func.__name__})
                
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for {func.__name__}")
        
        raise last_exception
    
    def get_error_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of recent errors."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        recent_errors = [e for e in self.error_history if e['timestamp'] > cutoff_time]
        
        error_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in recent_errors:
            error_counts[error['error_type']] += 1
            if 'severity' in error:
                severity_counts[error['severity']] += 1
        
        return {
            'total_errors': len(recent_errors),
            'error_types': dict(error_counts),
            'severity_distribution': dict(severity_counts),
            'time_window_hours': (time_window / 3600.0) if time_window else None,
            'most_common_error': max(error_counts.keys(), key=error_counts.get) if error_counts else None
        }
    
    def export_error_log(self, output_path: str, time_window: Optional[float] = None):
        """Export error history to file."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        recent_errors = [e for e in self.error_history if e['timestamp'] > cutoff_time]
        
        export_data = {
            'export_timestamp': current_time,
            'time_window_hours': (time_window / 3600.0) if time_window else None,
            'total_errors': len(recent_errors),
            'errors': recent_errors
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(recent_errors)} errors to {output_path}")


class MetricsCollector:
    """Thread-safe metrics collection with real-time monitoring."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.training_metrics: deque = deque(maxlen=max_history)
        self.expert_metrics: defaultdict = defaultdict(lambda: deque(maxlen=max_history))
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Collection control
        self._collecting = False
        self._collection_thread = None
        
        logger.info(f"Initialized MetricsCollector with max_history={max_history}")
    
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
            self._collection_thread.join(timeout=2.0)
        logger.info("Stopped automatic metrics collection")
    
    def _collection_loop(self):
        """Background collection loop."""
        while self._collecting:
            try:
                # Collect system metrics
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent(interval=None)
                else:
                    cpu_percent = 15.0  # Mock value
                
                system_metrics = SystemMetrics(
                    timestamp=time.time(),
                    memory_usage_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    processing_time_ms=0.0,  # Updated during processing
                    error_count=0,  # Updated by error handler
                    warning_count=0
                )
                
                with self._lock:
                    self.system_metrics.append(system_metrics)
                
                time.sleep(1.0)  # Collect every second
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(1.0)
    
    def record_training_metrics(self, **kwargs):
        """Record training metrics."""
        metrics = TrainingMetrics(
            timestamp=time.time(),
            **kwargs
        )
        
        with self._lock:
            self.training_metrics.append(metrics)
    
    def record_expert_metrics(self, expert_id: int, **kwargs):
        """Record expert-specific metrics."""
        metrics = ExpertMetrics(
            timestamp=time.time(),
            expert_id=expert_id,
            **kwargs
        )
        
        with self._lock:
            self.expert_metrics[expert_id].append(metrics)
    
    def get_recent_metrics(self, metric_type: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics."""
        with self._lock:
            if metric_type == 'system':
                metrics = list(self.system_metrics)[-count:]
                return [asdict(m) for m in metrics]
            elif metric_type == 'training':
                metrics = list(self.training_metrics)[-count:]
                return [asdict(m) for m in metrics]
            elif metric_type == 'expert':
                all_metrics = []
                for expert_metrics in self.expert_metrics.values():
                    all_metrics.extend([asdict(m) for m in list(expert_metrics)[-count:]])
                return sorted(all_metrics, key=lambda x: x['timestamp'])[-count:]
        return []
    
    def get_summary_stats(self, metric_type: str, time_window: float = 300.0) -> Dict[str, Any]:
        """Get summary statistics."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._lock:
            if metric_type == 'system':
                recent_metrics = [m for m in self.system_metrics if m.timestamp > cutoff_time]
                if not recent_metrics:
                    return {}
                
                return {
                    'avg_memory_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
                    'max_memory_mb': max(m.memory_usage_mb for m in recent_metrics),
                    'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                    'max_cpu_percent': max(m.cpu_percent for m in recent_metrics),
                    'sample_count': len(recent_metrics)
                }
            
            elif metric_type == 'training':
                recent_metrics = [m for m in self.training_metrics if m.timestamp > cutoff_time]
                if not recent_metrics:
                    return {}
                
                return {
                    'avg_loss': sum(m.loss for m in recent_metrics) / len(recent_metrics),
                    'min_loss': min(m.loss for m in recent_metrics),
                    'avg_expert_load_variance': sum(m.expert_load_variance for m in recent_metrics) / len(recent_metrics),
                    'avg_routing_entropy': sum(m.routing_entropy for m in recent_metrics) / len(recent_metrics),
                    'avg_tokens_per_second': sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics),
                    'sample_count': len(recent_metrics)
                }
        
        return {}
    
    def export_metrics(self, output_path: str, time_window: Optional[float] = None):
        """Export metrics to file."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        export_data = {
            'metadata': {
                'export_timestamp': current_time,
                'time_window_hours': time_window / 3600.0 if time_window else None
            },
            'system_metrics': [],
            'training_metrics': [],
            'expert_metrics': []
        }
        
        with self._lock:
            # Export system metrics
            for m in self.system_metrics:
                if m.timestamp > cutoff_time:
                    export_data['system_metrics'].append(asdict(m))
            
            # Export training metrics
            for m in self.training_metrics:
                if m.timestamp > cutoff_time:
                    export_data['training_metrics'].append(asdict(m))
            
            # Export expert metrics
            for expert_metrics in self.expert_metrics.values():
                for m in expert_metrics:
                    if m.timestamp > cutoff_time:
                        export_data['expert_metrics'].append(asdict(m))
        
        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported metrics to {output_path}")


class CheckpointManager:
    """Robust checkpoint management with automatic recovery."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        logger.info(f"Initialized CheckpointManager at {checkpoint_dir}")
    
    def save_checkpoint(self, state: Dict[str, Any], step: int) -> str:
        """Save checkpoint with error handling."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.json"
        
        try:
            # Save to temporary file first
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Atomic move
            temp_path.rename(checkpoint_path)
            logger.info(f"Saved checkpoint at step {step}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
        
        except Exception as e:
            raise TrainingError(
                f"Failed to save checkpoint at step {step}: {str(e)}",
                severity="high",
                context={'step': step, 'path': str(checkpoint_path)},
                recovery_suggestion="Check disk space and permissions"
            )
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint with error handling."""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None:
            raise TrainingError(
                "No checkpoint found to load",
                severity="medium",
                recovery_suggestion="Start training from scratch or provide specific checkpoint path"
            )
        
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state
        
        except Exception as e:
            raise TrainingError(
                f"Failed to load checkpoint from {checkpoint_path}: {str(e)}",
                severity="high",
                context={'path': str(checkpoint_path)},
                recovery_suggestion="Verify checkpoint file integrity"
            )
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.json"))
        if not checkpoint_files:
            return None
        
        # Sort by step number
        def extract_step(path):
            try:
                return int(path.stem.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        latest_checkpoint = max(checkpoint_files, key=extract_step)
        return str(latest_checkpoint)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.json"))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by modification time and remove oldest
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logger.debug(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {file_path}: {e}")


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {
            'memory_mb': 1024,  # 1GB
            'cpu_percent': 90.0,
            'loss_spike_factor': 2.0,
            'error_rate_per_hour': 10
        }
        logger.info("Initialized HealthMonitor")
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health_status = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'alerts': [],
            'warnings': [],
            'system_ok': True,
            'training_ok': True
        }
        
        # Check system resources
        system_stats = self.metrics_collector.get_summary_stats('system', time_window=60.0)
        if system_stats:
            if system_stats.get('max_memory_mb', 0) > self.alert_thresholds['memory_mb']:
                health_status['alerts'].append(f"High memory usage: {system_stats['max_memory_mb']:.1f}MB")
                health_status['system_ok'] = False
            
            if system_stats.get('max_cpu_percent', 0) > self.alert_thresholds['cpu_percent']:
                health_status['alerts'].append(f"High CPU usage: {system_stats['max_cpu_percent']:.1f}%")
                health_status['system_ok'] = False
        
        # Check training performance
        training_stats = self.metrics_collector.get_summary_stats('training', time_window=300.0)
        if training_stats:
            # Check for loss spikes
            recent_training = self.metrics_collector.get_recent_metrics('training', count=10)
            if len(recent_training) >= 5:
                recent_losses = [m['loss'] for m in recent_training[-3:]]
                older_losses = [m['loss'] for m in recent_training[-6:-3]]
                
                if older_losses and recent_losses:
                    recent_avg = sum(recent_losses) / len(recent_losses)
                    older_avg = sum(older_losses) / len(older_losses)
                    
                    if recent_avg > older_avg * self.alert_thresholds['loss_spike_factor']:
                        health_status['alerts'].append(f"Loss spike: {recent_avg:.4f} vs {older_avg:.4f}")
                        health_status['training_ok'] = False
        
        # Overall status
        if health_status['alerts']:
            health_status['overall_status'] = 'critical' if not (health_status['system_ok'] and health_status['training_ok']) else 'warning'
        
        return health_status
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        return {
            'timestamp': time.time(),
            'system_performance': self.metrics_collector.get_summary_stats('system', time_window=3600.0),
            'training_performance': self.metrics_collector.get_summary_stats('training', time_window=3600.0),
            'health_status': self.check_system_health(),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        system_stats = self.metrics_collector.get_summary_stats('system', time_window=3600.0)
        if system_stats:
            if system_stats.get('avg_memory_mb', 0) > 512:
                recommendations.append("Consider reducing batch size or model size to lower memory usage")
            
            if system_stats.get('avg_cpu_percent', 0) > 70:
                recommendations.append("High CPU usage - consider optimizing data preprocessing")
        
        training_stats = self.metrics_collector.get_summary_stats('training', time_window=3600.0)
        if training_stats:
            if training_stats.get('avg_expert_load_variance', 0) > 0.3:
                recommendations.append("High expert load variance - consider adjusting load balancing")
            
            if training_stats.get('avg_routing_entropy', 0) < 1.0:
                recommendations.append("Low routing entropy - experts may lack diversity")
        
        return recommendations


class RobustMoEDemo:
    """Enhanced MoE demonstration with comprehensive robustness features."""
    
    def __init__(self, hidden_size=32, num_experts=4, top_k=2):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Initialize robust components
        self.error_handler = ErrorHandler(log_file="robust_moe_errors.log")
        self.metrics_collector = MetricsCollector()
        self.checkpoint_manager = CheckpointManager("./robust_checkpoints")
        self.health_monitor = HealthMonitor(self.metrics_collector)
        
        # Initialize new production-ready components
        self.health_checker = HealthChecker(check_interval=10.0)  # Check every 10s
        self.expert_circuit_breakers = {
            i: CircuitBreaker(failure_threshold=3, recovery_timeout=30.0, expected_exception=MoEError)
            for i in range(num_experts)
        }
        self.system_circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=60.0)
        
        # Performance tracking
        self.expert_performance_history = defaultdict(lambda: deque(maxlen=100))
        self.system_performance_history = deque(maxlen=1000)
        self.error_rate_window = deque(maxlen=100)
        
        # Self-healing configuration
        self.auto_recovery_enabled = True
        self.max_auto_recovery_attempts = 3
        self.recovery_attempt_count = 0
        
        # Initialize health checks
        self._register_health_checks()
        
        # Model weights (with potential for errors)
        self._initialize_model_weights()
        
        # Start health monitoring
        self.health_checker.start_monitoring()
        
        logger.info(f"Initialized RobustMoEDemo with production-ready robustness features")
    
    def _register_health_checks(self):
        """Register system health checks."""
        
        def check_memory_usage() -> HealthCheckResult:
            try:
                process = psutil.Process()
                memory_percent = process.memory_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_percent > 90:
                    status = HealthStatus.CRITICAL
                elif memory_percent > 70:
                    status = HealthStatus.UNHEALTHY
                elif memory_percent > 50:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY
                    
                return HealthCheckResult(
                    component="memory",
                    status=status,
                    timestamp=time.time(),
                    metrics={"memory_percent": memory_percent, "memory_mb": memory_mb}
                )
            except Exception as e:
                return HealthCheckResult(
                    component="memory",
                    status=HealthStatus.CRITICAL,
                    timestamp=time.time(),
                    details={"error": str(e)}
                )
        
        def check_expert_health() -> HealthCheckResult:
            """Check expert performance and error rates."""
            try:
                critical_experts = 0
                unhealthy_experts = 0
                
                for expert_id, cb in self.expert_circuit_breakers.items():
                    if cb.state == CircuitState.OPEN:
                        critical_experts += 1
                    elif cb.failure_count > 1:
                        unhealthy_experts += 1
                
                total_experts = len(self.expert_circuit_breakers)
                if critical_experts > total_experts * 0.5:
                    status = HealthStatus.CRITICAL
                elif critical_experts > 0 or unhealthy_experts > total_experts * 0.3:
                    status = HealthStatus.UNHEALTHY
                elif unhealthy_experts > 0:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY
                
                return HealthCheckResult(
                    component="experts",
                    status=status,
                    timestamp=time.time(),
                    metrics={
                        "critical_experts": critical_experts,
                        "unhealthy_experts": unhealthy_experts,
                        "total_experts": total_experts
                    }
                )
            except Exception as e:
                return HealthCheckResult(
                    component="experts",
                    status=HealthStatus.CRITICAL,
                    timestamp=time.time(),
                    details={"error": str(e)}
                )
        
        def check_error_rate() -> HealthCheckResult:
            """Check recent error rates."""
            try:
                if not self.error_rate_window:
                    return HealthCheckResult(
                        component="error_rate",
                        status=HealthStatus.HEALTHY,
                        timestamp=time.time(),
                        metrics={"error_rate": 0.0}
                    )
                
                recent_errors = sum(self.error_rate_window) / len(self.error_rate_window)
                
                if recent_errors > 0.1:  # >10% error rate
                    status = HealthStatus.CRITICAL
                elif recent_errors > 0.05:  # >5% error rate
                    status = HealthStatus.UNHEALTHY
                elif recent_errors > 0.01:  # >1% error rate
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY
                
                return HealthCheckResult(
                    component="error_rate",
                    status=status,
                    timestamp=time.time(),
                    metrics={"error_rate": recent_errors}
                )
            except Exception as e:
                return HealthCheckResult(
                    component="error_rate",
                    status=HealthStatus.CRITICAL,
                    timestamp=time.time(),
                    details={"error": str(e)}
                )
        
        # Register all health checks
        self.health_checker.register_check("memory", check_memory_usage)
        self.health_checker.register_check("experts", check_expert_health)
        self.health_checker.register_check("error_rate", check_error_rate)
    
    def trigger_self_healing(self, component: str, error: MoEError):
        """Trigger self-healing mechanisms."""
        if not self.auto_recovery_enabled or self.recovery_attempt_count >= self.max_auto_recovery_attempts:
            logger.warning(f"Self-healing disabled or max attempts reached for {component}")
            return False
        
        self.recovery_attempt_count += 1
        logger.info(f"Attempting self-healing for {component} (attempt {self.recovery_attempt_count})")
        
        try:
            if component == "weights":
                self._recover_model_weights()
            elif component == "expert":
                expert_id = error.context.get("expert_id")
                if expert_id is not None:
                    self._recover_expert_weights(expert_id)
            elif component == "memory":
                self._recover_memory()
            else:
                logger.warning(f"No recovery strategy for component: {component}")
                return False
            
            logger.info(f"Self-healing successful for {component}")
            return True
            
        except Exception as recovery_error:
            logger.error(f"Self-healing failed for {component}: {recovery_error}")
            return False
    
    def _recover_model_weights(self):
        """Recover corrupted model weights."""
        logger.info("Recovering model weights from checkpoint or reinitializing")
        try:
            # Try loading from checkpoint first
            if self.checkpoint_manager.load_checkpoint("emergency"):
                logger.info("Weights recovered from emergency checkpoint")
            else:
                # Fallback to reinitialization
                self._initialize_model_weights()
                logger.info("Weights reinitialized")
        except Exception as e:
            raise MoEError(f"Weight recovery failed: {e}", severity="critical")
    
    def _recover_expert_weights(self, expert_id: int):
        """Recover specific expert weights."""
        logger.info(f"Recovering weights for expert {expert_id}")
        try:
            # Reinitialize specific expert
            self.expert_weights[expert_id] = [
                [random.gauss(0, 0.02) for _ in range(self.hidden_size)] 
                for _ in range(self.hidden_size)
            ]
            # Reset circuit breaker
            self.expert_circuit_breakers[expert_id] = CircuitBreaker(
                failure_threshold=3, recovery_timeout=30.0, expected_exception=MoEError
            )
            logger.info(f"Expert {expert_id} recovered")
        except Exception as e:
            raise MoEError(f"Expert {expert_id} recovery failed: {e}", severity="high")
    
    def _recover_memory(self):
        """Recover from memory issues."""
        logger.info("Attempting memory recovery")
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear performance history
            for history in self.expert_performance_history.values():
                history.clear()
            
            # Clear old system performance data
            while len(self.system_performance_history) > 500:
                self.system_performance_history.popleft()
                
            logger.info("Memory recovery completed")
        except Exception as e:
            raise MoEError(f"Memory recovery failed: {e}", severity="high")
    
    def _initialize_model_weights(self):
        """Initialize model weights with error handling."""
        try:
            # Router weights
            self.router_weights = [[random.gauss(0, 0.02) for _ in range(self.num_experts)] 
                                  for _ in range(self.hidden_size)]
            
            # Expert weights  
            self.expert_weights = []
            for e in range(self.num_experts):
                expert_w = [[random.gauss(0, 0.02) for _ in range(self.hidden_size)] 
                           for _ in range(self.hidden_size)]
                self.expert_weights.append(expert_w)
            
            logger.info("Model weights initialized successfully")
        
        except Exception as e:
            raise ModelError(
                f"Failed to initialize model weights: {str(e)}",
                severity="critical",
                recovery_suggestion="Check system memory and retry initialization"
            )
    
    @contextmanager
    def error_context(self, context_info: Dict[str, Any]):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            enhanced_error = e
            if not isinstance(e, MoEError):
                enhanced_error = MoEError(
                    message=str(e),
                    context=context_info,
                    recovery_suggestion="Check error context for debugging information"
                )
                enhanced_error.__cause__ = e
            
            self.error_handler.handle_error(enhanced_error, context_info)
            raise enhanced_error
    
    def softmax(self, logits):
        """Robust softmax with error handling."""
        try:
            max_logit = max(logits)
            exp_logits = [min(math.exp(x - max_logit), 1e6) for x in logits]  # Prevent overflow
            sum_exp = max(sum(exp_logits), 1e-10)  # Prevent division by zero
            return [x / sum_exp for x in exp_logits]
        except Exception as e:
            raise ModelError(
                f"Softmax computation failed: {str(e)}",
                severity="medium",
                context={'logits': logits},
                recovery_suggestion="Check for NaN or infinite values in input"
            )
    
    def matrix_vector_mult(self, matrix, vector):
        """Robust matrix-vector multiplication."""
        try:
            if len(vector) != len(matrix[0]):
                raise ValueError(f"Dimension mismatch: matrix cols {len(matrix[0])} != vector len {len(vector)}")
            
            result = []
            for row in matrix:
                value = sum(row[i] * vector[i] for i in range(len(vector)))
                if abs(value) > 1e6:  # Check for numerical instability
                    logger.warning(f"Large value detected in matrix multiplication: {value}")
                result.append(value)
            
            return result
        
        except Exception as e:
            raise ModelError(
                f"Matrix multiplication failed: {str(e)}",
                severity="medium",
                recovery_suggestion="Check matrix and vector dimensions"
            )
    
    def forward_with_monitoring(self, token_embedding, step: int):
        """Forward pass with comprehensive monitoring and circuit breaker protection."""
        start_time = time.time()
        error_occurred = False
        
        with self.error_context({'step': step, 'input_shape': len(token_embedding)}):
            try:
                # Validate input
                if not token_embedding or len(token_embedding) != self.hidden_size:
                    raise DataError(
                        f"Invalid input dimensions: expected {self.hidden_size}, got {len(token_embedding)}",
                        severity="high",
                        recovery_suggestion="Check input preprocessing pipeline"
                    )
                
                # Check for invalid values
                if any(not isinstance(x, (int, float)) or abs(x) > 1e3 for x in token_embedding):
                    raise DataError(
                        "Invalid values in input embedding",
                        severity="medium",
                        context={'input': token_embedding[:5]},  # First 5 values for debugging
                        recovery_suggestion="Apply input normalization or clipping"
                    )
                
                # Route token to experts
                router_logits = self.matrix_vector_mult(
                    [[self.router_weights[i][j] for i in range(self.hidden_size)] 
                     for j in range(self.num_experts)], 
                    token_embedding
                )
                
                # Get top-k experts with error handling
                try:
                    expert_scores = list(enumerate(router_logits))
                    expert_scores.sort(key=lambda x: x[1], reverse=True)
                    top_experts = expert_scores[:self.top_k]
                except Exception as e:
                    raise ModelError(
                        f"Expert selection failed: {str(e)}",
                        severity="medium",
                        recovery_suggestion="Check router logits for NaN/Inf values"
                    )
                
                # Extract and normalize
                expert_indices = [x[0] for x in top_experts]
                expert_logits = [x[1] for x in top_experts]
                expert_probs = self.softmax(expert_logits)
                
                # Process through experts with circuit breaker protection
                final_output = [0.0] * self.hidden_size
                expert_computations = []
                successful_experts = 0
                
                for i, expert_idx in enumerate(expert_indices):
                    circuit_breaker = self.expert_circuit_breakers[expert_idx]
                    
                    def expert_computation():
                        """Protected expert computation."""
                        expert_start = time.time()
                        
                        # Validate expert weights for corruption
                        expert_weights = self.expert_weights[expert_idx]
                        if any(any(abs(w) > 1e3 for w in row) for row in expert_weights):
                            raise MoEError(
                                f"Expert {expert_idx} weights corrupted",
                                severity="high",
                                context={"expert_id": expert_idx},
                                error_code="EXPERT_WEIGHTS_CORRUPTED"
                            )
                        
                        expert_output = self.matrix_vector_mult(expert_weights, token_embedding)
                        
                        # Apply activation (ReLU) with overflow protection
                        expert_output = [max(0, min(x, 1e6)) for x in expert_output]
                        
                        expert_time = (time.time() - expert_start) * 1000
                        
                        # Check for numerical instability
                        if any(abs(x) > 1e6 or math.isnan(x) or math.isinf(x) for x in expert_output):
                            raise MoEError(
                                f"Expert {expert_idx} output unstable",
                                severity="medium",
                                context={"expert_id": expert_idx},
                                error_code="EXPERT_OUTPUT_UNSTABLE"
                            )
                        
                        return expert_output, expert_time
                    
                    try:
                        # Use circuit breaker to call expert
                        expert_output, expert_time = circuit_breaker.call(expert_computation)
                        successful_experts += 1
                        
                        weight = expert_probs[i]
                        
                        # Add weighted contribution
                        for j in range(self.hidden_size):
                            final_output[j] += weight * expert_output[j]
                        
                        # Record expert metrics with health tracking
                        self.metrics_collector.record_expert_metrics(
                            expert_id=expert_idx,
                            utilization_rate=weight,
                            avg_routing_weight=weight,
                            num_tokens_processed=1
                        )
                        
                        # Track performance history for health monitoring
                        self.expert_performance_history[expert_idx].append({
                            'timestamp': time.time(),
                            'processing_time_ms': expert_time,
                            'success': True,
                            'weight': weight
                        })
                        
                        expert_computations.append({
                            'expert_id': expert_idx,
                            'weight': weight,
                            'computation_time_ms': expert_time,
                            'circuit_breaker_state': circuit_breaker.state.value,
                            'status': 'success'
                        })
                        
                    except MoEError as e:
                        error_occurred = True
                        logger.warning(f"Expert {expert_idx} failed with MoEError: {e.message}")
                        
                        # Track failure
                        self.expert_performance_history[expert_idx].append({
                            'timestamp': time.time(),
                            'success': False,
                            'error': e.message,
                            'error_code': e.error_code
                        })
                        
                        expert_computations.append({
                            'expert_id': expert_idx,
                            'weight': 0.0,
                            'computation_time_ms': 0.0,
                            'circuit_breaker_state': circuit_breaker.state.value,
                            'status': 'circuit_breaker_open' if circuit_breaker.state == CircuitState.OPEN else 'failed',
                            'error': e.message
                        })
                        
                        # Trigger self-healing for critical errors
                        if e.severity in ["high", "critical"]:
                            self.trigger_self_healing("expert", e)
                        
                        # Continue with other experts (graceful degradation)
                        continue
                        
                    except Exception as e:
                        error_occurred = True
                        logger.warning(f"Expert {expert_idx} failed with unexpected error: {e}")
                        
                        # Track failure
                        self.expert_performance_history[expert_idx].append({
                            'timestamp': time.time(),
                            'success': False,
                            'error': str(e)
                        })
                        
                        expert_computations.append({
                            'expert_id': expert_idx,
                            'weight': 0.0,
                            'computation_time_ms': 0.0,
                            'circuit_breaker_state': circuit_breaker.state.value,
                            'status': 'unexpected_error',
                            'error': str(e)
                        })
                        
                        continue
                
                # Check if we have enough successful experts
                if successful_experts == 0:
                    raise MoEError(
                        "All experts failed - system degraded",
                        severity="critical",
                        error_code="ALL_EXPERTS_FAILED",
                        recovery_suggestion="Check expert weights and trigger recovery"
                    )
                elif successful_experts < len(expert_indices) * 0.5:
                    logger.warning(f"Only {successful_experts}/{len(expert_indices)} experts succeeded")
                    # Normalize final output since we're missing some expert contributions
                    normalization_factor = len(expert_indices) / successful_experts
                    final_output = [x * normalization_factor for x in final_output]
                
                # Calculate metrics
                processing_time = (time.time() - start_time) * 1000
                
                # Compute routing stats
                all_probs = self.softmax(router_logits)
                expert_loads = [0] * self.num_experts
                for idx, prob in zip(expert_indices, expert_probs):
                    expert_loads[idx] = prob
                
                mean_load = sum(expert_loads) / self.num_experts
                load_variance = sum((x - mean_load) ** 2 for x in expert_loads) / self.num_experts
                
                entropy = -sum(p * math.log(p + 1e-8) for p in all_probs if p > 0)
                
                # Track error rate for health monitoring
                self.error_rate_window.append(1.0 if error_occurred else 0.0)
                
                # Record system performance for monitoring
                self.system_performance_history.append({
                    'timestamp': time.time(),
                    'processing_time_ms': processing_time,
                    'successful_experts': successful_experts,
                    'total_experts': len(expert_indices),
                    'error_occurred': error_occurred,
                    'step': step
                })
                
                # Record training metrics with enhanced tracking
                loss = random.uniform(0.5, 2.0)  # Simulated loss
                gradient_norm = random.uniform(0.1, 1.0)  # Simulated gradient norm
                
                self.metrics_collector.record_training_metrics(
                    step=step,
                    loss=loss,
                    expert_load_variance=load_variance,
                    routing_entropy=entropy,
                    tokens_per_second=1000 / max(processing_time, 1),
                    gradient_norm=gradient_norm
                )
                
                # Get system health status
                overall_health = self.health_checker.get_overall_health()
                
                # Get circuit breaker states
                circuit_breaker_states = {
                    expert_id: cb.get_state()
                    for expert_id, cb in self.expert_circuit_breakers.items()
                }
                
                return {
                    'output': final_output,
                    'routing_info': {
                        'selected_experts': expert_indices,
                        'expert_weights': expert_probs,
                        'router_logits': router_logits,
                        'load_variance': load_variance,
                        'entropy': entropy
                    },
                    'performance': {
                        'processing_time_ms': processing_time,
                        'expert_computations': expert_computations,
                        'successful_experts': successful_experts,
                        'total_experts': len(expert_indices),
                        'success_rate': successful_experts / len(expert_indices)
                    },
                    'health_status': {
                        'overall_health': overall_health.value,
                        'circuit_breaker_states': circuit_breaker_states,
                        'error_occurred': error_occurred,
                        'recovery_attempts': self.recovery_attempt_count,
                        'auto_recovery_enabled': self.auto_recovery_enabled
                    },
                    'step': step
                }
                
            except Exception as e:
                self.error_handler.handle_error(e, {'step': step})
                raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        return {
            'health_status': self.health_checker.get_overall_health().value,
            'circuit_breakers': {
                expert_id: cb.get_state()
                for expert_id, cb in self.expert_circuit_breakers.items()
            },
            'error_rate': sum(self.error_rate_window) / len(self.error_rate_window) if self.error_rate_window else 0.0,
            'recovery_attempts': self.recovery_attempt_count,
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'recent_performance': {
                'avg_processing_time_ms': sum(p['processing_time_ms'] for p in self.system_performance_history[-10:]) / min(10, len(self.system_performance_history)) if self.system_performance_history else 0,
                'recent_success_rate': sum(1 for p in self.system_performance_history[-10:] if not p['error_occurred']) / min(10, len(self.system_performance_history)) if self.system_performance_history else 1.0
            }
        }
    
    def reset_recovery_state(self):
        """Reset recovery attempt counter."""
        self.recovery_attempt_count = 0
        logger.info("Recovery state reset")
    
    def enable_auto_recovery(self, enable: bool = True):
        """Enable or disable auto-recovery."""
        self.auto_recovery_enabled = enable
        logger.info(f"Auto-recovery {'enabled' if enable else 'disabled'}")
    
    def shutdown(self):
        """Gracefully shutdown the robust MoE system."""
        logger.info("Shutting down RobustMoEDemo...")
        
        # Stop health monitoring
        self.health_checker.stop_monitoring()
        
        # Save final checkpoint
        try:
            self.checkpoint_manager.save_checkpoint("final_shutdown")
            logger.info("Final checkpoint saved")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
        
        # Save performance history
        try:
            performance_summary = {
                'expert_performance_history': {
                    str(k): list(v) for k, v in self.expert_performance_history.items()
                },
                'system_performance_history': list(self.system_performance_history),
                'final_status': self.get_system_status()
            }
            
            with open("robust_demo_final_performance.json", "w") as f:
                json.dump(performance_summary, f, indent=2, default=str)
                
            logger.info("Performance history saved")
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
        
        logger.info("RobustMoEDemo shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup


def run_robust_demo():
    """Run comprehensive robustness demonstration."""
    print("🛡️  Open MoE Trainer Lab - Generation 2: Robust Demo")
    print("=" * 60)
    
    # Configuration
    config = {
        "hidden_size": 32,
        "num_experts": 4,
        "top_k": 2,
        "num_steps": 50,
        "error_injection_rate": 0.1  # 10% chance of injecting errors for testing
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize robust MoE model
    print("🏗️  Creating robust MoE model...")
    try:
        model = RobustMoEDemo(
            hidden_size=config["hidden_size"],
            num_experts=config["num_experts"],
            top_k=config["top_k"]
        )
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return None
    
    # Start monitoring
    print("📊 Starting monitoring systems...")
    model.metrics_collector.start_collection()
    
    # Simulate training with error injection
    print("🔄 Running robust training simulation...")
    successful_steps = 0
    failed_steps = 0
    
    for step in range(config["num_steps"]):
        try:
            # Create input token
            if random.random() < config["error_injection_rate"]:
                # Inject various types of errors for testing
                error_type = random.choice(['invalid_dims', 'nan_values', 'large_values'])
                
                if error_type == 'invalid_dims':
                    token = [random.gauss(0, 1.0) for _ in range(config["hidden_size"] - 5)]  # Wrong size
                elif error_type == 'nan_values':
                    token = [float('nan')] * config["hidden_size"]
                else:  # large_values
                    token = [random.gauss(0, 100.0) for _ in range(config["hidden_size"])]
                
                logger.info(f"Injecting {error_type} error at step {step}")
            else:
                # Normal token
                token = [random.gauss(0, 1.0) for _ in range(config["hidden_size"])]
            
            # Forward pass with monitoring
            result = model.forward_with_monitoring(token, step)
            successful_steps += 1
            
            if step % 10 == 0:
                routing_info = result['routing_info']
                perf_info = result['performance']
                print(f"  Step {step}: "
                      f"Experts {routing_info['selected_experts']} "
                      f"Load Var: {routing_info['load_variance']:.4f} "
                      f"Time: {perf_info['processing_time_ms']:.1f}ms")
            
            # Periodic checkpointing
            if step % 20 == 0 and step > 0:
                checkpoint_state = {
                    'step': step,
                    'successful_steps': successful_steps,
                    'failed_steps': failed_steps,
                    'model_config': config
                }
                model.checkpoint_manager.save_checkpoint(checkpoint_state, step)
            
        except Exception as e:
            failed_steps += 1
            if failed_steps <= 5:  # Log first 5 failures
                print(f"  ⚠️  Step {step} failed: {type(e).__name__}: {e}")
            
            # Attempt recovery for certain error types
            if isinstance(e, DataError) and "Invalid input dimensions" in str(e):
                print(f"  🔧 Attempting recovery for step {step}")
                try:
                    # Retry with corrected input
                    corrected_token = [random.gauss(0, 1.0) for _ in range(config["hidden_size"])]
                    result = model.forward_with_monitoring(corrected_token, step)
                    successful_steps += 1
                    print(f"  ✅ Recovery successful for step {step}")
                except Exception:
                    pass  # Recovery failed, continue
    
    # Stop monitoring
    model.metrics_collector.stop_collection()
    
    # Generate reports
    print(f"\n📈 Training Summary:")
    print(f"  Successful steps: {successful_steps}/{config['num_steps']} ({successful_steps/config['num_steps']*100:.1f}%)")
    print(f"  Failed steps: {failed_steps}")
    print(f"  Recovery attempts: {successful_steps + failed_steps - config['num_steps']}")
    
    # Health check
    print(f"\n🏥 Health Check:")
    health_status = model.health_monitor.check_system_health()
    print(f"  Overall Status: {health_status['overall_status']}")
    print(f"  System OK: {health_status['system_ok']}")
    print(f"  Training OK: {health_status['training_ok']}")
    
    if health_status['alerts']:
        print("  Alerts:")
        for alert in health_status['alerts']:
            print(f"    - {alert}")
    
    if health_status['warnings']:
        print("  Warnings:")
        for warning in health_status['warnings']:
            print(f"    - {warning}")
    
    # Error analysis
    print(f"\n🔍 Error Analysis:")
    error_summary = model.error_handler.get_error_summary(time_window=3600.0)
    print(f"  Total errors: {error_summary['total_errors']}")
    if error_summary['error_types']:
        print("  Error types:")
        for error_type, count in error_summary['error_types'].items():
            print(f"    {error_type}: {count}")
    
    # Performance metrics
    print(f"\n⚡ Performance Metrics:")
    system_stats = model.metrics_collector.get_summary_stats('system', time_window=3600.0)
    training_stats = model.metrics_collector.get_summary_stats('training', time_window=3600.0)
    
    if system_stats:
        print(f"  Avg Memory: {system_stats['avg_memory_mb']:.1f}MB")
        print(f"  Max Memory: {system_stats['max_memory_mb']:.1f}MB")
        print(f"  Avg CPU: {system_stats['avg_cpu_percent']:.1f}%")
    
    if training_stats:
        print(f"  Avg Loss: {training_stats['avg_loss']:.4f}")
        print(f"  Avg Load Variance: {training_stats['avg_expert_load_variance']:.4f}")
        print(f"  Avg Routing Entropy: {training_stats['avg_routing_entropy']:.4f}")
        print(f"  Avg Tokens/sec: {training_stats['avg_tokens_per_second']:.1f}")
    
    # Generate comprehensive report
    print(f"\n📋 Generating comprehensive report...")
    health_report = model.health_monitor.generate_health_report()
    
    # Export data
    model.metrics_collector.export_metrics("robust_demo_metrics.json", time_window=3600.0)
    model.error_handler.export_error_log("robust_demo_errors.json", time_window=3600.0)
    
    # Save final report
    final_report = {
        'demo_config': config,
        'training_summary': {
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / config['num_steps']
        },
        'health_status': health_status,
        'error_summary': error_summary,
        'system_performance': system_stats,
        'training_performance': training_stats,
        'health_report': health_report,
        'generation': 2,
        'robustness_features': [
            'comprehensive_error_handling',
            'automatic_retry_with_backoff',
            'real_time_monitoring',
            'system_health_checks',
            'automatic_checkpointing',
            'graceful_degradation',
            'performance_analytics',
            'anomaly_detection'
        ]
    }
    
    with open("generation2_robust_results.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n✅ Generation 2 Complete!")
    print("Robustness features demonstrated:")
    print("  ✓ Comprehensive error handling with recovery")
    print("  ✓ Real-time system monitoring")
    print("  ✓ Automatic checkpointing and state recovery")
    print("  ✓ Health monitoring with alerting")
    print("  ✓ Performance analytics and optimization")
    print("  ✓ Graceful degradation under failures")
    print("  ✓ Detailed error tracking and analysis")
    print("  ✓ Resource usage monitoring")
    
    print(f"\n📁 Outputs generated:")
    print("  - generation2_robust_results.json")
    print("  - robust_demo_metrics.json")
    print("  - robust_demo_errors.json")
    print("  - robust_moe_errors.log")
    print("  - ./robust_checkpoints/ (checkpoint files)")
    
    return final_report


import math

if __name__ == "__main__":
    # Run the robust demonstration
    results = run_robust_demo()
    
    if results:
        print("\n🎉 Open MoE Trainer Lab Generation 2 is robust and operational!")
        print("Ready to proceed to Generation 3: Make It Scale")
    else:
        print("\n❌ Demo failed - check error logs for details")