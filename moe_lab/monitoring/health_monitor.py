"""Comprehensive health monitoring for MoE systems."""

import time
import threading
import logging
import psutil
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch for testing
    torch = type('torch', (), {
        'cuda': type('cuda', (), {
            'is_available': lambda: False,
            'device_count': lambda: 0,
            'memory_allocated': lambda device=None: 0,
            'memory_reserved': lambda device=None: 0,
            'get_device_properties': lambda device: type('prop', (), {'total_memory': 0})()
        })()
    })()


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemMetrics:
    """System-level metrics."""
    
    # CPU metrics
    cpu_percent: float
    cpu_count: int
    load_average: List[float]
    
    # Memory metrics
    memory_total: int
    memory_available: int
    memory_percent: float
    
    # GPU metrics (if available)
    gpu_count: int
    gpu_memory_used: List[int]
    gpu_memory_total: List[int]
    gpu_utilization: List[float]
    
    # Disk metrics
    disk_usage_percent: float
    disk_free: int
    
    # Network metrics
    network_bytes_sent: int
    network_bytes_recv: int
    
    # Timestamp
    timestamp: float
    
    
@dataclass
class HealthCheck:
    """Individual health check result."""
    
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    execution_time: float
    
    
class HealthMonitor:
    """Comprehensive system health monitor."""
    
    def __init__(
        self,
        check_interval: int = 60,
        log_file: Optional[str] = None,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.check_interval = check_interval
        self.running = False
        self.checks: Dict[str, Callable] = {}
        self.last_metrics: Optional[SystemMetrics] = None
        self.health_history: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = self._setup_logger(log_file)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0,
            "disk_usage_percent": 90.0,
            "load_average_ratio": 2.0
        }
        
        # Register default checks
        self._register_default_checks()
        
        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        
    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        """Setup health monitor logger."""
        logger = logging.getLogger("health_monitor")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("gpu_health", self._check_gpu_health)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_health", self._check_memory_health)
        
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a custom health check."""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_count = 0
        gpu_memory_used = []
        gpu_memory_total = []
        gpu_utilization = []
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            for i in range(gpu_count):
                memory_used = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                
                if hasattr(torch.cuda, 'get_device_properties'):
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory
                else:
                    total_memory = memory_reserved
                
                gpu_memory_used.append(memory_used)
                gpu_memory_total.append(total_memory)
                
                # Simple utilization estimate
                utilization = (memory_used / total_memory * 100) if total_memory > 0 else 0.0
                gpu_utilization.append(utilization)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            load_average=load_avg,
            memory_total=memory.total,
            memory_available=memory.available,
            memory_percent=memory.percent,
            gpu_count=gpu_count,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            disk_usage_percent=disk.percent,
            disk_free=disk.free,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            timestamp=time.time()
        )
    
    def _check_system_resources(self) -> HealthCheck:
        """Check overall system resource health."""
        start_time = time.time()
        
        try:
            metrics = self.collect_system_metrics()
            self.last_metrics = metrics
            
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check CPU
            if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
                status = HealthStatus.WARNING
                issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
            # Check memory
            if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
                status = HealthStatus.CRITICAL if metrics.memory_percent > 95 else HealthStatus.WARNING
                issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            
            # Check load average
            if metrics.load_average and metrics.cpu_count > 0:
                load_ratio = metrics.load_average[0] / metrics.cpu_count
                if load_ratio > self.alert_thresholds["load_average_ratio"]:
                    status = HealthStatus.WARNING
                    issues.append(f"High load average: {load_ratio:.2f}")
            
            message = "System resources healthy" if not issues else "; ".join(issues)
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details=asdict(metrics),
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def _check_gpu_health(self) -> HealthCheck:
        """Check GPU health and utilization."""
        start_time = time.time()
        
        try:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return HealthCheck(
                    name="gpu_health",
                    status=HealthStatus.HEALTHY,
                    message="No GPU available or PyTorch not installed",
                    details={"gpu_available": False},
                    timestamp=time.time(),
                    execution_time=time.time() - start_time
                )
            
            metrics = self.last_metrics or self.collect_system_metrics()
            status = HealthStatus.HEALTHY
            issues = []
            
            for i, (used, total) in enumerate(zip(metrics.gpu_memory_used, metrics.gpu_memory_total)):
                if total > 0:
                    usage_percent = (used / total) * 100
                    if usage_percent > self.alert_thresholds["gpu_memory_percent"]:
                        status = HealthStatus.WARNING
                        issues.append(f"GPU {i} memory high: {usage_percent:.1f}%")
            
            message = "GPU health good" if not issues else "; ".join(issues)
            
            return HealthCheck(
                name="gpu_health",
                status=status,
                message=message,
                details={
                    "gpu_count": metrics.gpu_count,
                    "gpu_memory_used": metrics.gpu_memory_used,
                    "gpu_memory_total": metrics.gpu_memory_total,
                    "gpu_utilization": metrics.gpu_utilization
                },
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="gpu_health",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check GPU health: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability."""
        start_time = time.time()
        
        try:
            metrics = self.last_metrics or self.collect_system_metrics()
            
            status = HealthStatus.HEALTHY
            if metrics.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
                status = HealthStatus.CRITICAL if metrics.disk_usage_percent > 95 else HealthStatus.WARNING
            
            message = f"Disk usage: {metrics.disk_usage_percent:.1f}%, {metrics.disk_free / 1e9:.1f}GB free"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "disk_usage_percent": metrics.disk_usage_percent,
                    "disk_free_bytes": metrics.disk_free,
                    "disk_free_gb": metrics.disk_free / 1e9
                },
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk space: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def _check_memory_health(self) -> HealthCheck:
        """Detailed memory health check."""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            status = HealthStatus.HEALTHY
            issues = []
            
            # Check virtual memory
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
                issues.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > 80:
                status = HealthStatus.WARNING
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Check swap usage
            if swap.percent > 50:
                status = HealthStatus.WARNING
                issues.append(f"High swap usage: {swap.percent:.1f}%")
            
            message = "Memory health good" if not issues else "; ".join(issues)
            
            return HealthCheck(
                name="memory_health",
                status=status,
                message=message,
                details={
                    "virtual_memory": {
                        "total_gb": memory.total / 1e9,
                        "available_gb": memory.available / 1e9,
                        "percent": memory.percent,
                        "used_gb": memory.used / 1e9
                    },
                    "swap_memory": {
                        "total_gb": swap.total / 1e9,
                        "used_gb": swap.used / 1e9,
                        "percent": swap.percent
                    }
                },
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory_health",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory health: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = result
                
                # Log critical issues
                if result.status == HealthStatus.CRITICAL:
                    self.logger.error(f"CRITICAL: {name} - {result.message}")
                elif result.status == HealthStatus.WARNING:
                    self.logger.warning(f"WARNING: {name} - {result.message}")
                    
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e}",
                    details={"error": str(e)},
                    timestamp=time.time(),
                    execution_time=0.0
                )
        
        return results
    
    def get_overall_health_status(self, check_results: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall health status from individual checks."""
        statuses = [check.status for check in check_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            self.logger.warning("Health monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                check_results = self.run_all_checks()
                overall_status = self.get_overall_health_status(check_results)
                
                # Store in history
                health_snapshot = {
                    "timestamp": time.time(),
                    "overall_status": overall_status.value,
                    "checks": {name: asdict(check) for name, check in check_results.items()}
                }
                
                self.health_history.append(health_snapshot)
                
                # Limit history size
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]
                
                # Log overall status
                self.logger.info(f"Health check complete - Overall status: {overall_status.value}")
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next check
            time.sleep(self.check_interval)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        check_results = self.run_all_checks()
        overall_status = self.get_overall_health_status(check_results)
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": {name: asdict(check) for name, check in check_results.items()},
            "system_metrics": asdict(self.last_metrics) if self.last_metrics else None,
            "alert_thresholds": self.alert_thresholds,
            "monitoring_config": {
                "check_interval": self.check_interval,
                "running": self.running,
                "checks_registered": list(self.checks.keys())
            }
        }
    
    def save_health_history(self, output_file: str):
        """Save health monitoring history to file."""
        with open(output_file, 'w') as f:
            json.dump(self.health_history, f, indent=2, default=str)
        
        self.logger.info(f"Health history saved to {output_file}")


class ModelHealthChecker:
    """Health checker specific to MoE models."""
    
    def __init__(self, model, model_name: str = "moe_model"):
        self.model = model
        self.model_name = model_name
        
    def check_model_health(self) -> HealthCheck:
        """Check MoE model-specific health."""
        start_time = time.time()
        
        try:
            issues = []
            status = HealthStatus.HEALTHY
            details = {}
            
            # Check model parameters
            total_params = sum(p.numel() for p in self.model.parameters() if hasattr(p, 'numel'))
            details["total_parameters"] = total_params
            
            # Check for NaN/Inf parameters
            nan_params = 0
            inf_params = 0
            
            if TORCH_AVAILABLE:
                for name, param in self.model.named_parameters():
                    if hasattr(param, 'data'):
                        if torch.isnan(param.data).any():
                            nan_params += 1
                        if torch.isinf(param.data).any():
                            inf_params += 1
            
            if nan_params > 0:
                status = HealthStatus.CRITICAL
                issues.append(f"Found {nan_params} parameters with NaN values")
            
            if inf_params > 0:
                status = HealthStatus.CRITICAL  
                issues.append(f"Found {inf_params} parameters with Inf values")
            
            details.update({
                "nan_parameters": nan_params,
                "inf_parameters": inf_params
            })
            
            # Check MoE-specific components
            if hasattr(self.model, 'moe_layers'):
                details["moe_layers"] = len(self.model.moe_layers) if hasattr(self.model.moe_layers, '__len__') else 0
                
            if hasattr(self.model, 'num_experts'):
                details["num_experts"] = self.model.num_experts
                
            if hasattr(self.model, 'experts_per_token'):
                details["experts_per_token"] = self.model.experts_per_token
            
            message = f"Model {self.model_name} healthy" if not issues else "; ".join(issues)
            
            return HealthCheck(
                name=f"model_health_{self.model_name}",
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheck(
                name=f"model_health_{self.model_name}",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check model health: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                execution_time=time.time() - start_time
            )