"""Advanced monitoring system for MoE models with comprehensive observability.

This module provides enterprise-grade monitoring capabilities:
1. Real-time performance metrics collection
2. Anomaly detection with statistical analysis
3. Distributed tracing for expert routing
4. Resource utilization monitoring
5. Automated alerting and incident response
6. Health check endpoints with detailed diagnostics
"""

import time
import threading
import queue
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import traceback
import psutil
import gc
import sys
from pathlib import Path


@dataclass
class MetricPoint:
    """Single metric measurement point."""
    name: str
    value: Union[float, int]
    timestamp: float
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Alert/incident information."""
    id: str
    severity: str  # 'critical', 'warning', 'info'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


class MetricsCollector:
    """Comprehensive metrics collection system."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.RLock()
        
        # Aggregation intervals
        self.aggregation_intervals = [60, 300, 900]  # 1min, 5min, 15min
        self.aggregated_metrics = defaultdict(lambda: defaultdict(dict))
        
        # Background aggregation
        self.aggregation_thread = None
        self.stop_aggregation = threading.Event()
        
    def record(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric measurement."""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata
        )
        
        with self.lock:
            self.metrics_buffer.append(metric)
            self.metric_history[name].append((metric.timestamp, metric.value))
            
    def get_recent_metrics(self, name: str, duration_seconds: int = 300) -> List[MetricPoint]:
        """Get recent metrics for a specific metric name."""
        cutoff_time = time.time() - duration_seconds
        
        with self.lock:
            return [
                metric for metric in self.metrics_buffer
                if metric.name == name and metric.timestamp >= cutoff_time
            ]
            
    def get_metric_statistics(self, name: str, duration_seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        recent_metrics = self.get_recent_metrics(name, duration_seconds)
        
        if not recent_metrics:
            return {}
            
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
        
    def start_aggregation(self):
        """Start background metric aggregation."""
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            return
            
        self.stop_aggregation.clear()
        self.aggregation_thread = threading.Thread(target=self._aggregation_worker)
        self.aggregation_thread.daemon = True
        self.aggregation_thread.start()
        
    def stop_background_aggregation(self):
        """Stop background aggregation."""
        if self.aggregation_thread:
            self.stop_aggregation.set()
            self.aggregation_thread.join(timeout=5)
            
    def _aggregation_worker(self):
        """Background worker for metric aggregation."""
        while not self.stop_aggregation.wait(10):  # Aggregate every 10 seconds
            try:
                current_time = time.time()
                
                for interval_seconds in self.aggregation_intervals:
                    self._aggregate_interval(current_time, interval_seconds)
                    
            except Exception as e:
                logging.error(f"Aggregation worker error: {e}")
                
    def _aggregate_interval(self, current_time: float, interval_seconds: int):
        """Aggregate metrics for a specific interval."""
        cutoff_time = current_time - interval_seconds
        
        # Get unique metric names in the interval
        metric_names = set()
        
        with self.lock:
            for metric in self.metrics_buffer:
                if metric.timestamp >= cutoff_time:
                    metric_names.add(metric.name)
                    
        # Aggregate each metric
        for name in metric_names:
            stats = self.get_metric_statistics(name, interval_seconds)
            if stats:
                self.aggregated_metrics[interval_seconds][name] = {
                    **stats,
                    'timestamp': current_time,
                    'interval_seconds': interval_seconds
                }


class AnomalyDetector:
    """Statistical anomaly detection for metrics."""
    
    def __init__(self, sensitivity: float = 3.0):
        self.sensitivity = sensitivity  # Z-score threshold
        self.baselines = {}  # metric_name -> {'mean': float, 'std': float, 'count': int}
        self.detection_history = defaultdict(list)
        
    def update_baseline(self, name: str, values: List[float]):
        """Update baseline statistics for a metric."""
        if len(values) < 2:
            return
            
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if name in self.baselines:
            # Exponential moving average update
            alpha = 0.1
            self.baselines[name]['mean'] = (1 - alpha) * self.baselines[name]['mean'] + alpha * mean_val
            self.baselines[name]['std'] = (1 - alpha) * self.baselines[name]['std'] + alpha * std_val
            self.baselines[name]['count'] += len(values)
        else:
            self.baselines[name] = {
                'mean': mean_val,
                'std': std_val,
                'count': len(values)
            }
            
    def detect_anomaly(self, name: str, value: float) -> Optional[Dict[str, Any]]:
        """Detect if a value is anomalous."""
        if name not in self.baselines or self.baselines[name]['count'] < 10:
            return None
            
        baseline = self.baselines[name]
        
        if baseline['std'] == 0:
            return None
            
        z_score = abs(value - baseline['mean']) / baseline['std']
        
        if z_score > self.sensitivity:
            anomaly_info = {
                'metric_name': name,
                'value': value,
                'baseline_mean': baseline['mean'],
                'baseline_std': baseline['std'],
                'z_score': z_score,
                'severity': 'critical' if z_score > 5 else 'warning',
                'timestamp': time.time()
            }
            
            self.detection_history[name].append(anomaly_info)
            return anomaly_info
            
        return None
        
    def get_anomaly_summary(self, duration_seconds: int = 3600) -> Dict[str, List[Dict]]:
        """Get anomaly summary for the last duration."""
        cutoff_time = time.time() - duration_seconds
        summary = {}
        
        for name, anomalies in self.detection_history.items():
            recent_anomalies = [
                a for a in anomalies
                if a['timestamp'] >= cutoff_time
            ]
            if recent_anomalies:
                summary[name] = recent_anomalies
                
        return summary


class AlertManager:
    """Alert management system with escalation."""
    
    def __init__(self, alert_handlers: Optional[List[Callable]] = None):
        self.alert_handlers = alert_handlers or []
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_history = []
        self.alert_queue = queue.Queue()
        
        # Escalation rules
        self.escalation_rules = {
            'critical': {'repeat_interval': 300, 'max_escalations': 5},  # 5 min
            'warning': {'repeat_interval': 900, 'max_escalations': 3},   # 15 min
            'info': {'repeat_interval': 3600, 'max_escalations': 1}      # 1 hour
        }
        
        # Background alert processing
        self.alert_thread = None
        self.stop_alerts = threading.Event()
        
    def trigger_alert(
        self,
        alert_id: str,
        severity: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        metadata: Optional[Dict] = None
    ):
        """Trigger a new alert."""
        alert = Alert(
            id=alert_id,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=time.time()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_queue.put(alert)
        
        # Add to history
        self.alert_history.append(alert)
        
        # Keep history manageable
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
            
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = time.time()
            del self.active_alerts[alert_id]
            
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
        
    def start_alert_processing(self):
        """Start background alert processing."""
        if self.alert_thread and self.alert_thread.is_alive():
            return
            
        self.stop_alerts.clear()
        self.alert_thread = threading.Thread(target=self._alert_worker)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
    def stop_alert_processing(self):
        """Stop background alert processing."""
        if self.alert_thread:
            self.stop_alerts.set()
            self.alert_thread.join(timeout=5)
            
    def _alert_worker(self):
        """Background alert processing worker."""
        while not self.stop_alerts.is_set():
            try:
                # Process alert queue with timeout
                try:
                    alert = self.alert_queue.get(timeout=1.0)
                    self._process_alert(alert)
                    self.alert_queue.task_done()
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logging.error(f"Alert worker error: {e}")
                
    def _process_alert(self, alert: Alert):
        """Process a single alert."""
        try:
            for handler in self.alert_handlers:
                handler(alert)
        except Exception as e:
            logging.error(f"Alert handler error: {e}")
            
    def get_active_alerts(self) -> Dict[str, Alert]:
        """Get currently active alerts."""
        return self.active_alerts.copy()
        
    def get_alert_summary(self, duration_seconds: int = 3600) -> Dict[str, Any]:
        """Get alert summary for the last duration."""
        cutoff_time = time.time() - duration_seconds
        
        recent_alerts = [
            a for a in self.alert_history
            if a.timestamp >= cutoff_time
        ]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': defaultdict(int),
            'by_metric': defaultdict(int),
            'resolution_times': [],
            'active_count': len(self.active_alerts)
        }
        
        for alert in recent_alerts:
            summary['by_severity'][alert.severity] += 1
            summary['by_metric'][alert.metric_name] += 1
            
            if alert.resolved and alert.resolution_time:
                resolution_time = alert.resolution_time - alert.timestamp
                summary['resolution_times'].append(resolution_time)
                
        if summary['resolution_times']:
            summary['avg_resolution_time'] = statistics.mean(summary['resolution_times'])
        else:
            summary['avg_resolution_time'] = 0.0
            
        return summary


class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        
        # Resource thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'gpu_memory_percent': 90.0
        }
        
        # Background monitoring
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        
    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
            
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start metrics aggregation
        self.metrics_collector.start_aggregation()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitor_thread:
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=5)
            
        self.metrics_collector.stop_background_aggregation()
        
    def _monitoring_worker(self):
        """Background monitoring worker."""
        while not self.stop_monitoring.wait(self.collection_interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record('system.cpu.percent', cpu_percent)
        
        cpu_count = psutil.cpu_count()
        self.metrics_collector.record('system.cpu.count', cpu_count)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.record('system.memory.percent', memory.percent)
        self.metrics_collector.record('system.memory.available_gb', memory.available / (1024**3))
        self.metrics_collector.record('system.memory.used_gb', memory.used / (1024**3))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics_collector.record('system.disk.usage_percent', disk_percent)
        self.metrics_collector.record('system.disk.free_gb', disk.free / (1024**3))
        
        # Process metrics
        process = psutil.Process()
        
        with process.oneshot():\n            process_memory = process.memory_info()\n            self.metrics_collector.record('process.memory.rss_mb', process_memory.rss / (1024**2))\n            self.metrics_collector.record('process.memory.vms_mb', process_memory.vms / (1024**2))\n            \n            self.metrics_collector.record('process.cpu.percent', process.cpu_percent())\n            self.metrics_collector.record('process.threads', process.num_threads())\n            self.metrics_collector.record('process.fds', process.num_fds() if hasattr(process, 'num_fds') else 0)\n            \n        # Python-specific metrics\n        self.metrics_collector.record('python.gc.objects', len(gc.get_objects()))\n        \n        # GPU metrics (if available)\n        try:\n            import torch\n            if torch.cuda.is_available():\n                for i in range(torch.cuda.device_count()):\n                    gpu_memory = torch.cuda.get_device_properties(i).total_memory\n                    gpu_allocated = torch.cuda.memory_allocated(i)\n                    gpu_percent = (gpu_allocated / gpu_memory) * 100\n                    \n                    self.metrics_collector.record(\n                        'gpu.memory.percent',\n                        gpu_percent,\n                        tags={'device': str(i)}\n                    )\n                    self.metrics_collector.record(\n                        'gpu.memory.allocated_gb',\n                        gpu_allocated / (1024**3),\n                        tags={'device': str(i)}\n                    )\n        except ImportError:\n            pass\n            \n    def get_resource_summary(self) -> Dict[str, Any]:\n        \"\"\"Get current resource utilization summary.\"\"\"\n        summary = {}\n        \n        # Get recent metrics for each resource type\n        resource_metrics = [\n            'system.cpu.percent',\n            'system.memory.percent', \n            'system.disk.usage_percent',\n            'process.memory.rss_mb',\n            'process.cpu.percent'\n        ]\n        \n        for metric in resource_metrics:\n            stats = self.metrics_collector.get_metric_statistics(metric, duration_seconds=60)\n            if stats:\n                summary[metric] = {\n                    'current': stats['mean'],\n                    'max_1min': stats['max'],\n                    'threshold_exceeded': stats['max'] > self.thresholds.get(metric.split('.')[-1], 100)\n                }\n                \n        return summary\n        \n    def check_resource_alerts(self, alert_manager: AlertManager):\n        \"\"\"Check resource metrics against thresholds and trigger alerts.\"\"\"\n        summary = self.get_resource_summary()\n        \n        for metric, data in summary.items():\n            metric_key = metric.split('.')[-1]\n            if metric_key in self.thresholds:\n                threshold = self.thresholds[metric_key]\n                current_value = data['current']\n                \n                alert_id = f\"resource_{metric_key}_high\"\n                \n                if current_value > threshold:\n                    alert_manager.trigger_alert(\n                        alert_id=alert_id,\n                        severity='critical' if current_value > threshold * 1.1 else 'warning',\n                        message=f\"{metric} is {current_value:.1f}% (threshold: {threshold}%)\",\n                        metric_name=metric,\n                        current_value=current_value,\n                        threshold=threshold\n                    )\n                else:\n                    # Resolve alert if it exists\n                    alert_manager.resolve_alert(alert_id)


class HealthChecker:
    \"\"\"Comprehensive health check system.\"\"\"\n    \n    def __init__(self):\n        self.health_checks = {}\n        self.last_results = {}\n        \n    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):\n        \"\"\"Register a health check function.\"\"\"\n        self.health_checks[name] = check_func\n        \n    def run_check(self, name: str) -> Dict[str, Any]:\n        \"\"\"Run a specific health check.\"\"\"\n        if name not in self.health_checks:\n            return {'status': 'error', 'message': f'Unknown health check: {name}'}\n            \n        try:\n            start_time = time.time()\n            result = self.health_checks[name]()\n            duration = time.time() - start_time\n            \n            result['duration_ms'] = duration * 1000\n            result['timestamp'] = time.time()\n            \n            self.last_results[name] = result\n            return result\n            \n        except Exception as e:\n            error_result = {\n                'status': 'error',\n                'message': str(e),\n                'traceback': traceback.format_exc(),\n                'timestamp': time.time()\n            }\n            \n            self.last_results[name] = error_result\n            return error_result\n            \n    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:\n        \"\"\"Run all registered health checks.\"\"\"\n        results = {}\n        \n        for name in self.health_checks:\n            results[name] = self.run_check(name)\n            \n        return results\n        \n    def get_overall_health(self) -> Dict[str, Any]:\n        \"\"\"Get overall system health status.\"\"\"\n        all_results = self.run_all_checks()\n        \n        statuses = [result.get('status', 'unknown') for result in all_results.values()]\n        \n        # Determine overall status\n        if 'error' in statuses:\n            overall_status = 'unhealthy'\n        elif 'warning' in statuses:\n            overall_status = 'degraded'\n        elif 'healthy' in statuses or 'ok' in statuses:\n            overall_status = 'healthy'\n        else:\n            overall_status = 'unknown'\n            \n        return {\n            'status': overall_status,\n            'checks': all_results,\n            'summary': {\n                'total_checks': len(all_results),\n                'healthy': len([s for s in statuses if s in ['healthy', 'ok']]),\n                'warnings': len([s for s in statuses if s == 'warning']),\n                'errors': len([s for s in statuses if s == 'error'])\n            },\n            'timestamp': time.time()\n        }


class AdvancedMonitoringSystem:
    \"\"\"Comprehensive monitoring system orchestrator.\"\"\"\n    \n    def __init__(self, config: Optional[Dict[str, Any]] = None):\n        self.config = config or {}\n        \n        # Initialize components\n        self.metrics_collector = MetricsCollector(\n            buffer_size=self.config.get('metrics_buffer_size', 50000)\n        )\n        \n        self.anomaly_detector = AnomalyDetector(\n            sensitivity=self.config.get('anomaly_sensitivity', 3.0)\n        )\n        \n        self.alert_manager = AlertManager()\n        self.resource_monitor = ResourceMonitor(\n            collection_interval=self.config.get('resource_collection_interval', 10.0)\n        )\n        \n        self.health_checker = HealthChecker()\n        \n        # Setup default health checks\n        self._register_default_health_checks()\n        \n        # Setup default alert handlers\n        self._setup_default_alert_handlers()\n        \n        # Background processing\n        self.anomaly_check_thread = None\n        self.stop_anomaly_checks = threading.Event()\n        \n    def _register_default_health_checks(self):\n        \"\"\"Register default health checks.\"\"\"\n        \n        def memory_check():\n            memory = psutil.virtual_memory()\n            if memory.percent > 90:\n                return {'status': 'error', 'message': f'Memory usage critical: {memory.percent}%'}\n            elif memory.percent > 80:\n                return {'status': 'warning', 'message': f'Memory usage high: {memory.percent}%'}\n            else:\n                return {'status': 'healthy', 'memory_percent': memory.percent}\n                \n        def cpu_check():\n            cpu_percent = psutil.cpu_percent(interval=1)\n            if cpu_percent > 90:\n                return {'status': 'error', 'message': f'CPU usage critical: {cpu_percent}%'}\n            elif cpu_percent > 80:\n                return {'status': 'warning', 'message': f'CPU usage high: {cpu_percent}%'}\n            else:\n                return {'status': 'healthy', 'cpu_percent': cpu_percent}\n                \n        def disk_check():\n            disk = psutil.disk_usage('/')\n            disk_percent = (disk.used / disk.total) * 100\n            if disk_percent > 95:\n                return {'status': 'error', 'message': f'Disk usage critical: {disk_percent:.1f}%'}\n            elif disk_percent > 85:\n                return {'status': 'warning', 'message': f'Disk usage high: {disk_percent:.1f}%'}\n            else:\n                return {'status': 'healthy', 'disk_percent': disk_percent}\n                \n        self.health_checker.register_check('memory', memory_check)\n        self.health_checker.register_check('cpu', cpu_check)\n        self.health_checker.register_check('disk', disk_check)\n        \n    def _setup_default_alert_handlers(self):\n        \"\"\"Setup default alert handlers.\"\"\"\n        \n        def log_alert_handler(alert: Alert):\n            logging.warning(f\"ALERT [{alert.severity.upper()}] {alert.message}\")\n            \n        def console_alert_handler(alert: Alert):\n            severity_emoji = {'critical': '🚨', 'warning': '⚠️', 'info': 'ℹ️'}\n            emoji = severity_emoji.get(alert.severity, '📢')\n            print(f\"{emoji} [{alert.severity.upper()}] {alert.message}\")\n            \n        self.alert_manager.add_alert_handler(log_alert_handler)\n        \n        if self.config.get('console_alerts', True):\n            self.alert_manager.add_alert_handler(console_alert_handler)\n            \n    def start_monitoring(self):\n        \"\"\"Start all monitoring components.\"\"\"\n        logging.info(\"Starting advanced monitoring system\")\n        \n        # Start metrics collection and aggregation\n        self.metrics_collector.start_aggregation()\n        \n        # Start resource monitoring\n        self.resource_monitor.start_monitoring()\n        \n        # Start alert processing\n        self.alert_manager.start_alert_processing()\n        \n        # Start anomaly detection checks\n        self._start_anomaly_checks()\n        \n        logging.info(\"Advanced monitoring system started successfully\")\n        \n    def stop_monitoring(self):\n        \"\"\"Stop all monitoring components.\"\"\"\n        logging.info(\"Stopping advanced monitoring system\")\n        \n        # Stop background threads\n        self.metrics_collector.stop_background_aggregation()\n        self.resource_monitor.stop_monitoring()\n        self.alert_manager.stop_alert_processing()\n        \n        if self.anomaly_check_thread:\n            self.stop_anomaly_checks.set()\n            self.anomaly_check_thread.join(timeout=5)\n            \n        logging.info(\"Advanced monitoring system stopped\")\n        \n    def _start_anomaly_checks(self):\n        \"\"\"Start background anomaly detection checks.\"\"\"\n        if self.anomaly_check_thread and self.anomaly_check_thread.is_alive():\n            return\n            \n        self.stop_anomaly_checks.clear()\n        self.anomaly_check_thread = threading.Thread(target=self._anomaly_check_worker)\n        self.anomaly_check_thread.daemon = True\n        self.anomaly_check_thread.start()\n        \n    def _anomaly_check_worker(self):\n        \"\"\"Background anomaly detection worker.\"\"\"\n        check_interval = 30  # Check every 30 seconds\n        \n        while not self.stop_anomaly_checks.wait(check_interval):\n            try:\n                self._check_metric_anomalies()\n                self.resource_monitor.check_resource_alerts(self.alert_manager)\n            except Exception as e:\n                logging.error(f\"Anomaly check worker error: {e}\")\n                \n    def _check_metric_anomalies(self):\n        \"\"\"Check for anomalies in collected metrics.\"\"\"\n        # Get recent metrics for anomaly detection\n        current_time = time.time()\n        recent_window = 300  # 5 minutes\n        \n        # Update baselines with recent data\n        with self.metrics_collector.lock:\n            metric_names = set()\n            for metric in self.metrics_collector.metrics_buffer:\n                if current_time - metric.timestamp <= recent_window:\n                    metric_names.add(metric.name)\n                    \n        # Check each metric for anomalies\n        for metric_name in metric_names:\n            recent_metrics = self.metrics_collector.get_recent_metrics(\n                metric_name, recent_window\n            )\n            \n            if len(recent_metrics) < 10:  # Need sufficient data\n                continue\n                \n            # Update baseline\n            values = [m.value for m in recent_metrics[:-5]]  # Use all but last 5 for baseline\n            if values:\n                self.anomaly_detector.update_baseline(metric_name, values)\n                \n            # Check last few values for anomalies\n            for metric in recent_metrics[-5:]:\n                anomaly = self.anomaly_detector.detect_anomaly(\n                    metric_name, metric.value\n                )\n                \n                if anomaly:\n                    self.alert_manager.trigger_alert(\n                        alert_id=f\"anomaly_{metric_name}_{int(anomaly['timestamp'])}\",\n                        severity=anomaly['severity'],\n                        message=f\"Anomaly detected in {metric_name}: {anomaly['value']:.3f} (z-score: {anomaly['z_score']:.2f})\",\n                        metric_name=metric_name,\n                        current_value=anomaly['value'],\n                        threshold=anomaly['baseline_mean'] + 3 * anomaly['baseline_std']\n                    )\n                    \n    def record_metric(self, name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None):\n        \"\"\"Record a metric measurement.\"\"\"\n        self.metrics_collector.record(name, value, tags)\n        \n    def get_monitoring_dashboard(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive monitoring dashboard data.\"\"\"\n        dashboard = {\n            'timestamp': time.time(),\n            'status': self.health_checker.get_overall_health(),\n            'resource_summary': self.resource_monitor.get_resource_summary(),\n            'alert_summary': self.alert_manager.get_alert_summary(),\n            'anomaly_summary': self.anomaly_detector.get_anomaly_summary(),\n            'metrics_summary': self._get_metrics_summary()\n        }\n        \n        return dashboard\n        \n    def _get_metrics_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of collected metrics.\"\"\"\n        with self.metrics_collector.lock:\n            total_metrics = len(self.metrics_collector.metrics_buffer)\n            \n            # Get unique metric names\n            metric_names = set()\n            for metric in self.metrics_collector.metrics_buffer:\n                metric_names.add(metric.name)\n                \n            # Recent activity (last 5 minutes)\n            cutoff_time = time.time() - 300\n            recent_metrics = sum(\n                1 for metric in self.metrics_collector.metrics_buffer\n                if metric.timestamp >= cutoff_time\n            )\n            \n        return {\n            'total_metrics_collected': total_metrics,\n            'unique_metric_names': len(metric_names),\n            'recent_metrics_5min': recent_metrics,\n            'buffer_utilization': len(self.metrics_collector.metrics_buffer) / self.metrics_collector.buffer_size\n        }\n        \n    def export_metrics(self, format_type: str = 'json', duration_seconds: int = 3600) -> str:\n        \"\"\"Export metrics in various formats.\"\"\"\n        cutoff_time = time.time() - duration_seconds\n        \n        with self.metrics_collector.lock:\n            metrics_to_export = [\n                asdict(metric) for metric in self.metrics_collector.metrics_buffer\n                if metric.timestamp >= cutoff_time\n            ]\n            \n        if format_type.lower() == 'json':\n            return json.dumps({\n                'metrics': metrics_to_export,\n                'export_timestamp': time.time(),\n                'duration_seconds': duration_seconds,\n                'total_metrics': len(metrics_to_export)\n            }, indent=2, default=str)\n        else:\n            raise ValueError(f\"Unsupported export format: {format_type}\")\n            \n    def save_monitoring_report(self, filepath: str):\n        \"\"\"Save comprehensive monitoring report to file.\"\"\"\n        report = {\n            'report_timestamp': time.time(),\n            'monitoring_dashboard': self.get_monitoring_dashboard(),\n            'system_info': {\n                'platform': sys.platform,\n                'python_version': sys.version,\n                'process_id': psutil.Process().pid,\n                'uptime_seconds': time.time() - psutil.Process().create_time()\n            }\n        }\n        \n        Path(filepath).parent.mkdir(parents=True, exist_ok=True)\n        \n        with open(filepath, 'w') as f:\n            json.dump(report, f, indent=2, default=str)\n            \n        logging.info(f\"Monitoring report saved to {filepath}\")\n\n\n# Export monitoring components\n__all__ = [\n    'MetricPoint',\n    'Alert',\n    'MetricsCollector',\n    'AnomalyDetector', \n    'AlertManager',\n    'ResourceMonitor',\n    'HealthChecker',\n    'AdvancedMonitoringSystem'\n]