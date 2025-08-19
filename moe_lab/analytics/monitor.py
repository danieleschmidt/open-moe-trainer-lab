"""Real-time router monitoring for MoE models."""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from dataclasses import dataclass
import json

from ..models import MoEModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouterStats:
    """Statistics about router performance."""
    
    load_variance: float
    drop_rate: float
    entropy: float
    expert_utilization: Dict[int, float]
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    timestamp: float


class RouterMonitor:
    """Real-time monitoring of MoE router decisions."""
    
    def __init__(
        self,
        model: MoEModel,
        window_size: int = 1000,
        collection_interval: float = 1.0
    ):
        self.model = model
        self.window_size = window_size
        self.collection_interval = collection_interval
        
        # Statistics storage
        self.routing_history = deque(maxlen=window_size)
        self.stats_history = deque(maxlen=window_size)
        
        # Counters
        self.token_count = 0
        self.expert_counts = defaultdict(int)
        self.dropped_tokens = 0
        self.start_time = time.time()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Dashboard state
        self.dashboard_data = {
            'expert_loads': [],
            'routing_entropy': [],
            'throughput': [],
            'memory_usage': []
        }
        
    def start_monitoring(self):
        """Start background monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Router monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Router monitoring stopped")
        
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                stats = self.get_current_stats()
                self.stats_history.append(stats)
                self._update_dashboard_data(stats)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    def track(self):
        """Context manager for tracking router decisions."""
        return self
        
    def __enter__(self):
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()
        
    def record_routing(self, routing_info, batch_size: int):
        """Record routing decisions from a forward pass."""
        self.routing_history.append(routing_info)
        self.token_count += batch_size
        
        # Update expert usage counts
        if hasattr(routing_info, 'selected_experts') and routing_info.selected_experts is not None:
            experts = routing_info.selected_experts.flatten()
            for expert_idx in experts:
                if expert_idx.item() >= 0:  # Valid expert
                    self.expert_counts[expert_idx.item()] += 1
                else:  # Dropped token
                    self.dropped_tokens += 1
                    
    def get_current_stats(self) -> RouterStats:
        """Get current router statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate throughput
        throughput = self.token_count / elapsed_time if elapsed_time > 0 else 0.0
        
        # Calculate expert utilization
        total_routed = sum(self.expert_counts.values())
        expert_utilization = {}
        for expert_idx, count in self.expert_counts.items():
            expert_utilization[expert_idx] = count / total_routed if total_routed > 0 else 0.0
            
        # Calculate drop rate
        total_tokens = total_routed + self.dropped_tokens
        drop_rate = self.dropped_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Get latest routing stats
        load_variance = 0.0
        entropy = 0.0
        if self.routing_history:
            latest_routing = self.routing_history[-1]
            load_variance = getattr(latest_routing, 'load_variance', 0.0)
            entropy = getattr(latest_routing, 'entropy', 0.0)
            
        # Memory usage
        memory_usage = 0.0
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
        return RouterStats(
            load_variance=load_variance,
            drop_rate=drop_rate,
            entropy=entropy,
            expert_utilization=expert_utilization,
            throughput_tokens_per_sec=throughput,
            memory_usage_mb=memory_usage,
            timestamp=current_time
        )
        
    def get_stats(self) -> RouterStats:
        """Get latest statistics."""
        return self.get_current_stats()
        
    def _update_dashboard_data(self, stats: RouterStats):
        """Update dashboard data for visualization."""
        # Keep only recent data points
        max_points = 100
        
        self.dashboard_data['expert_loads'].append({
            'timestamp': stats.timestamp,
            'utilization': stats.expert_utilization
        })
        
        self.dashboard_data['routing_entropy'].append({
            'timestamp': stats.timestamp,
            'entropy': stats.entropy
        })
        
        self.dashboard_data['throughput'].append({
            'timestamp': stats.timestamp,
            'tokens_per_sec': stats.throughput_tokens_per_sec
        })
        
        self.dashboard_data['memory_usage'].append({
            'timestamp': stats.timestamp,
            'memory_mb': stats.memory_usage_mb
        })
        
        # Trim old data
        for key in self.dashboard_data:
            if len(self.dashboard_data[key]) > max_points:
                self.dashboard_data[key] = self.dashboard_data[key][-max_points:]
                
    def start_dashboard(self, port: int = 8080, host: str = "localhost"):
        """Start web dashboard for live monitoring."""
        try:
            import flask
            from flask import Flask, jsonify, render_template_string
        except ImportError:
            logger.error("Flask not installed. Run: pip install flask")
            return
            
        app = Flask(__name__)
        
        @app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_html())
            
        @app.route('/api/stats')
        def api_stats():
            current_stats = self.get_current_stats()
            return jsonify({
                'load_variance': current_stats.load_variance,
                'drop_rate': current_stats.drop_rate,
                'entropy': current_stats.entropy,
                'expert_utilization': current_stats.expert_utilization,
                'throughput': current_stats.throughput_tokens_per_sec,
                'memory_usage': current_stats.memory_usage_mb
            })
            
        @app.route('/api/history')
        def api_history():
            return jsonify(self.dashboard_data)
            
        logger.info(f"Starting dashboard on http://{host}:{port}")
        app.run(host=host, port=port, debug=False, threaded=True)
        
    def _get_dashboard_html(self) -> str:
        """Get HTML template for dashboard."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MoE Router Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
                .chart { width: 45%; display: inline-block; margin: 10px; }
            </style>
        </head>
        <body>
            <h1>MoE Router Monitoring Dashboard</h1>
            
            <div id="metrics">
                <div class="metric" id="load-variance">Load Variance: --</div>
                <div class="metric" id="drop-rate">Drop Rate: --</div>
                <div class="metric" id="entropy">Entropy: --</div>
                <div class="metric" id="throughput">Throughput: --</div>
            </div>
            
            <div class="chart" id="entropy-chart"></div>
            <div class="chart" id="throughput-chart"></div>
            <div class="chart" id="expert-util-chart"></div>
            <div class="chart" id="memory-chart"></div>
            
            <script>
                function updateDashboard() {
                    fetch('/api/stats')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('load-variance').textContent = 
                                'Load Variance: ' + data.load_variance.toFixed(4);
                            document.getElementById('drop-rate').textContent = 
                                'Drop Rate: ' + (data.drop_rate * 100).toFixed(2) + '%';
                            document.getElementById('entropy').textContent = 
                                'Entropy: ' + data.entropy.toFixed(3);
                            document.getElementById('throughput').textContent = 
                                'Throughput: ' + data.throughput.toFixed(1) + ' tokens/sec';
                        });
                        
                    fetch('/api/history')
                        .then(response => response.json())
                        .then(data => {
                            updateCharts(data);
                        });
                }
                
                function updateCharts(data) {
                    // Entropy chart
                    const entropyTrace = {
                        x: data.routing_entropy.map(d => new Date(d.timestamp * 1000)),
                        y: data.routing_entropy.map(d => d.entropy),
                        type: 'scatter',
                        name: 'Routing Entropy'
                    };
                    Plotly.newPlot('entropy-chart', [entropyTrace], {title: 'Routing Entropy'});
                    
                    // Throughput chart
                    const throughputTrace = {
                        x: data.throughput.map(d => new Date(d.timestamp * 1000)),
                        y: data.throughput.map(d => d.tokens_per_sec),
                        type: 'scatter',
                        name: 'Tokens/sec'
                    };
                    Plotly.newPlot('throughput-chart', [throughputTrace], {title: 'Throughput'});
                    
                    // Memory chart
                    const memoryTrace = {
                        x: data.memory_usage.map(d => new Date(d.timestamp * 1000)),
                        y: data.memory_usage.map(d => d.memory_mb),
                        type: 'scatter',
                        name: 'Memory (MB)'
                    };
                    Plotly.newPlot('memory-chart', [memoryTrace], {title: 'Memory Usage'});
                }
                
                // Update every 2 seconds
                setInterval(updateDashboard, 2000);
                updateDashboard();
            </script>
        </body>
        </html>
        """
        
    def export_stats(self, filepath: str):
        """Export monitoring statistics to file."""
        stats_data = {
            'routing_history': [
                {
                    'load_variance': getattr(info, 'load_variance', 0.0),
                    'entropy': getattr(info, 'entropy', 0.0),
                    'timestamp': time.time()
                }
                for info in self.routing_history
            ],
            'expert_counts': dict(self.expert_counts),
            'total_tokens': self.token_count,
            'dropped_tokens': self.dropped_tokens,
            'monitoring_duration': time.time() - self.start_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Statistics exported to {filepath}")
        
    def reset_stats(self):
        """Reset all monitoring statistics."""
        self.routing_history.clear()
        self.stats_history.clear()
        self.expert_counts.clear()
        self.token_count = 0
        self.dropped_tokens = 0
        self.start_time = time.time()
        self.dashboard_data = {
            'expert_loads': [],
            'routing_entropy': [],
            'throughput': [],
            'memory_usage': []
        }
        logger.info("Monitoring statistics reset")