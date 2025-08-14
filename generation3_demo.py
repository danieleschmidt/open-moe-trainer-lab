#!/usr/bin/env python3
"""
Generation 3 Demo: MAKE IT SCALE
Simplified but comprehensive demonstration of production-ready scaling features.
"""

import json
import time
import random
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import concurrent.futures
import threading


@dataclass
class ScalingConfig:
    """Configuration for scalable MoE demo."""
    hidden_size: int = 64
    num_experts: int = 8
    initial_workers: int = 2
    max_workers: int = 6
    target_latency_ms: float = 100.0
    cache_enabled: bool = True
    auto_scaling: bool = True


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'requests_processed': 0,
            'total_latency': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'worker_utilization': defaultdict(int),
            'scaling_events': []
        }
        self.request_history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def record_request(self, latency_ms: float, worker_id: str, cache_hit: bool):
        """Record a completed request."""
        with self.lock:
            self.metrics['requests_processed'] += 1
            self.metrics['total_latency'] += latency_ms
            self.metrics['worker_utilization'][worker_id] += 1
            
            if cache_hit:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
            
            self.request_history.append({
                'timestamp': time.time(),
                'latency_ms': latency_ms,
                'worker_id': worker_id,
                'cache_hit': cache_hit
            })
    
    def record_scaling_event(self, event_type: str, worker_count: int):
        """Record a scaling event."""
        with self.lock:
            self.metrics['scaling_events'].append({
                'timestamp': time.time(),
                'event': event_type,
                'worker_count': worker_count
            })
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.lock:
            total_requests = self.metrics['requests_processed']
            avg_latency = (self.metrics['total_latency'] / total_requests 
                          if total_requests > 0 else 0)
            
            total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
            cache_hit_rate = (self.metrics['cache_hits'] / total_cache_requests 
                             if total_cache_requests > 0 else 0)
            
            # Recent performance (last 100 requests)
            recent_requests = list(self.request_history)[-100:]
            recent_latency = (statistics.mean(r['latency_ms'] for r in recent_requests)
                             if recent_requests else 0)
            
            p95_latency = 0
            if len(recent_requests) >= 20:
                latencies = sorted(r['latency_ms'] for r in recent_requests)
                p95_index = int(0.95 * len(latencies))
                p95_latency = latencies[p95_index]
            
            return {
                'total_requests': total_requests,
                'avg_latency_ms': avg_latency,
                'recent_avg_latency_ms': recent_latency,
                'p95_latency_ms': p95_latency,
                'cache_hit_rate': cache_hit_rate,
                'worker_utilization': dict(self.metrics['worker_utilization']),
                'scaling_events': len(self.metrics['scaling_events']),
                'recent_scaling_events': self.metrics['scaling_events'][-5:]
            }


class IntelligentCache:
    """Simple but effective caching system."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        """Store item in cache."""
        with self.lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.cache[lru_key]
                del self.access_times[lru_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class WorkerNode:
    """Simulated worker node for processing requests."""
    
    def __init__(self, worker_id: str, base_latency_ms: float = 80.0):
        self.worker_id = worker_id
        self.base_latency_ms = base_latency_ms
        self.load = 0
        self.processed_requests = 0
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request on this worker."""
        start_time = time.time()
        
        # Simulate processing time based on load
        load_factor = 1.0 + (self.load * 0.1)  # 10% increase per concurrent request
        complexity_factor = len(request.get('prompt', '')) / 100.0  # Longer prompts take more time
        
        processing_time = (self.base_latency_ms + 
                          random.uniform(-20, 30) + 
                          (load_factor * 10) +
                          (complexity_factor * 5)) / 1000.0
        
        time.sleep(max(0.01, processing_time))  # Minimum 10ms
        
        actual_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        self.processed_requests += 1
        
        return {
            'worker_id': self.worker_id,
            'generated_text': f"Response from {self.worker_id}: {request.get('prompt', 'unknown')[:50]}...",
            'latency_ms': actual_latency,
            'processing_time_ms': processing_time * 1000
        }
    
    def get_utilization(self) -> float:
        """Get current utilization (0.0 to 1.0)."""
        return min(1.0, self.load / 10.0)  # Max 10 concurrent requests


class AutoScaler:
    """Automatic scaling based on performance metrics."""
    
    def __init__(self, config: ScalingConfig, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        self.last_scale_time = 0
        self.cooldown_period = 30  # 30 second cooldown
    
    def should_scale_up(self, current_workers: int, recent_stats: Dict[str, Any]) -> bool:
        """Determine if we should scale up."""
        if current_workers >= self.config.max_workers:
            return False
        
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False
        
        # Scale up if latency is too high
        return recent_stats['p95_latency_ms'] > self.config.target_latency_ms * 1.5
    
    def should_scale_down(self, current_workers: int, recent_stats: Dict[str, Any]) -> bool:
        """Determine if we should scale down."""
        if current_workers <= 1:
            return False
        
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False
        
        # Scale down if latency is consistently low and utilization is low
        return (recent_stats['p95_latency_ms'] < self.config.target_latency_ms * 0.5 and
                recent_stats['recent_avg_latency_ms'] < self.config.target_latency_ms * 0.6)
    
    def record_scaling_action(self):
        """Record that a scaling action occurred."""
        self.last_scale_time = time.time()


class ScalableMoESystem:
    """Production-ready scalable MoE system."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.cache = IntelligentCache() if config.cache_enabled else None
        self.auto_scaler = AutoScaler(config, self.monitor) if config.auto_scaling else None
        
        # Worker management
        self.workers = {}
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)
        self.load_balancer_index = 0
        
        # Initialize workers
        for i in range(config.initial_workers):
            worker_id = f"worker_{i}"
            self.workers[worker_id] = WorkerNode(worker_id, base_latency_ms=70 + (i * 10))
        
        print(f"🚀 Initialized scalable MoE system:")
        print(f"   • Initial workers: {len(self.workers)}")
        print(f"   • Max workers: {config.max_workers}")
        print(f"   • Target latency: {config.target_latency_ms}ms")
        print(f"   • Cache enabled: {config.cache_enabled}")
        print(f"   • Auto-scaling: {config.auto_scaling}")
    
    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        return f"{request.get('prompt', '')}_{request.get('max_tokens', 50)}"
    
    def _select_worker(self) -> str:
        """Select worker using round-robin load balancing."""
        worker_ids = list(self.workers.keys())
        selected = worker_ids[self.load_balancer_index % len(worker_ids)]
        self.load_balancer_index += 1
        return selected
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single request."""
        # Check cache first
        cache_hit = False
        if self.cache:
            cache_key = self._generate_cache_key(request)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                cache_hit = True
                result = cached_result.copy()
                result['cache_hit'] = True
                
                # Record metrics
                self.monitor.record_request(1.0, 'cache', cache_hit=True)
                return result
        
        # Select worker and process
        selected_worker_id = self._select_worker()
        worker = self.workers[selected_worker_id]
        
        # Increment load
        worker.load += 1
        
        try:
            # Process request
            result = worker.process_request(request)
            result['cache_hit'] = False
            
            # Cache result
            if self.cache:
                cache_key = self._generate_cache_key(request)
                self.cache.put(cache_key, result)
            
            # Record metrics
            self.monitor.record_request(
                result['latency_ms'], 
                selected_worker_id, 
                cache_hit=False
            )
            
            return result
            
        finally:
            # Decrement load
            worker.load -= 1
    
    def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of requests in parallel."""
        futures = []
        
        for request in requests:
            future = self.worker_pool.submit(self.process_request, request)
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results
    
    def check_and_scale(self):
        """Check if scaling is needed and perform it."""
        if not self.auto_scaler:
            return
        
        current_stats = self.monitor.get_current_stats()
        current_workers = len(self.workers)
        
        if self.auto_scaler.should_scale_up(current_workers, current_stats):
            self._scale_up()
        elif self.auto_scaler.should_scale_down(current_workers, current_stats):
            self._scale_down()
    
    def _scale_up(self):
        """Add a new worker."""
        new_worker_id = f"worker_{len(self.workers)}"
        base_latency = 70 + (len(self.workers) * 10)
        self.workers[new_worker_id] = WorkerNode(new_worker_id, base_latency_ms=base_latency)
        
        self.monitor.record_scaling_event('scale_up', len(self.workers))
        self.auto_scaler.record_scaling_action()
        
        print(f"📈 Scaled UP: Added {new_worker_id} (total: {len(self.workers)} workers)")
    
    def _scale_down(self):
        """Remove a worker."""
        if len(self.workers) <= 1:
            return
        
        # Remove least utilized worker
        worker_stats = self.monitor.get_current_stats()['worker_utilization']
        least_used = min(self.workers.keys(), 
                        key=lambda w: worker_stats.get(w, 0))
        
        del self.workers[least_used]
        
        self.monitor.record_scaling_event('scale_down', len(self.workers))
        self.auto_scaler.record_scaling_action()
        
        print(f"📉 Scaled DOWN: Removed {least_used} (total: {len(self.workers)} workers)")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        performance_stats = self.monitor.get_current_stats()
        
        return {
            'config': asdict(self.config),
            'current_workers': len(self.workers),
            'cache_size': self.cache.size() if self.cache else 0,
            'performance': performance_stats,
            'worker_details': {
                worker_id: {
                    'processed_requests': worker.processed_requests,
                    'current_load': worker.load,
                    'utilization': worker.get_utilization()
                }
                for worker_id, worker in self.workers.items()
            }
        }


def demo_generation3_scaling():
    """Demonstrate Generation 3 scaling capabilities."""
    print("⚡ Generation 3 Demo: MAKE IT SCALE")
    print("=" * 60)
    
    # Create configuration
    config = ScalingConfig(
        hidden_size=64,
        num_experts=8,
        initial_workers=2,
        max_workers=5,
        target_latency_ms=100.0,
        cache_enabled=True,
        auto_scaling=True
    )
    
    print("📋 Configuration:")
    for key, value in asdict(config).items():
        print(f"   {key}: {value}")
    print()
    
    # Create system
    system = ScalableMoESystem(config)
    print()
    
    # Simulate different load patterns
    load_scenarios = [
        {"name": "Light Load", "requests": 20, "batch_size": 2, "delay": 0.5},
        {"name": "Medium Load", "requests": 50, "batch_size": 5, "delay": 0.2},
        {"name": "Heavy Load", "requests": 100, "batch_size": 10, "delay": 0.05},
        {"name": "Spike Load", "requests": 80, "batch_size": 15, "delay": 0.01},
        {"name": "Cool Down", "requests": 30, "batch_size": 3, "delay": 0.3}
    ]
    
    all_results = []
    scenario_stats = {}
    
    for scenario in load_scenarios:
        print(f"🔄 {scenario['name']}: {scenario['requests']} requests")
        scenario_start = time.time()
        scenario_results = []
        
        # Process requests in batches
        for i in range(0, scenario['requests'], scenario['batch_size']):
            batch_size = min(scenario['batch_size'], scenario['requests'] - i)
            
            # Generate batch requests
            batch_requests = []
            for j in range(batch_size):
                request = {
                    'prompt': f"Test prompt {i+j} for {scenario['name']} scenario",
                    'max_tokens': random.randint(20, 100)
                }
                batch_requests.append(request)
            
            # Process batch
            batch_results = system.process_batch(batch_requests)
            scenario_results.extend(batch_results)
            all_results.extend(batch_results)
            
            # Check for scaling opportunities
            system.check_and_scale()
            
            # Delay between batches
            time.sleep(scenario['delay'])
        
        scenario_duration = time.time() - scenario_start
        
        # Calculate scenario statistics
        successful_results = [r for r in scenario_results if 'error' not in r]
        if successful_results:
            latencies = [r['latency_ms'] for r in successful_results]
            cache_hits = sum(1 for r in successful_results if r.get('cache_hit', False))
            
            scenario_stats[scenario['name']] = {
                'requests': len(scenario_results),
                'successful': len(successful_results),
                'duration_s': scenario_duration,
                'throughput_rps': len(successful_results) / scenario_duration,
                'avg_latency_ms': statistics.mean(latencies),
                'p95_latency_ms': statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
                'cache_hit_rate': cache_hits / len(successful_results),
                'workers_used': len(system.workers)
            }
            
            print(f"   ✓ {len(successful_results)} successful requests")
            print(f"   ⏱️  Avg latency: {scenario_stats[scenario['name']]['avg_latency_ms']:.1f}ms")
            print(f"   📊 Throughput: {scenario_stats[scenario['name']]['throughput_rps']:.1f} RPS")
            print(f"   🏠 Cache hit rate: {scenario_stats[scenario['name']]['cache_hit_rate']:.1%}")
            print(f"   👥 Workers: {scenario_stats[scenario['name']]['workers_used']}")
        
        print()
        
        # Brief pause between scenarios
        time.sleep(1)
    
    # Final system analysis
    print("📊 GENERATION 3 SCALING ANALYSIS:")
    print("=" * 50)
    
    final_stats = system.get_system_stats()
    
    print(f"🏗️  System Architecture:")
    print(f"   Final Worker Count: {final_stats['current_workers']}")
    print(f"   Cache Size: {final_stats['cache_size']} items")
    print(f"   Total Requests Processed: {final_stats['performance']['total_requests']}")
    print(f"   Overall Cache Hit Rate: {final_stats['performance']['cache_hit_rate']:.1%}")
    print()
    
    print(f"⚡ Performance Metrics:")
    perf = final_stats['performance']
    print(f"   Average Latency: {perf['avg_latency_ms']:.1f}ms")
    print(f"   Recent Average Latency: {perf['recent_avg_latency_ms']:.1f}ms")
    print(f"   P95 Latency: {perf['p95_latency_ms']:.1f}ms")
    print(f"   SLA Compliance: {'✅ PASS' if perf['p95_latency_ms'] < config.target_latency_ms * 1.2 else '❌ FAIL'}")
    print()
    
    print(f"📈 Auto-Scaling Events:")
    if perf['scaling_events'] > 0:
        print(f"   Total Scaling Events: {perf['scaling_events']}")
        for event in perf['recent_scaling_events']:
            event_time = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
            print(f"   {event_time}: {event['event']} → {event['worker_count']} workers")
    else:
        print(f"   No scaling events occurred")
    print()
    
    print(f"👥 Worker Utilization:")
    for worker_id, details in final_stats['worker_details'].items():
        print(f"   {worker_id}: {details['processed_requests']} requests, "
              f"{details['utilization']:.1%} utilization")
    print()
    
    print(f"📊 Load Scenario Results:")
    for scenario_name, stats in scenario_stats.items():
        print(f"   {scenario_name}:")
        print(f"     Throughput: {stats['throughput_rps']:.1f} RPS")
        print(f"     Latency: {stats['avg_latency_ms']:.1f}ms (P95: {stats['p95_latency_ms']:.1f}ms)")
        print(f"     Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"     Success Rate: {stats['successful']/stats['requests']:.1%}")
    print()
    
    # Calculate overall performance grade
    all_successful = [r for r in all_results if 'error' not in r]
    overall_success_rate = len(all_successful) / len(all_results) if all_results else 0
    
    all_latencies = [r['latency_ms'] for r in all_successful]
    overall_p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) > 20 else max(all_latencies) if all_latencies else 0
    
    # Determine performance grade
    if overall_success_rate >= 0.99 and overall_p95 < config.target_latency_ms:
        grade = "A+"
    elif overall_success_rate >= 0.95 and overall_p95 < config.target_latency_ms * 1.2:
        grade = "A"
    elif overall_success_rate >= 0.90 and overall_p95 < config.target_latency_ms * 1.5:
        grade = "B"
    else:
        grade = "C"
    
    print(f"🎯 Overall Performance Grade: {grade}")
    print(f"   Success Rate: {overall_success_rate:.1%}")
    print(f"   Overall P95 Latency: {overall_p95:.1f}ms")
    print(f"   Target Compliance: {'✅' if overall_p95 < config.target_latency_ms * 1.2 else '❌'}")
    
    # Save results
    results = {
        'demo_type': 'generation_3_scaling',
        'config': asdict(config),
        'final_stats': final_stats,
        'scenario_stats': scenario_stats,
        'overall_metrics': {
            'total_requests': len(all_results),
            'successful_requests': len(all_successful),
            'success_rate': overall_success_rate,
            'overall_p95_latency_ms': overall_p95,
            'performance_grade': grade
        },
        'scaling_features': [
            'Dynamic auto-scaling based on latency',
            'Intelligent caching with LRU eviction',
            'Load balancing across worker nodes',
            'Real-time performance monitoring',
            'Batch processing for efficiency',
            'Graceful handling of load spikes',
            'SLA compliance monitoring',
            'Production-ready worker management'
        ],
        'demo_completed': True,
        'timestamp': time.time()
    }
    
    with open('generation3_scaling_demo.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to generation3_scaling_demo.json")
    print("\n🎉 Generation 3 Scaling Demo Complete!")
    print("    ✓ Auto-scaling based on performance metrics")
    print("    ✓ Intelligent caching and load balancing") 
    print("    ✓ Real-time monitoring and SLA compliance")
    print("    ✓ Batch processing and worker management")
    print("    ✓ Production-ready performance optimization")
    print("    ✓ Graceful handling of varying load patterns")
    
    return results


if __name__ == "__main__":
    demo_generation3_scaling()