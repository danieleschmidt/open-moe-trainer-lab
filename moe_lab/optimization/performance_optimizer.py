"""
Advanced performance optimization for MoE models.
Generation 3: Production-ready performance optimizations.
"""

import os
import time
import logging
import statistics
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import hashlib

try:
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.profiler import profile, ProfilerActivity
except ImportError:
    # Mock for environments without PyTorch
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    throughput_tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    expert_load_balance: float = 1.0  # 1.0 = perfect balance
    model_flops: int = 0
    inference_cost_usd: float = 0.0
    timestamp: float = 0.0


class IntelligentCache:
    """Multi-level intelligent caching system for MoE inference."""
    
    def __init__(
        self,
        l1_capacity_mb: int = 100,
        l2_capacity_mb: int = 500,
        l3_capacity_mb: int = 2000,
        enable_smart_prefetch: bool = True
    ):
        self.l1_capacity = l1_capacity_mb * 1024 * 1024  # Convert to bytes
        self.l2_capacity = l2_capacity_mb * 1024 * 1024
        self.l3_capacity = l3_capacity_mb * 1024 * 1024
        
        # Multi-level caches
        self.l1_cache = {}  # Hot data - in memory
        self.l2_cache = {}  # Warm data - compressed in memory
        self.l3_cache = {}  # Cold data - disk cache
        
        # Cache metadata
        self.access_patterns = defaultdict(deque)  # Track access patterns
        self.cache_stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0,
            "evictions": 0, "prefetches": 0
        }
        
        # Smart prefetching
        self.enable_smart_prefetch = enable_smart_prefetch
        self.access_sequences = deque(maxlen=10000)
        self.sequence_patterns = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background prefetching thread
        if enable_smart_prefetch:
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent promotion."""
        with self.lock:
            # Try L1 cache first
            if key in self.l1_cache:
                self.cache_stats["l1_hits"] += 1
                self._update_access_pattern(key)
                return self.l1_cache[key]
            
            # Try L2 cache
            if key in self.l2_cache:
                self.cache_stats["l2_hits"] += 1
                data = self._decompress(self.l2_cache[key])
                
                # Promote to L1 if frequently accessed
                if self._should_promote_to_l1(key):
                    self._put_l1(key, data)
                    del self.l2_cache[key]
                
                self._update_access_pattern(key)
                return data
            
            # Try L3 cache (disk)
            if key in self.l3_cache:
                self.cache_stats["l3_hits"] += 1
                data = self._load_from_disk(self.l3_cache[key])
                
                # Promote to L2
                if self._should_promote_to_l2(key):
                    self._put_l2(key, data)
                    del self.l3_cache[key]
                
                self._update_access_pattern(key)
                return data
            
            # Cache miss
            self.cache_stats["l1_misses"] += 1
            return None
    
    def put(self, key: str, value: Any, priority: str = "normal") -> None:
        """Put item in cache with intelligent placement."""
        with self.lock:
            if priority == "high" or self._is_hot_data(key):
                self._put_l1(key, value)
            elif priority == "medium":
                self._put_l2(key, value)
            else:
                self._put_l3(key, value)
    
    def _put_l1(self, key: str, value: Any) -> None:
        """Put item in L1 cache."""
        # Estimate size (simplified)
        size = self._estimate_size(value)
        
        # Evict if necessary
        while self._get_l1_size() + size > self.l1_capacity and self.l1_cache:
            lru_key = self._get_lru_key(self.l1_cache)
            evicted_value = self.l1_cache.pop(lru_key)
            
            # Demote to L2
            self._put_l2(lru_key, evicted_value)
            self.cache_stats["evictions"] += 1
        
        self.l1_cache[key] = value
    
    def _put_l2(self, key: str, value: Any) -> None:
        """Put item in L2 cache with compression."""
        compressed_value = self._compress(value)
        size = len(compressed_value)
        
        # Evict if necessary
        while self._get_l2_size() + size > self.l2_capacity and self.l2_cache:
            lru_key = self._get_lru_key(self.l2_cache)
            evicted_value = self.l2_cache.pop(lru_key)
            
            # Demote to L3
            self._put_l3_compressed(lru_key, evicted_value)
            self.cache_stats["evictions"] += 1
        
        self.l2_cache[key] = compressed_value
    
    def _put_l3(self, key: str, value: Any) -> None:
        """Put item in L3 cache (disk)."""
        file_path = self._get_cache_file_path(key)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            self.l3_cache[key] = str(file_path)
        except Exception as e:
            logger.warning(f"Failed to cache to disk: {e}")
    
    def _put_l3_compressed(self, key: str, compressed_value: bytes) -> None:
        """Put compressed item in L3 cache."""
        file_path = self._get_cache_file_path(key)
        try:
            with open(file_path, 'wb') as f:
                f.write(compressed_value)
            self.l3_cache[key] = str(file_path)
        except Exception as e:
            logger.warning(f"Failed to cache compressed data to disk: {e}")
    
    def _compress(self, data: Any) -> bytes:
        """Compress data for L2 cache."""
        import zlib
        return zlib.compress(pickle.dumps(data))
    
    def _decompress(self, compressed_data: bytes) -> Any:
        """Decompress data from L2 cache."""
        import zlib
        return pickle.loads(zlib.decompress(compressed_data))
    
    def _load_from_disk(self, file_path: str) -> Any:
        """Load data from disk cache."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def _update_access_pattern(self, key: str) -> None:
        """Update access patterns for smart prefetching."""
        self.access_sequences.append(key)
        
        if len(self.access_sequences) >= 2:
            # Update sequence patterns
            prev_key = self.access_sequences[-2]
            if prev_key not in self.sequence_patterns:
                self.sequence_patterns[prev_key] = defaultdict(int)
            self.sequence_patterns[prev_key][key] += 1
    
    def _prefetch_worker(self) -> None:
        """Background worker for smart prefetching."""
        while True:
            try:
                if len(self.access_sequences) > 0:
                    current_key = self.access_sequences[-1]
                    
                    # Predict next access
                    if current_key in self.sequence_patterns:
                        patterns = self.sequence_patterns[current_key]
                        # Get most likely next key
                        next_key = max(patterns.items(), key=lambda x: x[1])[0]
                        
                        # Prefetch if not in cache
                        if (next_key not in self.l1_cache and 
                            next_key not in self.l2_cache):
                            self._prefetch(next_key)
                
                time.sleep(0.1)  # Avoid busy waiting
                
            except Exception as e:
                logger.debug(f"Prefetch worker error: {e}")
                time.sleep(1)
    
    def _prefetch(self, key: str) -> None:
        """Prefetch data into cache."""
        # This would fetch data from the original source
        # For now, just track the prefetch attempt
        self.cache_stats["prefetches"] += 1
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def _get_l1_size(self) -> int:
        """Get current L1 cache size."""
        return sum(self._estimate_size(v) for v in self.l1_cache.values())
    
    def _get_l2_size(self) -> int:
        """Get current L2 cache size."""
        return sum(len(v) for v in self.l2_cache.values())
    
    def _get_lru_key(self, cache_dict: dict) -> str:
        """Get least recently used key (simplified LRU)."""
        return next(iter(cache_dict))  # First key (oldest in ordered dict)
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Decide if item should be promoted to L1."""
        access_count = len(self.access_patterns[key])
        return access_count > 3  # Promote if accessed more than 3 times
    
    def _should_promote_to_l2(self, key: str) -> bool:
        """Decide if item should be promoted to L2."""
        return True  # Always promote from L3 to L2
    
    def _is_hot_data(self, key: str) -> bool:
        """Determine if data is hot (frequently accessed)."""
        return len(self.access_patterns[key]) > 5
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Get file path for disk cache."""
        cache_dir = Path("/tmp/moe_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Create safe filename from key
        safe_name = hashlib.md5(key.encode()).hexdigest()
        return cache_dir / f"{safe_name}.cache"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = sum([
                self.cache_stats["l1_hits"], self.cache_stats["l1_misses"],
                self.cache_stats["l2_hits"], self.cache_stats["l2_misses"],
                self.cache_stats["l3_hits"], self.cache_stats["l3_misses"]
            ])
            
            total_hits = (self.cache_stats["l1_hits"] + 
                         self.cache_stats["l2_hits"] + 
                         self.cache_stats["l3_hits"])
            
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            return {
                **self.cache_stats,
                "total_requests": total_requests,
                "overall_hit_rate": hit_rate,
                "l1_size": len(self.l1_cache),
                "l2_size": len(self.l2_cache),
                "l3_size": len(self.l3_cache),
                "l1_size_mb": self._get_l1_size() / (1024 * 1024),
                "l2_size_mb": self._get_l2_size() / (1024 * 1024)
            }


class PerformanceProfiler:
    """Advanced performance profiler for MoE models."""
    
    def __init__(self, enable_detailed_profiling: bool = False):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.profiling_data = defaultdict(list)
        self.current_session = None
        
    def start_profiling(self, profile_name: str = "default"):
        """Start profiling session."""
        if torch is None:
            logger.warning("PyTorch not available, profiling disabled")
            return
        
        if self.enable_detailed_profiling:
            self.current_session = {
                "name": profile_name,
                "start_time": time.time(),
                "profiler": profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_stack=True,
                    record_shapes=True
                )
            }
            self.current_session["profiler"].start()
    
    def stop_profiling(self) -> Optional[Dict[str, Any]]:
        """Stop profiling and return results."""
        if self.current_session is None:
            return None
        
        self.current_session["profiler"].stop()
        self.current_session["end_time"] = time.time()
        
        # Analyze profiling data
        results = self._analyze_profiling_data()
        
        # Store for later analysis
        self.profiling_data[self.current_session["name"]].append(results)
        
        self.current_session = None
        return results
    
    def _analyze_profiling_data(self) -> Dict[str, Any]:
        """Analyze profiling data and extract insights."""
        if not self.current_session:
            return {}
        
        prof = self.current_session["profiler"]
        duration = self.current_session["end_time"] - self.current_session["start_time"]
        
        # Get top operations by time
        key_averages = prof.key_averages()
        top_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]
        
        # Memory usage
        memory_usage = []
        if hasattr(prof, 'profiler'):
            for event in prof.profiler.function_events:
                if hasattr(event, 'cpu_memory_usage'):
                    memory_usage.append(event.cpu_memory_usage)
        
        return {
            "duration": duration,
            "top_operations": [
                {
                    "name": op.key,
                    "cpu_time": op.cpu_time,
                    "cpu_time_total": op.cpu_time_total,
                    "count": op.count,
                    "input_shapes": getattr(op, 'input_shapes', [])
                }
                for op in top_ops
            ],
            "total_cpu_time": sum(op.cpu_time_total for op in key_averages),
            "memory_usage_mb": max(memory_usage) / (1024 * 1024) if memory_usage else 0
        }
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if not self.profiling_data:
            return ["No profiling data available"]
        
        # Analyze all profiling sessions
        all_sessions = []
        for sessions in self.profiling_data.values():
            all_sessions.extend(sessions)
        
        if not all_sessions:
            return recommendations
        
        # Check for common performance issues
        avg_duration = statistics.mean(s["duration"] for s in all_sessions)
        if avg_duration > 1.0:
            recommendations.append("Consider model quantization to reduce inference time")
        
        # Analyze top operations
        all_top_ops = []
        for session in all_sessions:
            all_top_ops.extend(session.get("top_operations", []))
        
        if all_top_ops:
            # Check for expensive operations
            expensive_ops = [op for op in all_top_ops if op["cpu_time_total"] > 100]
            if expensive_ops:
                recommendations.append("Consider optimizing expensive operations with CUDA kernels")
        
        # Memory usage recommendations
        memory_usage = [s.get("memory_usage_mb", 0) for s in all_sessions]
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0
        
        if avg_memory > 1000:  # 1GB+
            recommendations.append("High memory usage detected - consider gradient checkpointing")
        
        return recommendations or ["Performance appears optimal"]


class AdaptiveOptimizer:
    """Adaptive performance optimizer that learns from usage patterns."""
    
    def __init__(self, model, cache_system: IntelligentCache):
        self.model = model
        self.cache_system = cache_system
        self.performance_history = deque(maxlen=1000)
        self.optimization_strategies = {}
        
        # Adaptive parameters
        self.batch_size_range = (1, 32)
        self.current_batch_size = 8
        self.temperature_range = (0.1, 2.0)
        self.sequence_length_buckets = [128, 256, 512, 1024, 2048]
        
        # Learning system
        self.strategy_scores = defaultdict(lambda: {"success": 0, "total": 0})
        self.current_strategy = "baseline"
        
    def optimize_inference(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize inference based on request characteristics and history."""
        # Analyze request
        request_profile = self._profile_request(request_data)
        
        # Select optimization strategy
        strategy = self._select_strategy(request_profile)
        
        # Apply optimizations
        optimized_params = self._apply_strategy(strategy, request_data)
        
        # Track for learning
        self.current_strategy = strategy
        
        return optimized_params
    
    def _profile_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Profile the incoming request to understand its characteristics."""
        prompt = request_data.get("prompt", "")
        
        return {
            "prompt_length": len(prompt.split()),
            "complexity_score": self._estimate_complexity(prompt),
            "domain": self._detect_domain(prompt),
            "time_of_day": time.localtime().tm_hour,
            "max_tokens": request_data.get("max_new_tokens", 50)
        }
    
    def _estimate_complexity(self, prompt: str) -> float:
        """Estimate prompt complexity (0-1 scale)."""
        # Simple heuristics for complexity
        factors = [
            len(prompt) / 1000,  # Length factor
            len(set(prompt.split())) / len(prompt.split()) if prompt else 0,  # Vocabulary diversity
            prompt.count('?') * 0.1,  # Question complexity
            prompt.count(',') * 0.05,  # Structural complexity
        ]
        
        return min(1.0, sum(factors))
    
    def _detect_domain(self, prompt: str) -> str:
        """Detect domain of the prompt for specialized optimization."""
        domains = {
            "math": ["calculate", "solve", "equation", "math", "number"],
            "code": ["function", "code", "programming", "algorithm", "debug"],
            "creative": ["story", "poem", "creative", "imagine", "write"],
            "qa": ["what", "how", "why", "when", "where", "question"],
            "general": []
        }
        
        prompt_lower = prompt.lower()
        for domain, keywords in domains.items():
            if any(kw in prompt_lower for kw in keywords):
                return domain
        
        return "general"
    
    def _select_strategy(self, request_profile: Dict[str, Any]) -> str:
        """Select optimization strategy based on request profile and performance history."""
        
        # Get strategies ranked by success rate
        ranked_strategies = sorted(
            self.strategy_scores.items(),
            key=lambda x: x[1]["success"] / max(x[1]["total"], 1),
            reverse=True
        )
        
        # Adaptive selection based on request characteristics
        if request_profile["complexity_score"] > 0.7:
            return "high_complexity"
        elif request_profile["prompt_length"] > 100:
            return "long_prompt"
        elif request_profile["domain"] == "code":
            return "code_optimized"
        elif request_profile["max_tokens"] > 200:
            return "long_generation"
        
        # Use best performing strategy
        if ranked_strategies and ranked_strategies[0][1]["total"] > 10:
            return ranked_strategies[0][0]
        
        return "baseline"
    
    def _apply_strategy(self, strategy: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the selected optimization strategy."""
        optimized = request_data.copy()
        
        if strategy == "high_complexity":
            # Reduce temperature for more focused output
            optimized["temperature"] = max(0.3, optimized.get("temperature", 1.0) * 0.7)
            # Increase top_p for better quality
            optimized["top_p"] = min(0.95, optimized.get("top_p", 0.9) * 1.1)
            
        elif strategy == "long_prompt":
            # Optimize for long prompts
            optimized["do_sample"] = True
            optimized["repetition_penalty"] = 1.1
            
        elif strategy == "code_optimized":
            # Optimize for code generation
            optimized["temperature"] = 0.2
            optimized["top_p"] = 0.95
            optimized["repetition_penalty"] = 1.05
            
        elif strategy == "long_generation":
            # Optimize for long generation
            optimized["temperature"] = 0.8
            optimized["repetition_penalty"] = 1.1
            
        # Cache optimized parameters for similar requests
        request_hash = self._hash_request(request_data)
        self.cache_system.put(f"optimized_params_{request_hash}", optimized, priority="medium")
        
        return optimized
    
    def update_performance(self, strategy: str, success: bool, metrics: PerformanceMetrics):
        """Update performance tracking for learning."""
        self.strategy_scores[strategy]["total"] += 1
        if success:
            self.strategy_scores[strategy]["success"] += 1
        
        # Store performance metrics
        self.performance_history.append({
            "strategy": strategy,
            "success": success,
            "metrics": asdict(metrics),
            "timestamp": time.time()
        })
        
        # Adaptive parameter adjustment
        self._adjust_parameters(metrics)
    
    def _adjust_parameters(self, metrics: PerformanceMetrics):
        """Adjust optimization parameters based on performance."""
        # Adjust batch size based on throughput
        if metrics.throughput_tokens_per_sec > 100:
            self.current_batch_size = min(32, self.current_batch_size + 1)
        elif metrics.throughput_tokens_per_sec < 50:
            self.current_batch_size = max(1, self.current_batch_size - 1)
        
        # Memory-based adjustments
        if metrics.memory_usage_gb > 8:
            self.current_batch_size = max(1, self.current_batch_size - 2)
    
    def _hash_request(self, request_data: Dict[str, Any]) -> str:
        """Create hash for request caching."""
        key_data = {
            "prompt_hash": hashlib.md5(request_data.get("prompt", "").encode()).hexdigest(),
            "max_tokens": request_data.get("max_new_tokens", 50),
            "temperature": request_data.get("temperature", 1.0)
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_metrics = [entry["metrics"] for entry in self.performance_history[-100:]]
        
        return {
            "total_optimizations": len(self.performance_history),
            "strategy_performance": dict(self.strategy_scores),
            "average_throughput": statistics.mean(m["throughput_tokens_per_sec"] for m in recent_metrics),
            "average_latency": statistics.mean(m["latency_ms"] for m in recent_metrics),
            "cache_hit_rate": statistics.mean(m["cache_hit_rate"] for m in recent_metrics),
            "current_batch_size": self.current_batch_size,
            "optimization_recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if not self.performance_history:
            return ["Insufficient data for recommendations"]
        
        recent_metrics = [entry["metrics"] for entry in self.performance_history[-50:]]
        
        # Throughput analysis
        avg_throughput = statistics.mean(m["throughput_tokens_per_sec"] for m in recent_metrics)
        if avg_throughput < 50:
            recommendations.append("Consider increasing batch size or using model quantization")
        
        # Latency analysis
        avg_latency = statistics.mean(m["latency_ms"] for m in recent_metrics)
        if avg_latency > 500:
            recommendations.append("High latency detected - consider model compilation or caching")
        
        # Cache performance
        avg_cache_hit = statistics.mean(m["cache_hit_rate"] for m in recent_metrics)
        if avg_cache_hit < 0.5:
            recommendations.append("Low cache hit rate - consider increasing cache size or improving prefetching")
        
        # Memory usage
        avg_memory = statistics.mean(m["memory_usage_gb"] for m in recent_metrics)
        if avg_memory > 6:
            recommendations.append("High memory usage - consider gradient checkpointing or smaller batch sizes")
        
        return recommendations or ["Performance appears optimal"]


class ProductionOptimizer:
    """Production-ready performance optimizer orchestrating all optimization components."""
    
    def __init__(self, model, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        
        # Initialize components
        self.cache_system = IntelligentCache(
            l1_capacity_mb=self.config.get("l1_cache_mb", 100),
            l2_capacity_mb=self.config.get("l2_cache_mb", 500),
            l3_capacity_mb=self.config.get("l3_cache_mb", 2000),
            enable_smart_prefetch=self.config.get("enable_prefetch", True)
        )
        
        self.profiler = PerformanceProfiler(
            enable_detailed_profiling=self.config.get("enable_profiling", False)
        )
        
        self.adaptive_optimizer = AdaptiveOptimizer(model, self.cache_system)
        
        # Performance tracking
        self.global_metrics = PerformanceMetrics()
        self.optimization_sessions = []
        
        logger.info("Production optimizer initialized with advanced caching and profiling")
    
    def optimize_request(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Optimize a single inference request."""
        session_id = str(time.time())
        
        # Start profiling
        self.profiler.start_profiling(f"request_{session_id}")
        
        # Check cache first
        request_hash = self._hash_request(request_data)
        cached_result = self.cache_system.get(f"result_{request_hash}")
        
        if cached_result:
            logger.debug("Using cached result")
            return cached_result, session_id
        
        # Apply optimizations
        optimized_params = self.adaptive_optimizer.optimize_inference(request_data)
        
        return optimized_params, session_id
    
    def finalize_request(self, session_id: str, success: bool, result_data: Dict[str, Any], metrics: PerformanceMetrics):
        """Finalize request optimization and update learning systems."""
        
        # Stop profiling
        profiling_results = self.profiler.stop_profiling()
        
        # Update adaptive optimizer
        strategy = self.adaptive_optimizer.current_strategy
        self.adaptive_optimizer.update_performance(strategy, success, metrics)
        
        # Cache successful results
        if success and result_data:
            request_hash = self._hash_request(result_data)
            self.cache_system.put(f"result_{request_hash}", result_data, priority="high")
        
        # Update global metrics
        self._update_global_metrics(metrics)
        
        # Store session data
        session_data = {
            "session_id": session_id,
            "success": success,
            "metrics": asdict(metrics),
            "profiling": profiling_results,
            "timestamp": time.time()
        }
        self.optimization_sessions.append(session_data)
        
        # Cleanup old sessions
        if len(self.optimization_sessions) > 1000:
            self.optimization_sessions = self.optimization_sessions[-1000:]
    
    def _update_global_metrics(self, metrics: PerformanceMetrics):
        """Update global performance metrics."""
        # Simple moving average
        alpha = 0.1  # Learning rate
        
        self.global_metrics.throughput_tokens_per_sec = (
            alpha * metrics.throughput_tokens_per_sec + 
            (1 - alpha) * self.global_metrics.throughput_tokens_per_sec
        )
        
        self.global_metrics.latency_ms = (
            alpha * metrics.latency_ms + 
            (1 - alpha) * self.global_metrics.latency_ms
        )
        
        self.global_metrics.cache_hit_rate = (
            alpha * metrics.cache_hit_rate + 
            (1 - alpha) * self.global_metrics.cache_hit_rate
        )
        
        self.global_metrics.timestamp = time.time()
    
    def _hash_request(self, request_data: Dict[str, Any]) -> str:
        """Create consistent hash for request caching."""
        return self.adaptive_optimizer._hash_request(request_data)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            "global_metrics": asdict(self.global_metrics),
            "cache_stats": self.cache_system.get_stats(),
            "adaptive_optimizer": self.adaptive_optimizer.get_optimization_report(),
            "profiler_recommendations": self.profiler.get_recommendations(),
            "total_sessions": len(self.optimization_sessions),
            "recent_success_rate": self._calculate_recent_success_rate(),
            "timestamp": time.time()
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent success rate."""
        recent_sessions = self.optimization_sessions[-100:] if self.optimization_sessions else []
        if not recent_sessions:
            return 0.0
        
        successful = sum(1 for s in recent_sessions if s["success"])
        return successful / len(recent_sessions)
    
    def cleanup(self):
        """Cleanup resources."""
        # Clear caches
        self.cache_system.l1_cache.clear()
        self.cache_system.l2_cache.clear()
        
        # Clear old sessions
        self.optimization_sessions = self.optimization_sessions[-100:]
        
        logger.info("Production optimizer cleanup completed")


def create_production_optimizer(model, config: Dict[str, Any] = None) -> ProductionOptimizer:
    """Factory function to create production optimizer."""
    return ProductionOptimizer(model, config)


if __name__ == "__main__":
    # Example usage
    print("🚀 MoE Production Optimizer")
    print("Advanced caching, profiling, and adaptive optimization system")
    
    # Mock model for demonstration
    class MockModel:
        pass
    
    model = MockModel()
    
    # Create optimizer
    config = {
        "l1_cache_mb": 200,
        "l2_cache_mb": 800,
        "l3_cache_mb": 3000,
        "enable_prefetch": True,
        "enable_profiling": True
    }
    
    optimizer = create_production_optimizer(model, config)
    
    # Simulate request
    request = {
        "prompt": "Explain quantum computing",
        "max_new_tokens": 100,
        "temperature": 1.0
    }
    
    # Optimize request
    optimized_params, session_id = optimizer.optimize_request(request)
    print(f"Optimized parameters: {optimized_params}")
    
    # Simulate completion
    mock_metrics = PerformanceMetrics(
        throughput_tokens_per_sec=75.0,
        latency_ms=200.0,
        cache_hit_rate=0.8
    )
    
    optimizer.finalize_request(session_id, True, {"result": "optimized"}, mock_metrics)
    
    # Get report
    report = optimizer.get_comprehensive_report()
    print(f"Optimization report: {json.dumps(report, indent=2, default=str)}")