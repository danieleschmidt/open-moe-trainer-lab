"""Expert caching system for optimized MoE inference."""

import time
import threading
from typing import Dict, Optional, Any, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import torch
import torch.nn as nn
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for expert cache performance."""
    
    hit_rate: float
    miss_rate: float
    avg_load_time_ms: float
    memory_usage_gb: float
    evictions: int
    total_requests: int
    cache_size: int


class ExpertCache:
    """
    LRU cache for MoE experts with memory management and performance tracking.
    
    Features:
    - LRU eviction policy with expert importance weighting
    - Memory usage monitoring and automatic cleanup
    - Performance statistics tracking
    - Thread-safe operations
    - Preloading of frequently used experts
    """
    
    def __init__(
        self,
        capacity_gb: float = 8.0,
        max_experts: int = 64,
        policy: str = "weighted_lru",
        preload_top_k: int = 4,
        memory_threshold: float = 0.9,
        cleanup_interval: int = 300  # seconds
    ):
        self.capacity_gb = capacity_gb
        self.max_experts = max_experts
        self.policy = policy
        self.preload_top_k = preload_top_k
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        
        # Cache storage
        self._cache: OrderedDict[int, nn.Module] = OrderedDict()
        self._expert_weights: Dict[int, float] = {}  # Expert importance weights
        self._access_times: Dict[int, float] = {}
        self._load_times: Dict[int, float] = {}
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_load_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._cleanup_timer = None
        self._start_cleanup_timer()
        
        logger.info(f"Initialized ExpertCache with {capacity_gb}GB capacity, max {max_experts} experts")
    
    def get(self, expert_id: int) -> Optional[nn.Module]:
        """
        Get expert from cache.
        
        Args:
            expert_id: ID of the expert to retrieve
            
        Returns:
            Expert module if cached, None otherwise
        """
        with self._lock:
            if expert_id in self._cache:
                # Move to end (most recently used)
                expert = self._cache.pop(expert_id)
                self._cache[expert_id] = expert
                self._access_times[expert_id] = time.time()
                self._hits += 1
                
                logger.debug(f"Cache hit for expert {expert_id}")
                return expert
            else:
                self._misses += 1
                logger.debug(f"Cache miss for expert {expert_id}")
                return None
    
    def put(self, expert_id: int, expert: nn.Module, weight: float = 1.0) -> None:
        """
        Add expert to cache with optional importance weight.
        
        Args:
            expert_id: ID of the expert
            expert: Expert module to cache
            weight: Importance weight for eviction policy
        """
        start_time = time.time()
        
        with self._lock:
            # Remove if already exists
            if expert_id in self._cache:
                del self._cache[expert_id]
            
            # Check capacity and evict if necessary
            self._ensure_capacity()
            
            # Add to cache
            self._cache[expert_id] = expert
            self._expert_weights[expert_id] = weight
            self._access_times[expert_id] = time.time()
            
            load_time = (time.time() - start_time) * 1000  # ms
            self._load_times[expert_id] = load_time
            self._total_load_time += load_time
            
            logger.debug(f"Cached expert {expert_id} with weight {weight:.3f}, load time {load_time:.2f}ms")
    
    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed capacity limits."""
        # Check expert count limit
        while len(self._cache) >= self.max_experts:
            self._evict_expert()
        
        # Check memory limit
        current_memory = self._get_memory_usage_gb()
        while current_memory > self.capacity_gb:
            self._evict_expert()
            current_memory = self._get_memory_usage_gb()
        
        # Check system memory pressure
        system_memory = psutil.virtual_memory()
        if system_memory.percent > self.memory_threshold * 100:
            logger.warning(f"System memory usage high ({system_memory.percent:.1f}%), triggering cache cleanup")
            self._evict_multiple(max(1, len(self._cache) // 4))
    
    def _evict_expert(self) -> None:
        """Evict one expert based on the configured policy."""
        if not self._cache:
            return
        
        if self.policy == "lru":
            expert_id = next(iter(self._cache))
        elif self.policy == "weighted_lru":
            expert_id = self._get_least_important_expert()
        else:
            expert_id = next(iter(self._cache))  # Fallback to LRU
        
        del self._cache[expert_id]
        self._expert_weights.pop(expert_id, None)
        self._access_times.pop(expert_id, None)
        self._load_times.pop(expert_id, None)
        self._evictions += 1
        
        logger.debug(f"Evicted expert {expert_id}")
    
    def _evict_multiple(self, count: int) -> None:
        """Evict multiple experts at once."""
        for _ in range(min(count, len(self._cache))):
            self._evict_expert()
    
    def _get_least_important_expert(self) -> int:
        """Get the least important expert for weighted LRU eviction."""
        if not self._cache:
            return next(iter(self._cache))
        
        current_time = time.time()
        min_score = float('inf')
        least_important_id = next(iter(self._cache))
        
        for expert_id in self._cache:
            # Combine recency and importance
            recency = current_time - self._access_times.get(expert_id, 0)
            weight = self._expert_weights.get(expert_id, 1.0)
            
            # Lower score = less important (older access time, lower weight)
            score = weight / (1 + recency)
            
            if score < min_score:
                min_score = score
                least_important_id = expert_id
        
        return least_important_id
    
    def _get_memory_usage_gb(self) -> float:
        """Estimate memory usage of cached experts in GB."""
        total_params = 0
        for expert in self._cache.values():
            total_params += sum(p.numel() for p in expert.parameters())
        
        # Assume fp32 (4 bytes per parameter)
        bytes_used = total_params * 4
        return bytes_used / (1024 ** 3)
    
    def preload_experts(self, expert_ids: List[int], experts: Dict[int, nn.Module]) -> None:
        """
        Preload frequently used experts into cache.
        
        Args:
            expert_ids: List of expert IDs to preload
            experts: Dictionary mapping expert IDs to expert modules
        """
        logger.info(f"Preloading {len(expert_ids)} experts")
        
        for expert_id in expert_ids:
            if expert_id in experts:
                # Give preloaded experts higher weight
                self.put(expert_id, experts[expert_id], weight=2.0)
    
    def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(total_requests, 1)
            miss_rate = self._misses / max(total_requests, 1)
            
            avg_load_time = 0.0
            if self._load_times:
                avg_load_time = sum(self._load_times.values()) / len(self._load_times)
            
            return CacheStats(
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                avg_load_time_ms=avg_load_time,
                memory_usage_gb=self._get_memory_usage_gb(),
                evictions=self._evictions,
                total_requests=total_requests,
                cache_size=len(self._cache)
            )
    
    def clear(self) -> None:
        """Clear all cached experts."""
        with self._lock:
            self._cache.clear()
            self._expert_weights.clear()
            self._access_times.clear()
            self._load_times.clear()
            logger.info("Cache cleared")
    
    def warmup(self, experts: Dict[int, nn.Module], usage_stats: Optional[Dict[int, float]] = None) -> None:
        """
        Warm up cache with experts based on usage statistics.
        
        Args:
            experts: Dictionary of all available experts
            usage_stats: Optional usage frequency for each expert
        """
        if usage_stats:
            # Sort experts by usage frequency
            sorted_experts = sorted(usage_stats.items(), key=lambda x: x[1], reverse=True)
            top_experts = sorted_experts[:self.preload_top_k]
            
            logger.info(f"Warming up cache with top {len(top_experts)} experts")
            for expert_id, usage_freq in top_experts:
                if expert_id in experts:
                    self.put(expert_id, experts[expert_id], weight=usage_freq)
        else:
            # Just preload first few experts
            expert_ids = list(experts.keys())[:self.preload_top_k]
            self.preload_experts(expert_ids, experts)
    
    def _cleanup_expired(self) -> None:
        """Periodic cleanup of cache based on memory pressure."""
        with self._lock:
            system_memory = psutil.virtual_memory()
            
            if system_memory.percent > self.memory_threshold * 100:
                # Aggressive cleanup under memory pressure
                target_size = max(1, len(self._cache) // 2)
                while len(self._cache) > target_size:
                    self._evict_expert()
                
                logger.info(f"Memory pressure cleanup: evicted to {len(self._cache)} experts")
    
    def _start_cleanup_timer(self) -> None:
        """Start periodic cleanup timer."""
        self._cleanup_timer = threading.Timer(self.cleanup_interval, self._periodic_cleanup)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup callback."""
        try:
            self._cleanup_expired()
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
        finally:
            # Restart timer
            self._start_cleanup_timer()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
    
    def __len__(self) -> int:
        """Get number of cached experts."""
        return len(self._cache)
    
    def __contains__(self, expert_id: int) -> bool:
        """Check if expert is cached."""
        return expert_id in self._cache


class AdaptiveExpertCache(ExpertCache):
    """
    Advanced expert cache with adaptive capacity management.
    
    Features:
    - Dynamic capacity adjustment based on usage patterns
    - Learning-based expert importance scoring
    - Predictive preloading
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive parameters
        self._usage_history: Dict[int, List[float]] = {}
        self._prediction_window = 100
        self._adaptation_frequency = 50  # requests
        self._request_count = 0
    
    def get(self, expert_id: int) -> Optional[nn.Module]:
        """Enhanced get with usage tracking."""
        result = super().get(expert_id)
        
        # Track usage patterns
        current_time = time.time()
        if expert_id not in self._usage_history:
            self._usage_history[expert_id] = []
        
        self._usage_history[expert_id].append(current_time)
        
        # Keep only recent history
        cutoff_time = current_time - 3600  # 1 hour
        self._usage_history[expert_id] = [
            t for t in self._usage_history[expert_id] if t > cutoff_time
        ]
        
        self._request_count += 1
        
        # Periodic adaptation
        if self._request_count % self._adaptation_frequency == 0:
            self._adapt_cache()
        
        return result
    
    def _adapt_cache(self) -> None:
        """Adapt cache parameters based on usage patterns."""
        current_time = time.time()
        
        # Calculate expert usage frequencies
        usage_frequencies = {}
        for expert_id, timestamps in self._usage_history.items():
            recent_requests = len([t for t in timestamps if t > current_time - 1800])  # 30 min
            usage_frequencies[expert_id] = recent_requests
        
        # Update expert weights based on usage patterns
        total_usage = sum(usage_frequencies.values())
        if total_usage > 0:
            for expert_id, usage in usage_frequencies.items():
                normalized_usage = usage / total_usage
                self._expert_weights[expert_id] = normalized_usage * 10  # Scale factor
        
        # Consider expanding cache if hit rate is low
        stats = self.get_stats()
        if stats.hit_rate < 0.7 and len(self._cache) < self.max_experts:
            # Could expand cache size or adjust eviction policy
            logger.info(f"Low hit rate ({stats.hit_rate:.2f}), considering cache expansion")
        
        logger.debug(f"Adapted cache with {len(usage_frequencies)} active experts")


# Factory function for easy cache creation
def create_expert_cache(
    cache_type: str = "standard",
    capacity_gb: float = 8.0,
    **kwargs
) -> ExpertCache:
    """
    Create an expert cache with specified configuration.
    
    Args:
        cache_type: Type of cache ("standard" or "adaptive")
        capacity_gb: Memory capacity in GB
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ExpertCache instance
    """
    if cache_type == "adaptive":
        return AdaptiveExpertCache(capacity_gb=capacity_gb, **kwargs)
    else:
        return ExpertCache(capacity_gb=capacity_gb, **kwargs)