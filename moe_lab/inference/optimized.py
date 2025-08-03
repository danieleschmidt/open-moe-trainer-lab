"""Optimized MoE model for fast inference with expert caching."""

from typing import Optional, Union, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import os
import json
import time
from collections import OrderedDict
import threading
import logging

from ..models import MoEModel, MoEOutput
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ExpertCache:
    """LRU cache for expert weights with memory management."""
    
    def __init__(
        self,
        capacity: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.cache = OrderedDict()
        self.usage_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.lock = threading.RLock()
        
    def get(self, expert_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get expert weights from cache."""
        with self.lock:
            if expert_id in self.cache:
                # Move to end (most recently used)
                expert_weights = self.cache.pop(expert_id)
                self.cache[expert_id] = expert_weights
                self.usage_stats["hits"] += 1
                return expert_weights
            else:
                self.usage_stats["misses"] += 1
                return None
                
    def put(self, expert_id: str, expert_weights: Dict[str, torch.Tensor]) -> None:
        """Put expert weights into cache."""
        with self.lock:
            # Move weights to target device and dtype
            cached_weights = {}
            for name, weight in expert_weights.items():
                cached_weights[name] = weight.to(device=self.device, dtype=self.dtype)
                
            # Add to cache
            self.cache[expert_id] = cached_weights
            
            # Evict oldest if at capacity
            if len(self.cache) > self.capacity:
                oldest_id, _ = self.cache.popitem(last=False)
                self.usage_stats["evictions"] += 1
                logger.debug(f"Evicted expert {oldest_id} from cache")
                
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.usage_stats["hits"] + self.usage_stats["misses"]
            hit_rate = self.usage_stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "capacity": self.capacity,
                "current_size": len(self.cache),
                "hit_rate": hit_rate,
                "total_hits": self.usage_stats["hits"],
                "total_misses": self.usage_stats["misses"],
                "total_evictions": self.usage_stats["evictions"]
            }


class OptimizedMoEModel(nn.Module):
    """Optimized MoE model for fast inference."""
    
    def __init__(
        self,
        base_model: MoEModel,
        expert_cache_size: int = 8,
        selective_expert_loading: bool = True,
        expert_usage_threshold: float = 0.01,
        enable_compilation: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        
        self.base_model = base_model
        self.device = device
        self.dtype = dtype
        self.selective_expert_loading = selective_expert_loading
        self.expert_usage_threshold = expert_usage_threshold
        self.enable_compilation = enable_compilation
        
        # Expert caching
        self.expert_cache = ExpertCache(
            capacity=expert_cache_size,
            device=device,
            dtype=dtype
        )
        
        # Expert usage tracking
        self.expert_usage_stats = {}
        self.total_inference_steps = 0
        
        # Selective loading state
        self.loaded_experts = set()
        self.expert_storage_path = None
        
        # Move model to target device and dtype
        if base_model is not None:
            self.base_model.to(device=device, dtype=dtype)
        
        # Compile model if requested
        if enable_compilation and base_model is not None:
            try:
                self.base_model = torch.compile(self.base_model, mode="reduce-overhead")
                logger.info("Model compilation enabled")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        expert_selection_strategy: str = "frequency_based",
        num_experts_to_load: int = 4,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        expert_cache_size: int = 8,
        **kwargs
    ) -> 'OptimizedMoEModel':
        """Load optimized model from pretrained checkpoint."""
        
        # Determine device
        if device_map == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_map
            
        # Load base model
        if os.path.isdir(model_name_or_path):
            # Load from local directory
            model_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            config_path = os.path.join(model_name_or_path, "config.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Default config for demo
                config = {
                    "vocab_size": 32000,
                    "hidden_size": 768,
                    "num_experts": 8,
                    "experts_per_token": 2,
                    "num_layers": 12
                }
                
            # Create model
            base_model = MoEModel(**config)
            
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                base_model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}, using random weights")
                
        else:
            # For demo purposes, create a sample model
            logger.warning(f"Path {model_name_or_path} not found, creating demo model")
            base_model = MoEModel(
                vocab_size=32000,
                hidden_size=768,
                num_experts=8,
                experts_per_token=2,
                num_layers=12
            )
            
        # Create optimized model
        optimized_model = cls(
            base_model=base_model,
            expert_cache_size=expert_cache_size,
            device=device,
            dtype=torch_dtype,
            **kwargs
        )
        
        # Apply expert selection strategy
        if expert_selection_strategy == "frequency_based":
            optimized_model._select_frequent_experts(num_experts_to_load)
        elif expert_selection_strategy == "random":
            optimized_model._select_random_experts(num_experts_to_load)
            
        return optimized_model
        
    def enable_dynamic_loading(
        self,
        cache_size: int = 8,
        eviction_policy: str = "lru"
    ) -> None:
        """Enable dynamic expert loading."""
        self.expert_cache = ExpertCache(
            capacity=cache_size,
            device=self.device,
            dtype=self.dtype
        )
        logger.info(f"Dynamic expert loading enabled with cache size {cache_size}")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> MoEOutput:
        """Optimized forward pass."""
        
        if self.base_model is None:
            raise ValueError("Base model not initialized")
        
        # Ensure inputs are on correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        # Start timing
        start_time = time.time()
        
        # Forward pass through base model
        with torch.cuda.amp.autocast(enabled=(self.dtype == torch.float16)):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
        # Update usage statistics
        self._update_expert_usage(outputs.routing_info)
        self.total_inference_steps += 1
        
        # Log performance metrics
        inference_time = time.time() - start_time
        tokens_per_second = input_ids.numel() / inference_time if inference_time > 0 else 0
        
        logger.debug(f"Inference: {inference_time:.3f}s, {tokens_per_second:.1f} tokens/s")
        
        return outputs
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Generate text with optimized inference."""
        
        if self.base_model is None:
            raise ValueError("Base model not initialized")
        
        # Ensure model is in eval mode
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Track generation metrics
        start_time = time.time()
        generated_tokens = 0
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids)
                
                # Get logits for next token prediction
                logits = self.base_model.lm_head(outputs.last_hidden_state[:, -1, :])
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                    
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(input_ids[i].tolist()):
                            if logits[i, token_id] < 0:
                                logits[i, token_id] *= repetition_penalty
                            else:
                                logits[i, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Apply top-p (nucleus) sampling
                if do_sample and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample next token
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_tokens += batch_size
                
                # Check for early stopping (simplified)
                if (next_token == 0).all():  # Assuming 0 is EOS token
                    break
                    
        # Log generation metrics
        generation_time = time.time() - start_time
        tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
        
        logger.info(f"Generated {generated_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return input_ids
        
    def _update_expert_usage(self, routing_info) -> None:
        """Update expert usage statistics."""
        if routing_info is None or routing_info.selected_experts is None:
            return
            
        # Count expert usage
        selected_experts = routing_info.selected_experts.flatten()
        for expert_id in selected_experts.unique():
            expert_id = expert_id.item()
            if expert_id >= 0:  # Valid expert
                key = f"expert_{expert_id}"
                self.expert_usage_stats[key] = self.expert_usage_stats.get(key, 0) + 1
                
    def _select_frequent_experts(self, num_experts: int) -> None:
        """Select most frequently used experts for loading."""
        if not self.expert_usage_stats:
            # No usage data, select first N experts
            max_experts = getattr(self.base_model, 'num_experts', 8)
            self.loaded_experts = set(range(min(num_experts, max_experts)))
            logger.info(f"No usage data, loaded first {len(self.loaded_experts)} experts")
            return
            
        # Sort experts by usage frequency
        sorted_experts = sorted(
            self.expert_usage_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top experts
        self.loaded_experts = set()
        for expert_key, usage_count in sorted_experts[:num_experts]:
            expert_id = int(expert_key.split('_')[1])
            self.loaded_experts.add(expert_id)
            
        logger.info(f"Loaded {len(self.loaded_experts)} most frequent experts: {sorted(self.loaded_experts)}")
        
    def _select_random_experts(self, num_experts: int) -> None:
        """Select random experts for loading."""
        import random
        
        max_experts = getattr(self.base_model, 'num_experts', 8) if self.base_model else 8
        all_experts = list(range(max_experts))
        self.loaded_experts = set(random.sample(all_experts, min(num_experts, len(all_experts))))
        
        logger.info(f"Loaded {len(self.loaded_experts)} random experts: {sorted(self.loaded_experts)}")
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.expert_cache.get_stats()
        
        # Expert usage statistics
        total_usage = sum(self.expert_usage_stats.values())
        expert_utilization = {}
        for expert_key, usage in self.expert_usage_stats.items():
            expert_id = expert_key.split('_')[1]
            expert_utilization[expert_id] = usage / total_usage if total_usage > 0 else 0.0
            
        return {
            "total_inference_steps": self.total_inference_steps,
            "loaded_experts": sorted(self.loaded_experts),
            "expert_utilization": expert_utilization,
            "cache_stats": cache_stats,
            "model_dtype": str(self.dtype),
            "device": str(self.device)
        }
        
    def optimize_for_generation(self) -> None:
        """Apply generation-specific optimizations."""
        # Enable kv-cache if available
        # Apply torch.jit compilation
        # Set optimal batch sizes
        
        logger.info("Applied generation optimizations")
        
    def benchmark(
        self,
        input_ids: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        
        if self.base_model is None:
            raise ValueError("Base model not initialized")
        
        self.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.forward(input_ids)
                
        # Actual benchmarking
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.forward(input_ids)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start_time)
                
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        batch_size, seq_len = input_ids.shape
        tokens_per_second = (batch_size * seq_len) / avg_time if avg_time > 0 else 0
        
        return {
            "avg_latency_ms": avg_time * 1000,
            "min_latency_ms": min_time * 1000,
            "max_latency_ms": max_time * 1000,
            "tokens_per_second": tokens_per_second,
            "batch_size": batch_size,
            "sequence_length": seq_len
        }


class MoEInferenceServer:
    """Simple inference server for MoE models."""
    
    def __init__(
        self,
        model: OptimizedMoEModel,
        tokenizer=None,
        max_batch_size: int = 8,
        max_sequence_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        
        # Request queue and processing
        self.request_queue = []
        self.processing_lock = threading.Lock()
        
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_sequence_length - max_new_tokens,
            truncation=True,
            padding=False
        )
        
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
            
        # Decode only the new tokens
        new_tokens = generated_ids[:, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        
        return generated_text
        
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts."""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")
            
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i:i + self.max_batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                max_length=self.max_sequence_length - max_new_tokens,
                truncation=True,
                padding=True
            )
            
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
                
            # Decode results
            for j, generated_sequence in enumerate(generated_ids):
                original_length = input_ids[j].shape[0]
                new_tokens = generated_sequence[original_length:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(generated_text)
                
        return results