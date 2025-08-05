"""High-performance MoE inference server with dynamic batching and load balancing."""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import uuid

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..models import MoEModel
from ..inference import OptimizedMoEModel
from ..inference.caching import ExpertCache
from ..utils.monitoring import get_metrics_collector, MetricsCollector
from ..utils.error_handling import with_error_handling, InferenceError

logger = logging.getLogger(__name__)


@dataclass
class BatchingConfig:
    """Configuration for dynamic batching."""
    max_batch_size: int = 32
    batch_timeout_ms: int = 10
    max_queue_size: int = 1000
    padding_strategy: str = "left"  # "left", "right", "dynamic"
    enable_sequence_bucketing: bool = True
    bucket_sizes: List[int] = None
    
    def __post_init__(self):
        if self.bucket_sizes is None:
            self.bucket_sizes = [128, 256, 512, 1024, 2048]


@dataclass 
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30
    health_check_interval: int = 30
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_size_gb: float = 4.0


class GenerationRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(50, ge=1, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(False, description="Enable streaming response")
    request_id: Optional[str] = Field(None, description="Optional request ID")


class GenerationResponse(BaseModel):
    """Response for text generation."""
    generated_text: str
    prompt: str
    request_id: str
    generation_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    expert_utilization: Optional[Dict[str, Any]] = None
    model_stats: Optional[Dict[str, Any]] = None


class BatchProcessor:
    """Processes batched inference requests efficiently."""
    
    def __init__(self, model: nn.Module, batching_config: BatchingConfig, 
                 tokenizer, metrics_collector: Optional[MetricsCollector] = None):
        self.model = model
        self.config = batching_config
        self.tokenizer = tokenizer
        self.metrics_collector = metrics_collector
        
        # Request queues for different sequence lengths
        self.request_queues = {}
        for bucket_size in self.config.bucket_sizes:
            self.request_queues[bucket_size] = queue.Queue(maxsize=self.config.max_queue_size)
        
        # Processing state
        self.processing = True
        self.processor_thread = None
        
        # Performance tracking
        self.batch_times = []
        self.throughput_history = []
        
        logger.info(f"Initialized BatchProcessor with {len(self.request_queues)} buckets")
    
    def start(self):
        """Start batch processing."""
        self.processing = True
        self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processor_thread.start()
        logger.info("Started batch processor")
    
    def stop(self):
        """Stop batch processing."""
        self.processing = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        logger.info("Stopped batch processor")
    
    def submit_request(self, request: GenerationRequest, response_future: asyncio.Future):
        """Submit request for batched processing."""
        # Tokenize to determine sequence length bucket
        prompt_tokens = self.tokenizer.encode(request.prompt)
        seq_length = len(prompt_tokens) + request.max_new_tokens
        
        # Find appropriate bucket
        bucket_size = self._find_bucket(seq_length)
        if bucket_size is None:
            response_future.set_exception(
                InferenceError(f"Sequence length {seq_length} exceeds maximum bucket size")
            )
            return
        
        # Add to queue
        request_item = {
            'request': request,
            'response_future': response_future,
            'prompt_tokens': prompt_tokens,
            'submit_time': time.time()
        }
        
        try:
            self.request_queues[bucket_size].put_nowait(request_item)
        except queue.Full:
            response_future.set_exception(
                InferenceError("Request queue is full, please try again later")
            )
    
    def _find_bucket(self, seq_length: int) -> Optional[int]:
        """Find appropriate sequence length bucket."""
        for bucket_size in sorted(self.config.bucket_sizes):
            if seq_length <= bucket_size:
                return bucket_size
        return None
    
    def _process_loop(self):
        """Main processing loop."""
        while self.processing:
            try:
                # Process each bucket
                for bucket_size in self.config.bucket_sizes:
                    self._process_bucket(bucket_size)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                time.sleep(0.1)
    
    def _process_bucket(self, bucket_size: int):
        """Process requests in a specific bucket."""
        request_queue = self.request_queues[bucket_size]
        
        if request_queue.empty():
            return
        
        # Collect requests for batching
        batch_requests = []
        batch_start_time = time.time()
        
        # Collect up to max_batch_size requests or wait for timeout
        while (len(batch_requests) < self.config.max_batch_size and
               (time.time() - batch_start_time) * 1000 < self.config.batch_timeout_ms):
            
            try:
                request_item = request_queue.get_nowait()
                batch_requests.append(request_item)
            except queue.Empty:
                if batch_requests:
                    break  # Process what we have
                time.sleep(0.001)  # Short wait
                continue
        
        if not batch_requests:
            return
        
        # Process batch
        try:
            self._process_batch(batch_requests, bucket_size)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all requests in batch
            for request_item in batch_requests:
                if not request_item['response_future'].done():
                    request_item['response_future'].set_exception(e)
    
    @with_error_handling(InferenceError, recovery_suggestion="Check model and input validity")
    def _process_batch(self, batch_requests: List[Dict], bucket_size: int):
        """Process a batch of requests."""
        start_time = time.time()
        
        # Prepare batch inputs
        prompts = [item['request'].prompt for item in batch_requests]
        prompt_tokens_list = [item['prompt_tokens'] for item in batch_requests]
        
        # Pad sequences to bucket size
        max_input_length = max(len(tokens) for tokens in prompt_tokens_list)
        padded_inputs = []
        
        for tokens in prompt_tokens_list:
            if self.config.padding_strategy == "left":
                padded = [self.tokenizer.pad_token_id] * (max_input_length - len(tokens)) + tokens
            else:  # right padding
                padded = tokens + [self.tokenizer.pad_token_id] * (max_input_length - len(tokens))
            padded_inputs.append(padded)
        
        # Convert to tensor
        input_ids = torch.tensor(padded_inputs, dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Generate
        generation_config = {
            'max_new_tokens': batch_requests[0]['request'].max_new_tokens,
            'temperature': batch_requests[0]['request'].temperature,
            'top_p': batch_requests[0]['request'].top_p,
            'top_k': batch_requests[0]['request'].top_k,
            'do_sample': batch_requests[0]['request'].do_sample,
            'repetition_penalty': batch_requests[0]['request'].repetition_penalty
        }
        
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                outputs = self.model.generate(input_ids, **generation_config)
            else:
                # Fallback for models without generate method
                outputs = self._simple_generate(input_ids, **generation_config)
        
        generation_time = time.time() - start_time
        
        # Process outputs
        for i, (request_item, output_ids) in enumerate(zip(batch_requests, outputs)):
            try:
                # Decode generated text
                input_length = len(prompt_tokens_list[i])
                generated_tokens = output_ids[input_length:].cpu().tolist()
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Create response
                response = GenerationResponse(
                    generated_text=generated_text,
                    prompt=request_item['request'].prompt,
                    request_id=request_item['request'].request_id or str(uuid.uuid4()),
                    generation_time_ms=generation_time * 1000,
                    tokens_generated=len(generated_tokens),
                    tokens_per_second=len(generated_tokens) / generation_time if generation_time > 0 else 0
                )
                
                # Set response
                if not request_item['response_future'].done():
                    request_item['response_future'].set_result(response)
                    
            except Exception as e:
                logger.error(f"Error processing individual response: {e}")
                if not request_item['response_future'].done():
                    request_item['response_future'].set_exception(e)
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.record_model_metrics(
                forward_time_ms=generation_time * 1000,
                num_parameters=sum(p.numel() for p in self.model.parameters()),
                model_size_mb=sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)
            )
        
        # Track performance
        self.batch_times.append(generation_time)
        batch_throughput = len(batch_requests) / generation_time
        self.throughput_history.append(batch_throughput)
        
        # Keep limited history
        if len(self.batch_times) > 1000:
            self.batch_times = self.batch_times[-500:]
        if len(self.throughput_history) > 1000:
            self.throughput_history = self.throughput_history[-500:]
        
        logger.debug(f"Processed batch of {len(batch_requests)} requests in {generation_time:.3f}s")
    
    def _simple_generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                        temperature: float = 0.8, **kwargs) -> torch.Tensor:
        """Simple generation fallback."""
        batch_size, seq_len = input_ids.shape
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                if hasattr(outputs, 'last_hidden_state'):
                    logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
                else:
                    logits = outputs[:, -1, :] if len(outputs.shape) == 3 else outputs
            
            # Sample next token
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.batch_times:
            return {}
        
        import numpy as np
        return {
            'average_batch_time_ms': np.mean(self.batch_times) * 1000,
            'average_throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'total_batches_processed': len(self.batch_times),
            'queue_sizes': {size: q.qsize() for size, q in self.request_queues.items()}
        }


class MoEInferenceServer:
    """High-performance MoE inference server with advanced features."""
    
    def __init__(self, model_path: str, server_config: Optional[ServerConfig] = None,
                 batching_config: Optional[BatchingConfig] = None):
        self.model_path = model_path
        self.server_config = server_config or ServerConfig()
        self.batching_config = batching_config or BatchingConfig()
        
        # Initialize components
        self.app = FastAPI(title="MoE Inference Server", version="1.0.0")
        self.model = None
        self.tokenizer = None
        self.batch_processor = None
        self.metrics_collector = None
        
        # Server state
        self.server_start_time = None
        self.total_requests = 0
        self.active_requests = 0
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Initialized MoEInferenceServer for model: {model_path}")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            await self._load_model()
            await self._start_services()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self._cleanup()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "uptime_seconds": time.time() - self.server_start_time if self.server_start_time else 0,
                "active_requests": self.active_requests,
                "total_requests": self.total_requests
            }
        
        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            stats = {
                "server": {
                    "uptime_seconds": time.time() - self.server_start_time if self.server_start_time else 0,
                    "total_requests": self.total_requests,
                    "active_requests": self.active_requests,
                    "requests_per_second": self._calculate_rps()
                },
                "model": {
                    "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                    "size_mb": sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024) if self.model else 0
                }
            }
            
            if self.batch_processor:
                stats["batch_processor"] = self.batch_processor.get_stats()
            
            if self.metrics_collector:
                stats["system"] = self.metrics_collector.get_summary_stats("system", 300)
            
            return stats
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
            """Generate text from prompt."""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Increment counters
            self.total_requests += 1
            self.active_requests += 1
            
            try:
                # Create response future
                response_future = asyncio.Future()
                
                # Submit to batch processor
                self.batch_processor.submit_request(request, response_future)
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(
                        response_future, 
                        timeout=self.server_config.request_timeout_seconds
                    )
                    return response
                    
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=408, detail="Request timeout")
                
            finally:
                self.active_requests -= 1
        
        @self.app.get("/model/info")
        async def model_info():
            """Get model information."""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            info = {
                "model_path": self.model_path,
                "model_type": type(self.model).__name__,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "size_mb": sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)
            }
            
            # Add MoE-specific info
            if hasattr(self.model, 'num_experts'):
                info["num_experts"] = self.model.num_experts
                info["experts_per_token"] = getattr(self.model, 'experts_per_token', None)
                info["moe_layers"] = getattr(self.model, 'moe_layers', None)
            
            return info
    
    async def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load model
            if Path(self.model_path).is_dir():
                # Load from directory
                self.model = OptimizedMoEModel.from_pretrained(self.model_path)
            else:
                # Load from checkpoint
                checkpoint = torch.load(self.model_path, map_location='cpu')
                # Model loading logic here
                pass
            
            # Setup model for inference
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            # Setup expert caching if enabled
            if self.server_config.enable_caching:
                from ..inference.caching import create_expert_cache
                cache = create_expert_cache(
                    cache_type="adaptive",
                    capacity_gb=self.server_config.cache_size_gb
                )
                if hasattr(self.model, 'set_expert_cache'):
                    self.model.set_expert_cache(cache)
            
            # Mock tokenizer (in real implementation, load appropriate tokenizer)
            class MockTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                
                def encode(self, text):
                    return [hash(char) % 32000 for char in text[:100]]
                
                def decode(self, token_ids, skip_special_tokens=True):
                    return ''.join([chr(65 + (id % 26)) for id in token_ids])
            
            self.tokenizer = MockTokenizer()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def _start_services(self):
        """Start server services."""
        # Setup metrics collection
        if self.server_config.enable_metrics:
            from ..utils.monitoring import setup_monitoring
            self.metrics_collector = setup_monitoring(auto_start=True)
        
        # Start batch processor
        self.batch_processor = BatchProcessor(
            self.model, 
            self.batching_config, 
            self.tokenizer,
            self.metrics_collector
        )
        self.batch_processor.start()
        
        # Set server start time
        self.server_start_time = time.time()
        
        logger.info("Server services started")
    
    async def _cleanup(self):
        """Cleanup server resources."""
        if self.batch_processor:
            self.batch_processor.stop()
        
        if self.metrics_collector:
            from ..utils.monitoring import cleanup_monitoring
            cleanup_monitoring()
        
        logger.info("Server cleanup completed")
    
    def _calculate_rps(self) -> float:
        """Calculate requests per second."""
        if not self.server_start_time:
            return 0.0
        
        uptime = time.time() - self.server_start_time
        return self.total_requests / uptime if uptime > 0 else 0.0
    
    def run(self):
        """Run the server."""
        logger.info(f"Starting MoE Inference Server on {self.server_config.host}:{self.server_config.port}")
        
        uvicorn.run(
            self.app,
            host=self.server_config.host,
            port=self.server_config.port,
            workers=self.server_config.workers,
            log_level="info"
        )


def create_server(model_path: str, **kwargs) -> MoEInferenceServer:
    """Factory function to create MoE inference server."""
    return MoEInferenceServer(model_path, **kwargs)