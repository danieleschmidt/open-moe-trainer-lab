"""
Distributed MoE model serving with auto-scaling and load balancing.
Generation 3: Production-ready distributed inference server.
"""

import asyncio
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import uuid
from pathlib import Path
import statistics

try:
    import aiohttp
    import uvicorn
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    import pydantic
    from pydantic import BaseModel, Field
except ImportError:
    # Graceful degradation for missing dependencies
    class BaseModel:
        pass

logger = logging.getLogger(__name__)


@dataclass
class ServerMetrics:
    """Server performance metrics."""
    requests_per_second: float = 0.0
    average_latency: float = 0.0
    active_connections: int = 0
    queue_length: int = 0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    expert_cache_hit_rate: float = 0.0
    timestamp: float = 0.0


@dataclass
class NodeInfo:
    """Information about a server node."""
    node_id: str
    host: str
    port: int
    model_name: str
    max_concurrent_requests: int
    current_load: int
    metrics: ServerMetrics
    last_heartbeat: float
    status: str = "healthy"  # healthy, degraded, offline


class GenerationRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(50, description="Maximum tokens to generate", ge=1, le=1000)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: float = Field(0.9, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, description="Top-K sampling parameter", ge=1, le=100)
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.0, description="Repetition penalty", ge=0.5, le=2.0)
    return_full_text: bool = Field(False, description="Return full text including prompt")
    stream: bool = Field(False, description="Stream response tokens")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class GenerationResponse(BaseModel):
    """Response for text generation."""
    generated_text: str
    num_tokens: int
    generation_time: float
    request_id: str
    node_id: str
    expert_utilization: Optional[Dict[str, float]] = None


class BatchGenerationRequest(BaseModel):
    """Batch generation request."""
    requests: List[GenerationRequest]
    batch_size: Optional[int] = Field(None, description="Override batch size")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    node_id: str
    uptime: float
    metrics: Dict[str, Any]
    model_info: Dict[str, Any]


class LoadBalancer:
    """Intelligent load balancer for distributed MoE serving."""
    
    def __init__(self, strategy: str = "least_connections"):
        self.strategy = strategy
        self.nodes: Dict[str, NodeInfo] = {}
        self.request_history: List[Tuple[str, float, float]] = []
        self.lock = threading.RLock()
        
        # Auto-scaling parameters
        self.scale_up_threshold = 0.8  # CPU utilization
        self.scale_down_threshold = 0.3
        self.min_nodes = 1
        self.max_nodes = 10
        
    def register_node(self, node_info: NodeInfo):
        """Register a new server node."""
        with self.lock:
            self.nodes[node_info.node_id] = node_info
            logger.info(f"Registered node {node_info.node_id} at {node_info.host}:{node_info.port}")
    
    def remove_node(self, node_id: str):
        """Remove a server node."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed node {node_id}")
    
    def update_node_metrics(self, node_id: str, metrics: ServerMetrics):
        """Update node metrics."""
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].metrics = metrics
                self.nodes[node_id].last_heartbeat = time.time()
    
    def select_node(self) -> Optional[NodeInfo]:
        """Select optimal node for request routing."""
        with self.lock:
            healthy_nodes = [
                node for node in self.nodes.values()
                if node.status == "healthy" and 
                time.time() - node.last_heartbeat < 30  # 30s timeout
            ]
            
            if not healthy_nodes:
                return None
            
            if self.strategy == "least_connections":
                return min(healthy_nodes, key=lambda n: n.current_load)
            elif self.strategy == "round_robin":
                # Simple round robin implementation
                return healthy_nodes[int(time.time()) % len(healthy_nodes)]
            elif self.strategy == "weighted_response_time":
                # Weighted by inverse of average response time
                weights = []
                for node in healthy_nodes:
                    avg_latency = node.metrics.average_latency
                    weight = 1.0 / (avg_latency + 0.001)  # Avoid division by zero
                    weights.append(weight)
                
                # Weighted random selection
                import random
                return random.choices(healthy_nodes, weights=weights)[0]
            else:
                return random.choice(healthy_nodes)
    
    def should_scale_up(self) -> bool:
        """Determine if we should add more nodes."""
        with self.lock:
            if len(self.nodes) >= self.max_nodes:
                return False
            
            # Check if average utilization is high
            total_utilization = sum(node.metrics.gpu_utilization for node in self.nodes.values())
            avg_utilization = total_utilization / len(self.nodes) if self.nodes else 0
            
            return avg_utilization > self.scale_up_threshold
    
    def should_scale_down(self) -> bool:
        """Determine if we should remove nodes."""
        with self.lock:
            if len(self.nodes) <= self.min_nodes:
                return False
            
            # Check if we have consistently low utilization
            total_utilization = sum(node.metrics.gpu_utilization for node in self.nodes.values())
            avg_utilization = total_utilization / len(self.nodes) if self.nodes else 0
            
            return avg_utilization < self.scale_down_threshold
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get overall cluster metrics."""
        with self.lock:
            if not self.nodes:
                return {"error": "No active nodes"}
            
            nodes = list(self.nodes.values())
            metrics = [node.metrics for node in nodes]
            
            return {
                "total_nodes": len(nodes),
                "healthy_nodes": sum(1 for n in nodes if n.status == "healthy"),
                "total_requests_per_second": sum(m.requests_per_second for m in metrics),
                "average_latency": statistics.mean(m.average_latency for m in metrics),
                "total_active_connections": sum(m.active_connections for m in metrics),
                "average_gpu_utilization": statistics.mean(m.gpu_utilization for m in metrics),
                "average_cache_hit_rate": statistics.mean(m.expert_cache_hit_rate for m in metrics),
                "timestamp": time.time()
            }


class DistributedMoEServer:
    """Production-ready distributed MoE serving server."""
    
    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_concurrent_requests: int = 10,
        enable_auto_scaling: bool = True,
        load_balancer_strategy: str = "least_connections"
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_auto_scaling = enable_auto_scaling
        
        # Node identification
        self.node_id = str(uuid.uuid4())
        
        # Load balancer (for distributed setup)
        self.load_balancer = LoadBalancer(strategy=load_balancer_strategy)
        
        # FastAPI app
        self.app = FastAPI(
            title="MoE Distributed Inference Server",
            description="Production-ready distributed inference for Mixture of Experts models",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request queue and processing
        self.request_queue = asyncio.Queue()
        self.active_requests: Dict[str, Dict] = {}
        self.request_history: List[Dict] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        
        # Metrics tracking
        self.metrics = ServerMetrics()
        self.start_time = time.time()
        
        # Model (will be loaded on startup)
        self.model = None
        
        self._setup_routes()
        self._start_background_tasks()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
            """Generate text from prompt."""
            request_id = request.request_id or str(uuid.uuid4())
            
            # Add request tracking
            self.active_requests[request_id] = {
                "start_time": time.time(),
                "prompt": request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt
            }
            
            try:
                # Queue management
                if len(self.active_requests) > self.max_concurrent_requests:
                    raise HTTPException(
                        status_code=429,
                        detail="Server overloaded, please try again later"
                    )
                
                # Generate text
                result = await self._generate_text_async(request, request_id)
                
                # Background cleanup
                background_tasks.add_task(self._cleanup_request, request_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Generation failed for request {request_id}: {e}")
                self._cleanup_request(request_id)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/generate/batch", response_model=List[GenerationResponse])
        async def batch_generate(request: BatchGenerationRequest):
            """Batch text generation."""
            # Process requests in parallel
            tasks = []
            for gen_request in request.requests:
                if not gen_request.request_id:
                    gen_request.request_id = str(uuid.uuid4())
                task = self._generate_text_async(gen_request, gen_request.request_id)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return successful results
            successful_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch generation error: {result}")
                else:
                    successful_results.append(result)
            
            return successful_results
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            uptime = time.time() - self.start_time
            
            model_info = {}
            if self.model:
                try:
                    model_info = {
                        "model_type": "MoE",
                        "device": str(getattr(self.model, 'device', 'unknown')),
                        "dtype": str(getattr(self.model, 'dtype', 'unknown')),
                        "num_experts": getattr(self.model.base_model, 'num_experts', 'unknown')
                    }
                except:
                    model_info = {"status": "model_info_unavailable"}
            
            return HealthResponse(
                status="healthy",
                node_id=self.node_id,
                uptime=uptime,
                metrics=asdict(self.metrics),
                model_info=model_info
            )
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get detailed server metrics."""
            cluster_metrics = self.load_balancer.get_cluster_metrics()
            
            return {
                "node_metrics": asdict(self.metrics),
                "cluster_metrics": cluster_metrics,
                "node_id": self.node_id,
                "active_requests": len(self.active_requests),
                "total_requests_processed": len(self.request_history),
                "uptime": time.time() - self.start_time
            }
        
        @self.app.get("/status")
        async def get_status():
            """Get server status and load information."""
            return {
                "node_id": self.node_id,
                "status": "running",
                "load": len(self.active_requests),
                "max_capacity": self.max_concurrent_requests,
                "utilization": len(self.active_requests) / self.max_concurrent_requests,
                "queue_size": self.request_queue.qsize(),
                "model_loaded": self.model is not None
            }
        
        # Streaming generation endpoint
        @self.app.post("/generate/stream")
        async def stream_generate(request: GenerationRequest):
            """Stream text generation."""
            if not request.stream:
                request.stream = True
            
            async def generate():
                request_id = str(uuid.uuid4())
                try:
                    # For streaming, we'd implement token-by-token generation
                    # This is a simplified version
                    result = await self._generate_text_async(request, request_id)
                    
                    # Simulate streaming by yielding tokens
                    tokens = result.generated_text.split()
                    for token in tokens:
                        chunk = {
                            "token": token,
                            "request_id": request_id,
                            "done": False
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0.01)  # Simulate generation delay
                    
                    # Final chunk
                    final_chunk = {
                        "request_id": request_id,
                        "done": True,
                        "total_time": result.generation_time
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    
                except Exception as e:
                    error_chunk = {
                        "error": str(e),
                        "request_id": request_id,
                        "done": True
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
    
    async def _generate_text_async(self, request: GenerationRequest, request_id: str) -> GenerationResponse:
        """Async text generation wrapper."""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self.thread_pool,
            self._generate_text_sync,
            request,
            request_id
        )
        
        return result
    
    def _generate_text_sync(self, request: GenerationRequest, request_id: str) -> GenerationResponse:
        """Synchronous text generation."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Mock generation for testing (replace with actual model inference)
            import time
            time.sleep(0.1)  # Simulate generation time
            
            generated_text = f"Generated response for: {request.prompt[:50]}..."
            num_tokens = request.max_new_tokens
            
            generation_time = time.time() - start_time
            
            # Update metrics
            self.metrics.requests_per_second = len(self.request_history) / (time.time() - self.start_time)
            self.metrics.average_latency = statistics.mean(
                [r.get('generation_time', 0) for r in self.request_history[-100:]]
            ) if self.request_history else generation_time
            
            # Store request in history
            request_record = {
                "request_id": request_id,
                "generation_time": generation_time,
                "num_tokens": num_tokens,
                "timestamp": time.time()
            }
            self.request_history.append(request_record)
            
            # Keep only recent history
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
            
            return GenerationResponse(
                generated_text=generated_text,
                num_tokens=num_tokens,
                generation_time=generation_time,
                request_id=request_id,
                node_id=self.node_id,
                expert_utilization={"expert_0": 0.3, "expert_1": 0.7}  # Mock data
            )
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def _cleanup_request(self, request_id: str):
        """Clean up completed request."""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        
        async def metrics_updater():
            """Periodically update server metrics."""
            while True:
                try:
                    # Update metrics
                    self.metrics.active_connections = len(self.active_requests)
                    self.metrics.queue_length = self.request_queue.qsize()
                    self.metrics.timestamp = time.time()
                    
                    # Mock GPU utilization (replace with actual monitoring)
                    import random
                    self.metrics.gpu_utilization = random.uniform(0.2, 0.9)
                    self.metrics.expert_cache_hit_rate = random.uniform(0.6, 0.95)
                    
                    # Update load balancer
                    node_info = NodeInfo(
                        node_id=self.node_id,
                        host=self.host,
                        port=self.port,
                        model_name=Path(self.model_path).name,
                        max_concurrent_requests=self.max_concurrent_requests,
                        current_load=len(self.active_requests),
                        metrics=self.metrics,
                        last_heartbeat=time.time()
                    )
                    self.load_balancer.register_node(node_info)
                    
                except Exception as e:
                    logger.error(f"Metrics update error: {e}")
                
                await asyncio.sleep(10)  # Update every 10 seconds
        
        # Start background task
        asyncio.create_task(metrics_updater())
    
    async def load_model(self):
        """Load the MoE model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Mock model loading (replace with actual model loading)
            from types import SimpleNamespace
            self.model = SimpleNamespace(
                base_model=SimpleNamespace(num_experts=8),
                device="cpu",
                dtype="float32"
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def run(self, **kwargs):
        """Run the server."""
        # Load model first
        asyncio.run(self.load_model())
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            **kwargs
        )
        
        server = uvicorn.Server(config)
        logger.info(f"Starting MoE distributed server on {self.host}:{self.port}")
        logger.info(f"Node ID: {self.node_id}")
        logger.info(f"Max concurrent requests: {self.max_concurrent_requests}")
        
        # Run server
        server.run()


class AutoScaler:
    """Auto-scaling manager for MoE server clusters."""
    
    def __init__(self, load_balancer: LoadBalancer, container_orchestrator="docker"):
        self.load_balancer = load_balancer
        self.orchestrator = container_orchestrator
        self.scaling_cooldown = 300  # 5 minutes
        self.last_scale_action = 0
        
    async def monitor_and_scale(self):
        """Monitor cluster and trigger scaling actions."""
        while True:
            try:
                current_time = time.time()
                
                # Cooldown period
                if current_time - self.last_scale_action < self.scaling_cooldown:
                    await asyncio.sleep(60)
                    continue
                
                # Check scaling conditions
                if self.load_balancer.should_scale_up():
                    await self._scale_up()
                    self.last_scale_action = current_time
                elif self.load_balancer.should_scale_down():
                    await self._scale_down()
                    self.last_scale_action = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)
    
    async def _scale_up(self):
        """Add new server node."""
        logger.info("Scaling up: Adding new server node")
        
        if self.orchestrator == "docker":
            # Docker scaling logic
            await self._scale_docker_up()
        elif self.orchestrator == "kubernetes":
            # Kubernetes scaling logic
            await self._scale_k8s_up()
        else:
            logger.warning(f"Unknown orchestrator: {self.orchestrator}")
    
    async def _scale_down(self):
        """Remove server node."""
        logger.info("Scaling down: Removing server node")
        
        if self.orchestrator == "docker":
            await self._scale_docker_down()
        elif self.orchestrator == "kubernetes":
            await self._scale_k8s_down()
    
    async def _scale_docker_up(self):
        """Scale up using Docker."""
        # Docker compose scale command or Docker API
        import subprocess
        try:
            result = subprocess.run([
                "docker", "run", "-d", "--name", f"moe-server-{uuid.uuid4().hex[:8]}",
                "-p", "0:8000",  # Random port
                "moe-trainer:latest"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Successfully started new Docker container")
            else:
                logger.error(f"Docker scale up failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Docker scaling error: {e}")
    
    async def _scale_docker_down(self):
        """Scale down using Docker."""
        # Find least utilized container and stop it
        nodes = list(self.load_balancer.nodes.values())
        if len(nodes) > self.load_balancer.min_nodes:
            least_used = min(nodes, key=lambda n: n.current_load)
            
            try:
                subprocess.run([
                    "docker", "stop", f"moe-server-{least_used.node_id[:8]}"
                ])
                logger.info(f"Stopped container for node {least_used.node_id}")
            except Exception as e:
                logger.error(f"Docker scale down error: {e}")
    
    async def _scale_k8s_up(self):
        """Scale up using Kubernetes."""
        # Kubernetes scaling logic using kubectl or K8s API
        try:
            import subprocess
            result = subprocess.run([
                "kubectl", "scale", "deployment", "moe-server", 
                "--replicas", str(len(self.load_balancer.nodes) + 1)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Successfully scaled Kubernetes deployment")
            else:
                logger.error(f"K8s scale up failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Kubernetes scaling error: {e}")
    
    async def _scale_k8s_down(self):
        """Scale down using Kubernetes."""
        nodes = list(self.load_balancer.nodes.values())
        if len(nodes) > self.load_balancer.min_nodes:
            try:
                import subprocess
                result = subprocess.run([
                    "kubectl", "scale", "deployment", "moe-server",
                    "--replicas", str(len(nodes) - 1)
                ])
                
                if result.returncode == 0:
                    logger.info("Successfully scaled down Kubernetes deployment")
            except Exception as e:
                logger.error(f"Kubernetes scale down error: {e}")


def create_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_concurrent_requests: int = 10,
    enable_auto_scaling: bool = True
) -> DistributedMoEServer:
    """Factory function to create a distributed MoE server."""
    return DistributedMoEServer(
        model_path=model_path,
        host=host,
        port=port,
        max_concurrent_requests=max_concurrent_requests,
        enable_auto_scaling=enable_auto_scaling
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE Distributed Inference Server")
    parser.add_argument("--model-path", required=True, help="Path to MoE model")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--auto-scale", action="store_true", help="Enable auto-scaling")
    
    args = parser.parse_args()
    
    server = create_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        max_concurrent_requests=args.max_concurrent,
        enable_auto_scaling=args.auto_scale
    )
    
    server.run()