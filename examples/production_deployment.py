#!/usr/bin/env python3
"""
Production Deployment Example

This example demonstrates a complete production-ready deployment of a MoE model
with all advanced features: compilation optimization, distributed training,
monitoring, error handling, and serving infrastructure.
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

# Import MoE Lab components
from moe_lab import MoEModel, MoETrainer
from moe_lab.data import TextDataset, MoEDataCollator
from moe_lab.optimization import MoEModelCompiler, CompilationConfig, create_optimized_model
from moe_lab.serving import MoEInferenceServer, ServerConfig, BatchingConfig
from moe_lab.distributed import DistributedMoETrainer
from moe_lab.utils import (
    setup_logging, setup_monitoring, setup_error_handling,
    validate_and_suggest, get_metrics_collector, get_global_error_handler,
    CheckpointManager, install_exception_handler
)

console = Console()


class ProductionPipeline:
    """Complete production pipeline for MoE models."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.model = None
        self.optimized_model = None
        self.checkpoint_manager = None
        self.metrics_collector = None
        
        # Setup infrastructure
        self._setup_infrastructure()
        
        # Load and validate configuration
        self._load_configuration()
    
    def _setup_infrastructure(self):
        """Setup logging, monitoring, and error handling."""
        console.print("🔧 Setting up production infrastructure...", style="cyan")
        
        # Setup logging
        setup_logging(level="INFO")
        
        # Setup error handling with file logging
        setup_error_handling(log_file="production_errors.log", max_retries=3)
        install_exception_handler()
        
        # Setup monitoring
        self.metrics_collector = setup_monitoring(
            max_history=50000,
            collection_interval=1.0,
            auto_start=True
        )
        
        console.print("✅ Infrastructure setup completed", style="green")
    
    def _load_configuration(self):
        """Load and validate configuration."""
        console.print(f"📋 Loading configuration from {self.config_path}...", style="cyan")
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    import yaml
                    self.config = yaml.safe_load(f)
            
            # Validate configuration
            is_valid, suggested_config = validate_and_suggest(self.config)
            
            if not is_valid:
                console.print("❌ Configuration validation failed", style="red")
                sys.exit(1)
            
            console.print("✅ Configuration loaded and validated", style="green")
            
        except Exception as e:
            console.print(f"❌ Failed to load configuration: {e}", style="red")
            sys.exit(1)
    
    def train_model(self):
        """Train MoE model with production-grade features."""
        console.print(Panel("🚀 Starting Production Training", style="bold blue"))
        
        # Setup checkpoint manager
        output_dir = Path(self.config.get('training', {}).get('output_dir', './outputs'))
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(output_dir / 'checkpoints'),
            max_checkpoints=5
        )
        
        # Initialize model
        model_config = self.config.get('model', {})
        self.model = MoEModel(**model_config)
        
        # Setup distributed training if configured
        distributed_config = self.config.get('distributed', {})
        if distributed_config.get('enabled', False):
            self._setup_distributed_training()
        
        # Prepare data
        data_config = self.config.get('data', {})
        train_dataset, eval_dataset = self._prepare_datasets(data_config)
        
        # Initialize trainer
        training_config = self.config.get('training', {})
        if distributed_config.get('enabled', False):
            trainer = DistributedMoETrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=MoEDataCollator(),
                **training_config
            )
        else:
            trainer = MoETrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=MoEDataCollator(),
                **training_config
            )
        
        # Training with comprehensive monitoring
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training MoE model...", total=None)
            
            try:
                # Training loop with checkpointing
                result = trainer.train()
                
                # Save final model
                final_model_path = output_dir / "final_model"
                trainer.save_model(str(final_model_path))
                
                # Save training metadata
                metadata = {
                    'training_result': result.to_dict() if hasattr(result, 'to_dict') else str(result),
                    'model_config': model_config,
                    'training_config': training_config,
                    'final_loss': getattr(result, 'final_loss', None),
                    'training_time': getattr(result, 'total_time', None)
                }
                
                with open(output_dir / "training_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                console.print(Panel(
                    "✅ Training completed successfully!\n"
                    f"📁 Model saved to: {final_model_path}\n"
                    f"📊 Final loss: {getattr(result, 'final_loss', 'N/A')}\n"
                    f"⏱️  Training time: {getattr(result, 'total_time', 'N/A')}s",
                    style="bold green"
                ))
                
                return str(final_model_path)
                
            except Exception as e:
                console.print(f"❌ Training failed: {e}", style="red")
                
                # Save checkpoint on failure
                try:
                    failure_checkpoint = output_dir / "failure_checkpoint.pt"
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'error': str(e),
                        'timestamp': time.time()
                    }, failure_checkpoint)
                    console.print(f"💾 Failure checkpoint saved to: {failure_checkpoint}")
                except Exception as save_error:
                    console.print(f"❌ Failed to save failure checkpoint: {save_error}")
                
                raise
    
    def _setup_distributed_training(self):
        """Setup distributed training environment."""
        if not dist.is_initialized():
            # Initialize distributed training
            dist.init_process_group(backend='nccl')
            console.print("🌐 Distributed training initialized", style="green")
    
    def _prepare_datasets(self, data_config: Dict[str, Any]):
        """Prepare training and evaluation datasets."""
        # Mock dataset creation for example
        # In production, load from actual data files
        
        mock_texts = [
            "The field of machine learning continues to advance rapidly.",
            "Mixture of Experts models provide efficient scaling.",
            "Deep learning architectures enable complex pattern recognition.",
            "Natural language processing transforms how we interact with computers.",
            "Artificial intelligence systems require careful ethical consideration."
        ] * 1000  # Scale up for realistic training
        
        # Split into train/eval
        split_idx = int(0.8 * len(mock_texts))
        train_texts = mock_texts[:split_idx]
        eval_texts = mock_texts[split_idx:]
        
        train_dataset = TextDataset(
            texts=train_texts,
            max_length=data_config.get('max_seq_length', 512)
        )
        
        eval_dataset = TextDataset(
            texts=eval_texts,
            max_length=data_config.get('max_seq_length', 512)
        )
        
        return train_dataset, eval_dataset
    
    def optimize_model(self, model_path: str) -> str:
        """Optimize trained model for production inference."""
        console.print(Panel("⚡ Optimizing Model for Production", style="bold yellow"))
        
        # Load trained model
        self.model = MoEModel.from_pretrained(model_path)
        
        # Get optimization configuration
        optimization_config = self.config.get('optimization', {})
        optimization_level = optimization_config.get('level', 'balanced')
        
        # Create example inputs for compilation
        example_inputs = {
            'input_ids': torch.randint(0, 32000, (4, 256))
        }
        
        if torch.cuda.is_available():
            example_inputs = {k: v.cuda() for k, v in example_inputs.items()}
            self.model = self.model.cuda()
        
        # Optimize model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Optimizing model...", total=None)
            
            self.optimized_model = create_optimized_model(
                self.model,
                optimization_level=optimization_level,
                example_inputs=example_inputs
            )
        
        # Benchmark optimization
        if optimization_config.get('benchmark', True):
            self._benchmark_optimization(example_inputs)
        
        # Export optimized model
        optimized_path = Path(model_path).parent / "optimized_model"
        
        # Setup compiler with custom config
        compiler_config = CompilationConfig(
            backend=optimization_config.get('backend', 'inductor'),
            mode=optimization_config.get('mode', 'default'),
            optimize_routing=True,
            fuse_expert_ops=True
        )
        
        compiler = MoEModelCompiler(compiler_config)
        compiler.export_optimized_model(
            self.optimized_model,
            str(optimized_path),
            example_inputs
        )
        
        console.print(Panel(
            f"✅ Model optimization completed!\n"
            f"📁 Optimized model saved to: {optimized_path}",
            style="bold green"
        ))
        
        return str(optimized_path)
    
    def _benchmark_optimization(self, example_inputs: Dict[str, torch.Tensor]):
        """Benchmark optimization improvements."""
        console.print("📊 Benchmarking optimization improvements...")
        
        # Benchmark configuration
        num_runs = 50
        
        # Benchmark original model
        self.model.eval()
        original_times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = self.model(**example_inputs)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(**example_inputs)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                original_times.append(time.time() - start_time)
        
        # Benchmark optimized model
        self.optimized_model.eval()
        optimized_times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = self.optimized_model(**example_inputs)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.optimized_model(**example_inputs)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                optimized_times.append(time.time() - start_time)
        
        # Calculate statistics
        import numpy as np
        original_avg = np.mean(original_times) * 1000  # ms
        optimized_avg = np.mean(optimized_times) * 1000  # ms
        speedup = original_avg / optimized_avg
        
        # Display results
        table = Table(title="Optimization Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Original", style="magenta")
        table.add_column("Optimized", style="green")
        table.add_column("Improvement", style="yellow")
        
        table.add_row("Avg Latency (ms)", f"{original_avg:.2f}", f"{optimized_avg:.2f}", f"{speedup:.2f}x")
        table.add_row("Min Latency (ms)", f"{min(original_times)*1000:.2f}", f"{min(optimized_times)*1000:.2f}", "")
        table.add_row("Max Latency (ms)", f"{max(original_times)*1000:.2f}", f"{max(optimized_times)*1000:.2f}", "")
        
        console.print(table)
    
    def deploy_server(self, model_path: str):
        """Deploy production inference server."""
        console.print(Panel("🚀 Deploying Production Server", style="bold green"))
        
        # Server configuration
        server_config_dict = self.config.get('serving', {})
        server_config = ServerConfig(
            host=server_config_dict.get('host', '0.0.0.0'),
            port=server_config_dict.get('port', 8000),
            workers=server_config_dict.get('workers', 1),
            max_concurrent_requests=server_config_dict.get('max_concurrent_requests', 100),
            enable_metrics=server_config_dict.get('enable_metrics', True),
            enable_caching=server_config_dict.get('enable_caching', True)
        )
        
        # Batching configuration
        batching_config_dict = server_config_dict.get('batching', {})
        batching_config = BatchingConfig(
            max_batch_size=batching_config_dict.get('max_batch_size', 32),
            batch_timeout_ms=batching_config_dict.get('batch_timeout_ms', 10),
            enable_sequence_bucketing=batching_config_dict.get('enable_sequence_bucketing', True)
        )
        
        # Create and run server
        server = MoEInferenceServer(
            model_path=model_path,
            server_config=server_config,
            batching_config=batching_config
        )
        
        console.print(f"🎯 Starting server on {server_config.host}:{server_config.port}")
        console.print("🔍 Available endpoints:")
        console.print("  • POST /generate - Generate text")
        console.print("  • GET /health - Health check")
        console.print("  • GET /stats - Server statistics")
        console.print("  • GET /model/info - Model information")
        
        # Run server (this will block)
        server.run()
    
    def run_end_to_end_demo(self):
        """Run complete end-to-end production pipeline."""
        console.print(Panel("🎯 Starting End-to-End Production Pipeline", style="bold blue"))
        
        try:
            # Step 1: Train model
            model_path = self.train_model()
            
            # Step 2: Optimize model
            optimized_path = self.optimize_model(model_path)
            
            # Step 3: Generate deployment report
            self._generate_deployment_report(model_path, optimized_path)
            
            # Step 4: Optional server deployment
            deploy_server = input("\n🚀 Deploy inference server? (y/n): ").lower() == 'y'
            if deploy_server:
                self.deploy_server(optimized_path)
            else:
                console.print("✅ Pipeline completed successfully!", style="green")
                console.print(f"📁 Optimized model ready for deployment: {optimized_path}")
            
        except KeyboardInterrupt:
            console.print("\n⚠️  Pipeline interrupted by user", style="yellow")
        except Exception as e:
            console.print(f"\n❌ Pipeline failed: {e}", style="red")
            
            # Generate error report
            error_handler = get_global_error_handler()
            error_summary = error_handler.get_error_summary(time_window=3600)
            
            console.print("\n📊 Error Summary:")
            for error_type, count in error_summary.get('error_types', {}).items():
                console.print(f"  • {error_type}: {count} occurrences")
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _generate_deployment_report(self, model_path: str, optimized_path: str):
        """Generate comprehensive deployment report."""
        console.print("📋 Generating deployment report...")
        
        # Collect metrics
        system_stats = self.metrics_collector.get_summary_stats('system', 3600)
        training_stats = self.metrics_collector.get_summary_stats('training', 3600)
        
        # Model information
        model_info = {
            'original_model_path': model_path,
            'optimized_model_path': optimized_path,
            'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'model_size_mb': sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024) if self.model else 0
        }
        
        # Error summary
        error_handler = get_global_error_handler()
        error_summary = error_handler.get_error_summary(time_window=3600)
        
        # Create comprehensive report
        report = {
            'timestamp': time.time(),
            'pipeline_config': self.config,
            'model_info': model_info,
            'system_performance': system_stats,
            'training_performance': training_stats,
            'error_summary': error_summary,
            'deployment_ready': True,
            'recommendations': self._generate_recommendations(system_stats, training_stats)
        }
        
        # Save report
        report_path = Path(optimized_path).parent / "deployment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"📊 Deployment report saved to: {report_path}")
    
    def _generate_recommendations(self, system_stats: Dict, training_stats: Dict) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []
        
        if system_stats and system_stats.get('max_memory_percent', 0) > 85:
            recommendations.append("Consider increasing server memory or reducing batch size")
        
        if system_stats and system_stats.get('avg_gpu_utilization', 0) < 70:
            recommendations.append("GPU utilization is low - consider increasing batch size")
        
        if training_stats and training_stats.get('avg_expert_load_variance', 0) > 0.5:
            recommendations.append("Expert load balancing could be improved")
        
        if not recommendations:
            recommendations.append("Model is ready for production deployment")
        
        return recommendations
    
    def _cleanup(self):
        """Cleanup resources."""
        if self.metrics_collector:
            from moe_lab.utils.monitoring import cleanup_monitoring
            cleanup_monitoring()


async def test_deployed_server(server_url: str = "http://localhost:8000"):
    """Test deployed server with sample requests."""
    import aiohttp
    
    console.print(f"🧪 Testing deployed server at {server_url}")
    
    test_requests = [
        {
            "prompt": "The future of artificial intelligence",
            "max_new_tokens": 50,
            "temperature": 0.8
        },
        {
            "prompt": "Machine learning has revolutionized",
            "max_new_tokens": 30,
            "temperature": 0.7
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, request in enumerate(test_requests):
            try:
                async with session.post(f"{server_url}/generate", json=request) as response:
                    if response.status == 200:
                        result = await response.json()
                        console.print(f"✅ Test {i+1} successful:")
                        console.print(f"   Generated: {result.get('generated_text', 'N/A')}")
                        console.print(f"   Tokens/sec: {result.get('tokens_per_second', 'N/A'):.2f}")
                    else:
                        console.print(f"❌ Test {i+1} failed: {response.status}")
            except Exception as e:
                console.print(f"❌ Test {i+1} error: {e}")


def main():
    """Main function to run production pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE Production Deployment Pipeline")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--mode", choices=["train", "optimize", "serve", "full"], 
                       default="full", help="Pipeline mode")
    parser.add_argument("--model-path", help="Path to trained model (for optimize/serve modes)")
    parser.add_argument("--test-server", action="store_true", help="Test deployed server")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionPipeline(args.config)
    
    if args.mode == "full":
        pipeline.run_end_to_end_demo()
    elif args.mode == "train":
        model_path = pipeline.train_model()
        console.print(f"✅ Training completed. Model saved to: {model_path}")
    elif args.mode == "optimize":
        if not args.model_path:
            console.print("❌ --model-path required for optimize mode", style="red")
            sys.exit(1)
        optimized_path = pipeline.optimize_model(args.model_path)
        console.print(f"✅ Optimization completed. Model saved to: {optimized_path}")
    elif args.mode == "serve":
        if not args.model_path:
            console.print("❌ --model-path required for serve mode", style="red")
            sys.exit(1)
        pipeline.deploy_server(args.model_path)
    
    if args.test_server:
        asyncio.run(test_deployed_server())


if __name__ == "__main__":
    main()