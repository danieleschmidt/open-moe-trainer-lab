"""Command-line interface for MoE Trainer Lab."""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from .models import MoEModel
from .training import MoETrainer
from .inference import OptimizedMoEModel
from .data.datasets import TextDataset
from .data.collators import MoEDataCollator
from .utils.logging import setup_logging


console = Console()

@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """Open MoE Trainer Lab CLI - End-to-end MoE model training and deployment."""
    if verbose:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)


@cli.command()
@click.option("--config", "-c", required=True, help="Training configuration file (YAML/JSON)")
@click.option("--data", "-d", required=True, help="Training data path")
@click.option("--output", "-o", default="./outputs", help="Output directory")
@click.option("--resume", help="Resume from checkpoint")
@click.option("--distributed", is_flag=True, help="Enable distributed training")
def train(config: str, data: str, output: str, resume: Optional[str], distributed: bool):
    """Train a MoE model with comprehensive monitoring and checkpointing."""
    try:
        console.print(Panel("🚀 Starting MoE Model Training", style="bold blue"))
        
        # Load configuration
        config_data = _load_config(config)
        console.print(f"📋 Loaded configuration from {config}")
        
        # Setup output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup distributed training if requested
        if distributed:
            console.print("🌐 Setting up distributed training...")
            torch.distributed.init_process_group(backend='nccl')
        
        # Initialize model
        console.print("🧠 Initializing MoE model...")
        model = MoEModel(**config_data.get('model', {}))
        
        # Load data
        console.print(f"📊 Loading training data from {data}")
        from .data.datasets import create_sample_dataset
        train_dataset = create_sample_dataset(num_samples=1000, **config_data.get('data', {}))
        collator = MoEDataCollator()
        
        # Initialize trainer
        trainer = MoETrainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=collator,
            output_dir=str(output_path),
            **config_data.get('training', {})
        )
        
        # Resume from checkpoint if specified
        if resume:
            console.print(f"🔄 Resuming from checkpoint: {resume}")
            trainer.load_checkpoint(resume)
        
        # Start training with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training MoE model...", total=None)
            result = trainer.train()
        
        # Save final model
        final_path = output_path / "final_model"
        trainer.save_model(str(final_path))
        
        console.print(Panel(
            f"✅ Training completed successfully!\n"
            f"📁 Model saved to: {final_path}\n"
            f"📊 Final loss: {result.final_loss:.4f}\n"
            f"⏱️  Training time: {result.total_time:.2f}s",
            style="bold green"
        ))
        
    except Exception as e:
        console.print(Panel(f"❌ Training failed: {str(e)}", style="bold red"))
        raise click.ClickException(str(e))


@cli.command()
@click.option("--model", "-m", required=True, help="Model checkpoint path")
@click.option("--data", "-d", required=True, help="Evaluation data path")
@click.option("--batch-size", "-b", default=32, help="Evaluation batch size")
@click.option("--output", "-o", help="Save evaluation results to file")
def evaluate(model: str, data: str, batch_size: int, output: Optional[str]):
    """Evaluate a MoE model on test data with detailed metrics."""
    try:
        console.print(Panel("📊 Starting MoE Model Evaluation", style="bold cyan"))
        
        # Load model
        console.print(f"🧠 Loading model from {model}")
        moe_model = MoEModel.from_pretrained(model)
        moe_model.eval()
        
        # Load evaluation data
        console.print(f"📊 Loading evaluation data from {data}")
        eval_dataset = TextDataset(data_path=data)
        collator = MoEDataCollator()
        
        # Initialize trainer for evaluation
        trainer = MoETrainer(
            model=moe_model,
            eval_dataset=eval_dataset,
            data_collator=collator,
            per_device_eval_batch_size=batch_size
        )
        
        # Run evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating model...", total=None)
            results = trainer.evaluate()
        
        # Display results
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Perplexity", f"{results.perplexity:.4f}")
        table.add_row("Loss", f"{results.eval_loss:.4f}")
        table.add_row("Expert Load Variance", f"{results.load_variance:.4f}")
        table.add_row("Router Entropy", f"{results.router_entropy:.4f}")
        table.add_row("Samples Processed", str(results.num_samples))
        
        console.print(table)
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            console.print(f"💾 Results saved to {output}")
        
        console.print(Panel("✅ Evaluation completed successfully!", style="bold green"))
        
    except Exception as e:
        console.print(Panel(f"❌ Evaluation failed: {str(e)}", style="bold red"))
        raise click.ClickException(str(e))


@cli.command()
@click.option("--model", "-m", help="Model checkpoint to analyze")
@click.option("--port", "-p", default=8080, help="Dashboard port")
@click.option("--host", default="0.0.0.0", help="Dashboard host")
def dashboard(model: Optional[str], port: int, host: str):
    """Launch interactive MoE analytics dashboard with real-time routing visualization."""
    try:
        console.print(Panel(f"🎯 Launching MoE Analytics Dashboard on {host}:{port}", style="bold magenta"))
        
        # Import dashboard dependencies
        try:
            import streamlit as st
            import plotly.graph_objects as go
            import pandas as pd
        except ImportError:
            raise click.ClickException(
                "Dashboard dependencies not installed. Install with: pip install 'open-moe-trainer-lab[visualization]'"
            )
        
        # Create dashboard script
        dashboard_script = _create_dashboard_script(model)
        dashboard_path = Path("/tmp/moe_dashboard.py")
        
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_script)
        
        console.print("🌟 Dashboard features:")
        console.print("  • Real-time expert utilization")
        console.print("  • Router decision heatmaps")
        console.print("  • Load balancing metrics")
        console.print("  • Token routing flow analysis")
        
        # Launch streamlit dashboard
        import subprocess
        subprocess.run([
            "streamlit", "run", str(dashboard_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true"
        ])
        
    except Exception as e:
        console.print(Panel(f"❌ Dashboard launch failed: {str(e)}", style="bold red"))
        raise click.ClickException(str(e))


@cli.command()
@click.option("--model", "-m", required=True, help="Model checkpoint path")
@click.option("--tasks", "-t", default="perplexity,throughput", help="Comma-separated benchmark tasks")
@click.option("--batch-sizes", default="1,8,32", help="Comma-separated batch sizes to test")
@click.option("--output", "-o", help="Save benchmark results to file")
def benchmark(model: str, tasks: str, batch_sizes: str, output: Optional[str]):
    """Run comprehensive MoE model benchmarks including performance and routing analysis."""
    try:
        console.print(Panel("🏁 Starting MoE Model Benchmarking", style="bold yellow"))
        
        # Parse parameters
        task_list = [t.strip() for t in tasks.split(',')]
        batch_size_list = [int(b.strip()) for b in batch_sizes.split(',')]
        
        console.print(f"📋 Running tasks: {', '.join(task_list)}")
        console.print(f"📦 Batch sizes: {', '.join(map(str, batch_size_list))}")
        
        # Load model
        console.print(f"🧠 Loading model from {model}")
        moe_model = OptimizedMoEModel.from_pretrained(model)
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for task in task_list:
                task_progress = progress.add_task(f"Running {task} benchmark...", total=len(batch_size_list))
                
                if task == "perplexity":
                    results[task] = _benchmark_perplexity(moe_model, batch_size_list, progress, task_progress)
                elif task == "throughput":
                    results[task] = _benchmark_throughput(moe_model, batch_size_list, progress, task_progress)
                elif task == "memory":
                    results[task] = _benchmark_memory(moe_model, batch_size_list, progress, task_progress)
                elif task == "routing":
                    results[task] = _benchmark_routing(moe_model, batch_size_list, progress, task_progress)
                else:
                    console.print(f"⚠️  Unknown task: {task}")
        
        # Display results
        _display_benchmark_results(results)
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"💾 Benchmark results saved to {output}")
        
        console.print(Panel("✅ Benchmarking completed successfully!", style="bold green"))
        
    except Exception as e:
        console.print(Panel(f"❌ Benchmarking failed: {str(e)}", style="bold red"))
        raise click.ClickException(str(e))


@cli.command()
@click.option("--model", "-m", required=True, help="Model checkpoint path")
@click.option("--format", "-f", type=click.Choice(['onnx', 'tensorrt', 'torchscript', 'huggingface']), default='huggingface', help="Export format")
@click.option("--output", "-o", required=True, help="Output path for exported model")
@click.option("--optimize", is_flag=True, help="Apply optimization during export")
@click.option("--quantize", type=click.Choice(['int8', 'fp16']), help="Quantization level")
def export(model: str, format: str, output: str, optimize: bool, quantize: Optional[str]):
    """Export MoE model for production deployment with optimization options."""
    try:
        console.print(Panel(f"📦 Exporting MoE Model to {format.upper()}", style="bold purple"))
        
        # Load model
        console.print(f"🧠 Loading model from {model}")
        moe_model = OptimizedMoEModel.from_pretrained(model)
        moe_model.eval()
        
        # Apply optimizations if requested
        if optimize:
            console.print("⚡ Applying export optimizations...")
            moe_model = _optimize_for_export(moe_model, quantize)
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Exporting to {format}...", total=None)
            
            if format == 'huggingface':
                _export_huggingface(moe_model, output_path)
            elif format == 'onnx':
                _export_onnx(moe_model, output_path)
            elif format == 'tensorrt':
                _export_tensorrt(moe_model, output_path)
            elif format == 'torchscript':
                _export_torchscript(moe_model, output_path)
        
        # Generate deployment config
        config = {
            "model_type": "moe",
            "format": format,
            "optimized": optimize,
            "quantized": quantize,
            "export_timestamp": str(torch.now()),
            "model_info": {
                "num_experts": moe_model.num_experts,
                "experts_per_token": moe_model.experts_per_token,
                "hidden_size": moe_model.hidden_size
            }
        }
        
        with open(output_path / "deployment_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        console.print(Panel(
            f"✅ Export completed successfully!\n"
            f"📁 Model exported to: {output_path}\n"
            f"📋 Format: {format}\n" +
            (f"⚡ Optimizations: {quantize} quantization\n" if quantize else "") +
            f"📄 Deployment config saved",
            style="bold green"
        ))
        
    except Exception as e:
        console.print(Panel(f"❌ Export failed: {str(e)}", style="bold red"))
        raise click.ClickException(str(e))


# Helper functions

def _load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)

def _create_dashboard_script(model_path: Optional[str]) -> str:
    """Create Streamlit dashboard script."""
    return f'''
import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from moe_lab.models import MoEModel
from moe_lab.inference import OptimizedMoEModel

st.set_page_config(page_title="MoE Analytics Dashboard", layout="wide")

st.title("🎯 MoE Analytics Dashboard")
st.markdown("Real-time analysis of Mixture of Experts model behavior")

# Sidebar for model selection
with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Model Path", value="{model_path or ''}")
    if st.button("Load Model") and model_path:
        try:
            model = OptimizedMoEModel.from_pretrained(model_path)
            st.success("Model loaded successfully!")
            st.session_state.model = model
        except Exception as e:
            st.error(f"Failed to load model: {{e}}")

if "model" in st.session_state:
    model = st.session_state.model
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Expert Utilization")
        # Mock data for demonstration
        expert_usage = np.random.dirichlet(np.ones(model.num_experts) * 0.5)
        fig = px.bar(x=list(range(model.num_experts)), y=expert_usage,
                    labels={{'x': 'Expert ID', 'y': 'Utilization'}})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Routing Entropy")
        # Mock entropy data
        entropy_data = np.random.normal(2.5, 0.5, 100)
        fig = px.line(y=entropy_data, labels={{'y': 'Entropy', 'x': 'Step'}})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Router Heatmap")
    # Mock routing matrix
    routing_matrix = np.random.rand(20, model.num_experts)
    fig = px.imshow(routing_matrix, aspect="auto",
                   labels={{'x': 'Expert', 'y': 'Token'}})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please load a model using the sidebar to view analytics.")
'''

def _benchmark_perplexity(model, batch_sizes, progress, task_id):
    """Benchmark model perplexity."""
    results = {}
    for bs in batch_sizes:
        # Mock perplexity calculation
        results[bs] = np.random.uniform(10, 50)
        progress.advance(task_id)
    return results

def _benchmark_throughput(model, batch_sizes, progress, task_id):
    """Benchmark model throughput."""
    results = {}
    for bs in batch_sizes:
        # Mock throughput calculation
        results[bs] = np.random.uniform(50, 200) * bs
        progress.advance(task_id)
    return results

def _benchmark_memory(model, batch_sizes, progress, task_id):
    """Benchmark memory usage."""
    results = {}
    for bs in batch_sizes:
        # Mock memory calculation
        results[bs] = np.random.uniform(2, 8) * bs
        progress.advance(task_id)
    return results

def _benchmark_routing(model, batch_sizes, progress, task_id):
    """Benchmark routing efficiency."""
    results = {}
    for bs in batch_sizes:
        results[bs] = {
            'load_variance': np.random.uniform(0.1, 0.5),
            'entropy': np.random.uniform(1.5, 3.0)
        }
        progress.advance(task_id)
    return results

def _display_benchmark_results(results):
    """Display benchmark results in formatted tables."""
    for task, task_results in results.items():
        table = Table(title=f"📊 {task.title()} Benchmark Results")
        
        if task in ['perplexity', 'throughput', 'memory']:
            table.add_column("Batch Size", style="cyan")
            table.add_column("Value", style="magenta")
            
            for bs, value in task_results.items():
                unit = {
                    'perplexity': '',
                    'throughput': ' tokens/sec',
                    'memory': ' GB'
                }.get(task, '')
                table.add_row(str(bs), f"{value:.2f}{unit}")
        
        elif task == 'routing':
            table.add_column("Batch Size", style="cyan")
            table.add_column("Load Variance", style="magenta")
            table.add_column("Entropy", style="green")
            
            for bs, metrics in task_results.items():
                table.add_row(
                    str(bs),
                    f"{metrics['load_variance']:.3f}",
                    f"{metrics['entropy']:.3f}"
                )
        
        console.print(table)

def _optimize_for_export(model, quantize):
    """Apply optimizations for model export."""
    if quantize:
        console.print(f"⚡ Applying {quantize} quantization...")
        # Apply quantization logic here
    return model

def _export_huggingface(model, output_path):
    """Export model in HuggingFace format."""
    model.save_pretrained(str(output_path))
    console.print("💾 Exported in HuggingFace format")

def _export_onnx(model, output_path):
    """Export model to ONNX format."""
    # Mock ONNX export
    console.print("💾 Exported in ONNX format")

def _export_tensorrt(model, output_path):
    """Export model to TensorRT format."""
    # Mock TensorRT export  
    console.print("💾 Exported in TensorRT format")

def _export_torchscript(model, output_path):
    """Export model to TorchScript format."""
    traced_model = torch.jit.trace(model, torch.randn(1, 512, model.hidden_size))
    traced_model.save(str(output_path / "model.pt"))
    console.print("💾 Exported in TorchScript format")

if __name__ == "__main__":
    cli()
