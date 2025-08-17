#!/usr/bin/env python3
"""
GENERATION 2: MAKE IT ROBUST (Reliable)
Autonomous SDLC - Robust MoE with error handling, monitoring, security, validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import logging
import hashlib
import warnings
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import traceback

from moe_lab import MoEModel

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation2_robust.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RobustConfig:
    """Robust configuration with validation."""
    vocab_size: int = 1000
    hidden_size: int = 256
    num_experts: int = 4
    experts_per_token: int = 2
    num_layers: int = 4
    num_attention_heads: int = 4
    max_position_embeddings: int = 512
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    checkpoint_every: int = 5
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.experts_per_token <= 0 or self.experts_per_token > self.num_experts:
            raise ValueError(f"experts_per_token must be in range [1, {self.num_experts}], got {self.experts_per_token}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

class SecurityManager:
    """Handle security concerns for MoE training."""
    
    @staticmethod
    def sanitize_input(data: torch.Tensor) -> torch.Tensor:
        """Sanitize input data for security."""
        # Check for NaN or infinite values
        if torch.isnan(data).any() or torch.isinf(data).any():
            logger.warning("Found NaN or infinite values in input, replacing with zeros")
            data = torch.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clamp values to reasonable range
        data = torch.clamp(data, min=-1e8, max=1e8)
        return data
    
    @staticmethod
    def validate_model_state(model: torch.nn.Module) -> bool:
        """Validate model state for security issues."""
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.error(f"Invalid gradients found in {name}")
                        return False
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"Invalid parameters found in {name}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error validating model state: {e}")
            return False

class AdvancedMonitor:
    """Advanced monitoring for MoE training."""
    
    def __init__(self):
        self.metrics = {
            'training_losses': [],
            'routing_entropy': [],
            'expert_utilization': [],
            'gradient_norms': [],
            'memory_usage': [],
            'execution_times': [],
            'error_counts': 0,
            'warning_counts': 0
        }
        
    def log_training_step(self, step: int, loss: float, model: torch.nn.Module, 
                         routing_info: Any = None) -> None:
        """Log comprehensive training metrics."""
        try:
            self.metrics['training_losses'].append({
                'step': step,
                'loss': float(loss),
                'timestamp': time.time()
            })
            
            # Monitor gradient norms
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.metrics['gradient_norms'].append(grad_norm)
            
            # Monitor memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                self.metrics['memory_usage'].append(memory_used)
            
            # Monitor routing statistics
            if routing_info is not None:
                self.metrics['routing_entropy'].append(float(routing_info.entropy))
                
        except Exception as e:
            logger.warning(f"Error logging training step: {e}")
            self.metrics['warning_counts'] += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        if not self.metrics['training_losses']:
            return {"status": "no_data", "health": "unknown"}
        
        recent_losses = [m['loss'] for m in self.metrics['training_losses'][-10:]]
        avg_recent_loss = np.mean(recent_losses)
        loss_trend = "stable"
        
        if len(recent_losses) > 5:
            first_half = np.mean(recent_losses[:len(recent_losses)//2])
            second_half = np.mean(recent_losses[len(recent_losses)//2:])
            if second_half < first_half * 0.95:
                loss_trend = "improving"
            elif second_half > first_half * 1.05:
                loss_trend = "degrading"
        
        return {
            "status": "healthy" if avg_recent_loss < 20.0 else "warning",
            "average_recent_loss": avg_recent_loss,
            "loss_trend": loss_trend,
            "error_count": self.metrics['error_counts'],
            "warning_count": self.metrics['warning_counts'],
            "total_steps": len(self.metrics['training_losses'])
        }

class RobustTrainer:
    """Robust trainer with comprehensive error handling."""
    
    def __init__(self, config: RobustConfig):
        self.config = config
        self.monitor = AdvancedMonitor()
        self.security = SecurityManager()
        self.checkpoint_dir = Path("robust_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def create_robust_model(self) -> torch.nn.Module:
        """Create model with robust initialization."""
        try:
            logger.info("Creating robust MoE model...")
            model = MoEModel(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_experts=self.config.num_experts,
                experts_per_token=self.config.experts_per_token,
                num_layers=self.config.num_layers,
                num_attention_heads=self.config.num_attention_heads,
                max_position_embeddings=self.config.max_position_embeddings,
                moe_layers=[1, 3] if self.config.num_layers >= 4 else [1]
            )
            
            # Initialize with Xavier uniform for better stability
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
            
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model created with {total_params:,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def create_robust_dataset(self, num_samples: int = 128) -> torch.Tensor:
        """Create robust synthetic dataset with validation."""
        try:
            logger.info(f"Creating robust dataset with {num_samples} samples...")
            
            # Create more realistic text-like sequences
            data = []
            vocab_size = self.config.vocab_size
            
            for i in range(num_samples):
                # Create sequences with some structure (beginning, middle, end tokens)
                seq_len = np.random.randint(32, 65)  # Variable length
                
                # Start token
                sequence = [vocab_size - 1]  # Special start token
                
                # Middle tokens with some patterns
                for _ in range(seq_len - 2):
                    if np.random.random() < 0.1:  # 10% special tokens
                        token = np.random.randint(vocab_size - 10, vocab_size)
                    else:
                        token = np.random.randint(0, vocab_size - 10)
                    sequence.append(token)
                
                # End token  
                sequence.append(vocab_size - 2)  # Special end token
                
                # Pad to max length
                max_len = 64
                if len(sequence) < max_len:
                    sequence.extend([0] * (max_len - len(sequence)))  # Pad with 0
                else:
                    sequence = sequence[:max_len]
                
                data.append(sequence)
            
            dataset = torch.tensor(data, dtype=torch.long)
            dataset = self.security.sanitize_input(dataset.float()).long()
            
            logger.info(f"Dataset created: {dataset.shape}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    def robust_training_loop(self, model: torch.nn.Module, data: torch.Tensor) -> List[float]:
        """Robust training loop with comprehensive error handling."""
        try:
            logger.info("Starting robust training...")
            
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=0.01,
                eps=1e-8
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.config.warmup_steps, T_mult=2
            )
            
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
            
            model.train()
            losses = []
            global_step = 0
            
            for epoch in range(self.config.num_epochs):
                epoch_losses = []
                logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Shuffle data
                indices = torch.randperm(len(data))
                data_shuffled = data[indices]
                
                for i in range(0, len(data_shuffled), self.config.batch_size):
                    try:
                        batch = data_shuffled[i:i+self.config.batch_size]
                        
                        # Validate batch
                        if batch.size(0) == 0:
                            continue
                        
                        # Create targets (next token prediction)
                        input_ids = batch[:, :-1]
                        targets = batch[:, 1:]
                        
                        # Security validation
                        input_ids = self.security.sanitize_input(input_ids.float()).long()
                        targets = self.security.sanitize_input(targets.float()).long()
                        
                        optimizer.zero_grad()
                        
                        # Forward pass with error handling
                        try:
                            outputs = model(input_ids)
                        except RuntimeError as e:
                            logger.warning(f"Forward pass error: {e}")
                            self.monitor.metrics['error_counts'] += 1
                            continue
                        
                        # Get logits for language modeling
                        logits = model.lm_head(outputs.last_hidden_state)
                        
                        # Compute loss
                        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                        
                        # Add MoE losses with proper weighting
                        if outputs.load_balancing_loss is not None:
                            loss += 0.01 * outputs.load_balancing_loss
                        
                        # Validate loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning("Invalid loss detected, skipping step")
                            self.monitor.metrics['error_counts'] += 1
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        
                        # Validate model state before optimizer step
                        if not self.security.validate_model_state(model):
                            logger.warning("Invalid model state detected, skipping optimizer step")
                            self.monitor.metrics['error_counts'] += 1
                            continue
                        
                        optimizer.step()
                        scheduler.step()
                        
                        epoch_losses.append(loss.item())
                        global_step += 1
                        
                        # Log metrics
                        self.monitor.log_training_step(global_step, loss.item(), model, outputs.routing_info)
                        
                        # Checkpoint saving
                        if global_step % self.config.checkpoint_every == 0:
                            self.save_checkpoint(model, optimizer, epoch, global_step, loss.item())
                        
                    except Exception as e:
                        logger.error(f"Error in training step: {e}")
                        logger.error(traceback.format_exc())
                        self.monitor.metrics['error_counts'] += 1
                        continue
                
                if epoch_losses:
                    avg_loss = np.mean(epoch_losses)
                    losses.append(avg_loss)
                    logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
                    
                    # Health check
                    health = self.monitor.get_health_status()
                    logger.info(f"Health status: {health['status']}")
                    
                    if health['status'] == "warning":
                        logger.warning("Training health degraded, consider stopping")
                else:
                    logger.warning(f"No valid losses in epoch {epoch + 1}")
                    losses.append(float('inf'))
            
            logger.info("Robust training completed")
            return losses
            
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, step: int, loss: float) -> None:
        """Save training checkpoint with metadata."""
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'config': asdict(self.config),
                'metrics': self.monitor.metrics,
                'timestamp': time.time()
            }
            
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save metrics separately as JSON
            metrics_path = self.checkpoint_dir / f"checkpoint_step_{step}.json"
            with open(metrics_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_metrics = {}
                for key, value in self.monitor.metrics.items():
                    if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                        serializable_metrics[key] = [v.tolist() for v in value]
                    elif isinstance(value, np.ndarray):
                        serializable_metrics[key] = value.tolist()
                    else:
                        serializable_metrics[key] = value
                json.dump(serializable_metrics, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

def run_generation2_robust():
    """Run Generation 2 robust MoE demonstration."""
    logger.info("🚀 Starting Generation 2: MAKE IT ROBUST (Reliable)")
    logger.info("=" * 70)
    
    start_time = time.time()
    results = {}
    
    try:
        # 1. Create robust configuration
        logger.info("1. Creating robust configuration...")
        config = RobustConfig(
            vocab_size=1000,
            hidden_size=256,
            num_experts=6,
            experts_per_token=2,
            num_layers=6,
            learning_rate=5e-4,
            batch_size=8,
            num_epochs=5
        )
        logger.info("   ✅ Configuration validated")
        
        # 2. Initialize robust trainer
        logger.info("2. Initializing robust trainer...")
        trainer = RobustTrainer(config)
        logger.info("   ✅ Trainer initialized with monitoring and security")
        
        # 3. Create robust model
        logger.info("3. Creating robust model...")
        model = trainer.create_robust_model()
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   ✅ Robust model created with {total_params:,} parameters")
        
        # 4. Create robust dataset
        logger.info("4. Creating robust dataset...")
        dataset = trainer.create_robust_dataset(num_samples=128)
        logger.info(f"   ✅ Robust dataset created: {dataset.shape}")
        
        # 5. Robust training with monitoring
        logger.info("5. Running robust training with monitoring...")
        losses = trainer.robust_training_loop(model, dataset)
        logger.info(f"   ✅ Robust training complete. Final loss: {losses[-1]:.4f}")
        
        # 6. Comprehensive validation
        logger.info("6. Running comprehensive validation...")
        health_status = trainer.monitor.get_health_status()
        model_valid = trainer.security.validate_model_state(model)
        logger.info(f"   ✅ Health status: {health_status['status']}")
        logger.info(f"   ✅ Model validation: {'PASSED' if model_valid else 'FAILED'}")
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        results = {
            "generation": 2,
            "status": "ROBUST" if model_valid and health_status['status'] != "warning" else "DEGRADED",
            "model_params": total_params,
            "training_epochs": config.num_epochs,
            "final_loss": losses[-1] if losses else float('inf'),
            "health_status": health_status,
            "model_validation": model_valid,
            "error_count": trainer.monitor.metrics['error_counts'],
            "warning_count": trainer.monitor.metrics['warning_counts'],
            "execution_time_seconds": elapsed_time,
            "features_implemented": [
                "Robust configuration validation",
                "Comprehensive error handling", 
                "Advanced monitoring system",
                "Security validation",
                "Gradient clipping and stabilization",
                "Checkpoint saving with metadata",
                "Health status monitoring",
                "Learning rate scheduling",
                "Input sanitization",
                "Model state validation"
            ]
        }
        
        # Save comprehensive results
        with open("generation2_robust_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n🎉 Generation 2 Complete: ROBUST FUNCTIONALITY IMPLEMENTED")
        logger.info(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
        logger.info(f"🛡️  Security validation: {'PASSED' if model_valid else 'FAILED'}")
        logger.info(f"📊 Health status: {health_status['status']}")
        logger.info("📊 Results saved to: generation2_robust_results.json")
        logger.info("🚀 Proceeding to Generation 3: MAKE IT SCALE...")
        
        return results
        
    except Exception as e:
        logger.error(f"Generation 2 failed: {e}")
        logger.error(traceback.format_exc())
        
        results = {
            "generation": 2,
            "status": "FAILED",
            "error": str(e),
            "execution_time_seconds": time.time() - start_time
        }
        
        with open("generation2_robust_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        raise

if __name__ == "__main__":
    # Run autonomous Generation 2 robust demo
    results = run_generation2_robust()
    
    # Validate success
    assert results["status"] in ["ROBUST", "DEGRADED"], f"Generation 2 failed: {results.get('error', 'Unknown error')}"
    assert results["final_loss"] < 15.0, "Training loss too high"
    assert results["model_validation"], "Model validation failed"
    
    logger.info("✅ Generation 2 validation passed - Ready for Generation 3")