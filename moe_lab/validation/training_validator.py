"""Training validation utilities for MoE models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time

from ..models import MoEModel
from ..training import MoETrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingValidationResult:
    """Result of training validation."""
    
    is_valid: bool
    convergence_score: float
    stability_score: float
    efficiency_score: float
    errors: List[str]
    warnings: List[str]
    training_metrics: Dict[str, Any]


class TrainingValidator:
    """Validator for MoE training process and convergence."""
    
    def __init__(self, model: MoEModel, trainer: Optional[MoETrainer] = None):
        self.model = model
        self.trainer = trainer
        self.device = next(model.parameters()).device
        
    def validate_training_setup(self) -> TrainingValidationResult:
        """Validate training configuration and setup."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Validate model for training
            model_errors, model_warnings, model_metrics = self._validate_model_for_training()
            errors.extend(model_errors)
            warnings.extend(model_warnings)
            metrics.update(model_metrics)
            
            # Validate optimizer configuration
            if self.trainer:
                opt_errors, opt_warnings, opt_metrics = self._validate_optimizer_config()
                errors.extend(opt_errors)
                warnings.extend(opt_warnings)
                metrics.update(opt_metrics)
                
            # Test gradient computation
            grad_errors, grad_warnings, grad_metrics = self._validate_gradient_computation()
            errors.extend(grad_errors)
            warnings.extend(grad_warnings)
            metrics.update(grad_metrics)
            
            # Test loss computation
            loss_errors, loss_warnings, loss_metrics = self._validate_loss_computation()
            errors.extend(loss_errors)
            warnings.extend(loss_warnings)
            metrics.update(loss_metrics)
            
            # Test memory usage
            mem_errors, mem_warnings, mem_metrics = self._validate_memory_usage()
            errors.extend(mem_errors)
            warnings.extend(mem_warnings)
            metrics.update(mem_metrics)
            
        except Exception as e:
            errors.append(f"Training setup validation failed: {str(e)}")
            
        is_valid = len(errors) == 0
        
        return TrainingValidationResult(
            is_valid=is_valid,
            convergence_score=0.0,  # Not applicable for setup validation
            stability_score=0.0,
            efficiency_score=0.0,
            errors=errors,
            warnings=warnings,
            training_metrics=metrics
        )
        
    def validate_training_convergence(
        self,
        num_steps: int = 50,
        batch_size: int = 4,
        learning_rate: float = 1e-4
    ) -> TrainingValidationResult:
        """Validate training convergence behavior."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Create synthetic training data
            train_data = self._create_synthetic_dataset(num_samples=num_steps * batch_size)
            
            # Setup simple trainer if not provided
            if self.trainer is None:
                trainer = MoETrainer(
                    model=self.model,
                    logging_steps=10,
                    output_dir="./test_checkpoints"
                )
            else:
                trainer = self.trainer
                
            # Monitor training metrics
            initial_loss = None
            loss_history = []
            aux_loss_history = []
            gradient_norms = []
            
            # Training loop with monitoring
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            self.model.train()
            
            for step in range(num_steps):
                optimizer.zero_grad()
                
                # Get batch
                batch_data = []
                for i in range(batch_size):
                    idx = (step * batch_size + i) % len(train_data)
                    batch_data.append(train_data[idx])
                    
                # Convert to tensor
                input_ids = torch.stack([item['input_ids'].squeeze() for item in batch_data]).to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, return_routing_info=True)
                
                # Compute loss
                logits = self.model.lm_head(outputs.last_hidden_state)
                targets = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1].contiguous()
                
                lm_loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                aux_loss = outputs.load_balancing_loss if outputs.load_balancing_loss is not None else 0.0
                total_loss = lm_loss + 0.01 * aux_loss if isinstance(aux_loss, torch.Tensor) else lm_loss
                
                # Backward pass
                total_loss.backward()
                
                # Compute gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                gradient_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                
                optimizer.step()
                
                # Record metrics
                if initial_loss is None:
                    initial_loss = lm_loss.item()
                    
                loss_history.append(lm_loss.item())
                aux_loss_history.append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0)
                
            # Analyze convergence
            convergence_score, conv_errors, conv_warnings, conv_metrics = self._analyze_convergence(
                loss_history, aux_loss_history, gradient_norms, initial_loss
            )
            errors.extend(conv_errors)
            warnings.extend(conv_warnings)
            metrics.update(conv_metrics)
            
            # Analyze stability
            stability_score, stab_errors, stab_warnings, stab_metrics = self._analyze_stability(
                loss_history, gradient_norms
            )
            errors.extend(stab_errors)
            warnings.extend(stab_warnings)
            metrics.update(stab_metrics)
            
            # Analyze efficiency
            efficiency_score, eff_errors, eff_warnings, eff_metrics = self._analyze_efficiency(
                loss_history, gradient_norms, num_steps
            )
            errors.extend(eff_errors)
            warnings.extend(eff_warnings)
            metrics.update(eff_metrics)
            
        except Exception as e:
            errors.append(f"Training convergence validation failed: {str(e)}")
            convergence_score = stability_score = efficiency_score = 0.0
            
        is_valid = len(errors) == 0
        
        return TrainingValidationResult(
            is_valid=is_valid,
            convergence_score=convergence_score,
            stability_score=stability_score,
            efficiency_score=efficiency_score,
            errors=errors,
            warnings=warnings,
            training_metrics=metrics
        )
        
    def _validate_model_for_training(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate model readiness for training."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check model mode
        if not self.model.training:
            warnings.append("Model is in eval mode, switching to train mode")
            self.model.train()
            
        # Check parameter requires_grad
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
                
        if trainable_params == 0:
            errors.append("No trainable parameters found")
        elif frozen_params > trainable_params:
            warnings.append(f"More frozen ({frozen_params}) than trainable ({trainable_params}) parameters")
            
        metrics.update({
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'total_parameters': trainable_params + frozen_params
        })
        
        # Check device consistency
        device_types = set()
        for param in self.model.parameters():
            device_types.add(str(param.device))
            
        if len(device_types) > 1:
            errors.append(f"Model parameters on multiple devices: {device_types}")
            
        return errors, warnings, metrics
        
    def _validate_optimizer_config(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate optimizer configuration."""
        errors = []
        warnings = []
        metrics = {}
        
        # This would need access to trainer's optimizer
        # For now, just validate trainer configuration
        if hasattr(self.trainer, 'aux_loss_coef'):
            if self.trainer.aux_loss_coef <= 0:
                warnings.append("Auxiliary loss coefficient is zero or negative")
            elif self.trainer.aux_loss_coef > 0.1:
                warnings.append(f"High auxiliary loss coefficient: {self.trainer.aux_loss_coef}")
                
        if hasattr(self.trainer, 'router_z_loss_coef'):
            if self.trainer.router_z_loss_coef <= 0:
                warnings.append("Router z-loss coefficient is zero or negative")
                
        if hasattr(self.trainer, 'max_grad_norm'):
            if self.trainer.max_grad_norm <= 0:
                warnings.append("Gradient clipping disabled (max_grad_norm <= 0)")
            elif self.trainer.max_grad_norm > 10.0:
                warnings.append(f"Very high gradient clipping threshold: {self.trainer.max_grad_norm}")
                
        return errors, warnings, metrics
        
    def _validate_gradient_computation(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate gradient computation."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Create test input
            test_input = torch.randint(0, 1000, (2, 16), device=self.device)
            
            self.model.train()
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(test_input)
            
            # Compute test loss
            logits = self.model.lm_head(outputs.last_hidden_state)
            targets = test_input[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad_count = 0
            total_grad_norm = 0.0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        errors.append(f"No gradient computed for {name}")
                    else:
                        grad_count += 1
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm ** 2
                        
                        if torch.isnan(param.grad).any():
                            errors.append(f"NaN gradients in {name}")
                        if torch.isinf(param.grad).any():
                            errors.append(f"Inf gradients in {name}")
                            
            total_grad_norm = total_grad_norm ** 0.5
            
            metrics.update({
                'parameters_with_gradients': grad_count,
                'total_gradient_norm': total_grad_norm
            })
            
            if total_grad_norm == 0:
                errors.append("All gradients are zero")
            elif total_grad_norm > 100:
                warnings.append(f"Very large gradient norm: {total_grad_norm:.2f}")
                
        except Exception as e:
            errors.append(f"Gradient validation failed: {str(e)}")
            
        return errors, warnings, metrics
        
    def _validate_loss_computation(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate loss computation."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Test different input sizes
            test_inputs = [
                torch.randint(0, 1000, (1, 8), device=self.device),
                torch.randint(0, 1000, (4, 16), device=self.device),
                torch.randint(0, 1000, (2, 32), device=self.device)
            ]
            
            losses = []
            aux_losses = []
            
            self.model.eval()
            
            with torch.no_grad():
                for i, test_input in enumerate(test_inputs):
                    try:
                        outputs = self.model(test_input, return_routing_info=True)
                        
                        # Compute language modeling loss
                        logits = self.model.lm_head(outputs.last_hidden_state)
                        targets = test_input[:, 1:].contiguous()
                        logits = logits[:, :-1].contiguous()
                        
                        lm_loss = nn.CrossEntropyLoss()(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        )
                        
                        if torch.isnan(lm_loss):
                            errors.append(f"NaN loss for input {i}")
                        elif torch.isinf(lm_loss):
                            errors.append(f"Inf loss for input {i}")
                        else:
                            losses.append(lm_loss.item())
                            
                        # Check auxiliary losses
                        if outputs.load_balancing_loss is not None:
                            aux_loss = outputs.load_balancing_loss
                            if torch.isnan(aux_loss):
                                errors.append(f"NaN auxiliary loss for input {i}")
                            elif torch.isinf(aux_loss):
                                errors.append(f"Inf auxiliary loss for input {i}")
                            else:
                                aux_losses.append(aux_loss.item())
                                
                    except Exception as e:
                        errors.append(f"Loss computation failed for input {i}: {str(e)}")
                        
            if losses:
                metrics.update({
                    'mean_loss': np.mean(losses),
                    'std_loss': np.std(losses),
                    'min_loss': np.min(losses),
                    'max_loss': np.max(losses)
                })
                
            if aux_losses:
                metrics.update({
                    'mean_aux_loss': np.mean(aux_losses),
                    'std_aux_loss': np.std(aux_losses)
                })
                
        except Exception as e:
            errors.append(f"Loss validation failed: {str(e)}")
            
        return errors, warnings, metrics
        
    def _validate_memory_usage(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate memory usage during training."""
        errors = []
        warnings = []
        metrics = {}
        
        if not torch.cuda.is_available():
            warnings.append("CUDA not available, skipping memory validation")
            return errors, warnings, metrics
            
        try:
            # Clear cache and get baseline
            torch.cuda.empty_cache()
            baseline_memory = torch.cuda.memory_allocated()
            
            # Test forward pass memory
            test_input = torch.randint(0, 1000, (4, 64), device=self.device)
            
            self.model.train()
            outputs = self.model(test_input, return_routing_info=True)
            
            forward_memory = torch.cuda.memory_allocated()
            
            # Test backward pass memory
            logits = self.model.lm_head(outputs.last_hidden_state)
            targets = test_input[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            loss.backward()
            
            backward_memory = torch.cuda.memory_allocated()
            
            metrics.update({
                'baseline_memory_mb': baseline_memory / (1024 * 1024),
                'forward_memory_mb': forward_memory / (1024 * 1024),
                'backward_memory_mb': backward_memory / (1024 * 1024),
                'forward_memory_increase_mb': (forward_memory - baseline_memory) / (1024 * 1024),
                'backward_memory_increase_mb': (backward_memory - forward_memory) / (1024 * 1024)
            })
            
            # Check for memory issues
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            
            if backward_memory > total_memory * 0.9:
                warnings.append(f"High memory usage: {backward_memory / (1024**3):.1f}GB")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                errors.append("GPU out of memory during validation")
            else:
                errors.append(f"Memory validation failed: {str(e)}")
        except Exception as e:
            errors.append(f"Memory validation failed: {str(e)}")
            
        return errors, warnings, metrics
        
    def _create_synthetic_dataset(self, num_samples: int, seq_len: int = 32):
        """Create synthetic dataset for testing."""
        
        class SyntheticDataset:
            def __init__(self, num_samples, seq_len):
                self.num_samples = num_samples
                self.seq_len = seq_len
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                input_ids = torch.randint(0, 1000, (self.seq_len,))
                return {'input_ids': input_ids.unsqueeze(0)}
                
        return SyntheticDataset(num_samples, seq_len)
        
    def _analyze_convergence(
        self, 
        loss_history: List[float], 
        aux_loss_history: List[float],
        gradient_norms: List[float],
        initial_loss: float
    ) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Analyze training convergence."""
        errors = []
        warnings = []
        metrics = {}
        
        if len(loss_history) < 10:
            warnings.append("Insufficient training steps for convergence analysis")
            return 0.0, errors, warnings, metrics
            
        # Compute loss reduction
        final_loss = np.mean(loss_history[-5:])  # Average of last 5 steps
        loss_reduction = (initial_loss - final_loss) / initial_loss
        
        # Compute loss trend
        x = np.arange(len(loss_history))
        loss_slope = np.polyfit(x, loss_history, 1)[0]
        
        # Convergence score based on loss reduction and trend
        convergence_score = max(0.0, min(1.0, loss_reduction + abs(loss_slope) * 10))
        
        metrics.update({
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction': loss_reduction,
            'loss_slope': loss_slope,
            'loss_history': loss_history[-20:],  # Last 20 steps
            'aux_loss_history': aux_loss_history[-20:],
            'gradient_norms': gradient_norms[-20:]
        })
        
        # Check convergence issues
        if loss_reduction < 0.01:
            warnings.append(f"Poor convergence: loss reduction {loss_reduction:.3f}")
            
        if loss_slope > 0:
            warnings.append("Loss is increasing during training")
            
        return convergence_score, errors, warnings, metrics
        
    def _analyze_stability(
        self, 
        loss_history: List[float],
        gradient_norms: List[float]
    ) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Analyze training stability."""
        errors = []
        warnings = []
        metrics = {}
        
        if len(loss_history) < 5:
            return 0.0, errors, warnings, metrics
            
        # Loss variance (lower is more stable)
        loss_variance = np.var(loss_history)
        loss_cv = np.std(loss_history) / (np.mean(loss_history) + 1e-8)
        
        # Gradient norm variance
        grad_variance = np.var(gradient_norms)
        grad_cv = np.std(gradient_norms) / (np.mean(gradient_norms) + 1e-8)
        
        # Stability score (higher is more stable)
        stability_score = 1.0 / (1.0 + loss_cv + grad_cv)
        
        metrics.update({
            'loss_variance': loss_variance,
            'loss_cv': loss_cv,
            'gradient_variance': grad_variance,
            'gradient_cv': grad_cv
        })
        
        # Check stability issues
        if loss_cv > 0.5:
            warnings.append(f"High loss variance (CV: {loss_cv:.3f})")
            
        if grad_cv > 1.0:
            warnings.append(f"High gradient variance (CV: {grad_cv:.3f})")
            
        if any(gn > 100 for gn in gradient_norms):
            warnings.append("Gradient explosion detected")
            
        return stability_score, errors, warnings, metrics
        
    def _analyze_efficiency(
        self, 
        loss_history: List[float],
        gradient_norms: List[float],
        num_steps: int
    ) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        """Analyze training efficiency."""
        errors = []
        warnings = []
        metrics = {}
        
        if len(loss_history) < 5:
            return 0.0, errors, warnings, metrics
            
        # Loss reduction per step
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        
        if initial_loss > 0:
            loss_reduction_rate = (initial_loss - final_loss) / (initial_loss * num_steps)
        else:
            loss_reduction_rate = 0.0
            
        # Gradient efficiency (how much loss reduction per unit gradient)
        avg_grad_norm = np.mean(gradient_norms)
        if avg_grad_norm > 0:
            gradient_efficiency = loss_reduction_rate / avg_grad_norm
        else:
            gradient_efficiency = 0.0
            
        # Efficiency score
        efficiency_score = min(1.0, loss_reduction_rate * 1000)  # Scale appropriately
        
        metrics.update({
            'loss_reduction_rate': loss_reduction_rate,
            'gradient_efficiency': gradient_efficiency,
            'average_gradient_norm': avg_grad_norm
        })
        
        # Check efficiency issues
        if loss_reduction_rate < 1e-5:
            warnings.append(f"Low learning efficiency: {loss_reduction_rate:.2e}")
            
        if avg_grad_norm < 1e-6:
            warnings.append("Very small gradients - may indicate vanishing gradients")
            
        return efficiency_score, errors, warnings, metrics