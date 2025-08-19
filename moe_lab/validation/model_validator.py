"""Model validation utilities for MoE models."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from ..models import MoEModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    

class MoEModelValidator:
    """Comprehensive validator for MoE models."""
    
    def __init__(self, model: MoEModel, strict: bool = True):
        self.model = model
        self.strict = strict
        self.device = next(model.parameters()).device
        
    def validate_model(self) -> ValidationResult:
        """Run comprehensive model validation."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Validate model architecture
            arch_errors, arch_warnings, arch_metrics = self._validate_architecture()
            errors.extend(arch_errors)
            warnings.extend(arch_warnings)
            metrics.update(arch_metrics)
            
            # Validate parameter initialization
            param_errors, param_warnings, param_metrics = self._validate_parameters()
            errors.extend(param_errors)
            warnings.extend(param_warnings)
            metrics.update(param_metrics)
            
            # Validate forward pass
            forward_errors, forward_warnings, forward_metrics = self._validate_forward_pass()
            errors.extend(forward_errors)
            warnings.extend(forward_warnings)
            metrics.update(forward_metrics)
            
            # Validate routing consistency
            routing_errors, routing_warnings, routing_metrics = self._validate_routing()
            errors.extend(routing_errors)
            warnings.extend(routing_warnings)
            metrics.update(routing_metrics)
            
            # Validate gradient flow
            grad_errors, grad_warnings, grad_metrics = self._validate_gradients()
            errors.extend(grad_errors)
            warnings.extend(grad_warnings)
            metrics.update(grad_metrics)
            
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
        
    def _validate_architecture(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate model architecture."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check basic architecture properties
        if self.model.hidden_size <= 0:
            errors.append("Hidden size must be positive")
            
        if self.model.num_experts <= 1:
            errors.append("Number of experts must be > 1")
            
        if self.model.experts_per_token <= 0:
            errors.append("Experts per token must be positive")
            
        if self.model.experts_per_token > self.model.num_experts:
            errors.append("Experts per token cannot exceed total number of experts")
            
        # Check layer configuration
        if self.model.num_layers <= 0:
            errors.append("Number of layers must be positive")
            
        if len(self.model.moe_layers) == 0:
            warnings.append("No MoE layers found - model behaves like standard transformer")
            
        # Check for reasonable parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        metrics['total_parameters'] = total_params
        
        if total_params == 0:
            errors.append("Model has no parameters")
        elif total_params > 1e12:  # 1T parameters
            warnings.append(f"Very large model with {total_params:.2e} parameters")
            
        # Validate MoE layer configuration
        for layer_idx in self.model.moe_layers:
            if layer_idx >= self.model.num_layers:
                errors.append(f"MoE layer index {layer_idx} exceeds number of layers")
                
        return errors, warnings, metrics
        
    def _validate_parameters(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate parameter initialization and properties."""
        errors = []
        warnings = []
        metrics = {}
        
        param_stats = {
            'nan_params': 0,
            'inf_params': 0,
            'zero_params': 0,
            'total_params': 0,
            'param_norms': []
        }
        
        for name, param in self.model.named_parameters():
            param_stats['total_params'] += param.numel()
            
            # Check for NaN/Inf values
            if torch.isnan(param).any():
                param_stats['nan_params'] += torch.isnan(param).sum().item()
                errors.append(f"Parameter {name} contains NaN values")
                
            if torch.isinf(param).any():
                param_stats['inf_params'] += torch.isinf(param).sum().item()
                errors.append(f"Parameter {name} contains Inf values")
                
            # Check for all-zero parameters
            if torch.all(param == 0):
                param_stats['zero_params'] += param.numel()
                warnings.append(f"Parameter {name} is all zeros")
                
            # Compute parameter norm
            param_norm = param.norm().item()
            param_stats['param_norms'].append(param_norm)
            
            # Check for extremely large/small norms
            if param_norm > 100:
                warnings.append(f"Parameter {name} has very large norm: {param_norm:.2f}")
            elif param_norm < 1e-6 and param.numel() > 1:
                warnings.append(f"Parameter {name} has very small norm: {param_norm:.2e}")
                
        # Compute aggregate statistics
        if param_stats['param_norms']:
            metrics.update({
                'mean_param_norm': np.mean(param_stats['param_norms']),
                'std_param_norm': np.std(param_stats['param_norms']),
                'max_param_norm': np.max(param_stats['param_norms']),
                'min_param_norm': np.min(param_stats['param_norms'])
            })
            
        metrics.update(param_stats)
        
        return errors, warnings, metrics
        
    def _validate_forward_pass(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate forward pass functionality."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Create test input
            batch_size = 2
            seq_len = 16
            test_input = torch.randint(
                0, 1000, 
                (batch_size, seq_len), 
                device=self.device
            )
            
            # Test forward pass
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(test_input, return_routing_info=True)
                
            # Validate output shape
            expected_shape = (batch_size, seq_len, self.model.hidden_size)
            actual_shape = outputs.last_hidden_state.shape
            
            if actual_shape != expected_shape:
                errors.append(
                    f"Output shape mismatch: expected {expected_shape}, got {actual_shape}"
                )
                
            # Check for NaN/Inf in outputs
            if torch.isnan(outputs.last_hidden_state).any():
                errors.append("Forward pass produces NaN outputs")
                
            if torch.isinf(outputs.last_hidden_state).any():
                errors.append("Forward pass produces Inf outputs")
                
            # Validate routing info
            routing_info = outputs.routing_info
            if routing_info is None:
                warnings.append("No routing information returned")
            else:
                if routing_info.expert_weights is None:
                    warnings.append("No expert weights in routing info")
                if routing_info.selected_experts is None:
                    warnings.append("No selected experts in routing info")
                    
            # Test with different input sizes
            test_inputs = [
                torch.randint(0, 1000, (1, 8), device=self.device),
                torch.randint(0, 1000, (4, 32), device=self.device)
            ]
            
            for i, test_inp in enumerate(test_inputs):
                try:
                    with torch.no_grad():
                        out = self.model(test_inp)
                    metrics[f'forward_pass_test_{i}'] = True
                except Exception as e:
                    errors.append(f"Forward pass failed for input {i}: {str(e)}")
                    metrics[f'forward_pass_test_{i}'] = False
                    
        except Exception as e:
            errors.append(f"Forward pass validation failed: {str(e)}")
            
        return errors, warnings, metrics
        
    def _validate_routing(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate routing behavior."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Test routing consistency
            test_input = torch.randint(0, 1000, (2, 16), device=self.device)
            
            self.model.eval()
            routing_results = []
            
            # Run multiple times to check consistency
            for _ in range(5):
                with torch.no_grad():
                    outputs = self.model(test_input, return_routing_info=True)
                    routing_results.append(outputs.routing_info)
                    
            # Validate routing properties
            for i, routing_info in enumerate(routing_results):
                if routing_info is None:
                    continue
                    
                # Check load variance
                if hasattr(routing_info, 'load_variance'):
                    if routing_info.load_variance < 0:
                        errors.append(f"Negative load variance: {routing_info.load_variance}")
                    elif routing_info.load_variance > 1.0:
                        warnings.append(f"Very high load variance: {routing_info.load_variance}")
                        
                # Check entropy
                if hasattr(routing_info, 'entropy'):
                    if routing_info.entropy < 0:
                        errors.append(f"Negative entropy: {routing_info.entropy}")
                    elif routing_info.entropy < 0.1:
                        warnings.append(f"Very low routing entropy: {routing_info.entropy}")
                        
                # Check expert selection validity
                if routing_info.selected_experts is not None:
                    experts = routing_info.selected_experts.flatten()
                    invalid_experts = (experts >= self.model.num_experts) | (experts < -1)
                    if invalid_experts.any():
                        errors.append("Invalid expert indices found in routing")
                        
            # Compute routing consistency metrics
            if len(routing_results) > 1:
                metrics['routing_consistency'] = self._compute_routing_consistency(routing_results)
                
        except Exception as e:
            errors.append(f"Routing validation failed: {str(e)}")
            
        return errors, warnings, metrics
        
    def _validate_gradients(self) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Validate gradient flow."""
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Create test input with requires_grad
            test_input = torch.randint(0, 1000, (2, 16), device=self.device)
            
            self.model.train()
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(test_input)
            
            # Compute dummy loss
            target = torch.randint(0, 1000, (2, 16), device=self.device)
            logits = self.model.lm_head(outputs.last_hidden_state)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), 
                target.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Check gradient statistics
            grad_stats = {
                'params_with_grad': 0,
                'params_without_grad': 0,
                'nan_grads': 0,
                'inf_grads': 0,
                'zero_grads': 0,
                'grad_norms': []
            }
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        grad_stats['params_without_grad'] += 1
                        warnings.append(f"Parameter {name} has no gradient")
                    else:
                        grad_stats['params_with_grad'] += 1
                        
                        # Check for NaN/Inf gradients
                        if torch.isnan(param.grad).any():
                            grad_stats['nan_grads'] += 1
                            errors.append(f"Parameter {name} has NaN gradients")
                            
                        if torch.isinf(param.grad).any():
                            grad_stats['inf_grads'] += 1
                            errors.append(f"Parameter {name} has Inf gradients")
                            
                        # Check for zero gradients
                        if torch.all(param.grad == 0):
                            grad_stats['zero_grads'] += 1
                            warnings.append(f"Parameter {name} has all-zero gradients")
                            
                        grad_norm = param.grad.norm().item()
                        grad_stats['grad_norms'].append(grad_norm)
                        
            # Compute gradient statistics
            if grad_stats['grad_norms']:
                metrics.update({
                    'mean_grad_norm': np.mean(grad_stats['grad_norms']),
                    'std_grad_norm': np.std(grad_stats['grad_norms']),
                    'max_grad_norm': np.max(grad_stats['grad_norms']),
                    'total_grad_norm': np.linalg.norm(grad_stats['grad_norms'])
                })
                
            metrics.update(grad_stats)
            
        except Exception as e:
            errors.append(f"Gradient validation failed: {str(e)}")
            
        return errors, warnings, metrics
        
    def _compute_routing_consistency(self, routing_results: List) -> float:
        """Compute consistency score for routing decisions."""
        if len(routing_results) < 2:
            return 1.0
            
        # Compare selected experts across runs
        expert_selections = []
        for routing_info in routing_results:
            if routing_info and routing_info.selected_experts is not None:
                expert_selections.append(routing_info.selected_experts.cpu().numpy())
                
        if len(expert_selections) < 2:
            return 1.0
            
        # Compute pairwise consistency
        consistencies = []
        for i in range(len(expert_selections)):
            for j in range(i + 1, len(expert_selections)):
                consistency = np.mean(expert_selections[i] == expert_selections[j])
                consistencies.append(consistency)
                
        return float(np.mean(consistencies))
        
    def validate_expert_utilization(
        self, 
        num_samples: int = 100,
        target_utilization: float = 1.0 / None  # Will be set to 1/num_experts
    ) -> ValidationResult:
        """Validate expert utilization balance."""
        errors = []
        warnings = []
        metrics = {}
        
        if target_utilization is None:
            target_utilization = 1.0 / self.model.num_experts
            
        try:
            expert_counts = torch.zeros(self.model.num_experts)
            total_tokens = 0
            
            self.model.eval()
            
            for _ in range(num_samples):
                # Generate random input
                seq_len = np.random.randint(8, 64)
                test_input = torch.randint(0, 1000, (1, seq_len), device=self.device)
                
                with torch.no_grad():
                    outputs = self.model(test_input, return_routing_info=True)
                    
                if outputs.routing_info and outputs.routing_info.selected_experts is not None:
                    experts = outputs.routing_info.selected_experts.flatten()
                    total_tokens += len(experts)
                    
                    for expert_idx in range(self.model.num_experts):
                        count = (experts == expert_idx).sum().item()
                        expert_counts[expert_idx] += count
                        
            # Compute utilization statistics
            if total_tokens > 0:
                utilizations = expert_counts / total_tokens
                
                metrics.update({
                    'expert_utilizations': utilizations.tolist(),
                    'mean_utilization': utilizations.mean().item(),
                    'std_utilization': utilizations.std().item(),
                    'min_utilization': utilizations.min().item(),
                    'max_utilization': utilizations.max().item(),
                    'utilization_variance': utilizations.var().item()
                })
                
                # Check for severely under/over-utilized experts
                for i, util in enumerate(utilizations):
                    if util < target_utilization * 0.1:  # Less than 10% of target
                        warnings.append(f"Expert {i} severely under-utilized: {util:.3f}")
                    elif util > target_utilization * 5.0:  # More than 5x target
                        warnings.append(f"Expert {i} severely over-utilized: {util:.3f}")
                        
                # Check overall balance
                if utilizations.std() > target_utilization * 0.5:
                    warnings.append(
                        f"High utilization variance: {utilizations.std().item():.3f}"
                    )
                    
        except Exception as e:
            errors.append(f"Expert utilization validation failed: {str(e)}")
            
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )