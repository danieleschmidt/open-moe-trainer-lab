"""Input validation and configuration checking utilities."""

import os
import re
import torch
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    def __post_init__(self):
        if not self.errors:
            self.errors = []
        if not self.warnings:
            self.warnings = []
        if not self.suggestions:
            self.suggestions = []
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    def add_suggestion(self, message: str):
        """Add improvement suggestion."""
        self.suggestions.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.suggestions.extend(other.suggestions)
        if not other.is_valid:
            self.is_valid = False


class ConfigValidator:
    """Comprehensive configuration validator for MoE training."""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        # Required fields
        required_fields = ['hidden_size', 'num_experts', 'experts_per_token', 'num_layers']
        for field in required_fields:
            if field not in config:
                result.add_error(f"Missing required model config field: {field}")
        
        if not result.is_valid:
            return result
        
        # Validate numeric constraints
        hidden_size = config.get('hidden_size', 768)
        if hidden_size <= 0 or hidden_size % 64 != 0:
            result.add_error(f"hidden_size must be positive and divisible by 64, got {hidden_size}")
        
        num_experts = config.get('num_experts', 8)
        if num_experts <= 1:
            result.add_error(f"num_experts must be > 1, got {num_experts}")
        elif num_experts > 1024:
            result.add_warning(f"Very large num_experts ({num_experts}) may cause memory issues")
        
        experts_per_token = config.get('experts_per_token', 2)
        if experts_per_token <= 0 or experts_per_token > num_experts:
            result.add_error(f"experts_per_token must be in range [1, {num_experts}], got {experts_per_token}")
        
        num_layers = config.get('num_layers', 12)
        if num_layers <= 0:
            result.add_error(f"num_layers must be positive, got {num_layers}")
        
        # Validate MoE layer configuration
        moe_layers = config.get('moe_layers')
        if moe_layers is not None:
            if not isinstance(moe_layers, list):
                result.add_error("moe_layers must be a list of layer indices")
            else:
                for layer_idx in moe_layers:
                    if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= num_layers:
                        result.add_error(f"Invalid MoE layer index {layer_idx}, must be in range [0, {num_layers-1}]")
                
                if len(moe_layers) == 0:
                    result.add_warning("No MoE layers specified, model will be standard transformer")
                elif len(moe_layers) == num_layers:
                    result.add_warning("All layers are MoE, consider using some standard layers for stability")
        
        # Validate attention configuration
        num_attention_heads = config.get('num_attention_heads', 12)
        if hidden_size % num_attention_heads != 0:
            result.add_error(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        
        # Validate vocabulary size
        vocab_size = config.get('vocab_size', 32000)
        if vocab_size <= 0:
            result.add_error(f"vocab_size must be positive, got {vocab_size}")
        elif vocab_size < 1000:
            result.add_warning(f"Small vocab_size ({vocab_size}) may limit model expressiveness")
        
        # Loss coefficient validation
        aux_loss_coef = config.get('aux_loss_coef', 0.01)
        if aux_loss_coef < 0 or aux_loss_coef > 1:
            result.add_warning(f"aux_loss_coef ({aux_loss_coef}) outside typical range [0, 1]")
        
        z_loss_coef = config.get('z_loss_coef', 0.001)
        if z_loss_coef < 0 or z_loss_coef > 0.1:
            result.add_warning(f"z_loss_coef ({z_loss_coef}) outside typical range [0, 0.1]")
        
        # Router configuration
        router_type = config.get('router_type', 'top_k')
        if router_type not in ['top_k', 'switch', 'expert_choice']:
            result.add_error(f"Unsupported router_type: {router_type}")
        
        # Performance suggestions
        if num_experts > 64 and experts_per_token == 2:
            result.add_suggestion("Consider increasing experts_per_token for better utilization with many experts")
        
        if hidden_size < 512:
            result.add_suggestion("Small hidden_size may limit model capacity for complex tasks")
        
        return result
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        # Learning rate validation
        learning_rate = config.get('learning_rate', 3e-4)
        if learning_rate <= 0 or learning_rate > 1:
            result.add_error(f"learning_rate must be in range (0, 1], got {learning_rate}")
        elif learning_rate > 1e-2:
            result.add_warning(f"High learning_rate ({learning_rate}) may cause training instability")
        elif learning_rate < 1e-6:
            result.add_warning(f"Very low learning_rate ({learning_rate}) may slow convergence")
        
        # Batch size validation
        batch_size = config.get('per_device_train_batch_size', 8)
        if batch_size <= 0:
            result.add_error(f"per_device_train_batch_size must be positive, got {batch_size}")
        elif batch_size > 128:
            result.add_warning(f"Large batch_size ({batch_size}) may require significant memory")
        
        # Gradient accumulation
        grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        if grad_accum_steps <= 0:
            result.add_error(f"gradient_accumulation_steps must be positive, got {grad_accum_steps}")
        
        effective_batch_size = batch_size * grad_accum_steps
        if effective_batch_size > 1024:
            result.add_warning(f"Very large effective batch size ({effective_batch_size}) may hurt convergence")
        
        # Epoch validation
        num_epochs = config.get('num_epochs', 10)
        if num_epochs <= 0:
            result.add_error(f"num_epochs must be positive, got {num_epochs}")
        elif num_epochs > 1000:
            result.add_warning(f"Very large num_epochs ({num_epochs}) - consider early stopping")
        
        # Mixed precision validation
        fp16 = config.get('fp16', False)
        bf16 = config.get('bf16', False)
        if fp16 and bf16:
            result.add_error("Cannot enable both fp16 and bf16 simultaneously")
        
        # Optimizer validation
        optimizer = config.get('optimizer', 'adamw')
        if optimizer not in ['adamw', 'adam', 'sgd', 'adafactor']:
            result.add_warning(f"Uncommon optimizer: {optimizer}")
        
        # Gradient clipping
        max_grad_norm = config.get('max_grad_norm', 1.0)
        if max_grad_norm <= 0:
            result.add_error(f"max_grad_norm must be positive, got {max_grad_norm}")
        elif max_grad_norm > 10:
            result.add_warning(f"High max_grad_norm ({max_grad_norm}) may indicate unstable training")
        
        # Load balancing coefficients
        lb_coef = config.get('load_balancing_loss_coef', 0.01)
        if lb_coef < 0 or lb_coef > 1:
            result.add_warning(f"load_balancing_loss_coef ({lb_coef}) outside typical range [0, 1]")
        
        # Suggestions
        if not fp16 and not bf16:
            result.add_suggestion("Consider enabling mixed precision (bf16 recommended for MoE) for faster training")
        
        if config.get('warmup_steps', 0) == 0:
            result.add_suggestion("Consider adding warmup steps for more stable training")
        
        return result
    
    @staticmethod
    def validate_data_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate data configuration parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        # Sequence length validation
        max_seq_length = config.get('max_seq_length', 1024)
        if max_seq_length <= 0:
            result.add_error(f"max_seq_length must be positive, got {max_seq_length}")
        elif max_seq_length > 32768:
            result.add_warning(f"Very long max_seq_length ({max_seq_length}) requires significant memory")
        
        # Data loading
        num_workers = config.get('num_workers', 4)
        if num_workers < 0:
            result.add_error(f"num_workers must be non-negative, got {num_workers}")
        elif num_workers > 16:
            result.add_warning(f"Many data workers ({num_workers}) may cause overhead")
        
        return result
    
    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> ValidationResult:
        """Validate file and directory paths in configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        # Check data paths
        data_paths = ['train_data_path', 'eval_data_path', 'test_data_path']
        for path_key in data_paths:
            if path_key in config:
                path = Path(config[path_key])
                if not path.exists():
                    if path_key == 'train_data_path':
                        result.add_error(f"Training data path does not exist: {path}")
                    else:
                        result.add_warning(f"Data path does not exist: {path}")
        
        # Check output directory
        output_dir = config.get('output_dir')
        if output_dir:
            output_path = Path(output_dir)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                result.add_error(f"Cannot create output directory {output_path}: {e}")
        
        return result
    
    @staticmethod
    def validate_hardware_requirements(config: Dict[str, Any]) -> ValidationResult:
        """Validate hardware requirements against available resources."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            result.add_warning("CUDA not available, training will be slow on CPU")
            return result
        
        # Memory estimation
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        hidden_size = model_config.get('hidden_size', 768)
        num_experts = model_config.get('num_experts', 8)
        num_layers = model_config.get('num_layers', 12)
        batch_size = training_config.get('per_device_train_batch_size', 8)
        seq_length = config.get('data', {}).get('max_seq_length', 1024)
        
        # Rough memory estimation (in GB)
        param_memory = (hidden_size * hidden_size * num_layers * 4 * 4) / (1024**3)  # Transformer params
        expert_memory = (hidden_size * hidden_size * 4 * num_experts * len(model_config.get('moe_layers', [1]))) / (1024**3)
        activation_memory = (batch_size * seq_length * hidden_size * num_layers * 4) / (1024**3)
        
        total_memory_gb = param_memory + expert_memory + activation_memory * 2  # Factor for gradients
        
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if total_memory_gb > available_memory * 0.9:
            result.add_error(f"Estimated memory usage ({total_memory_gb:.1f}GB) exceeds available GPU memory ({available_memory:.1f}GB)")
            result.add_suggestion("Consider reducing batch_size, seq_length, or model size")
        elif total_memory_gb > available_memory * 0.7:
            result.add_warning(f"High memory usage estimated ({total_memory_gb:.1f}GB / {available_memory:.1f}GB available)")
        
        return result
    
    @classmethod
    def validate_full_config(cls, config: Dict[str, Any]) -> ValidationResult:
        """Validate complete configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        # Validate each section
        if 'model' in config:
            model_result = cls.validate_model_config(config['model'])
            result.merge(model_result)
        
        if 'training' in config:
            training_result = cls.validate_training_config(config['training'])
            result.merge(training_result)
        
        if 'data' in config:
            data_result = cls.validate_data_config(config['data'])
            result.merge(data_result)
        
        # Validate paths
        path_result = cls.validate_paths(config)
        result.merge(path_result)
        
        # Validate hardware requirements
        hardware_result = cls.validate_hardware_requirements(config)
        result.merge(hardware_result)
        
        return result


class InputValidator:
    """Validate user inputs and command line arguments."""
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> ValidationResult:
        """Validate file path."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        try:
            path = Path(path)
        except Exception as e:
            result.add_error(f"Invalid path format: {e}")
            return result
        
        if must_exist and not path.exists():
            result.add_error(f"File does not exist: {path}")
        elif must_exist and path.is_dir():
            result.add_error(f"Expected file but got directory: {path}")
        elif not must_exist:
            # Check if parent directory exists and is writable
            parent = path.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    result.add_error(f"Cannot create parent directory {parent}: {e}")
            elif not os.access(parent, os.W_OK):
                result.add_error(f"No write permission for directory: {parent}")
        
        return result
    
    @staticmethod
    def validate_model_name(name: str) -> ValidationResult:
        """Validate model name format."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        if not name:
            result.add_error("Model name cannot be empty")
            return result
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', name):
            result.add_error("Model name can only contain letters, numbers, hyphens, underscores, and dots")
        
        # Check length
        if len(name) > 100:
            result.add_error("Model name too long (max 100 characters)")
        elif len(name) < 3:
            result.add_warning("Very short model name, consider more descriptive naming")
        
        return result
    
    @staticmethod
    def validate_port(port: Union[str, int]) -> ValidationResult:
        """Validate network port number."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        try:
            port = int(port)
        except (ValueError, TypeError):
            result.add_error(f"Port must be an integer, got {port}")
            return result
        
        if port < 1 or port > 65535:
            result.add_error(f"Port must be in range [1, 65535], got {port}")
        elif port < 1024:
            result.add_warning(f"Port {port} is reserved, may require root privileges")
        
        return result
    
    @staticmethod  
    def validate_batch_size(batch_size: Union[str, int]) -> ValidationResult:
        """Validate batch size parameter."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        try:
            batch_size = int(batch_size)
        except (ValueError, TypeError):
            result.add_error(f"Batch size must be an integer, got {batch_size}")
            return result
        
        if batch_size <= 0:
            result.add_error(f"Batch size must be positive, got {batch_size}")
        elif batch_size > 256:
            result.add_warning(f"Large batch size ({batch_size}) may require significant memory")
        
        return result


def validate_and_suggest(config: Dict[str, Any], silent: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate configuration and provide suggestions for improvement.
    
    Args:
        config: Configuration dictionary to validate
        silent: If True, only log errors (not warnings/suggestions)
        
    Returns:
        Tuple of (is_valid, suggested_config)
    """
    result = ConfigValidator.validate_full_config(config)
    
    # Log validation results
    if result.errors:
        for error in result.errors:
            logger.error(f"❌ Validation Error: {error}")
    
    if not silent:
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"⚠️  Validation Warning: {warning}")
        
        if result.suggestions:
            for suggestion in result.suggestions:
                logger.info(f"💡 Suggestion: {suggestion}")
    
    # Generate suggested config (could be enhanced with automatic fixes)
    suggested_config = config.copy()
    
    return result.is_valid, suggested_config


def safe_config_load(config_path: str) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """
    Safely load and validate configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (success, config_dict, error_messages)
    """
    import yaml
    import json
    
    errors = []
    
    # Validate path
    path_result = InputValidator.validate_file_path(config_path, must_exist=True)
    if not path_result.is_valid:
        return False, None, path_result.errors
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    except Exception as e:
        errors.append(f"Failed to load configuration file: {e}")
        return False, None, errors
    
    # Validate configuration
    is_valid, _ = validate_and_suggest(config)
    
    return is_valid, config, errors