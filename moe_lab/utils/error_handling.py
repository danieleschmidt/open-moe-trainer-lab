"""Robust error handling and recovery mechanisms for MoE training."""

import sys
import traceback
import logging
import functools
from typing import Optional, Callable, Any, Dict, Type, Union, List
from pathlib import Path
import torch
import json
import time
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MoETrainingError(Exception):
    """Base exception for MoE training errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None, recovery_suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.timestamp = time.time()


class ModelConfigurationError(MoETrainingError):
    """Errors related to model configuration."""
    pass


class DataLoadingError(MoETrainingError):
    """Errors related to data loading and preprocessing."""
    pass


class TrainingError(MoETrainingError):
    """Errors during training process."""
    pass


class InferenceError(MoETrainingError):
    """Errors during inference."""
    pass


class ResourceError(MoETrainingError):
    """Errors related to system resources."""
    pass


class CheckpointError(MoETrainingError):
    """Errors related to model checkpointing."""
    pass


class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self, log_file: Optional[str] = None, max_retries: int = 3):
        self.max_retries = max_retries
        self.log_file = log_file
        self.error_history: List[Dict[str, Any]] = []
        
        # Setup file logging if specified
        if log_file:
            self._setup_file_logging(log_file)
        
        logger.info(f"Initialized ErrorHandler with max_retries={max_retries}")
    
    def _setup_file_logging(self, log_file: str):
        """Setup file logging for errors."""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle and log error with context information."""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'traceback': traceback.format_exception(type(error), error, error.__traceback__)
        }
        
        # Add MoE-specific error information
        if isinstance(error, MoETrainingError):
            error_info.update({
                'severity': error.severity.value,
                'recovery_suggestion': error.recovery_suggestion,
                'moe_context': error.context
            })
        
        # Store in error history
        self.error_history.append(error_info)
        
        # Log error
        log_level = self._get_log_level(error)
        logger.log(log_level, f"Error handled: {error_info['error_type']} - {error_info['message']}")
        
        if isinstance(error, MoETrainingError) and error.recovery_suggestion:
            logger.info(f"Recovery suggestion: {error.recovery_suggestion}")
        
        return error_info
    
    def _get_log_level(self, error: Exception) -> int:
        """Get appropriate log level for error."""
        if isinstance(error, MoETrainingError):
            severity_to_level = {
                ErrorSeverity.LOW: logging.INFO,
                ErrorSeverity.MEDIUM: logging.WARNING,
                ErrorSeverity.HIGH: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL
            }
            return severity_to_level.get(error.severity, logging.ERROR)
        return logging.ERROR
    
    def retry_with_backoff(self, func: Callable, *args, backoff_factor: float = 2.0, **kwargs) -> Any:
        """Retry function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.handle_error(e, {'attempt': attempt + 1, 'function': func.__name__})
                
                if attempt < self.max_retries - 1:
                    sleep_time = backoff_factor ** attempt
                    logger.info(f"Retrying in {sleep_time:.1f} seconds (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for {func.__name__}")
        
        raise last_exception
    
    def get_error_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of recent errors."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        recent_errors = [e for e in self.error_history if e['timestamp'] > cutoff_time]
        
        error_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            error_type = error['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            if 'severity' in error:
                severity = error['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'error_types': error_counts,
            'severity_distribution': severity_counts,
            'time_window_hours': (time_window / 3600.0) if time_window else None,
            'most_common_error': max(error_counts.keys(), key=error_counts.get) if error_counts else None
        }
    
    def export_error_log(self, output_path: str, time_window: Optional[float] = None):
        """Export error history to file."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        recent_errors = [e for e in self.error_history if e['timestamp'] > cutoff_time]
        
        export_data = {
            'export_timestamp': current_time,
            'time_window_hours': (time_window / 3600.0) if time_window else None,
            'total_errors': len(recent_errors),
            'errors': recent_errors
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(recent_errors)} errors to {output_path}")


def with_error_handling(error_types: Union[Type[Exception], List[Type[Exception]]] = Exception,
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       recovery_suggestion: Optional[str] = None,
                       reraise: bool = True):
    """Decorator for automatic error handling."""
    if not isinstance(error_types, (list, tuple)):
        error_types = [error_types]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if any(isinstance(e, error_type) for error_type in error_types):
                    # Convert to MoE error if not already
                    if not isinstance(e, MoETrainingError):
                        moe_error = MoETrainingError(
                            message=str(e),
                            severity=severity,
                            context={'function': func.__name__, 'args': str(args), 'kwargs': str(kwargs)},
                            recovery_suggestion=recovery_suggestion
                        )
                        moe_error.__cause__ = e
                        e = moe_error
                    
                    # Handle error
                    error_handler = get_global_error_handler()
                    error_handler.handle_error(e)
                    
                    if reraise:
                        raise e
                    return None
                else:
                    raise
        return wrapper
    return decorator


@contextmanager
def error_context(context_info: Dict[str, Any]):
    """Context manager for adding context to errors."""
    try:
        yield
    except Exception as e:
        if isinstance(e, MoETrainingError):
            e.context.update(context_info)
        else:
            # Wrap in MoE error with context
            moe_error = MoETrainingError(
                message=str(e),
                context=context_info,
                recovery_suggestion="Check the error context for debugging information"
            )
            moe_error.__cause__ = e
            raise moe_error from e
        raise


class CheckpointManager:
    """Robust checkpoint management with error recovery."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        logger.info(f"Initialized CheckpointManager at {checkpoint_dir}")
    
    @with_error_handling(CheckpointError, recovery_suggestion="Check disk space and permissions")
    def save_checkpoint(self, state_dict: Dict[str, Any], step: int, is_best: bool = False) -> str:
        """Save checkpoint with error handling."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        try:
            # Save to temporary file first
            temp_path = checkpoint_path.with_suffix('.tmp')
            torch.save(state_dict, temp_path)
            
            # Atomic move
            temp_path.rename(checkpoint_path)
            
            logger.info(f"Saved checkpoint at step {step}")
            
            # Save as best if specified
            if is_best:
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                torch.save(state_dict, best_path)
                logger.info("Saved as best checkpoint")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to save checkpoint at step {step}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context={'step': step, 'path': str(checkpoint_path)},
                recovery_suggestion="Check disk space, permissions, and model state"
            )
    
    @with_error_handling(CheckpointError, recovery_suggestion="Check checkpoint file integrity")
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint with error handling."""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None:
            raise CheckpointError(
                "No checkpoint found to load",
                severity=ErrorSeverity.MEDIUM,
                recovery_suggestion="Check checkpoint directory or provide specific checkpoint path"
            )
        
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state_dict
            
        except Exception as e:
            raise CheckpointError(
                f"Failed to load checkpoint from {checkpoint_path}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context={'path': str(checkpoint_path)},
                recovery_suggestion="Verify checkpoint file integrity or use different checkpoint"
            )
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by step number
        def extract_step(path):
            try:
                return int(path.stem.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        latest_checkpoint = max(checkpoint_files, key=extract_step)
        return str(latest_checkpoint)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest files
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logger.debug(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {file_path}: {e}")


class GradientMonitor:
    """Monitor gradients for anomalies and instabilities."""
    
    def __init__(self, clip_threshold: float = 10.0, nan_check: bool = True):
        self.clip_threshold = clip_threshold
        self.nan_check = nan_check
        self.gradient_history = []
        self.max_history = 1000
    
    def check_gradients(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check model gradients for anomalies."""
        gradient_stats = {
            'timestamp': time.time(),
            'total_norm': 0.0,
            'max_grad': 0.0,
            'min_grad': 0.0,
            'nan_count': 0,
            'inf_count': 0,
            'zero_grad_params': 0,
            'param_count': 0
        }
        
        all_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                gradient_stats['param_count'] += 1
                
                # Check for NaN/Inf
                if self.nan_check:
                    nan_count = torch.isnan(grad).sum().item()
                    inf_count = torch.isinf(grad).sum().item()
                    gradient_stats['nan_count'] += nan_count
                    gradient_stats['inf_count'] += inf_count
                
                # Collect gradient values
                grad_flat = grad.flatten()
                all_grads.append(grad_flat)
                
                # Check for zero gradients
                if torch.all(grad == 0):
                    gradient_stats['zero_grad_params'] += 1
            else:
                # Parameter has no gradient
                gradient_stats['zero_grad_params'] += 1
                gradient_stats['param_count'] += 1
        
        if all_grads:
            all_grads_tensor = torch.cat(all_grads)
            gradient_stats['total_norm'] = torch.norm(all_grads_tensor).item()
            gradient_stats['max_grad'] = torch.max(all_grads_tensor).item()
            gradient_stats['min_grad'] = torch.min(all_grads_tensor).item()
        
        # Store in history
        self.gradient_history.append(gradient_stats)
        if len(self.gradient_history) > self.max_history:
            self.gradient_history.pop(0)
        
        # Check for anomalies
        anomalies = self._detect_anomalies(gradient_stats)
        gradient_stats['anomalies'] = anomalies
        
        return gradient_stats
    
    def _detect_anomalies(self, stats: Dict[str, Any]) -> List[str]:
        """Detect gradient anomalies."""
        anomalies = []
        
        if stats['nan_count'] > 0:
            anomalies.append(f"NaN gradients detected: {stats['nan_count']}")
        
        if stats['inf_count'] > 0:
            anomalies.append(f"Infinite gradients detected: {stats['inf_count']}")
        
        if stats['total_norm'] > self.clip_threshold:
            anomalies.append(f"Large gradient norm: {stats['total_norm']:.4f}")
        
        zero_grad_ratio = stats['zero_grad_params'] / max(stats['param_count'], 1)
        if zero_grad_ratio > 0.5:
            anomalies.append(f"Many parameters with zero gradients: {zero_grad_ratio:.1%}")
        
        return anomalies


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def setup_error_handling(log_file: Optional[str] = None, max_retries: int = 3) -> ErrorHandler:
    """Setup global error handling."""
    global _global_error_handler
    _global_error_handler = ErrorHandler(log_file, max_retries)
    logger.info("Setup global error handling")
    return _global_error_handler


def install_exception_handler():
    """Install global exception handler."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_handler = get_global_error_handler()
        error_handler.handle_error(exc_value)
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    logger.info("Installed global exception handler")