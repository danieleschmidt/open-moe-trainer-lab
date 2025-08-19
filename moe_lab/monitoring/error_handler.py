"""Robust error handling and recovery for MoE systems."""

import time
import logging
import threading
import traceback
import functools
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch for testing
    torch = type('torch', (), {
        'cuda': type('cuda', (), {
            'OutOfMemoryError': Exception,
            'empty_cache': lambda: None,
            'is_available': lambda: False
        })()
    })()


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery action types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    ABORT = "abort"
    IGNORE = "ignore"


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""
    
    error_type: str
    error_message: str
    traceback_str: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    recovery_action: Optional[RecoveryAction] = None
    recovery_successful: Optional[bool] = None
    
    
@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    
    name: str
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half_open
    total_requests: int = 0
    successful_requests: int = 0
    

class MoEErrorHandler:
    """Centralized error handler for MoE systems."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_error_history: int = 1000,
        auto_recovery: bool = True
    ):
        self.max_error_history = max_error_history
        self.auto_recovery = auto_recovery
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_counts = defaultdict(int)
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        
        # Setup logging
        self.logger = self._setup_logger(log_file)
        
        # Register default recovery strategies
        self._register_default_strategies()
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        """Setup error handler logger."""
        logger = logging.getLogger("moe_error_handler")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        
        # CUDA out of memory
        if TORCH_AVAILABLE:
            self.register_recovery_strategy(
                torch.cuda.OutOfMemoryError,
                self._handle_cuda_oom
            )
        
        # General memory error
        self.register_recovery_strategy(
            MemoryError,
            self._handle_memory_error
        )
        
        # Runtime errors
        self.register_recovery_strategy(
            RuntimeError,
            self._handle_runtime_error
        )
        
        # Value errors
        self.register_recovery_strategy(
            ValueError,
            self._handle_value_error
        )
        
        # Network/connection errors
        self.register_recovery_strategy(
            ConnectionError,
            self._handle_connection_error
        )
    
    def register_recovery_strategy(
        self,
        error_type: Type[Exception],
        recovery_func: Callable
    ):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = recovery_func
        self.logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy."""
        
        with self._lock:
            # Create error info
            error_info = ErrorInfo(
                error_type=type(error).__name__,
                error_message=str(error),
                traceback_str=traceback.format_exc(),
                timestamp=time.time(),
                context=context or {},
                severity=severity
            )
            
            # Log error
            self.logger.error(f"Error handled: {error_info.error_type} - {error_info.error_message}")
            
            # Update statistics
            self.error_counts[error_info.error_type] += 1
            
            # Attempt recovery if enabled
            recovery_result = None
            if self.auto_recovery:
                recovery_result = self._attempt_recovery(error, error_info)
            
            # Store in history
            self.error_history.append(error_info)
            
            return recovery_result
    
    def _attempt_recovery(self, error: Exception, error_info: ErrorInfo) -> Optional[Any]:
        """Attempt to recover from an error."""
        
        # Find appropriate recovery strategy
        recovery_func = None
        for error_type, func in self.recovery_strategies.items():
            if isinstance(error, error_type):
                recovery_func = func
                break
        
        if recovery_func is None:
            self.logger.warning(f"No recovery strategy found for {type(error).__name__}")
            error_info.recovery_action = RecoveryAction.ABORT
            return None
        
        try:
            self.logger.info(f"Attempting recovery for {type(error).__name__}")
            recovery_result = recovery_func(error, error_info.context)
            error_info.recovery_successful = True
            self.logger.info("Recovery successful")
            return recovery_result
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            error_info.recovery_successful = False
            return None
    
    def _handle_cuda_oom(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Handle CUDA out of memory errors."""
        self.logger.info("Handling CUDA OOM error")
        
        if TORCH_AVAILABLE:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Suggest batch size reduction
            suggested_batch_size = context.get('batch_size', 32) // 2
            self.logger.info(f"Suggested batch size reduction: {suggested_batch_size}")
            
            return {
                "recovery_action": RecoveryAction.RETRY,
                "suggested_batch_size": suggested_batch_size,
                "cuda_cache_cleared": True
            }
        
        return {"recovery_action": RecoveryAction.FALLBACK}
    
    def _handle_memory_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Handle general memory errors."""
        self.logger.info("Handling memory error")
        
        # Suggest memory optimization
        return {
            "recovery_action": RecoveryAction.RETRY,
            "suggestions": [
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Use mixed precision training",
                "Reduce model size"
            ]
        }
    
    def _handle_runtime_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Handle runtime errors."""
        error_message = str(error).lower()
        
        if "dimension" in error_message or "shape" in error_message:
            return {
                "recovery_action": RecoveryAction.FALLBACK,
                "error_type": "shape_mismatch",
                "suggestion": "Check tensor dimensions and reshaping operations"
            }
        elif "device" in error_message:
            return {
                "recovery_action": RecoveryAction.RETRY,
                "error_type": "device_mismatch", 
                "suggestion": "Ensure all tensors are on the same device"
            }
        else:
            return {
                "recovery_action": RecoveryAction.FALLBACK,
                "error_type": "general_runtime"
            }
    
    def _handle_value_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Handle value errors."""
        return {
            "recovery_action": RecoveryAction.RETRY,
            "suggestion": "Check input values and parameters"
        }
    
    def _handle_connection_error(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]:
        """Handle connection errors."""
        return {
            "recovery_action": RecoveryAction.RETRY,
            "retry_delay": 5.0,
            "max_retries": 3
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        with self._lock:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {"total_errors": 0}
            
            # Count by type
            error_type_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            recovery_success_rate = defaultdict(list)
            
            for error_info in self.error_history:
                error_type_counts[error_info.error_type] += 1
                severity_counts[error_info.severity.value] += 1
                
                if error_info.recovery_successful is not None:
                    recovery_success_rate[error_info.error_type].append(error_info.recovery_successful)
            
            # Calculate success rates
            success_rates = {}
            for error_type, successes in recovery_success_rate.items():
                success_rates[error_type] = sum(successes) / len(successes) if successes else 0.0
            
            # Recent error trend (last hour)
            recent_errors = [
                error for error in self.error_history
                if time.time() - error.timestamp < 3600
            ]
            
            return {
                "total_errors": total_errors,
                "error_type_counts": dict(error_type_counts),
                "severity_counts": dict(severity_counts),
                "recovery_success_rates": success_rates,
                "recent_errors_count": len(recent_errors),
                "error_rate_per_hour": len(recent_errors)
            }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        recovery_timeout: float = 30.0
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitBreakerState(name=name)
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(f"circuit_breaker_{name}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        
        with self._lock:
            self.state.total_requests += 1
            
            # Check if circuit is open
            if self._is_open():
                if time.time() - self.state.last_failure_time > self.recovery_timeout:
                    # Try half-open state
                    self.state.state = "half_open"
                    self.logger.info(f"Circuit breaker {self.name} entering half-open state")
                else:
                    raise Exception(f"Circuit breaker {self.name} is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise e
    
    def _is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state.state == "open"
    
    def _on_success(self):
        """Handle successful execution."""
        self.state.successful_requests += 1
        
        if self.state.state == "half_open":
            # Recovery successful, close circuit
            self.state.state = "closed"
            self.state.failure_count = 0
            self.logger.info(f"Circuit breaker {self.name} closed after successful recovery")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()
        
        if self.state.failure_count >= self.failure_threshold:
            self.state.state = "open"
            self.logger.warning(f"Circuit breaker {self.name} opened after {self.state.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "name": self.state.name,
                "state": self.state.state,
                "failure_count": self.state.failure_count,
                "total_requests": self.state.total_requests,
                "successful_requests": self.state.successful_requests,
                "success_rate": self.state.successful_requests / self.state.total_requests if self.state.total_requests > 0 else 0.0,
                "last_failure_time": self.state.last_failure_time
            }


class RetryHandler:
    """Configurable retry handler with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        self.logger = logging.getLogger("retry_handler")
    
    def retry(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ) -> Any:
        """Retry function with exponential backoff."""
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Function succeeded after {attempt} retries")
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.logger.error(f"Function failed after {self.max_retries} retries")
                    break
                
                # Calculate delay
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Add jitter
                if self.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        # All retries exhausted
        raise last_exception


class ErrorRecovery:
    """Advanced error recovery with context-aware strategies."""
    
    def __init__(self):
        self.recovery_history: List[Dict[str, Any]] = []
        self.success_rates: Dict[str, float] = {}
        self.logger = logging.getLogger("error_recovery")
    
    def register_recovery_context(self, context_name: str, recovery_func: Callable):
        """Register recovery function for specific context."""
        setattr(self, f"_recover_{context_name}", recovery_func)
        self.logger.info(f"Registered recovery function for context: {context_name}")
    
    def attempt_recovery(
        self,
        error: Exception,
        context: str,
        **context_kwargs
    ) -> Optional[Any]:
        """Attempt context-aware recovery."""
        
        recovery_func_name = f"_recover_{context}"
        if not hasattr(self, recovery_func_name):
            self.logger.warning(f"No recovery function for context: {context}")
            return None
        
        recovery_func = getattr(self, recovery_func_name)
        
        try:
            self.logger.info(f"Attempting recovery for {context}")
            result = recovery_func(error, **context_kwargs)
            
            # Record success
            self._record_recovery_attempt(context, True, str(error))
            return result
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed for {context}: {recovery_error}")
            self._record_recovery_attempt(context, False, str(error))
            return None
    
    def _record_recovery_attempt(self, context: str, success: bool, error_message: str):
        """Record recovery attempt for analysis."""
        attempt = {
            "context": context,
            "success": success,
            "error_message": error_message,
            "timestamp": time.time()
        }
        
        self.recovery_history.append(attempt)
        
        # Update success rate
        context_attempts = [a for a in self.recovery_history if a["context"] == context]
        context_successes = [a for a in context_attempts if a["success"]]
        self.success_rates[context] = len(context_successes) / len(context_attempts)
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "total_attempts": len(self.recovery_history),
            "success_rates_by_context": self.success_rates,
            "recent_attempts": self.recovery_history[-10:] if self.recovery_history else []
        }


def robust_execution(
    retries: int = 3,
    circuit_breaker: Optional[CircuitBreaker] = None,
    error_handler: Optional[MoEErrorHandler] = None
):
    """Decorator for robust function execution."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(max_retries=retries)
            
            def execute():
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            try:
                return retry_handler.retry(execute)
            except Exception as e:
                if error_handler:
                    recovery_result = error_handler.handle_error(e)
                    if recovery_result and recovery_result.get("recovery_action") == RecoveryAction.RETRY:
                        # Try once more with recovery suggestions
                        return execute()
                raise e
        
        return wrapper
    return decorator