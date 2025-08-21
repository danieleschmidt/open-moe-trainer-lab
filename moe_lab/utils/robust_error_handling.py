"""Robust error handling system for MoE models with comprehensive recovery mechanisms.

This module provides enterprise-grade error handling:
1. Hierarchical exception handling with context preservation
2. Automatic retry mechanisms with exponential backoff
3. Circuit breaker pattern for fault isolation
4. Graceful degradation strategies
5. Error recovery and rollback mechanisms
6. Comprehensive error reporting and analysis
"""

import time
import logging
import traceback
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import random
import json
from pathlib import Path
import inspect


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    ROLLBACK = "rollback"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str
    timestamp: float
    exception_type: str
    exception_message: str
    traceback_info: str
    function_name: str
    module_name: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    context_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempted: bool = False
    recovery_successful: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class CircuitBreakerState:
    """Circuit breaker state information."""
    name: str
    state: str  # 'closed', 'open', 'half_open'
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    success_count: int = 0
    total_calls: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        backoff_strategy: str = "exponential"
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.backoff_strategy = backoff_strategy
        
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.backoff_strategy == "exponential":
            delay = self.base_delay * (self.exponential_base ** attempt)
        elif self.backoff_strategy == "linear":
            delay = self.base_delay * (attempt + 1)
        elif self.backoff_strategy == "constant":
            delay = self.base_delay
        else:
            delay = self.base_delay
            
        # Apply jitter
        if self.jitter:
            jitter_factor = random.uniform(0.8, 1.2)
            delay *= jitter_factor
            
        return min(delay, self.max_delay)


class ErrorRecoveryManager:
    """Manages error recovery strategies and mechanisms."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.error_history = deque(maxlen=1000)
        self.recovery_stats = defaultdict(lambda: {"attempted": 0, "successful": 0})
        self.lock = threading.RLock()
        
    def register_recovery_strategy(
        self,
        exception_type: Type[Exception],
        recovery_func: Callable[[ErrorContext], Any],
        strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    ):
        """Register a recovery strategy for specific exception types."""
        strategy_key = exception_type.__name__
        self.recovery_strategies[strategy_key] = {
            'function': recovery_func,
            'strategy': strategy,
            'exception_type': exception_type
        }
        
    def attempt_recovery(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Attempt to recover from an error."""
        exception_name = error_context.exception_type
        
        with self.lock:
            self.recovery_stats[exception_name]["attempted"] += 1
            
        if exception_name in self.recovery_strategies:
            strategy_info = self.recovery_strategies[exception_name]
            recovery_func = strategy_info['function']
            
            try:
                result = recovery_func(error_context)
                
                with self.lock:
                    self.recovery_stats[exception_name]["successful"] += 1
                    
                error_context.recovery_attempted = True
                error_context.recovery_successful = True
                
                return True, result
                
            except Exception as e:
                error_context.recovery_attempted = True
                error_context.recovery_successful = False
                error_context.metadata['recovery_error'] = str(e)
                
                return False, None
        else:
            return False, None
            
    def get_recovery_stats(self) -> Dict[str, Dict[str, int]]:
        """Get recovery attempt statistics."""
        with self.lock:
            return dict(self.recovery_stats)
            
    def add_error_to_history(self, error_context: ErrorContext):
        """Add error to history for analysis."""
        with self.lock:
            self.error_history.append(error_context)


class CircuitBreaker:
    """Circuit breaker implementation for fault isolation."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.state = CircuitBreakerState(
            name=name,
            state='closed',
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold
        )
        self.lock = threading.RLock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            current_time = time.time()
            
            # Check if circuit should transition from open to half-open
            if (self.state.state == 'open' and
                self.state.last_failure_time and
                current_time - self.state.last_failure_time > self.state.recovery_timeout):
                self.state.state = 'half_open'
                self.state.success_count = 0
                
            # If circuit is open, fail fast
            if self.state.state == 'open':
                raise CircuitBreakerOpenException(f"Circuit breaker {self.state.name} is open")
                
            self.state.total_calls += 1
            
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
            
    def _record_success(self):
        """Record successful call."""
        with self.lock:
            if self.state.state == 'half_open':
                self.state.success_count += 1
                if self.state.success_count >= self.state.success_threshold:
                    self.state.state = 'closed'
                    self.state.failure_count = 0
            elif self.state.state == 'closed':\n                self.state.failure_count = max(0, self.state.failure_count - 1)\n                \n    def _record_failure(self):\n        \"\"\"Record failed call.\"\"\"\n        with self.lock:\n            self.state.failure_count += 1\n            self.state.last_failure_time = time.time()\n            \n            if self.state.failure_count >= self.state.failure_threshold:\n                self.state.state = 'open'\n                \n    def get_state(self) -> CircuitBreakerState:\n        \"\"\"Get current circuit breaker state.\"\"\"\n        with self.lock:\n            return self.state\n            \n    def reset(self):\n        \"\"\"Reset circuit breaker to closed state.\"\"\"\n        with self.lock:\n            self.state.state = 'closed'\n            self.state.failure_count = 0\n            self.state.success_count = 0\n            self.state.last_failure_time = None\n\n\nclass CircuitBreakerOpenException(Exception):\n    \"\"\"Exception raised when circuit breaker is open.\"\"\"\n    pass\n\n\nclass GracefulDegradation:\n    \"\"\"Manages graceful degradation strategies.\"\"\"\n    \n    def __init__(self):\n        self.degradation_strategies = {}\n        self.current_degradations = set()\n        self.lock = threading.RLock()\n        \n    def register_degradation_strategy(\n        self,\n        service_name: str,\n        degraded_func: Callable,\n        conditions: Optional[List[Callable[[], bool]]] = None\n    ):\n        \"\"\"Register a degradation strategy for a service.\"\"\"\n        self.degradation_strategies[service_name] = {\n            'degraded_function': degraded_func,\n            'conditions': conditions or [],\n            'activated': False\n        }\n        \n    def should_degrade(self, service_name: str) -> bool:\n        \"\"\"Check if service should operate in degraded mode.\"\"\"\n        if service_name not in self.degradation_strategies:\n            return False\n            \n        strategy = self.degradation_strategies[service_name]\n        \n        # Check all conditions\n        for condition in strategy['conditions']:\n            try:\n                if condition():\n                    with self.lock:\n                        self.current_degradations.add(service_name)\n                    return True\n            except Exception:\n                # If condition check fails, assume degradation needed\n                with self.lock:\n                    self.current_degradations.add(service_name)\n                return True\n                \n        # Remove from current degradations if no conditions met\n        with self.lock:\n            self.current_degradations.discard(service_name)\n            \n        return False\n        \n    def get_degraded_function(self, service_name: str) -> Optional[Callable]:\n        \"\"\"Get degraded function for a service.\"\"\"\n        if service_name in self.degradation_strategies:\n            return self.degradation_strategies[service_name]['degraded_function']\n        return None\n        \n    def get_active_degradations(self) -> List[str]:\n        \"\"\"Get list of currently active degradations.\"\"\"\n        with self.lock:\n            return list(self.current_degradations)\n\n\nclass RobustErrorHandler:\n    \"\"\"Comprehensive error handling system orchestrator.\"\"\"\n    \n    def __init__(self, config: Optional[Dict[str, Any]] = None):\n        self.config = config or {}\n        \n        # Initialize components\n        self.error_recovery_manager = ErrorRecoveryManager()\n        self.circuit_breakers = {}\n        self.graceful_degradation = GracefulDegradation()\n        \n        # Error tracking\n        self.error_counts = defaultdict(int)\n        self.error_patterns = defaultdict(list)\n        self.lock = threading.RLock()\n        \n        # Default retry configurations\n        self.default_retry_configs = {\n            'ConnectionError': RetryConfig(max_attempts=5, base_delay=1.0),\n            'TimeoutError': RetryConfig(max_attempts=3, base_delay=2.0),\n            'RuntimeError': RetryConfig(max_attempts=2, base_delay=0.5)\n        }\n        \n        self._setup_default_recovery_strategies()\n        \n    def _setup_default_recovery_strategies(self):\n        \"\"\"Setup default error recovery strategies.\"\"\"\n        \n        def connection_error_recovery(error_context: ErrorContext) -> Any:\n            \"\"\"Recovery strategy for connection errors.\"\"\"\n            # Implement connection retry logic\n            logging.info(f\"Attempting connection recovery for error: {error_context.error_id}\")\n            time.sleep(1.0)  # Brief pause before retry\n            return None\n            \n        def memory_error_recovery(error_context: ErrorContext) -> Any:\n            \"\"\"Recovery strategy for memory errors.\"\"\"\n            logging.warning(f\"Attempting memory cleanup for error: {error_context.error_id}\")\n            import gc\n            gc.collect()  # Force garbage collection\n            return None\n            \n        def timeout_error_recovery(error_context: ErrorContext) -> Any:\n            \"\"\"Recovery strategy for timeout errors.\"\"\"\n            logging.info(f\"Implementing timeout recovery for error: {error_context.error_id}\")\n            # Could implement request chunking or timeout adjustment\n            return None\n            \n        # Register default strategies\n        self.error_recovery_manager.register_recovery_strategy(\n            ConnectionError, connection_error_recovery, RecoveryStrategy.RETRY\n        )\n        \n        self.error_recovery_manager.register_recovery_strategy(\n            MemoryError, memory_error_recovery, RecoveryStrategy.GRACEFUL_DEGRADE\n        )\n        \n        self.error_recovery_manager.register_recovery_strategy(\n            TimeoutError, timeout_error_recovery, RecoveryStrategy.RETRY\n        )\n        \n    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:\n        \"\"\"Get or create circuit breaker for a service.\"\"\"\n        if name not in self.circuit_breakers:\n            self.circuit_breakers[name] = CircuitBreaker(name, **kwargs)\n        return self.circuit_breakers[name]\n        \n    def handle_error(\n        self,\n        exception: Exception,\n        context_data: Optional[Dict[str, Any]] = None,\n        severity: ErrorSeverity = ErrorSeverity.MEDIUM,\n        recovery_strategy: Optional[RecoveryStrategy] = None\n    ) -> ErrorContext:\n        \"\"\"Handle an error with comprehensive context and recovery.\"\"\"\n        \n        # Generate error context\n        error_context = ErrorContext(\n            error_id=self._generate_error_id(),\n            timestamp=time.time(),\n            exception_type=type(exception).__name__,\n            exception_message=str(exception),\n            traceback_info=traceback.format_exc(),\n            function_name=self._get_calling_function(),\n            module_name=self._get_calling_module(),\n            severity=severity,\n            recovery_strategy=recovery_strategy or RecoveryStrategy.RETRY,\n            context_data=context_data or {}\n        )\n        \n        # Update error tracking\n        with self.lock:\n            self.error_counts[error_context.exception_type] += 1\n            self.error_patterns[error_context.exception_type].append({\n                'timestamp': error_context.timestamp,\n                'function': error_context.function_name,\n                'message': error_context.exception_message\n            })\n            \n        # Add to error history\n        self.error_recovery_manager.add_error_to_history(error_context)\n        \n        # Log error with appropriate level\n        self._log_error(error_context)\n        \n        return error_context\n        \n    def _generate_error_id(self) -> str:\n        \"\"\"Generate unique error identifier.\"\"\"\n        import uuid\n        return str(uuid.uuid4())[:8]\n        \n    def _get_calling_function(self) -> str:\n        \"\"\"Get name of function that called error handler.\"\"\"\n        try:\n            frame = inspect.currentframe()\n            # Go up the call stack to find the actual caller\n            for _ in range(3):  # Skip handle_error and intermediate frames\n                frame = frame.f_back\n                if frame is None:\n                    return \"unknown\"\n            return frame.f_code.co_name\n        except Exception:\n            return \"unknown\"\n            \n    def _get_calling_module(self) -> str:\n        \"\"\"Get name of module that called error handler.\"\"\"\n        try:\n            frame = inspect.currentframe()\n            # Go up the call stack to find the actual caller\n            for _ in range(3):\n                frame = frame.f_back\n                if frame is None:\n                    return \"unknown\"\n            return frame.f_globals.get('__name__', 'unknown')\n        except Exception:\n            return \"unknown\"\n            \n    def _log_error(self, error_context: ErrorContext):\n        \"\"\"Log error with appropriate level based on severity.\"\"\"\n        log_message = (\n            f\"Error [{error_context.error_id}] in {error_context.function_name}: \"\n            f\"{error_context.exception_message}\"\n        )\n        \n        if error_context.severity == ErrorSeverity.CRITICAL:\n            logging.critical(log_message)\n        elif error_context.severity == ErrorSeverity.HIGH:\n            logging.error(log_message)\n        elif error_context.severity == ErrorSeverity.MEDIUM:\n            logging.warning(log_message)\n        else:\n            logging.info(log_message)\n            \n    def get_error_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive error statistics.\"\"\"\n        with self.lock:\n            total_errors = sum(self.error_counts.values())\n            \n            # Calculate error rates\n            error_rates = {}\n            for error_type, count in self.error_counts.items():\n                if total_errors > 0:\n                    error_rates[error_type] = (count / total_errors) * 100\n                else:\n                    error_rates[error_type] = 0.0\n                    \n            # Get recent error patterns (last hour)\n            recent_cutoff = time.time() - 3600\n            recent_patterns = {}\n            \n            for error_type, patterns in self.error_patterns.items():\n                recent = [\n                    p for p in patterns\n                    if p['timestamp'] >= recent_cutoff\n                ]\n                if recent:\n                    recent_patterns[error_type] = {\n                        'count': len(recent),\n                        'functions': list(set(p['function'] for p in recent)),\n                        'messages': list(set(p['message'] for p in recent))\n                    }\n                    \n            return {\n                'total_errors': total_errors,\n                'error_counts': dict(self.error_counts),\n                'error_rates': error_rates,\n                'recent_patterns': recent_patterns,\n                'recovery_stats': self.error_recovery_manager.get_recovery_stats(),\n                'circuit_breaker_states': {\n                    name: {\n                        'state': cb.get_state().state,\n                        'failure_count': cb.get_state().failure_count,\n                        'total_calls': cb.get_state().total_calls\n                    }\n                    for name, cb in self.circuit_breakers.items()\n                },\n                'active_degradations': self.graceful_degradation.get_active_degradations()\n            }\n            \n    def export_error_report(self, filepath: str, include_details: bool = True):\n        \"\"\"Export comprehensive error report.\"\"\"\n        report = {\n            'report_timestamp': time.time(),\n            'statistics': self.get_error_statistics(),\n        }\n        \n        if include_details:\n            # Include recent error history\n            recent_errors = [\n                {\n                    'error_id': error.error_id,\n                    'timestamp': error.timestamp,\n                    'exception_type': error.exception_type,\n                    'exception_message': error.exception_message,\n                    'function_name': error.function_name,\n                    'severity': error.severity.value,\n                    'recovery_attempted': error.recovery_attempted,\n                    'recovery_successful': error.recovery_successful\n                }\n                for error in list(self.error_recovery_manager.error_history)\n                if time.time() - error.timestamp <= 3600  # Last hour\n            ]\n            \n            report['recent_errors'] = recent_errors\n            \n        Path(filepath).parent.mkdir(parents=True, exist_ok=True)\n        \n        with open(filepath, 'w') as f:\n            json.dump(report, f, indent=2, default=str)\n            \n        logging.info(f\"Error report exported to {filepath}\")\n\n\n# Decorators for robust error handling\n\ndef robust_execution(\n    retry_config: Optional[RetryConfig] = None,\n    circuit_breaker_name: Optional[str] = None,\n    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,\n    severity: ErrorSeverity = ErrorSeverity.MEDIUM,\n    fallback_function: Optional[Callable] = None,\n    error_handler: Optional[RobustErrorHandler] = None\n):\n    \"\"\"Decorator for robust function execution with error handling.\"\"\"\n    \n    def decorator(func: Callable) -> Callable:\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            # Use global error handler if not provided\n            handler = error_handler or _get_global_error_handler()\n            \n            # Setup retry configuration\n            retry_conf = retry_config or RetryConfig()\n            \n            # Setup circuit breaker if specified\n            circuit_breaker = None\n            if circuit_breaker_name:\n                circuit_breaker = handler.get_circuit_breaker(circuit_breaker_name)\n                \n            last_exception = None\n            \n            for attempt in range(retry_conf.max_attempts):\n                try:\n                    # Execute with circuit breaker if specified\n                    if circuit_breaker:\n                        return circuit_breaker.call(func, *args, **kwargs)\n                    else:\n                        return func(*args, **kwargs)\n                        \n                except Exception as e:\n                    last_exception = e\n                    \n                    # Handle the error\n                    error_context = handler.handle_error(\n                        exception=e,\n                        context_data={\n                            'function_name': func.__name__,\n                            'attempt': attempt + 1,\n                            'max_attempts': retry_conf.max_attempts,\n                            'args_provided': len(args) > 0,\n                            'kwargs_provided': len(kwargs) > 0\n                        },\n                        severity=severity,\n                        recovery_strategy=recovery_strategy\n                    )\n                    \n                    error_context.retry_count = attempt + 1\n                    \n                    # If this is the last attempt, don't retry\n                    if attempt == retry_conf.max_attempts - 1:\n                        break\n                        \n                    # Attempt recovery\n                    recovery_success, recovery_result = handler.error_recovery_manager.attempt_recovery(\n                        error_context\n                    )\n                    \n                    if recovery_success and recovery_result is not None:\n                        return recovery_result\n                        \n                    # Wait before retry\n                    delay = retry_conf.get_delay(attempt)\n                    if delay > 0:\n                        time.sleep(delay)\n                        \n            # All retries exhausted, try fallback if available\n            if fallback_function:\n                try:\n                    logging.info(f\"Attempting fallback for {func.__name__}\")\n                    return fallback_function(*args, **kwargs)\n                except Exception as fallback_error:\n                    logging.error(f\"Fallback also failed: {fallback_error}\")\n                    \n            # No recovery possible, re-raise last exception\n            raise last_exception\n            \n        return wrapper\n    return decorator\n\n\ndef circuit_breaker(\n    name: str,\n    failure_threshold: int = 5,\n    recovery_timeout: float = 60.0,\n    error_handler: Optional[RobustErrorHandler] = None\n):\n    \"\"\"Decorator for circuit breaker protection.\"\"\"\n    \n    def decorator(func: Callable) -> Callable:\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            handler = error_handler or _get_global_error_handler()\n            cb = handler.get_circuit_breaker(\n                name,\n                failure_threshold=failure_threshold,\n                recovery_timeout=recovery_timeout\n            )\n            \n            return cb.call(func, *args, **kwargs)\n            \n        return wrapper\n    return decorator\n\n\ndef graceful_degradation(\n    service_name: str,\n    degraded_function: Callable,\n    conditions: Optional[List[Callable[[], bool]]] = None,\n    error_handler: Optional[RobustErrorHandler] = None\n):\n    \"\"\"Decorator for graceful degradation.\"\"\"\n    \n    def decorator(func: Callable) -> Callable:\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            handler = error_handler or _get_global_error_handler()\n            \n            # Register degradation strategy if not already registered\n            handler.graceful_degradation.register_degradation_strategy(\n                service_name, degraded_function, conditions\n            )\n            \n            # Check if should operate in degraded mode\n            if handler.graceful_degradation.should_degrade(service_name):\n                logging.info(f\"Operating in degraded mode for {service_name}\")\n                return degraded_function(*args, **kwargs)\n            else:\n                return func(*args, **kwargs)\n                \n        return wrapper\n    return decorator\n\n\n# Global error handler instance\n_global_error_handler = None\n\n\ndef _get_global_error_handler() -> RobustErrorHandler:\n    \"\"\"Get or create global error handler instance.\"\"\"\n    global _global_error_handler\n    if _global_error_handler is None:\n        _global_error_handler = RobustErrorHandler()\n    return _global_error_handler\n\n\ndef set_global_error_handler(handler: RobustErrorHandler):\n    \"\"\"Set global error handler instance.\"\"\"\n    global _global_error_handler\n    _global_error_handler = handler\n\n\n# Export error handling components\n__all__ = [\n    'ErrorSeverity',\n    'RecoveryStrategy', \n    'ErrorContext',\n    'CircuitBreakerState',\n    'RetryConfig',\n    'ErrorRecoveryManager',\n    'CircuitBreaker',\n    'CircuitBreakerOpenException',\n    'GracefulDegradation',\n    'RobustErrorHandler',\n    'robust_execution',\n    'circuit_breaker',\n    'graceful_degradation',\n    'set_global_error_handler'\n]