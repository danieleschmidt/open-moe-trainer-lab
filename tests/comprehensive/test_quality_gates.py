"""
Quality gates and acceptance criteria tests.
Ensures all components meet production readiness standards.
"""

import pytest
import torch
import numpy as np
import time
import os
import tempfile
from typing import Dict, Any, List, Tuple

from moe_lab import (
    MoEModel,
    MoETrainer,
    RouterMonitor,
    RouterAnalyzer,
    MoECostAnalyzer
)
from moe_lab.validation import (
    MoEModelValidator,
    RoutingValidator,
    TrainingValidator
)
from moe_lab.optimization.efficient_training import EfficientMoETrainer
from moe_lab.optimization.auto_scaling import AutoScalingMoEModel


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def check(self, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        """Check quality gate. Returns (passed, message, metrics)."""
        raise NotImplementedError


class ModelArchitectureGate(QualityGate):
    """Quality gate for model architecture."""
    
    def __init__(self):
        super().__init__(
            "Model Architecture",
            "Validates model architecture and parameter correctness"
        )
        
    def check(self, model: MoEModel, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        validator = MoEModelValidator(model)
        result = validator.validate_model()
        
        metrics = {
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'total_parameters': result.metrics.get('total_parameters', 0)
        }
        
        if not result.is_valid:
            return False, f"Model validation failed: {result.errors[0] if result.errors else 'Unknown error'}", metrics
            
        if len(result.warnings) > 5:
            return False, f"Too many warnings ({len(result.warnings)}), model may be problematic", metrics
            
        return True, "Model architecture validation passed", metrics


class RoutingQualityGate(QualityGate):
    """Quality gate for routing behavior."""
    
    def __init__(self):
        super().__init__(
            "Routing Quality",
            "Validates routing efficiency and load balancing"
        )
        
    def check(self, model: MoEModel, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        validator = RoutingValidator(model)
        result = validator.validate_routing_behavior(num_samples=100)
        
        metrics = {
            'load_balance_score': result.load_balance_score,
            'routing_efficiency': result.routing_efficiency,
            'expert_specialization_score': result.expert_specialization_score,
            'error_count': len(result.errors)
        }
        
        # Quality thresholds
        if not result.is_valid:
            return False, f"Routing validation failed: {result.errors[0] if result.errors else 'Unknown error'}", metrics
            
        if result.load_balance_score < 0.3:
            return False, f"Poor load balancing: {result.load_balance_score:.3f} < 0.3", metrics
            
        if result.routing_efficiency < 0.2:
            return False, f"Poor routing efficiency: {result.routing_efficiency:.3f} < 0.2", metrics
            
        return True, "Routing quality validation passed", metrics


class TrainingStabilityGate(QualityGate):
    """Quality gate for training stability."""
    
    def __init__(self):
        super().__init__(
            "Training Stability",
            "Validates training convergence and stability"
        )
        
    def check(self, model: MoEModel, trainer: MoETrainer = None, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        if trainer is None:
            trainer = MoETrainer(model, output_dir=tempfile.mkdtemp())
            
        validator = TrainingValidator(model, trainer)
        
        # Test training setup
        setup_result = validator.validate_training_setup()
        if not setup_result.is_valid:
            return False, f"Training setup invalid: {setup_result.errors[0]}", {}
            
        # Test convergence
        convergence_result = validator.validate_training_convergence(
            num_steps=30,
            batch_size=4,
            learning_rate=1e-3
        )
        
        metrics = {
            'convergence_score': convergence_result.convergence_score,
            'stability_score': convergence_result.stability_score,
            'efficiency_score': convergence_result.efficiency_score
        }
        
        # Quality thresholds
        if convergence_result.convergence_score < 0.1:
            return False, f"Poor convergence: {convergence_result.convergence_score:.3f} < 0.1", metrics
            
        if convergence_result.stability_score < 0.3:
            return False, f"Poor stability: {convergence_result.stability_score:.3f} < 0.3", metrics
            
        return True, "Training stability validation passed", metrics


class PerformanceGate(QualityGate):
    """Quality gate for performance requirements."""
    
    def __init__(self):
        super().__init__(
            "Performance",
            "Validates performance meets minimum requirements"
        )
        
    def check(self, model: MoEModel, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        device = next(model.parameters()).device
        model.eval()
        
        # Performance benchmarks
        batch_sizes = [1, 4, 8]
        sequence_length = 64
        num_iterations = 10
        
        performance_metrics = {}
        
        for batch_size in batch_sizes:
            test_input = torch.randint(0, 1000, (batch_size, sequence_length), device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    model(test_input)
                    
            # Benchmark
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    outputs = model(test_input)
                    
            total_time = time.time() - start_time
            
            tokens_per_sec = (batch_size * sequence_length * num_iterations) / total_time
            latency_ms = (total_time / num_iterations) * 1000
            
            performance_metrics[f'batch_{batch_size}'] = {
                'tokens_per_sec': tokens_per_sec,
                'latency_ms': latency_ms
            }
            
        # Performance requirements
        min_throughput = 50 if torch.cuda.is_available() else 10
        max_latency = 500  # ms
        
        batch_1_perf = performance_metrics['batch_1']
        
        if batch_1_perf['tokens_per_sec'] < min_throughput:
            return False, f"Throughput too low: {batch_1_perf['tokens_per_sec']:.1f} < {min_throughput}", performance_metrics
            
        if batch_1_perf['latency_ms'] > max_latency:
            return False, f"Latency too high: {batch_1_perf['latency_ms']:.1f}ms > {max_latency}ms", performance_metrics
            
        return True, "Performance requirements met", performance_metrics


class MemoryEfficiencyGate(QualityGate):
    """Quality gate for memory efficiency."""
    
    def __init__(self):
        super().__init__(
            "Memory Efficiency",
            "Validates memory usage is within acceptable bounds"
        )
        
    def check(self, model: MoEModel, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        if not torch.cuda.is_available():
            return True, "Memory check skipped (no CUDA)", {}
            
        device = next(model.parameters()).device
        
        # Clear cache and measure baseline
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()
        
        # Test memory usage with different batch sizes
        batch_sizes = [1, 4, 8, 16]
        memory_metrics = {}
        
        model.train()
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
            
            try:
                test_input = torch.randint(0, 1000, (batch_size, 64), device=device)
                
                # Forward pass
                outputs = model(test_input)
                forward_memory = torch.cuda.memory_allocated()
                
                # Backward pass
                logits = model.lm_head(outputs.last_hidden_state)
                targets = test_input[:, 1:].contiguous()
                logits = logits[:, :-1].contiguous()
                
                loss = torch.nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                loss.backward()
                
                backward_memory = torch.cuda.memory_allocated()
                
                memory_metrics[f'batch_{batch_size}'] = {
                    'forward_memory_mb': (forward_memory - start_memory) / (1024**2),
                    'backward_memory_mb': (backward_memory - forward_memory) / (1024**2),
                    'total_memory_mb': (backward_memory - start_memory) / (1024**2)
                }
                
                # Clear gradients
                model.zero_grad()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    memory_metrics[f'batch_{batch_size}'] = {
                        'oom': True,
                        'forward_memory_mb': 0,
                        'backward_memory_mb': 0,
                        'total_memory_mb': 0
                    }
                else:
                    raise e
                    
        # Memory requirements
        max_memory_per_token = 2.0  # MB per token for batch_size=1
        
        if 'batch_1' in memory_metrics and not memory_metrics['batch_1'].get('oom', False):
            memory_per_token = memory_metrics['batch_1']['total_memory_mb'] / 64
            
            if memory_per_token > max_memory_per_token:
                return False, f"Memory per token too high: {memory_per_token:.2f}MB > {max_memory_per_token}MB", memory_metrics
                
        # Check for OOM at reasonable batch sizes
        if memory_metrics.get('batch_4', {}).get('oom', False):
            return False, "Out of memory at batch size 4", memory_metrics
            
        return True, "Memory efficiency requirements met", memory_metrics


class SecurityGate(QualityGate):
    """Quality gate for security requirements."""
    
    def __init__(self):
        super().__init__(
            "Security",
            "Validates security requirements and safe practices"
        )
        
    def check(self, model: MoEModel, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        metrics = {
            'parameter_validation': True,
            'input_validation': True,
            'gradient_security': True
        }
        
        device = next(model.parameters()).device
        
        # Test parameter validation
        try:
            # Test with extreme values
            extreme_input = torch.full((2, 32), 999999, device=device)
            with torch.no_grad():
                outputs = model(extreme_input)
                
            # Check for NaN/Inf in outputs
            if torch.isnan(outputs.last_hidden_state).any() or torch.isinf(outputs.last_hidden_state).any():
                metrics['parameter_validation'] = False
                return False, "Model produces NaN/Inf with extreme inputs", metrics
                
        except Exception as e:
            metrics['parameter_validation'] = False
            return False, f"Model fails with extreme inputs: {str(e)}", metrics
            
        # Test input validation
        try:
            # Test with invalid shapes
            invalid_input = torch.randint(0, 1000, (0, 32), device=device)
            with torch.no_grad():
                outputs = model(invalid_input)
                
        except Exception:
            # Expected to fail - this is good
            pass
        else:
            metrics['input_validation'] = False
            return False, "Model accepts invalid input shapes", metrics
            
        # Test gradient security
        try:
            model.train()
            test_input = torch.randint(0, 1000, (2, 32), device=device)
            
            outputs = model(test_input)
            logits = model.lm_head(outputs.last_hidden_state)
            
            # Check for gradient explosion indicators
            loss = logits.sum()
            loss.backward()
            
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_grad_norm > 1000:
                metrics['gradient_security'] = False
                return False, f"Potential gradient explosion: norm={total_grad_norm:.2f}", metrics
                
        except Exception as e:
            metrics['gradient_security'] = False
            return False, f"Gradient computation failed: {str(e)}", metrics
            
        return True, "Security requirements met", metrics


class ReproducibilityGate(QualityGate):
    """Quality gate for reproducibility."""
    
    def __init__(self):
        super().__init__(
            "Reproducibility",
            "Validates results are reproducible across runs"
        )
        
    def check(self, model: MoEModel, **kwargs) -> Tuple[bool, str, Dict[str, Any]]:
        device = next(model.parameters()).device
        
        # Test reproducibility with fixed seeds
        test_input = torch.randint(0, 1000, (4, 32), device=device)
        
        results = []
        
        for run in range(3):
            # Set seed
            torch.manual_seed(42 + run)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + run)
                
            model.eval()
            with torch.no_grad():
                outputs = model(test_input)
                results.append(outputs.last_hidden_state.cpu().numpy())
                
        # Check consistency across runs
        reproducibility_scores = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                # Compute correlation
                corr = np.corrcoef(results[i].flatten(), results[j].flatten())[0, 1]
                reproducibility_scores.append(corr)
                
        avg_reproducibility = np.mean(reproducibility_scores)
        
        metrics = {
            'avg_reproducibility': avg_reproducibility,
            'min_reproducibility': np.min(reproducibility_scores),
            'reproducibility_scores': reproducibility_scores
        }
        
        # Reproducibility threshold
        if avg_reproducibility < 0.95:
            return False, f"Poor reproducibility: {avg_reproducibility:.3f} < 0.95", metrics
            
        return True, "Reproducibility requirements met", metrics


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_model(device):
    """Create test model."""
    model = MoEModel(
        vocab_size=1000,
        hidden_size=256,
        num_experts=8,
        experts_per_token=2,
        num_layers=4,
        num_attention_heads=8
    )
    return model.to(device)


class TestQualityGates:
    """Test suite for quality gates."""
    
    def test_model_architecture_gate(self, test_model):
        """Test model architecture quality gate."""
        gate = ModelArchitectureGate()
        passed, message, metrics = gate.check(model=test_model)
        
        assert passed, f"Architecture gate failed: {message}"
        assert metrics['is_valid']
        assert metrics['total_parameters'] > 0
        assert metrics['error_count'] == 0
        
    def test_routing_quality_gate(self, test_model):
        """Test routing quality gate."""
        gate = RoutingQualityGate()
        passed, message, metrics = gate.check(model=test_model)
        
        assert passed, f"Routing gate failed: {message}"
        assert metrics['load_balance_score'] >= 0.3
        assert metrics['routing_efficiency'] >= 0.2
        assert metrics['error_count'] == 0
        
    def test_training_stability_gate(self, test_model):
        """Test training stability gate."""
        gate = TrainingStabilityGate()
        passed, message, metrics = gate.check(model=test_model)
        
        assert passed, f"Training stability gate failed: {message}"
        assert metrics['convergence_score'] >= 0.1
        assert metrics['stability_score'] >= 0.3
        
    def test_performance_gate(self, test_model):
        """Test performance gate."""
        gate = PerformanceGate()
        passed, message, metrics = gate.check(model=test_model)
        
        assert passed, f"Performance gate failed: {message}"
        
        # Check metrics exist
        assert 'batch_1' in metrics
        assert 'tokens_per_sec' in metrics['batch_1']
        assert 'latency_ms' in metrics['batch_1']
        
        # Performance should be reasonable
        assert metrics['batch_1']['tokens_per_sec'] > 0
        assert metrics['batch_1']['latency_ms'] > 0
        
    def test_memory_efficiency_gate(self, test_model):
        """Test memory efficiency gate."""
        gate = MemoryEfficiencyGate()
        passed, message, metrics = gate.check(model=test_model)
        
        assert passed, f"Memory efficiency gate failed: {message}"
        
        # Should not OOM at reasonable batch sizes
        if torch.cuda.is_available():
            assert not metrics.get('batch_4', {}).get('oom', False)
            
    def test_security_gate(self, test_model):
        """Test security gate."""
        gate = SecurityGate()
        passed, message, metrics = gate.check(model=test_model)
        
        assert passed, f"Security gate failed: {message}"
        assert metrics['parameter_validation']
        assert metrics['input_validation']
        assert metrics['gradient_security']
        
    def test_reproducibility_gate(self, test_model):
        """Test reproducibility gate."""
        gate = ReproducibilityGate()
        passed, message, metrics = gate.check(model=test_model)
        
        assert passed, f"Reproducibility gate failed: {message}"
        assert metrics['avg_reproducibility'] >= 0.95
        
    def test_all_quality_gates(self, test_model):
        """Test all quality gates together."""
        gates = [
            ModelArchitectureGate(),
            RoutingQualityGate(),
            TrainingStabilityGate(),
            PerformanceGate(),
            MemoryEfficiencyGate(),
            SecurityGate(),
            ReproducibilityGate()
        ]
        
        results = {}
        
        for gate in gates:
            passed, message, metrics = gate.check(model=test_model)
            
            results[gate.name] = {
                'passed': passed,
                'message': message,
                'metrics': metrics
            }
            
            # All gates must pass
            assert passed, f"Quality gate '{gate.name}' failed: {message}"
            
        # Summary report
        total_gates = len(gates)
        passed_gates = sum(1 for r in results.values() if r['passed'])
        
        assert passed_gates == total_gates, f"Only {passed_gates}/{total_gates} quality gates passed"
        
        # Collect overall metrics
        overall_metrics = {
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'gate_results': results
        }
        
        return overall_metrics
        
    def test_quality_gates_with_different_models(self, device):
        """Test quality gates with different model configurations."""
        configs = [
            {'num_experts': 4, 'experts_per_token': 2, 'hidden_size': 128},
            {'num_experts': 8, 'experts_per_token': 2, 'hidden_size': 256},
            {'num_experts': 6, 'experts_per_token': 3, 'hidden_size': 192}
        ]
        
        critical_gates = [
            ModelArchitectureGate(),
            RoutingQualityGate(),
            SecurityGate()
        ]
        
        for i, config in enumerate(configs):
            model = MoEModel(
                vocab_size=1000,
                num_layers=3,
                **config
            ).to(device)
            
            for gate in critical_gates:
                passed, message, metrics = gate.check(model=model)
                assert passed, f"Config {i} failed gate '{gate.name}': {message}"
                
    @pytest.mark.slow
    def test_quality_gates_under_stress(self, device):
        """Test quality gates under stress conditions."""
        # Large model for stress testing
        model = MoEModel(
            vocab_size=2000,
            hidden_size=512,
            num_experts=16,
            experts_per_token=4,
            num_layers=6
        ).to(device)
        
        # Critical gates that must pass even under stress
        stress_gates = [
            ModelArchitectureGate(),
            SecurityGate(),
            MemoryEfficiencyGate()
        ]
        
        for gate in stress_gates:
            passed, message, metrics = gate.check(model=model)
            
            # Some gates may have different thresholds under stress
            if gate.name == "Memory Efficiency" and not passed:
                # Allow memory gate to fail for very large models
                continue
                
            assert passed, f"Stress test failed for gate '{gate.name}': {message}"


class TestProductionReadiness:
    """Test production readiness criteria."""
    
    def test_production_checklist(self, test_model):
        """Comprehensive production readiness checklist."""
        checklist = {
            'model_validation': False,
            'routing_quality': False,
            'training_stability': False,
            'performance_requirements': False,
            'memory_efficiency': False,
            'security_compliance': False,
            'reproducibility': False,
            'error_handling': False,
            'monitoring_integration': False
        }
        
        # Model validation
        validator = MoEModelValidator(test_model)
        result = validator.validate_model()
        checklist['model_validation'] = result.is_valid and len(result.errors) == 0
        
        # Routing quality
        routing_validator = RoutingValidator(test_model)
        routing_result = routing_validator.validate_routing_behavior(num_samples=50)
        checklist['routing_quality'] = (
            routing_result.load_balance_score >= 0.3 and
            routing_result.routing_efficiency >= 0.2
        )
        
        # Training stability
        trainer = MoETrainer(test_model, output_dir=tempfile.mkdtemp())
        training_validator = TrainingValidator(test_model, trainer)
        convergence_result = training_validator.validate_training_convergence(num_steps=20)
        checklist['training_stability'] = convergence_result.stability_score >= 0.3
        
        # Performance requirements
        device = next(test_model.parameters()).device
        test_input = torch.randint(0, 1000, (1, 64), device=device)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                test_model(test_input)
        avg_time = (time.time() - start_time) / 10
        
        checklist['performance_requirements'] = avg_time < 0.5  # 500ms
        
        # Memory efficiency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                test_model(test_input)
                
            memory_used = (torch.cuda.memory_allocated() - start_memory) / (1024**2)
            checklist['memory_efficiency'] = memory_used < 100  # 100MB for small batch
        else:
            checklist['memory_efficiency'] = True
            
        # Security compliance
        try:
            extreme_input = torch.full((1, 64), 999999, device=device)
            with torch.no_grad():
                outputs = test_model(extreme_input)
            checklist['security_compliance'] = not (
                torch.isnan(outputs.last_hidden_state).any() or 
                torch.isinf(outputs.last_hidden_state).any()
            )
        except:
            checklist['security_compliance'] = False
            
        # Reproducibility
        torch.manual_seed(42)
        output1 = test_model(test_input)
        torch.manual_seed(42)
        output2 = test_model(test_input)
        
        diff = torch.abs(output1.last_hidden_state - output2.last_hidden_state).max().item()
        checklist['reproducibility'] = diff < 1e-6
        
        # Error handling
        try:
            invalid_input = torch.randint(0, 1000, (0, 64), device=device)
            test_model(invalid_input)
            checklist['error_handling'] = False  # Should have failed
        except:
            checklist['error_handling'] = True  # Expected failure
            
        # Monitoring integration
        monitor = RouterMonitor(test_model)
        try:
            with monitor.track():
                with torch.no_grad():
                    test_model(test_input)
            stats = monitor.get_stats()
            checklist['monitoring_integration'] = stats.entropy >= 0
        except:
            checklist['monitoring_integration'] = False
            
        # Check all criteria met
        failed_criteria = [k for k, v in checklist.items() if not v]
        
        assert len(failed_criteria) == 0, f"Production readiness failed for: {failed_criteria}"
        
        return checklist


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])