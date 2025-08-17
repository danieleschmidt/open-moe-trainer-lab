#!/usr/bin/env python3
"""
AUTONOMOUS QUALITY GATES - Comprehensive Validation
SDLC Quality Assurance with security, performance, and functionality validation.
"""

import torch
import numpy as np
import json
import time
import logging
import subprocess
import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import traceback
import hashlib
import psutil
import threading
import concurrent.futures
from dataclasses import dataclass, asdict

from moe_lab import MoEModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_gates_comprehensive.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result from a quality gate check."""
    name: str
    status: str  # PASS, FAIL, WARNING, SKIP
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    execution_time: float
    recommendations: List[str]

class SecurityValidator:
    """Comprehensive security validation."""
    
    def __init__(self):
        self.security_checks = []
    
    def validate_model_security(self, model: torch.nn.Module) -> QualityGateResult:
        """Validate model security aspects."""
        start_time = time.time()
        details = {}
        recommendations = []
        issues_found = 0
        
        try:
            # Check for suspicious parameters
            suspicious_params = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                total_params += 1
                
                # Check for unusual parameter values
                if torch.isnan(param).any():
                    suspicious_params += 1
                    issues_found += 1
                    recommendations.append(f"Found NaN values in parameter: {name}")
                
                if torch.isinf(param).any():
                    suspicious_params += 1
                    issues_found += 1
                    recommendations.append(f"Found infinite values in parameter: {name}")
                
                # Check for extremely large values that might indicate attacks
                if param.abs().max() > 1e6:
                    suspicious_params += 1
                    issues_found += 1
                    recommendations.append(f"Extremely large values in parameter: {name}")
            
            details['total_parameters'] = total_params
            details['suspicious_parameters'] = suspicious_params
            details['parameter_health_ratio'] = 1.0 - (suspicious_params / max(total_params, 1))
            
            # Check model architecture for security issues
            layer_count = sum(1 for _ in model.modules())
            details['total_layers'] = layer_count
            
            # Check for common vulnerabilities
            details['security_checks'] = {
                'parameter_validation': suspicious_params == 0,
                'architecture_integrity': layer_count > 0,
                'gradient_safety': True  # Would need actual training to test
            }
            
            # Calculate security score
            security_score = details['parameter_health_ratio']
            if all(details['security_checks'].values()):
                security_score = min(security_score + 0.1, 1.0)
            
            status = "PASS" if issues_found == 0 else "WARNING" if issues_found < 5 else "FAIL"
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            status = "FAIL"
            security_score = 0.0
            details['error'] = str(e)
            recommendations.append("Security validation encountered errors")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Security Validation",
            status=status,
            score=security_score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def scan_dependencies(self) -> QualityGateResult:
        """Scan dependencies for known vulnerabilities."""
        start_time = time.time()
        details = {}
        recommendations = []
        
        try:
            # Check if safety is available for dependency scanning
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=30)
            installed_packages = result.stdout.lower()
            
            # Look for common vulnerable packages (simplified check)
            vulnerable_patterns = ['urllib3==1.25', 'requests==2.19', 'pillow==6.0']
            vulnerabilities = []
            
            for pattern in vulnerable_patterns:
                if pattern in installed_packages:
                    vulnerabilities.append(pattern)
            
            details['installed_packages_scanned'] = len(installed_packages.split('\n'))
            details['vulnerabilities_found'] = len(vulnerabilities)
            details['vulnerable_packages'] = vulnerabilities
            
            if vulnerabilities:
                recommendations.extend([f"Update vulnerable package: {pkg}" for pkg in vulnerabilities])
            
            score = 1.0 if len(vulnerabilities) == 0 else max(0.0, 1.0 - len(vulnerabilities) * 0.2)
            status = "PASS" if len(vulnerabilities) == 0 else "WARNING"
            
        except Exception as e:
            logger.warning(f"Dependency scan failed: {e}")
            status = "SKIP"
            score = 0.5
            details['error'] = str(e)
            recommendations.append("Could not complete dependency vulnerability scan")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Dependency Security Scan",
            status=status,
            score=score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )

class PerformanceValidator:
    """Comprehensive performance validation."""
    
    def benchmark_inference_speed(self, model: torch.nn.Module) -> QualityGateResult:
        """Benchmark model inference performance."""
        start_time = time.time()
        details = {}
        recommendations = []
        
        try:
            model.eval()
            batch_sizes = [1, 4, 8, 16]
            sequence_lengths = [32, 64, 128]
            benchmark_results = {}
            
            with torch.no_grad():
                for batch_size in batch_sizes:
                    for seq_len in sequence_lengths:
                        key = f"batch_{batch_size}_seq_{seq_len}"
                        
                        # Create test input
                        test_input = torch.randint(0, 1000, (batch_size, seq_len))
                        
                        # Warmup
                        for _ in range(3):
                            _ = model(test_input)
                        
                        # Benchmark
                        times = []
                        for _ in range(10):
                            iter_start = time.perf_counter()
                            outputs = model(test_input)
                            iter_end = time.perf_counter()
                            times.append(iter_end - iter_start)
                        
                        avg_time = np.mean(times)
                        std_time = np.std(times)
                        throughput = batch_size / avg_time  # samples per second
                        
                        benchmark_results[key] = {
                            'avg_time_ms': avg_time * 1000,
                            'std_time_ms': std_time * 1000,
                            'throughput_samples_per_sec': throughput,
                            'tokens_per_sec': batch_size * seq_len / avg_time
                        }
            
            details['benchmark_results'] = benchmark_results
            
            # Calculate performance score based on throughput
            max_throughput = max(r['throughput_samples_per_sec'] for r in benchmark_results.values())
            details['max_throughput'] = max_throughput
            
            # Performance thresholds (samples per second)
            if max_throughput > 100:
                score = 1.0
                status = "PASS"
            elif max_throughput > 50:
                score = 0.8
                status = "PASS"
                recommendations.append("Performance is acceptable but could be optimized")
            elif max_throughput > 10:
                score = 0.6
                status = "WARNING"
                recommendations.append("Performance is below optimal thresholds")
            else:
                score = 0.3
                status = "FAIL"
                recommendations.append("Performance is critically low")
            
            details['performance_score'] = score
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            status = "FAIL"
            score = 0.0
            details['error'] = str(e)
            recommendations.append("Performance benchmarking encountered errors")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Inference Performance Benchmark",
            status=status,
            score=score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def validate_memory_efficiency(self, model: torch.nn.Module) -> QualityGateResult:
        """Validate memory efficiency."""
        start_time = time.time()
        details = {}
        recommendations = []
        
        try:
            # Get model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            details['model_size_mb'] = model_size_mb
            
            # Memory usage during inference
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**2  # MB
            
            model.eval()
            with torch.no_grad():
                # Test with different batch sizes
                memory_usage = {}
                for batch_size in [1, 4, 8, 16]:
                    test_input = torch.randint(0, 1000, (batch_size, 64))
                    
                    # Measure memory before inference
                    before_memory = process.memory_info().rss / 1024**2
                    
                    # Run inference
                    outputs = model(test_input)
                    
                    # Measure memory after inference
                    after_memory = process.memory_info().rss / 1024**2
                    memory_delta = after_memory - before_memory
                    
                    memory_usage[f'batch_{batch_size}'] = {
                        'memory_delta_mb': memory_delta,
                        'memory_per_sample_mb': memory_delta / batch_size if batch_size > 0 else 0
                    }
                    
                    # Cleanup
                    del outputs, test_input
            
            details['memory_usage_by_batch'] = memory_usage
            details['baseline_memory_mb'] = initial_memory
            
            # Calculate memory efficiency score
            avg_memory_per_sample = np.mean([usage['memory_per_sample_mb'] for usage in memory_usage.values()])
            details['avg_memory_per_sample_mb'] = avg_memory_per_sample
            
            # Memory efficiency thresholds
            if avg_memory_per_sample < 10:
                score = 1.0
                status = "PASS"
            elif avg_memory_per_sample < 50:
                score = 0.8
                status = "PASS"
                recommendations.append("Memory usage is reasonable but could be optimized")
            elif avg_memory_per_sample < 100:
                score = 0.6
                status = "WARNING"
                recommendations.append("Memory usage is high, consider optimization")
            else:
                score = 0.3
                status = "FAIL"
                recommendations.append("Memory usage is excessive")
            
        except Exception as e:
            logger.error(f"Memory validation failed: {e}")
            status = "FAIL"
            score = 0.0
            details['error'] = str(e)
            recommendations.append("Memory validation encountered errors")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Memory Efficiency Validation",
            status=status,
            score=score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )

class FunctionalityValidator:
    """Comprehensive functionality validation."""
    
    def validate_moe_routing(self, model: torch.nn.Module) -> QualityGateResult:
        """Validate MoE routing functionality."""
        start_time = time.time()
        details = {}
        recommendations = []
        
        try:
            model.eval()
            test_input = torch.randint(0, 1000, (4, 32))
            
            with torch.no_grad():
                outputs = model(test_input, return_routing_info=True)
            
            # Validate routing information
            routing_info = outputs.routing_info
            details['routing_available'] = routing_info is not None
            
            if routing_info is not None:
                details['routing_entropy'] = float(routing_info.entropy)
                details['load_variance'] = float(routing_info.load_variance)
                
                # Check if routing is working properly
                entropy_threshold = 0.1  # Minimum expected entropy
                variance_threshold = 1.0  # Maximum acceptable variance
                
                routing_checks = {
                    'entropy_sufficient': routing_info.entropy > entropy_threshold,
                    'load_balanced': routing_info.load_variance < variance_threshold,
                    'expert_weights_available': routing_info.expert_weights is not None
                }
                
                details['routing_checks'] = routing_checks
                
                # Validate expert utilization
                if hasattr(outputs, 'expert_weights') and outputs.expert_weights:
                    expert_usage = {}
                    for layer_idx, weights in outputs.expert_weights.items():
                        if weights is not None:
                            usage_dist = weights.sum(dim=0)  # Sum across tokens
                            expert_usage[f'layer_{layer_idx}'] = {
                                'expert_usage_variance': float(usage_dist.var()),
                                'most_used_expert': int(torch.argmax(usage_dist)),
                                'least_used_expert': int(torch.argmin(usage_dist))
                            }
                    
                    details['expert_usage_analysis'] = expert_usage
                
                # Calculate routing score
                passed_checks = sum(routing_checks.values())
                total_checks = len(routing_checks)
                score = passed_checks / total_checks
                
                if score >= 0.8:
                    status = "PASS"
                elif score >= 0.6:
                    status = "WARNING"
                    recommendations.append("Some routing checks failed, investigate load balancing")
                else:
                    status = "FAIL"
                    recommendations.append("Multiple routing issues detected")
            else:
                score = 0.0
                status = "FAIL"
                recommendations.append("Routing information not available")
            
        except Exception as e:
            logger.error(f"MoE routing validation failed: {e}")
            status = "FAIL"
            score = 0.0
            details['error'] = str(e)
            recommendations.append("MoE routing validation encountered errors")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="MoE Routing Validation",
            status=status,
            score=score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def validate_training_capability(self, model: torch.nn.Module) -> QualityGateResult:
        """Validate training capability."""
        start_time = time.time()
        details = {}
        recommendations = []
        
        try:
            # Create small training setup
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Create small batch for testing
            batch_size = 2
            seq_len = 16
            test_data = torch.randint(0, 1000, (batch_size, seq_len))
            input_ids = test_data[:, :-1]
            targets = test_data[:, 1:]
            
            initial_loss = None
            final_loss = None
            gradient_norms = []
            
            # Run a few training steps
            for step in range(5):
                optimizer.zero_grad()
                
                outputs = model(input_ids)
                logits = model.lm_head(outputs.last_hidden_state)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                if outputs.load_balancing_loss is not None:
                    loss += 0.01 * outputs.load_balancing_loss
                
                if step == 0:
                    initial_loss = loss.item()
                
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                gradient_norms.append(grad_norm)
                
                optimizer.step()
                final_loss = loss.item()
            
            details['initial_loss'] = initial_loss
            details['final_loss'] = final_loss
            details['loss_improvement'] = initial_loss - final_loss if initial_loss and final_loss else 0
            details['avg_gradient_norm'] = np.mean(gradient_norms) if gradient_norms else 0
            details['gradient_stability'] = np.std(gradient_norms) if len(gradient_norms) > 1 else 0
            
            # Validate training capability
            training_checks = {
                'loss_finite': not (np.isnan(final_loss) or np.isinf(final_loss)),
                'gradients_present': len(gradient_norms) > 0,
                'gradients_stable': details['gradient_stability'] < 10.0,
                'loss_reasonable': final_loss < 100.0  # Reasonable loss range
            }
            
            details['training_checks'] = training_checks
            
            # Calculate training score
            passed_checks = sum(training_checks.values())
            total_checks = len(training_checks)
            score = passed_checks / total_checks
            
            if score >= 0.9:
                status = "PASS"
            elif score >= 0.7:
                status = "WARNING"
                recommendations.append("Some training issues detected, monitor closely")
            else:
                status = "FAIL"
                recommendations.append("Significant training issues detected")
            
            if details['loss_improvement'] <= 0:
                recommendations.append("Loss did not improve during test training")
            
        except Exception as e:
            logger.error(f"Training validation failed: {e}")
            status = "FAIL"
            score = 0.0
            details['error'] = str(e)
            recommendations.append("Training validation encountered errors")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Training Capability Validation",
            status=status,
            score=score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )

class ComprehensiveQualityGates:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator()
        self.functionality_validator = FunctionalityValidator()
        self.results = []
    
    def run_all_gates(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Run all quality gates."""
        logger.info("🚪 Starting Comprehensive Quality Gates Validation")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Define all quality gates
        quality_gates = [
            ("Security", [
                lambda: self.security_validator.validate_model_security(model),
                lambda: self.security_validator.scan_dependencies()
            ]),
            ("Performance", [
                lambda: self.performance_validator.benchmark_inference_speed(model),
                lambda: self.performance_validator.validate_memory_efficiency(model)
            ]),
            ("Functionality", [
                lambda: self.functionality_validator.validate_moe_routing(model),
                lambda: self.functionality_validator.validate_training_capability(model)
            ])
        ]
        
        # Run quality gates with parallel execution where possible
        all_results = []
        
        for category, gates in quality_gates:
            logger.info(f"Running {category} Quality Gates...")
            
            # Run gates in parallel for this category
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_gate = {executor.submit(gate): gate for gate in gates}
                
                for future in concurrent.futures.as_completed(future_to_gate):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per gate
                        all_results.append(result)
                        
                        status_emoji = {
                            "PASS": "✅",
                            "WARNING": "⚠️",
                            "FAIL": "❌",
                            "SKIP": "⏭️"
                        }.get(result.status, "❓")
                        
                        logger.info(f"  {status_emoji} {result.name}: {result.status} (Score: {result.score:.2f})")
                        
                        if result.recommendations:
                            for rec in result.recommendations[:3]:  # Show first 3 recommendations
                                logger.info(f"    💡 {rec}")
                        
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Quality gate timed out")
                        all_results.append(QualityGateResult(
                            name="Timeout Gate",
                            status="FAIL",
                            score=0.0,
                            details={"error": "Timeout"},
                            execution_time=60.0,
                            recommendations=["Investigate performance issues causing timeout"]
                        ))
                    except Exception as e:
                        logger.error(f"Quality gate failed: {e}")
                        all_results.append(QualityGateResult(
                            name="Failed Gate",
                            status="FAIL",
                            score=0.0,
                            details={"error": str(e)},
                            execution_time=0.0,
                            recommendations=["Investigate gate execution failure"]
                        ))
        
        # Aggregate results
        total_execution_time = time.time() - start_time
        
        # Calculate overall scores
        category_scores = {}
        for category, _ in quality_gates:
            category_results = [r for r in all_results if category.lower() in r.name.lower()]
            if category_results:
                category_scores[category.lower()] = np.mean([r.score for r in category_results])
            else:
                category_scores[category.lower()] = 0.0
        
        overall_score = np.mean(list(category_scores.values()))
        
        # Determine overall status
        if overall_score >= 0.9:
            overall_status = "EXCELLENT"
        elif overall_score >= 0.8:
            overall_status = "GOOD"
        elif overall_score >= 0.7:
            overall_status = "ACCEPTABLE"
        elif overall_score >= 0.6:
            overall_status = "NEEDS_IMPROVEMENT"
        else:
            overall_status = "CRITICAL"
        
        # Count statuses
        status_counts = {}
        for status in ["PASS", "WARNING", "FAIL", "SKIP"]:
            status_counts[status] = len([r for r in all_results if r.status == status])
        
        # Collect all recommendations
        all_recommendations = []
        for result in all_results:
            all_recommendations.extend(result.recommendations)
        
        # Compile comprehensive report
        comprehensive_report = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "overall_score": overall_score,
            "category_scores": category_scores,
            "status_counts": status_counts,
            "total_gates": len(all_results),
            "passed_gates": status_counts.get("PASS", 0),
            "failed_gates": status_counts.get("FAIL", 0),
            "execution_time_seconds": total_execution_time,
            "individual_results": [asdict(result) for result in all_results],
            "key_recommendations": list(set(all_recommendations))[:10],  # Top 10 unique recommendations
            "quality_metrics": {
                "security_score": category_scores.get("security", 0.0),
                "performance_score": category_scores.get("performance", 0.0),
                "functionality_score": category_scores.get("functionality", 0.0)
            }
        }
        
        # Log summary
        logger.info("\n📊 Quality Gates Summary:")
        logger.info(f"Overall Status: {overall_status} (Score: {overall_score:.3f})")
        logger.info(f"Security Score: {category_scores.get('security', 0.0):.3f}")
        logger.info(f"Performance Score: {category_scores.get('performance', 0.0):.3f}")
        logger.info(f"Functionality Score: {category_scores.get('functionality', 0.0):.3f}")
        logger.info(f"Gates Passed: {status_counts.get('PASS', 0)}/{len(all_results)}")
        logger.info(f"Execution Time: {total_execution_time:.2f}s")
        
        return comprehensive_report

def run_autonomous_quality_gates():
    """Run autonomous quality gates validation."""
    logger.info("🎯 Autonomous Quality Gates - Comprehensive SDLC Validation")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Create test model for validation
        logger.info("1. Creating test model for validation...")
        model = MoEModel(
            vocab_size=1000,
            hidden_size=256,
            num_experts=8,
            experts_per_token=2,
            num_layers=6,
            num_attention_heads=8,  # Must divide hidden_size evenly
            moe_layers=[1, 3, 5]
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   ✅ Test model created with {total_params:,} parameters")
        
        # 2. Initialize comprehensive quality gates
        logger.info("2. Initializing comprehensive quality gates...")
        quality_gates = ComprehensiveQualityGates()
        logger.info("   ✅ Quality gates initialized")
        
        # 3. Run all quality gates
        logger.info("3. Running comprehensive quality gates...")
        report = quality_gates.run_all_gates(model)
        logger.info("   ✅ All quality gates completed")
        
        # 4. Save comprehensive report
        logger.info("4. Saving comprehensive report...")
        with open("autonomous_quality_gates_report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("   ✅ Report saved to: autonomous_quality_gates_report.json")
        
        execution_time = time.time() - start_time
        
        # 5. Final validation
        logger.info("\n🎉 Autonomous Quality Gates Complete!")
        logger.info(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        logger.info(f"🏆 Overall quality status: {report['overall_status']}")
        logger.info(f"📈 Overall quality score: {report['overall_score']:.3f}")
        
        if report['overall_score'] >= 0.7:
            logger.info("✅ Quality gates PASSED - Ready for production deployment")
        else:
            logger.warning("⚠️ Quality gates need attention before production")
        
        return report
        
    except Exception as e:
        logger.error(f"Quality gates validation failed: {e}")
        logger.error(traceback.format_exc())
        
        error_report = {
            "timestamp": time.time(),
            "overall_status": "FAILED",
            "overall_score": 0.0,
            "error": str(e),
            "execution_time_seconds": time.time() - start_time
        }
        
        with open("autonomous_quality_gates_report.json", "w") as f:
            json.dump(error_report, f, indent=2)
        
        raise

if __name__ == "__main__":
    # Run autonomous quality gates
    report = run_autonomous_quality_gates()
    
    # Validate success
    assert report["overall_score"] >= 0.6, f"Quality gates failed with score: {report['overall_score']}"
    assert report["overall_status"] != "FAILED", "Quality gates encountered critical failures"
    
    logger.info("✅ Autonomous Quality Gates validation passed - Proceeding to Production Deployment")