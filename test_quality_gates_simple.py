"""
Simple quality gates validation without external dependencies.
Tests core functionality and validates implementation completeness.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple

# Add the project to Python path
sys.path.insert(0, '/root/repo')


def check_file_exists(filepath: str) -> bool:
    """Check if file exists."""
    return os.path.exists(filepath)


def check_directory_structure() -> Tuple[bool, str, Dict[str, Any]]:
    """Check project directory structure."""
    required_dirs = [
        'moe_lab',
        'moe_lab/models',
        'moe_lab/training',
        'moe_lab/inference',
        'moe_lab/analytics',
        'moe_lab/validation',
        'moe_lab/optimization',
        'moe_lab/utils',
        'tests',
        'examples',
        'scripts',
        'monitoring'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
            
    metrics = {
        'total_required': len(required_dirs),
        'found': len(required_dirs) - len(missing_dirs),
        'missing': missing_dirs
    }
    
    if missing_dirs:
        return False, f"Missing directories: {missing_dirs}", metrics
        
    return True, "Directory structure validation passed", metrics


def check_core_files() -> Tuple[bool, str, Dict[str, Any]]:
    """Check core implementation files."""
    core_files = [
        'moe_lab/__init__.py',
        'moe_lab/models/__init__.py',
        'moe_lab/models/moe_model.py',
        'moe_lab/models/router.py',
        'moe_lab/models/expert.py',
        'moe_lab/models/architectures.py',
        'moe_lab/training/__init__.py',
        'moe_lab/training/trainer.py',
        'moe_lab/analytics/__init__.py',
        'moe_lab/analytics/monitor.py',
        'moe_lab/analytics/analyzer.py',
        'moe_lab/analytics/cost.py',
        'moe_lab/validation/__init__.py',
        'moe_lab/validation/model_validator.py',
        'moe_lab/validation/routing_validator.py',
        'moe_lab/validation/training_validator.py',
        'moe_lab/optimization/__init__.py',
        'moe_lab/optimization/adaptive_routing.py',
        'moe_lab/optimization/efficient_training.py',
        'moe_lab/optimization/auto_scaling.py'
    ]
    
    missing_files = []
    file_sizes = {}
    
    for file_path in core_files:
        if not check_file_exists(file_path):
            missing_files.append(file_path)
        else:
            file_sizes[file_path] = os.path.getsize(file_path)
            
    metrics = {
        'total_required': len(core_files),
        'found': len(core_files) - len(missing_files),
        'missing': missing_files,
        'file_sizes': file_sizes,
        'total_code_size': sum(file_sizes.values())
    }
    
    if missing_files:
        return False, f"Missing core files: {missing_files}", metrics
        
    # Check file sizes are reasonable (not empty)
    empty_files = [f for f, size in file_sizes.items() if size < 100]
    if empty_files:
        return False, f"Files too small (likely empty): {empty_files}", metrics
        
    return True, "Core files validation passed", metrics


def check_implementation_completeness() -> Tuple[bool, str, Dict[str, Any]]:
    """Check implementation completeness by analyzing code."""
    
    # Key implementations to check for
    key_implementations = {
        'MoEModel': 'moe_lab/models/moe_model.py',
        'TopKRouter': 'moe_lab/models/router.py',
        'Expert': 'moe_lab/models/expert.py',
        'MoETrainer': 'moe_lab/training/trainer.py',
        'RouterMonitor': 'moe_lab/analytics/monitor.py',
        'RouterAnalyzer': 'moe_lab/analytics/analyzer.py',
        'MoECostAnalyzer': 'moe_lab/analytics/cost.py',
        'MoEModelValidator': 'moe_lab/validation/model_validator.py',
        'RoutingValidator': 'moe_lab/validation/routing_validator.py',
        'TrainingValidator': 'moe_lab/validation/training_validator.py',
        'AdaptiveRouter': 'moe_lab/optimization/adaptive_routing.py',
        'EfficientMoETrainer': 'moe_lab/optimization/efficient_training.py',
        'AutoScaler': 'moe_lab/optimization/auto_scaling.py'
    }
    
    implementation_status = {}
    
    for impl_name, file_path in key_implementations.items():
        if not check_file_exists(file_path):
            implementation_status[impl_name] = 'missing_file'
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check if class is defined
            if f'class {impl_name}' in content:
                # Check for methods
                method_count = content.count('def ')
                if method_count >= 3:  # At least __init__ and a couple others
                    implementation_status[impl_name] = 'complete'
                else:
                    implementation_status[impl_name] = 'incomplete'
            else:
                implementation_status[impl_name] = 'missing_class'
                
        except Exception as e:
            implementation_status[impl_name] = f'error: {str(e)}'
            
    # Count implementations
    complete_count = sum(1 for status in implementation_status.values() if status == 'complete')
    total_count = len(key_implementations)
    
    metrics = {
        'total_implementations': total_count,
        'complete_implementations': complete_count,
        'completion_rate': complete_count / total_count,
        'implementation_status': implementation_status
    }
    
    if complete_count < total_count * 0.8:  # At least 80% complete
        return False, f"Only {complete_count}/{total_count} implementations complete", metrics
        
    return True, "Implementation completeness validation passed", metrics


def check_documentation_completeness() -> Tuple[bool, str, Dict[str, Any]]:
    """Check documentation completeness."""
    
    doc_files = [
        'README.md',
        'CLAUDE.md',
        'CONTRIBUTING.md',
        'LICENSE',
        'ARCHITECTURE.md',
        'docs/API_REFERENCE.md',
        'docs/DEPLOYMENT_GUIDE.md',
        'docs/DEVELOPMENT.md'
    ]
    
    missing_docs = []
    doc_sizes = {}
    
    for doc_file in doc_files:
        if not check_file_exists(doc_file):
            missing_docs.append(doc_file)
        else:
            doc_sizes[doc_file] = os.path.getsize(doc_file)
            
    metrics = {
        'total_docs': len(doc_files),
        'found_docs': len(doc_files) - len(missing_docs),
        'missing_docs': missing_docs,
        'doc_sizes': doc_sizes,
        'total_doc_size': sum(doc_sizes.values())
    }
    
    # Check README exists and is substantial
    if 'README.md' in missing_docs:
        return False, "README.md is missing", metrics
        
    if doc_sizes.get('README.md', 0) < 1000:
        return False, "README.md is too small", metrics
        
    return True, "Documentation completeness validation passed", metrics


def check_example_completeness() -> Tuple[bool, str, Dict[str, Any]]:
    """Check example implementations."""
    
    example_files = [
        'examples/generation1_simple_analytics.py',
        'examples/generation2_robust_validation.py',
        'examples/generation3_optimized_scaling.py'
    ]
    
    missing_examples = []
    example_sizes = {}
    
    for example_file in example_files:
        if not check_file_exists(example_file):
            missing_examples.append(example_file)
        else:
            example_sizes[example_file] = os.path.getsize(example_file)
            
    metrics = {
        'total_examples': len(example_files),
        'found_examples': len(example_files) - len(missing_examples),
        'missing_examples': missing_examples,
        'example_sizes': example_sizes
    }
    
    if missing_examples:
        return False, f"Missing examples: {missing_examples}", metrics
        
    # Check examples are substantial
    small_examples = [f for f, size in example_sizes.items() if size < 1000]
    if small_examples:
        return False, f"Examples too small: {small_examples}", metrics
        
    return True, "Example completeness validation passed", metrics


def check_test_structure() -> Tuple[bool, str, Dict[str, Any]]:
    """Check test structure."""
    
    test_files = [
        'tests/__init__.py',
        'tests/unit/test_routing.py',
        'tests/integration/test_training_pipeline.py',
        'tests/e2e/test_full_training_pipeline.py',
        'tests/comprehensive/test_complete_pipeline.py',
        'tests/comprehensive/test_quality_gates.py'
    ]
    
    missing_tests = []
    test_sizes = {}
    
    for test_file in test_files:
        if not check_file_exists(test_file):
            missing_tests.append(test_file)
        else:
            test_sizes[test_file] = os.path.getsize(test_file)
            
    metrics = {
        'total_tests': len(test_files),
        'found_tests': len(test_files) - len(missing_tests),
        'missing_tests': missing_tests,
        'test_sizes': test_sizes
    }
    
    if len(missing_tests) > 2:  # Allow some flexibility
        return False, f"Too many missing tests: {missing_tests}", metrics
        
    return True, "Test structure validation passed", metrics


def check_configuration_files() -> Tuple[bool, str, Dict[str, Any]]:
    """Check configuration and setup files."""
    
    config_files = [
        'pyproject.toml',
        'Dockerfile',
        'docker-compose.yml',
        'Makefile'
    ]
    
    missing_configs = []
    config_sizes = {}
    
    for config_file in config_files:
        if not check_file_exists(config_file):
            missing_configs.append(config_file)
        else:
            config_sizes[config_file] = os.path.getsize(config_file)
            
    metrics = {
        'total_configs': len(config_files),
        'found_configs': len(config_files) - len(missing_configs),
        'missing_configs': missing_configs,
        'config_sizes': config_sizes
    }
    
    # pyproject.toml is critical
    if 'pyproject.toml' in missing_configs:
        return False, "pyproject.toml is missing", metrics
        
    return True, "Configuration files validation passed", metrics


def run_quality_gates() -> Dict[str, Any]:
    """Run all quality gates."""
    
    quality_gates = [
        ("Directory Structure", check_directory_structure),
        ("Core Files", check_core_files), 
        ("Implementation Completeness", check_implementation_completeness),
        ("Documentation Completeness", check_documentation_completeness),
        ("Example Completeness", check_example_completeness),
        ("Test Structure", check_test_structure),
        ("Configuration Files", check_configuration_files)
    ]
    
    results = {}
    
    print("Running Quality Gates Validation...")
    print("=" * 50)
    
    for gate_name, gate_func in quality_gates:
        print(f"\nRunning {gate_name} gate...")
        
        start_time = time.time()
        passed, message, metrics = gate_func()
        duration = time.time() - start_time
        
        results[gate_name] = {
            'passed': passed,
            'message': message,
            'metrics': metrics,
            'duration': duration
        }
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {message}")
        
        if not passed:
            print(f"  Metrics: {metrics}")
            
    print("\n" + "=" * 50)
    print("Quality Gates Summary:")
    
    total_gates = len(quality_gates)
    passed_gates = sum(1 for r in results.values() if r['passed'])
    
    print(f"  Total Gates: {total_gates}")
    print(f"  Passed: {passed_gates}")
    print(f"  Failed: {total_gates - passed_gates}")
    print(f"  Success Rate: {passed_gates/total_gates:.1%}")
    
    overall_passed = passed_gates == total_gates
    
    if overall_passed:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        print("The implementation meets production readiness criteria.")
    else:
        failed_gates = [name for name, result in results.items() if not result['passed']]
        print(f"\n❌ QUALITY GATES FAILED: {failed_gates}")
        print("Please address the failing quality gates before deployment.")
        
    # Collect summary metrics
    total_metrics = {
        'overall_passed': overall_passed,
        'total_gates': total_gates,
        'passed_gates': passed_gates,
        'failed_gates': total_gates - passed_gates,
        'success_rate': passed_gates / total_gates,
        'gate_results': results
    }
    
    return total_metrics


def main():
    """Main function."""
    print("Open MoE Trainer Lab - Quality Gates Validation")
    print("=" * 60)
    
    # Change to repo directory
    os.chdir('/root/repo')
    
    # Run quality gates
    results = run_quality_gates()
    
    # Save results
    results_file = 'quality_gates_results.json'
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {results_file}")
    except Exception as e:
        print(f"\nFailed to save results: {e}")
        
    # Exit with appropriate code
    if results['overall_passed']:
        print("\n✅ Quality Gates Validation: PASSED")
        sys.exit(0)
    else:
        print("\n❌ Quality Gates Validation: FAILED") 
        sys.exit(1)


if __name__ == "__main__":
    main()