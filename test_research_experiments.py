#!/usr/bin/env python3
"""
Test Research Experiments - Validate Novel Algorithm Implementations

Tests the experimental framework and novel algorithms without heavy dependencies.
Validates the research contributions and comparative studies.
"""

import sys
import json
import time
import math
from pathlib import Path


def test_cadr_implementation():
    """Test Complexity-Aware Dynamic Routing implementation."""
    print("Testing CADR implementation...")
    
    try:
        # Import locally to test availability
        sys.path.append('examples')
        from research_experiments import ComplexityAwareDynamicRouter
        
        # Create router
        router = ComplexityAwareDynamicRouter(num_experts=8, min_k=1, max_k=4)
        
        # Test routing with different complexity inputs
        test_inputs = [
            "Short",
            "Medium length text input",
            "Very long and complex text input with multiple sentences and clauses that should trigger high complexity routing"
        ]
        
        routing_decisions = router.route(test_inputs)
        
        # Validate results
        if len(routing_decisions) != len(test_inputs):
            print(f"❌ Expected {len(test_inputs)} routing decisions")
            return False
        
        # Check adaptive k values
        adaptive_k_values = [d['adaptive_k'] for d in routing_decisions]
        if not all(router.min_k <= k <= router.max_k for k in adaptive_k_values):
            print(f"❌ Adaptive k values out of range")
            return False
        
        # Check complexity scores
        complexity_scores = [d['complexity'] for d in routing_decisions]
        if not all(0.0 <= c <= 1.0 for c in complexity_scores):
            print(f"❌ Complexity scores out of range")
            return False
        
        # Test performance stats
        stats = router.get_performance_stats()
        required_stats = ['avg_routing_time', 'avg_experts_per_token', 'computational_efficiency']
        
        for stat in required_stats:
            if stat not in stats:
                print(f"❌ Missing performance stat: {stat}")
                return False
        
        print(f"✅ CADR: {len(routing_decisions)} inputs routed, efficiency: {stats['computational_efficiency']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ CADR test failed: {e}")
        return False


def test_hmr_implementation():
    """Test Hierarchical Multi-Level Routing implementation."""
    print("Testing HMR implementation...")
    
    try:
        from research_experiments import HierarchicalMultiLevelRouter
        
        # Create router
        router = HierarchicalMultiLevelRouter(num_experts=16, num_groups=4)
        
        # Test routing
        test_inputs = [f"input_{i}" for i in range(12)]
        routing_decisions = router.route(test_inputs)
        
        # Validate results
        if len(routing_decisions) != len(test_inputs):
            print(f"❌ Expected {len(test_inputs)} routing decisions")
            return False
        
        # Check group assignments
        selected_groups = [d['selected_group'] for d in routing_decisions]
        if not all(0 <= g < router.num_groups for g in selected_groups):
            print(f"❌ Invalid group assignments")
            return False
        
        # Check expert assignments
        expert_indices = [d['global_expert_idx'] for d in routing_decisions]
        if not all(0 <= e < router.num_experts for e in expert_indices):
            print(f"❌ Invalid expert assignments")
            return False
        
        # Test performance stats
        stats = router.get_performance_stats()
        if 'group_utilization_balance' not in stats:
            print(f"❌ Missing group utilization balance")
            return False
        
        print(f"✅ HMR: {len(routing_decisions)} inputs routed, balance: {stats['group_utilization_balance']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ HMR test failed: {e}")
        return False


def test_casr_implementation():
    """Test Context-Aware Sequential Routing implementation."""
    print("Testing CASR implementation...")
    
    try:
        from research_experiments import ContextAwareSequentialRouter
        
        # Create router
        router = ContextAwareSequentialRouter(num_experts=8, context_window=5)
        
        # Test with sequence
        sequence = ["The", "cat", "sat", "on", "the", "mat"]
        routing_decisions = router.route(sequence)
        
        # Validate results
        if len(routing_decisions) != len(sequence):
            print(f"❌ Expected {len(sequence)} routing decisions")
            return False
        
        # Check attention scores
        for decision in routing_decisions:
            if 'attention_scores' not in decision:
                print(f"❌ Missing attention scores")
                return False
            
            attention_scores = decision['attention_scores']
            if not isinstance(attention_scores, list):
                print(f"❌ Invalid attention scores format")
                return False
        
        # Test performance stats
        stats = router.get_performance_stats()
        if 'avg_context_influence' not in stats:
            print(f"❌ Missing context influence metric")
            return False
        
        print(f"✅ CASR: {len(sequence)} tokens processed, context influence: {stats['avg_context_influence']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ CASR test failed: {e}")
        return False


def test_pec_implementation():
    """Test Predictive Expert Caching implementation."""
    print("Testing PEC implementation...")
    
    try:
        from research_experiments import PredictiveExpertCache
        
        # Create cache
        cache = PredictiveExpertCache(cache_size=4, prediction_window=5)
        
        # Test access pattern with locality
        access_pattern = [0, 1, 2, 0, 1, 3, 0, 2, 4, 1, 0]
        
        for expert_id in access_pattern:
            expert = cache.get_expert(expert_id)
            if expert is None:
                print(f"❌ Expert {expert_id} should be loaded")
                return False
            
            # Test prefetching
            prefetched = cache.prefetch_experts()
            
        # Test cache stats
        stats = cache.get_cache_stats()
        required_stats = ['cache_hit_rate', 'total_accesses', 'cache_utilization']
        
        for stat in required_stats:
            if stat not in stats:
                print(f"❌ Missing cache stat: {stat}")
                return False
        
        # Should have some cache hits due to locality
        if stats['cache_hit_rate'] < 0.1:
            print(f"❌ Hit rate too low: {stats['cache_hit_rate']:.2%}")
            return False
        
        print(f"✅ PEC: Hit rate {stats['cache_hit_rate']:.1%}, utilization {stats['cache_utilization']:.1%}")
        return True
        
    except Exception as e:
        print(f"❌ PEC test failed: {e}")
        return False


def test_dea_implementation():
    """Test Dynamic Expert Allocation implementation."""
    print("Testing DEA implementation...")
    
    try:
        from research_experiments import DynamicExpertAllocator
        
        # Create allocator
        allocator = DynamicExpertAllocator(initial_experts=6, max_experts=12)
        
        # Test workload scenarios
        scenarios = [
            ["short"] * 5,  # Low complexity
            ["short", "medium length text", "very long complex text"] * 2,  # High variance
            [f"type_{i}_input" for i in range(10)]  # High diversity
        ]
        
        allocation_decisions = []
        
        for scenario in scenarios:
            workload_stats = allocator.analyze_workload(scenario)
            decision = allocator.adapt_expert_allocation(workload_stats)
            allocation_decisions.append(decision)
            
            # Validate workload stats
            required_stats = ['complexity_variance', 'input_diversity', 'batch_size']
            for stat in required_stats:
                if stat not in workload_stats:
                    print(f"❌ Missing workload stat: {stat}")
                    return False
        
        # Test allocation stats
        stats = allocator.get_allocation_stats()
        if 'current_experts' not in stats:
            print(f"❌ Missing current experts count")
            return False
        
        if stats['current_experts'] < 1:
            print(f"❌ Invalid expert count: {stats['current_experts']}")
            return False
        
        print(f"✅ DEA: {len(allocation_decisions)} adaptations, {stats['current_experts']} experts")
        return True
        
    except Exception as e:
        print(f"❌ DEA test failed: {e}")
        return False


def test_baseline_comparison():
    """Test baseline comparison framework."""
    print("Testing baseline comparison...")
    
    try:
        from research_experiments import run_comparative_baseline_study
        
        # This should run without errors and return method comparisons
        baseline_results = run_comparative_baseline_study()
        
        expected_methods = ['CADR', 'Random', 'Round_Robin', 'HMR']
        
        for method in expected_methods:
            if method not in baseline_results:
                print(f"❌ Missing baseline method: {method}")
                return False
            
            method_stats = baseline_results[method]
            required_stats = ['routing_time', 'avg_experts_per_token', 'efficiency']
            
            for stat in required_stats:
                if stat not in method_stats:
                    print(f"❌ Missing stat for {method}: {stat}")
                    return False
        
        print(f"✅ Baseline comparison: {len(baseline_results)} methods compared")
        return True
        
    except Exception as e:
        print(f"❌ Baseline comparison test failed: {e}")
        return False


def test_research_report_generation():
    """Test research report generation."""
    print("Testing research report generation...")
    
    try:
        from research_experiments import generate_research_report
        
        # Mock results for testing
        mock_results = {
            'CADR': {'Balanced': {'computational_efficiency': 1.5}},
            'HMR': {'4x4 Hierarchy': {'group_utilization_balance': 0.85}},
            'CASR': {'Sequence_1': {'interpretability_score': 0.7}},
            'PEC': {'Large Cache': {'cache_hit_rate': 0.82}},
            'DEA': {'Overall': {'allocation_efficiency': 0.9}}
        }
        
        report = generate_research_report(mock_results)
        
        # Validate report structure
        required_sections = [
            'experiment_timestamp',
            'research_summary',
            'experimental_results',
            'performance_analysis',
            'research_contributions',
            'validation_results'
        ]
        
        for section in required_sections:
            if section not in report:
                print(f"❌ Missing report section: {section}")
                return False
        
        # Check research summary
        summary = report['research_summary']
        if summary['novel_algorithms_tested'] != 5:
            print(f"❌ Incorrect algorithm count")
            return False
        
        print(f"✅ Research report: {len(required_sections)} sections generated")
        return True
        
    except Exception as e:
        print(f"❌ Research report test failed: {e}")
        return False


def test_math_operations():
    """Test math operations used in algorithms."""
    print("Testing math operations...")
    
    try:
        # Test variance calculation (manual implementation)
        data = [0.1, 0.3, 0.5, 0.7, 0.9]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        
        if not isinstance(variance, (int, float)):
            print(f"❌ Invalid variance calculation")
            return False
        
        # Test mean calculation
        if not isinstance(mean, (int, float)):
            print(f"❌ Invalid mean calculation")
            return False
        
        # Test standard deviation
        std = math.sqrt(variance)
        if not isinstance(std, (int, float)):
            print(f"❌ Invalid std calculation")
            return False
        
        print(f"✅ Math operations: variance={variance:.3f}, mean={mean:.3f}, std={std:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Math operations test failed: {e}")
        return False


def main():
    """Run all research experiment tests."""
    print("🧪 Research Experiments Test Suite")
    print("=" * 60)
    
    tests = [
        ("CADR Implementation", test_cadr_implementation),
        ("HMR Implementation", test_hmr_implementation),
        ("CASR Implementation", test_casr_implementation),
        ("PEC Implementation", test_pec_implementation),
        ("DEA Implementation", test_dea_implementation),
        ("Baseline Comparison", test_baseline_comparison),
        ("Research Report Generation", test_research_report_generation),
        ("Math Operations", test_math_operations)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results[test_name] = {"passed": success}
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = {"passed": False, "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESEARCH EXPERIMENTS TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)} tests")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not result["passed"] and "error" in result:
            print(f"    Error: {result['error']}")
    
    # Save results
    output_file = Path("research_experiments_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    if passed == len(tests):
        print("\n🎉 Research experiments implementation SUCCESSFUL!")
        print("Novel algorithms validated:")
        print("  • Complexity-Aware Dynamic Routing (CADR)")
        print("  • Hierarchical Multi-Level Routing (HMR)")
        print("  • Context-Aware Sequential Routing (CASR)")
        print("  • Predictive Expert Caching (PEC)")
        print("  • Dynamic Expert Allocation (DEA)")
        print("  • Comprehensive baseline comparisons")
        print("  • Research report generation")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Research experiments need fixes.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)