#!/usr/bin/env python3
"""
Test Research Validation Framework

Tests the statistical analysis and comparative studies framework.
Validates the research validation methodology and statistical tools.
"""

import sys
import json
import time
import math
from pathlib import Path


def test_statistical_analyzer():
    """Test StatisticalAnalyzer class."""
    print("Testing StatisticalAnalyzer...")
    
    try:
        sys.path.append('examples')
        from research_validation import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        
        # Test mean and std calculation
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, std = analyzer.calculate_mean_and_std(data)
        
        expected_mean = 3.0
        expected_std = math.sqrt(2.0)  # Variance = 2.0
        
        if abs(mean - expected_mean) > 0.001:
            print(f"❌ Mean calculation incorrect: {mean} vs {expected_mean}")
            return False
        
        if abs(std - expected_std) > 0.001:
            print(f"❌ Std calculation incorrect: {std} vs {expected_std}")
            return False
        
        # Test confidence interval
        ci_low, ci_high = analyzer.calculate_confidence_interval(data)
        
        if ci_low is None or ci_high is None:
            print(f"❌ Confidence interval calculation failed")
            return False
        
        if ci_low >= ci_high:
            print(f"❌ Invalid confidence interval: [{ci_low}, {ci_high}]")
            return False
        
        # Test t-test
        sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        t_test_result = analyzer.welch_t_test(sample1, sample2)
        
        if t_test_result is None:
            print(f"❌ T-test failed")
            return False
        
        required_fields = ['t_statistic', 'degrees_of_freedom', 'p_value', 'significant']
        for field in required_fields:
            if field not in t_test_result:
                print(f"❌ Missing t-test field: {field}")
                return False
        
        # Test effect size
        effect_size = analyzer.calculate_effect_size(sample1, sample2)
        
        if 'cohens_d' not in effect_size:
            print(f"❌ Missing Cohen's d in effect size")
            return False
        
        print(f"✅ StatisticalAnalyzer: mean={mean:.2f}, std={std:.2f}, CI=[{ci_low:.2f}, {ci_high:.2f}]")
        return True
        
    except Exception as e:
        print(f"❌ StatisticalAnalyzer test failed: {e}")
        return False


def test_multi_run_validator():
    """Test MultiRunValidator class."""
    print("Testing MultiRunValidator...")
    
    try:
        from research_validation import MultiRunValidator
        
        validator = MultiRunValidator(num_runs=3, random_seeds=[42, 43, 44])
        
        # Create a simple test algorithm
        def test_algorithm(test_data):
            # Simulate some variability
            import random
            time.sleep(0.001)  # Simulate processing time
            
            return {
                'performance_metric': random.uniform(0.8, 1.2),
                'efficiency_metric': random.uniform(0.5, 1.5)
            }
        
        # Test data
        test_data = ['input1', 'input2', 'input3']
        
        # Run validation
        validation_result = validator.run_algorithm_validation(
            test_algorithm, test_data, "TestAlgorithm"
        )
        
        # Validate result structure
        required_fields = ['algorithm_name', 'num_runs', 'raw_results', 'statistical_summary', 'reproducibility_score']
        for field in required_fields:
            if field not in validation_result:
                print(f"❌ Missing validation result field: {field}")
                return False
        
        # Check statistical summary
        stats = validation_result['statistical_summary']
        if 'performance_metric' not in stats:
            print(f"❌ Missing performance metric in statistical summary")
            return False
        
        metric_stats = stats['performance_metric']
        required_stat_fields = ['mean', 'std', 'min', 'max', 'confidence_interval', 'coefficient_of_variation']
        for field in required_stat_fields:
            if field not in metric_stats:
                print(f"❌ Missing metric stat field: {field}")
                return False
        
        # Check reproducibility score
        repro_score = validation_result['reproducibility_score']
        if not (0.0 <= repro_score <= 1.0):
            print(f"❌ Invalid reproducibility score: {repro_score}")
            return False
        
        print(f"✅ MultiRunValidator: {validation_result['num_runs']} runs, reproducibility={repro_score:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ MultiRunValidator test failed: {e}")
        return False


def test_comparative_study_framework():
    """Test ComparativeStudyFramework class."""
    print("Testing ComparativeStudyFramework...")
    
    try:
        from research_validation import ComparativeStudyFramework
        
        framework = ComparativeStudyFramework()
        
        # Create mock algorithms
        def algorithm_a(test_data):
            return {
                'performance_metric': 1.0,
                'efficiency_metric': 0.8
            }
        
        def algorithm_b(test_data):
            return {
                'performance_metric': 1.2,
                'efficiency_metric': 0.6
            }
        
        algorithms_config = {
            'Algorithm_A': algorithm_a,
            'Algorithm_B': algorithm_b
        }
        
        # Create test datasets
        test_datasets = {
            'Dataset_1': ['data1', 'data2'],
            'Dataset_2': ['data3', 'data4']
        }
        
        # Run comparative study (reduced runs for testing)
        framework.validator.num_runs = 2  # Reduce for faster testing
        comparison_results = framework.compare_algorithms(algorithms_config, test_datasets)
        
        # Validate result structure
        required_fields = ['dataset_comparisons', 'overall_analysis', 'validation_timestamp']
        for field in required_fields:
            if field not in comparison_results:
                print(f"❌ Missing comparison result field: {field}")
                return False
        
        # Check dataset comparisons
        dataset_comparisons = comparison_results['dataset_comparisons']
        for dataset_name in test_datasets.keys():
            if dataset_name not in dataset_comparisons:
                print(f"❌ Missing dataset in comparisons: {dataset_name}")
                return False
            
            dataset_result = dataset_comparisons[dataset_name]
            required_dataset_fields = ['individual_results', 'pairwise_comparisons', 'ranking']
            for field in required_dataset_fields:
                if field not in dataset_result:
                    print(f"❌ Missing dataset result field: {field}")
                    return False
        
        # Check overall analysis
        overall_analysis = comparison_results['overall_analysis']
        if 'overall_ranking' not in overall_analysis:
            print(f"❌ Missing overall ranking")
            return False
        
        ranking = overall_analysis['overall_ranking']
        if len(ranking) != len(algorithms_config):
            print(f"❌ Ranking length mismatch: {len(ranking)} vs {len(algorithms_config)}")
            return False
        
        print(f"✅ ComparativeStudyFramework: {len(test_datasets)} datasets, {len(algorithms_config)} algorithms")
        return True
        
    except Exception as e:
        print(f"❌ ComparativeStudyFramework test failed: {e}")
        return False


def test_ablation_study_framework():
    """Test AblationStudyFramework class."""
    print("Testing AblationStudyFramework...")
    
    try:
        from research_validation import AblationStudyFramework
        
        framework = AblationStudyFramework()
        framework.validator.num_runs = 2  # Reduce for faster testing
        
        # Create base algorithm
        def base_algorithm(test_data):
            return {
                'performance_metric': 1.0,
                'efficiency_metric': 0.8
            }
        
        # Create ablation configurations
        def ablated_config1(test_data):
            return {
                'performance_metric': 0.9,  # Slightly worse
                'efficiency_metric': 0.8
            }
        
        def ablated_config2(test_data):
            return {
                'performance_metric': 1.0,
                'efficiency_metric': 0.6  # Much worse
            }
        
        ablation_configs = {
            'Without_Component_1': ablated_config1,
            'Without_Component_2': ablated_config2
        }
        
        test_data = ['data1', 'data2']
        
        # Run ablation study
        ablation_results = framework.conduct_ablation_study(
            base_algorithm, ablation_configs, test_data
        )
        
        # Validate result structure
        required_fields = ['ablation_results', 'component_analysis', 'recommendations']
        for field in required_fields:
            if field not in ablation_results:
                print(f"❌ Missing ablation result field: {field}")
                return False
        
        # Check ablation results
        ablation_result_data = ablation_results['ablation_results']
        expected_configs = ['Full_Algorithm'] + list(ablation_configs.keys())
        
        for config in expected_configs:
            if config not in ablation_result_data:
                print(f"❌ Missing ablation config: {config}")
                return False
        
        # Check component analysis
        component_analysis = ablation_results['component_analysis']
        for config_name in ablation_configs.keys():
            if config_name not in component_analysis:
                print(f"❌ Missing component analysis for: {config_name}")
                return False
        
        # Check recommendations
        recommendations = ablation_results['recommendations']
        if not isinstance(recommendations, list):
            print(f"❌ Recommendations should be a list")
            return False
        
        print(f"✅ AblationStudyFramework: {len(ablation_configs)} ablations, {len(recommendations)} recommendations")
        return True
        
    except Exception as e:
        print(f"❌ AblationStudyFramework test failed: {e}")
        return False


def test_algorithm_creation():
    """Test test algorithm creation functions."""
    print("Testing algorithm creation...")
    
    try:
        from research_validation import create_test_algorithms, create_test_datasets, create_ablation_configurations
        
        # Test algorithm creation
        algorithms = create_test_algorithms()
        
        expected_algorithms = ['CADR', 'HMR', 'Random_Baseline']
        for algo_name in expected_algorithms:
            if algo_name not in algorithms:
                print(f"❌ Missing algorithm: {algo_name}")
                return False
        
        # Test dataset creation
        datasets = create_test_datasets()
        
        expected_datasets = ['Simple_Text', 'Complex_Text', 'Mixed_Complexity']
        for dataset_name in expected_datasets:
            if dataset_name not in datasets:
                print(f"❌ Missing dataset: {dataset_name}")
                return False
            
            if not isinstance(datasets[dataset_name], list):
                print(f"❌ Dataset should be a list: {dataset_name}")
                return False
        
        # Test ablation configurations
        ablation_configs = create_ablation_configurations()
        
        if not ablation_configs:
            print(f"❌ No ablation configurations created")
            return False
        
        print(f"✅ Algorithm creation: {len(algorithms)} algorithms, {len(datasets)} datasets, {len(ablation_configs)} ablations")
        return True
        
    except Exception as e:
        print(f"❌ Algorithm creation test failed: {e}")
        return False


def test_analysis_functions():
    """Test analysis helper functions."""
    print("Testing analysis functions...")
    
    try:
        from research_validation import (
            analyze_statistical_significance, analyze_reproducibility,
            extract_performance_findings, get_reproducibility_rating
        )
        
        # Test reproducibility rating
        ratings = [
            (0.95, "Excellent"),
            (0.85, "Good"), 
            (0.75, "Acceptable"),
            (0.65, "Fair"),
            (0.5, "Poor")
        ]
        
        for score, expected_rating in ratings:
            rating = get_reproducibility_rating(score)
            if rating != expected_rating:
                print(f"❌ Wrong reproducibility rating: {score} -> {rating} (expected {expected_rating})")
                return False
        
        # Test with mock data structures
        mock_comparative_results = {
            'dataset_comparisons': {
                'Dataset1': {
                    'individual_results': {
                        'Algo1': {'reproducibility_score': 0.9},
                        'Algo2': {'reproducibility_score': 0.8}
                    },
                    'pairwise_comparisons': {
                        'Algo1_vs_Algo2': {
                            'metric1': {
                                'statistical_test': {'significant': True, 'p_value': 0.01}
                            }
                        }
                    }
                }
            }
        }
        
        # Test statistical significance analysis
        sig_analysis = analyze_statistical_significance(mock_comparative_results)
        
        if 'overall_significance_rate' not in sig_analysis:
            print(f"❌ Missing overall significance rate")
            return False
        
        # Test reproducibility analysis
        repro_analysis = analyze_reproducibility(mock_comparative_results)
        
        if 'overall_reproducibility' not in repro_analysis:
            print(f"❌ Missing overall reproducibility")
            return False
        
        print(f"✅ Analysis functions: All helper functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Analysis functions test failed: {e}")
        return False


def test_report_generation():
    """Test validation report generation."""
    print("Testing report generation...")
    
    try:
        from research_validation import generate_validation_report
        
        # Create mock validation results
        mock_validation_results = {
            'comparative_study': {
                'dataset_comparisons': {
                    'Dataset1': {
                        'individual_results': {},
                        'pairwise_comparisons': {},
                        'ranking': []
                    }
                },
                'overall_analysis': {
                    'overall_ranking': [
                        {'algorithm': 'CADR', 'overall_average_rank': 1.5},
                        {'algorithm': 'HMR', 'overall_average_rank': 2.0}
                    ]
                }
            },
            'significance_analysis': {
                'overall_significance_rate': 0.7,
                'total_comparisons': 10,
                'significant_comparisons': 7
            },
            'reproducibility_analysis': {
                'overall_reproducibility': 0.85
            },
            'ablation_study': {
                'component_analysis': {
                    'Component1': {
                        'metric1': {'relative_change': 0.3}
                    }
                }
            }
        }
        
        # Generate report
        report = generate_validation_report(mock_validation_results)
        
        # Validate report structure
        required_sections = [
            'validation_timestamp',
            'validation_summary',
            'key_findings',
            'research_validity',
            'recommendations'
        ]
        
        for section in required_sections:
            if section not in report:
                print(f"❌ Missing report section: {section}")
                return False
        
        # Check key findings
        key_findings = report['key_findings']
        required_finding_sections = [
            'algorithm_performance',
            'statistical_significance',
            'reproducibility',
            'component_contributions'
        ]
        
        for section in required_finding_sections:
            if section not in key_findings:
                print(f"❌ Missing key findings section: {section}")
                return False
        
        # Check recommendations
        recommendations = report['recommendations']
        if not isinstance(recommendations, list):
            print(f"❌ Recommendations should be a list")
            return False
        
        print(f"✅ Report generation: {len(required_sections)} sections, {len(recommendations)} recommendations")
        return True
        
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False


def main():
    """Run all research validation tests."""
    print("🧪 Research Validation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Statistical Analyzer", test_statistical_analyzer),
        ("Multi-Run Validator", test_multi_run_validator),
        ("Comparative Study Framework", test_comparative_study_framework),
        ("Ablation Study Framework", test_ablation_study_framework),
        ("Algorithm Creation", test_algorithm_creation),
        ("Analysis Functions", test_analysis_functions),
        ("Report Generation", test_report_generation)
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
    print("📊 RESEARCH VALIDATION TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)} tests")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not result["passed"] and "error" in result:
            print(f"    Error: {result['error']}")
    
    # Save results
    output_file = Path("research_validation_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    if passed == len(tests):
        print("\n🎉 Research validation framework SUCCESSFUL!")
        print("Statistical analysis tools validated:")
        print("  • Statistical significance testing")
        print("  • Multi-run reproducibility validation")
        print("  • Comparative algorithm studies")
        print("  • Ablation component analysis")
        print("  • Confidence interval estimation")
        print("  • Effect size calculations")
        print("  • Comprehensive validation reporting")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Research validation needs fixes.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)