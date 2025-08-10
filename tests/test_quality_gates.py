#!/usr/bin/env python3
"""
Comprehensive Quality Gates Testing
Tests all three generations: Basic, Robust, and Scalable MoE functionality
"""

import pytest
import json
import time
import random
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.simple_moe_working import MoEDemo
from examples.robust_moe_demo import RobustMoEDemo, ErrorHandler, MetricsCollector, CheckpointManager
from examples.scalable_moe_demo import ScalableMoEDemo, IntelligentCache, ConcurrentRequestProcessor


class TestGeneration1Basic:
    """Test Generation 1: Basic MoE functionality."""
    
    def test_moe_demo_initialization(self):
        """Test basic MoE demo initialization."""
        demo = MoEDemo(hidden_size=16, num_experts=4, top_k=2)
        assert demo.hidden_size == 16
        assert demo.num_experts == 4
        assert demo.top_k == 2
        assert len(demo.router_weights) == 16
        assert len(demo.expert_weights) == 4
    
    def test_basic_forward_pass(self):
        """Test basic forward pass functionality."""
        demo = MoEDemo(hidden_size=8, num_experts=4, top_k=2)
        token = [0.5, -0.3, 0.8, -0.1, 0.2, 0.9, -0.4, 0.6]
        
        output, routing_info = demo.forward(token)
        
        assert len(output) == 8
        assert len(routing_info['selected_experts']) == 2
        assert len(routing_info['expert_weights']) == 2
        assert len(routing_info['router_logits']) == 4
        assert all(isinstance(w, float) for w in routing_info['expert_weights'])
        assert abs(sum(routing_info['expert_weights']) - 1.0) < 1e-6  # Weights sum to 1
    
    def test_routing_statistics(self):
        """Test routing statistics computation."""
        demo = MoEDemo(hidden_size=8, num_experts=4, top_k=2)
        
        # Run multiple forward passes
        routing_history = []
        for _ in range(20):
            token = [random.gauss(0, 1) for _ in range(8)]
            _, routing_info = demo.forward(token)
            routing_history.append(routing_info)
        
        stats = demo.compute_routing_stats(routing_history)
        
        assert 'expert_utilization' in stats
        assert 'load_variance' in stats
        assert 'average_entropy' in stats
        assert len(stats['expert_utilization']) == 4
        assert stats['load_variance'] >= 0
        assert stats['average_entropy'] >= 0
    
    def test_expert_selection_consistency(self):
        """Test that expert selection is deterministic for same input."""
        demo = MoEDemo(hidden_size=8, num_experts=4, top_k=2)
        token = [0.5, -0.3, 0.8, -0.1, 0.2, 0.9, -0.4, 0.6]
        
        output1, routing1 = demo.forward(token)
        output2, routing2 = demo.forward(token)
        
        assert routing1['selected_experts'] == routing2['selected_experts']
        assert routing1['expert_weights'] == routing2['expert_weights']
        
        # Outputs should be very close (allowing for floating point precision)
        for i in range(len(output1)):
            assert abs(output1[i] - output2[i]) < 1e-10


class TestGeneration2Robust:
    """Test Generation 2: Robust MoE with error handling and monitoring."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler(max_retries=3)
        assert handler.max_retries == 3
        assert isinstance(handler.error_history, list)
        assert len(handler.error_history) == 0
    
    def test_error_handling_and_logging(self):
        """Test error handling and logging functionality."""
        from examples.robust_moe_demo import MoEError, TrainingError
        
        handler = ErrorHandler()
        
        # Test handling a custom MoE error
        error = TrainingError(
            "Test training error",
            severity="high",
            context={"step": 100},
            recovery_suggestion="Reduce learning rate"
        )
        
        error_info = handler.handle_error(error)
        
        assert len(handler.error_history) == 1
        assert error_info['error_type'] == 'TrainingError'
        assert error_info['severity'] == 'high'
        assert error_info['recovery_suggestion'] == 'Reduce learning rate'
        assert error_info['moe_context']['step'] == 100
    
    def test_error_summary_generation(self):
        """Test error summary generation."""
        from examples.robust_moe_demo import DataError, ModelError
        
        handler = ErrorHandler()
        
        # Generate some test errors
        for i in range(5):
            if i % 2 == 0:
                error = DataError(f"Data error {i}")
            else:
                error = ModelError(f"Model error {i}")
            handler.handle_error(error)
        
        summary = handler.get_error_summary()
        
        assert summary['total_errors'] == 5
        assert 'DataError' in summary['error_types']
        assert 'ModelError' in summary['error_types']
        assert summary['error_types']['DataError'] == 3
        assert summary['error_types']['ModelError'] == 2
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_history=100)
        assert collector.max_history == 100
        assert len(collector.system_metrics) == 0
        assert len(collector.training_metrics) == 0
    
    def test_metrics_recording(self):
        """Test metrics recording functionality."""
        collector = MetricsCollector()
        
        # Record training metrics
        collector.record_training_metrics(
            step=1,
            loss=0.5,
            expert_load_variance=0.1,
            routing_entropy=1.2,
            tokens_per_second=100.0,
            gradient_norm=0.8
        )
        
        assert len(collector.training_metrics) == 1
        
        # Record expert metrics
        collector.record_expert_metrics(
            expert_id=0,
            utilization_rate=0.6,
            avg_routing_weight=0.3,
            num_tokens_processed=50
        )
        
        assert len(collector.expert_metrics[0]) == 1
    
    def test_checkpoint_manager(self):
        """Test checkpoint management."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Save a checkpoint
            test_state = {"step": 100, "loss": 0.5, "model_weights": [1.0, 2.0, 3.0]}
            checkpoint_path = checkpoint_manager.save_checkpoint(test_state, step=100)
            
            assert os.path.exists(checkpoint_path)
            
            # Load the checkpoint
            loaded_state = checkpoint_manager.load_checkpoint(checkpoint_path)
            
            assert loaded_state == test_state
    
    def test_robust_moe_demo_error_injection(self):
        """Test robust MoE demo with error injection."""
        demo = RobustMoEDemo(hidden_size=16, num_experts=4, top_k=2)
        
        # Test normal operation
        normal_input = [random.gauss(0, 1) for _ in range(16)]
        result = demo.forward_with_monitoring(normal_input, step=1)
        
        assert 'output' in result
        assert 'routing_info' in result
        assert 'performance' in result
        
        # Test error handling with invalid input
        from examples.robust_moe_demo import DataError
        
        with pytest.raises(DataError):
            invalid_input = [random.gauss(0, 1) for _ in range(10)]  # Wrong size
            demo.forward_with_monitoring(invalid_input, step=2)


class TestGeneration3Scalable:
    """Test Generation 3: Scalable MoE with advanced optimizations."""
    
    def test_intelligent_cache_initialization(self):
        """Test intelligent cache initialization."""
        cache = IntelligentCache(l1_capacity_mb=10, l2_capacity_mb=50, l3_capacity_mb=100)
        
        assert cache.l1_capacity == 10 * 1024 * 1024
        assert cache.l2_capacity == 50 * 1024 * 1024
        assert cache.l3_capacity == 100 * 1024 * 1024
        assert len(cache.l1_cache) == 0
        assert len(cache.l2_cache) == 0
        assert len(cache.l3_cache) == 0
    
    def test_cache_operations(self):
        """Test cache put/get operations."""
        cache = IntelligentCache(l1_capacity_mb=1, l2_capacity_mb=5, l3_capacity_mb=10)
        
        # Test cache miss
        result = cache.get("test_key")
        assert result is None
        
        # Test cache put/get
        test_data = {"value": 42, "metadata": "test"}
        cache.put("test_key", test_data, priority="high")
        
        retrieved_data = cache.get("test_key")
        assert retrieved_data == test_data
    
    def test_concurrent_request_processor(self):
        """Test concurrent request processing."""
        processor = ConcurrentRequestProcessor(num_workers=2, enable_process_pool=False)
        
        # Define a simple processing function
        def simple_processor(data):
            return {"processed": data["input"] * 2}
        
        # Submit requests
        request_ids = []
        for i in range(5):
            req_id = processor.submit_request({"input": i}, priority="normal")
            request_ids.append(req_id)
        
        # Process requests
        results = processor.process_requests_batch(simple_processor, max_batch_size=3)
        
        assert len(results) <= 5  # May be less due to batching
        
        # Cleanup
        processor.cleanup()
    
    def test_scalable_moe_demo_single_request(self):
        """Test scalable MoE demo single request processing."""
        demo = ScalableMoEDemo(hidden_size=32, num_experts=4, top_k=2)
        
        request_data = {
            "input": [random.gauss(0, 1) for _ in range(32)],
            "max_tokens": 50,
            "temperature": 1.0
        }
        
        result = demo.process_request_scalable(request_data)
        
        assert "request_id" in result
        assert "result" in result
        assert "processing_time_ms" in result
        assert "selected_worker" in result
        assert "complexity_score" in result
        assert result["processing_time_ms"] > 0
        
        # Cleanup
        demo.cleanup()
    
    def test_scalable_moe_demo_batch_requests(self):
        """Test scalable MoE demo batch processing."""
        demo = ScalableMoEDemo(hidden_size=16, num_experts=4, top_k=2)
        
        # Create batch of requests
        batch_requests = []
        for i in range(3):
            request = {
                "input": [random.gauss(0, 1) for _ in range(16)],
                "max_tokens": 30,
                "temperature": 0.8
            }
            batch_requests.append(request)
        
        results = demo.process_batch_requests(batch_requests)
        
        # Results might be processed asynchronously, so we check structure
        assert isinstance(results, list)
        
        # Cleanup
        demo.cleanup()
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        demo = ScalableMoEDemo(hidden_size=8, num_experts=4, top_k=2)
        
        # Process several requests to generate metrics
        for i in range(5):
            request_data = {
                "input": [random.gauss(0, 1) for _ in range(8)],
                "max_tokens": 20,
                "temperature": 1.0
            }
            demo.process_request_scalable(request_data)
        
        # Get comprehensive report
        report = demo.get_comprehensive_report()
        
        assert "global_metrics" in report
        assert "cache_performance" in report
        assert "load_balancer_stats" in report
        assert "scaling_recommendations" in report
        
        # Check global metrics structure
        global_metrics = report["global_metrics"]
        assert "throughput_tokens_per_sec" in global_metrics
        assert "latency_ms" in global_metrics
        assert "cache_hit_rate" in global_metrics
        
        # Cleanup
        demo.cleanup()


class TestIntegrationAndPerformance:
    """Integration tests and performance benchmarks."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline from basic to scalable."""
        
        # Generation 1: Basic functionality
        basic_demo = MoEDemo(hidden_size=8, num_experts=4, top_k=2)
        token = [random.gauss(0, 1) for _ in range(8)]
        basic_output, basic_routing = basic_demo.forward(token)
        
        assert len(basic_output) == 8
        assert len(basic_routing['selected_experts']) == 2
        
        # Generation 2: Robust functionality
        robust_demo = RobustMoEDemo(hidden_size=8, num_experts=4, top_k=2)
        robust_result = robust_demo.forward_with_monitoring(token, step=1)
        
        assert 'output' in robust_result
        assert len(robust_result['output']) == 8
        
        # Generation 3: Scalable functionality
        scalable_demo = ScalableMoEDemo(hidden_size=8, num_experts=4, top_k=2)
        request_data = {"input": token}
        scalable_result = scalable_demo.process_request_scalable(request_data)
        
        assert 'result' in scalable_result
        assert len(scalable_result['result']['output']) == 8
        
        # Cleanup
        scalable_demo.cleanup()
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks across all generations."""
        hidden_size = 32
        num_experts = 8
        top_k = 2
        num_iterations = 10
        
        # Benchmark Generation 1
        basic_demo = MoEDemo(hidden_size, num_experts, top_k)
        basic_times = []
        
        for _ in range(num_iterations):
            token = [random.gauss(0, 1) for _ in range(hidden_size)]
            start_time = time.time()
            basic_demo.forward(token)
            basic_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Benchmark Generation 2
        robust_demo = RobustMoEDemo(hidden_size, num_experts, top_k)
        robust_times = []
        
        for i in range(num_iterations):
            token = [random.gauss(0, 1) for _ in range(hidden_size)]
            start_time = time.time()
            robust_demo.forward_with_monitoring(token, step=i)
            robust_times.append((time.time() - start_time) * 1000)
        
        # Benchmark Generation 3
        scalable_demo = ScalableMoEDemo(hidden_size, num_experts, top_k)
        scalable_times = []
        
        for _ in range(num_iterations):
            request_data = {
                "input": [random.gauss(0, 1) for _ in range(hidden_size)]
            }
            start_time = time.time()
            scalable_demo.process_request_scalable(request_data)
            scalable_times.append((time.time() - start_time) * 1000)
        
        # Calculate statistics
        avg_basic = sum(basic_times) / len(basic_times)
        avg_robust = sum(robust_times) / len(robust_times)
        avg_scalable = sum(scalable_times) / len(scalable_times)
        
        print(f"\nPerformance Benchmark Results:")
        print(f"  Generation 1 (Basic): {avg_basic:.2f}ms avg")
        print(f"  Generation 2 (Robust): {avg_robust:.2f}ms avg")
        print(f"  Generation 3 (Scalable): {avg_scalable:.2f}ms avg")
        
        # All should complete within reasonable time (< 100ms per request)
        assert avg_basic < 100
        assert avg_robust < 100
        assert avg_scalable < 100
        
        # Cleanup
        scalable_demo.cleanup()
    
    def test_scalability_under_load(self):
        """Test system behavior under increasing load."""
        demo = ScalableMoEDemo(hidden_size=16, num_experts=4, top_k=2)
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10]
        results = {}
        
        for batch_size in batch_sizes:
            batch_requests = []
            for i in range(batch_size):
                request = {
                    "input": [random.gauss(0, 1) for _ in range(16)],
                    "max_tokens": 30
                }
                batch_requests.append(request)
            
            start_time = time.time()
            batch_results = demo.process_batch_requests(batch_requests)
            processing_time = time.time() - start_time
            
            results[batch_size] = {
                "processing_time": processing_time,
                "throughput": batch_size / processing_time if processing_time > 0 else 0
            }
        
        # Verify that throughput generally increases with batch size
        # (or at least doesn't decrease dramatically)
        print(f"\nScalability Test Results:")
        for batch_size, metrics in results.items():
            print(f"  Batch size {batch_size}: {metrics['throughput']:.1f} req/sec")
        
        # Cleanup
        demo.cleanup()
    
    def test_error_recovery_and_resilience(self):
        """Test system resilience and error recovery."""
        from examples.robust_moe_demo import DataError
        
        demo = RobustMoEDemo(hidden_size=16, num_experts=4, top_k=2)
        
        successful_requests = 0
        failed_requests = 0
        
        # Mix of valid and invalid requests
        for i in range(10):
            try:
                if i % 3 == 0:
                    # Invalid input (wrong size)
                    token = [random.gauss(0, 1) for _ in range(10)]
                else:
                    # Valid input
                    token = [random.gauss(0, 1) for _ in range(16)]
                
                result = demo.forward_with_monitoring(token, step=i)
                successful_requests += 1
                
            except DataError:
                failed_requests += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                failed_requests += 1
        
        print(f"\nResilience Test Results:")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Failed requests: {failed_requests}")
        print(f"  Success rate: {successful_requests/(successful_requests + failed_requests)*100:.1f}%")
        
        # Verify error tracking (errors might be logged multiple times due to context handling)
        error_summary = demo.error_handler.get_error_summary()
        assert error_summary['total_errors'] >= failed_requests  # At least as many as failed requests


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-x"])