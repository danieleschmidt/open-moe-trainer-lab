#!/usr/bin/env python3
"""Generation 2 Robust Testing - Comprehensive test suite for reliability features.

Tests advanced error handling, monitoring, and production robustness capabilities.
"""

import os
import sys
import pytest
import torch
import threading
import time
import psutil
from unittest.mock import Mock, patch
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe_lab.utils.robust_error_handling import (
    RobustErrorHandler, ErrorSeverity, RecoveryStrategy,
    CircuitBreaker, robust_execution, RetryConfig
)
from moe_lab.monitoring.advanced_monitoring import (
    AdvancedMonitoringSystem, MetricsCollector, AnomalyDetector,
    AlertManager, ResourceMonitor
)


class TestRobustErrorHandling:
    """Test suite for robust error handling capabilities."""

    def setup_method(self):
        """Setup for each test method."""
        self.error_handler = RobustErrorHandler()

    def test_basic_error_handling(self):
        """Test basic error context creation and handling."""
        test_exception = ValueError("Test error")
        
        error_context = self.error_handler.handle_error(
            exception=test_exception,
            severity=ErrorSeverity.MEDIUM,
            context_data={'test_key': 'test_value'}
        )
        
        assert error_context.exception_type == "ValueError"
        assert error_context.exception_message == "Test error"
        assert error_context.severity == ErrorSeverity.MEDIUM
        assert error_context.context_data['test_key'] == 'test_value'
        assert len(error_context.error_id) == 8

    def test_retry_configuration(self):
        """Test retry configuration calculations."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Test exponential backoff
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        
        # Test max delay cap
        config.max_delay = 3.0
        assert config.get_delay(2) == 3.0

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        cb = CircuitBreaker("test_circuit", failure_threshold=2, recovery_timeout=1.0)
        
        # Test closed state - successful calls
        def successful_func():
            return "success"
        
        result = cb.call(successful_func)
        assert result == "success"
        assert cb.get_state().state == "closed"
        
        # Test failures leading to open state
        def failing_func():
            raise ValueError("Test failure")
        
        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.get_state().failure_count == 1
        
        # Second failure should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.get_state().state == "open"
        
        # Circuit should now fail fast
        with pytest.raises(Exception) as exc_info:
            cb.call(successful_func)
        assert "Circuit breaker" in str(exc_info.value)

    def test_robust_execution_decorator(self):
        """Test robust execution decorator with retries."""
        call_count = 0
        
        @robust_execution(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
            severity=ErrorSeverity.LOW
        )
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_error_statistics(self):
        """Test error statistics collection."""
        # Generate some test errors
        for i in range(5):
            try:
                raise ValueError(f"Test error {i}")
            except ValueError as e:
                self.error_handler.handle_error(e)
        
        for i in range(3):
            try:
                raise ConnectionError(f"Connection error {i}")
            except ConnectionError as e:
                self.error_handler.handle_error(e)
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 8
        assert stats['error_counts']['ValueError'] == 5
        assert stats['error_counts']['ConnectionError'] == 3
        assert stats['error_rates']['ValueError'] == 62.5  # 5/8 * 100
        assert stats['error_rates']['ConnectionError'] == 37.5  # 3/8 * 100


class TestAdvancedMonitoring:
    """Test suite for advanced monitoring system."""

    def setup_method(self):
        """Setup for each test method."""
        self.monitoring = AdvancedMonitoringSystem({
            'console_alerts': False,  # Disable console output during tests
            'metrics_buffer_size': 1000
        })

    def teardown_method(self):
        """Cleanup after each test."""
        self.monitoring.stop_monitoring()

    def test_metrics_collection(self):
        """Test basic metrics collection."""
        # Record some test metrics
        self.monitoring.record_metric('test.metric', 100.0, {'service': 'test'})
        self.monitoring.record_metric('test.metric', 150.0, {'service': 'test'})
        self.monitoring.record_metric('test.metric', 125.0, {'service': 'test'})
        
        # Get recent metrics
        recent_metrics = self.monitoring.metrics_collector.get_recent_metrics('test.metric', 60)
        assert len(recent_metrics) == 3
        
        # Check statistics
        stats = self.monitoring.metrics_collector.get_metric_statistics('test.metric', 60)
        assert stats['count'] == 3
        assert stats['mean'] == 125.0  # (100 + 150 + 125) / 3
        assert stats['min'] == 100.0
        assert stats['max'] == 150.0

    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        detector = AnomalyDetector(sensitivity=2.0)  # Lower threshold for testing
        
        # Create baseline with normal values
        normal_values = [100.0] * 20 + [101.0] * 10 + [99.0] * 10  # Mean ≈ 100
        detector.update_baseline('test.metric', normal_values)
        
        # Test normal value - should not be anomaly
        anomaly = detector.detect_anomaly('test.metric', 102.0)
        assert anomaly is None
        
        # Test anomalous value - should be detected
        anomaly = detector.detect_anomaly('test.metric', 200.0)  # Way outside normal range
        assert anomaly is not None
        assert anomaly['metric_name'] == 'test.metric'
        assert anomaly['value'] == 200.0
        assert anomaly['z_score'] > 2.0

    def test_alert_management(self):
        """Test alert triggering and management."""
        alert_manager = AlertManager()
        triggered_alerts = []
        
        def test_alert_handler(alert):
            triggered_alerts.append(alert)
        
        alert_manager.add_alert_handler(test_alert_handler)
        alert_manager.start_alert_processing()
        
        # Trigger test alert
        alert_manager.trigger_alert(
            alert_id="test_alert",
            severity="warning",
            message="Test alert message",
            metric_name="test.metric",
            current_value=95.0,
            threshold=90.0
        )
        
        # Wait for alert processing
        time.sleep(0.1)
        
        # Check alert was processed
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].id == "test_alert"
        assert triggered_alerts[0].severity == "warning"
        
        # Check active alerts
        active_alerts = alert_manager.get_active_alerts()
        assert "test_alert" in active_alerts
        
        # Resolve alert
        alert_manager.resolve_alert("test_alert")
        active_alerts = alert_manager.get_active_alerts()
        assert "test_alert" not in active_alerts
        
        alert_manager.stop_alert_processing()

    def test_resource_monitoring(self):
        """Test system resource monitoring."""
        resource_monitor = ResourceMonitor(collection_interval=0.1)
        
        # Start monitoring briefly
        resource_monitor.start_monitoring()
        time.sleep(0.3)  # Let it collect a few samples
        resource_monitor.stop_monitoring()
        
        # Check resource summary
        summary = resource_monitor.get_resource_summary()
        
        # Should have collected some basic system metrics
        assert 'system.cpu.percent' in summary
        assert 'system.memory.percent' in summary
        
        # Values should be reasonable
        cpu_data = summary['system.cpu.percent']
        assert 0 <= cpu_data['current'] <= 100
        
        memory_data = summary['system.memory.percent']
        assert 0 <= memory_data['current'] <= 100

    def test_health_checks(self):
        """Test health check functionality."""
        health_checker = self.monitoring.health_checker
        
        # Register custom health check
        def custom_check():
            return {'status': 'healthy', 'custom_metric': 42}
        
        health_checker.register_check('custom', custom_check)
        
        # Run individual check
        result = health_checker.run_check('custom')
        assert result['status'] == 'healthy'
        assert result['custom_metric'] == 42
        assert 'duration_ms' in result
        
        # Run all checks
        all_results = health_checker.run_all_checks()
        assert 'custom' in all_results
        assert 'memory' in all_results  # Default check
        assert 'cpu' in all_results     # Default check
        
        # Get overall health
        overall = health_checker.get_overall_health()
        assert overall['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'checks' in overall
        assert 'summary' in overall

    def test_monitoring_dashboard(self):
        """Test comprehensive monitoring dashboard."""
        # Start monitoring to collect some data
        self.monitoring.start_monitoring()
        
        # Record some test metrics
        self.monitoring.record_metric('test.latency', 50.0)
        self.monitoring.record_metric('test.throughput', 1000.0)
        
        time.sleep(0.2)  # Allow some data collection
        
        # Get dashboard
        dashboard = self.monitoring.get_monitoring_dashboard()
        
        # Check structure
        assert 'timestamp' in dashboard
        assert 'status' in dashboard
        assert 'resource_summary' in dashboard
        assert 'alert_summary' in dashboard
        assert 'metrics_summary' in dashboard
        
        # Check status structure
        status = dashboard['status']
        assert 'status' in status
        assert 'checks' in status
        assert 'summary' in status
        
        self.monitoring.stop_monitoring()

    def test_full_monitoring_integration(self):
        """Test full monitoring system integration."""
        # Start monitoring
        self.monitoring.start_monitoring()
        
        # Simulate some application metrics
        for i in range(10):
            self.monitoring.record_metric('app.requests', 100 + i * 10)
            self.monitoring.record_metric('app.latency', 50 + i * 5)
            time.sleep(0.01)
        
        # Let monitoring run briefly
        time.sleep(0.5)
        
        # Check that everything is working
        dashboard = self.monitoring.get_monitoring_dashboard()
        
        # Should have collected metrics
        metrics_summary = dashboard['metrics_summary']
        assert metrics_summary['total_metrics_collected'] > 0
        assert metrics_summary['unique_metric_names'] >= 2
        
        # Health checks should be running
        health_status = dashboard['status']
        assert health_status['summary']['total_checks'] > 0
        
        self.monitoring.stop_monitoring()


class TestRobustnessIntegration:
    """Integration tests for robustness features."""

    def test_error_handling_with_monitoring(self):
        """Test integration of error handling with monitoring."""
        monitoring = AdvancedMonitoringSystem({'console_alerts': False})
        error_handler = RobustErrorHandler()
        
        monitoring.start_monitoring()
        
        try:
            # Simulate errors that should trigger monitoring
            for i in range(5):
                try:
                    if i % 2 == 0:
                        raise ValueError(f"Test error {i}")
                    else:
                        raise ConnectionError(f"Connection error {i}")
                except Exception as e:
                    error_context = error_handler.handle_error(e)
                    
                    # Record error as metric
                    monitoring.record_metric(
                        'errors.count',
                        1,
                        {'error_type': error_context.exception_type}
                    )
                    
                    # Record error severity
                    severity_score = {
                        ErrorSeverity.LOW: 1,
                        ErrorSeverity.MEDIUM: 2,
                        ErrorSeverity.HIGH: 3,
                        ErrorSeverity.CRITICAL: 4
                    }[error_context.severity]
                    
                    monitoring.record_metric('errors.severity', severity_score)
            
            # Let monitoring process
            time.sleep(0.2)
            
            # Check that errors were recorded as metrics
            error_metrics = monitoring.metrics_collector.get_recent_metrics('errors.count', 60)
            assert len(error_metrics) == 5
            
            severity_metrics = monitoring.metrics_collector.get_recent_metrics('errors.severity', 60)
            assert len(severity_metrics) == 5
            
            # Check error handler statistics
            error_stats = error_handler.get_error_statistics()
            assert error_stats['total_errors'] == 5
            assert 'ValueError' in error_stats['error_counts']
            assert 'ConnectionError' in error_stats['error_counts']
            
        finally:
            monitoring.stop_monitoring()

    def test_circuit_breaker_with_monitoring(self):
        """Test circuit breaker integration with monitoring."""
        monitoring = AdvancedMonitoringSystem({'console_alerts': False})
        monitoring.start_monitoring()
        
        circuit_name = "test_service"
        failure_count = 0
        success_count = 0
        
        @robust_execution(
            circuit_breaker_name=circuit_name,
            retry_config=RetryConfig(max_attempts=1)  # No retries for this test
        )
        def unreliable_service():
            nonlocal failure_count, success_count
            
            # Fail first 3 times, then succeed
            if failure_count < 3:
                failure_count += 1
                monitoring.record_metric(f'circuit_breaker.{circuit_name}.failures', 1)
                raise ConnectionError("Service unavailable")
            else:
                success_count += 1
                monitoring.record_metric(f'circuit_breaker.{circuit_name}.successes', 1)
                return "success"
        
        try:
            # Should fail first 3 times due to circuit breaker
            for i in range(3):
                with pytest.raises(ConnectionError):
                    unreliable_service()
            
            # Circuit should be open now - next call should fail fast
            with pytest.raises(Exception) as exc_info:
                unreliable_service()
            assert "Circuit breaker" in str(exc_info.value) or "Connection" in str(exc_info.value)
            
            # Check monitoring data
            time.sleep(0.1)
            failure_metrics = monitoring.metrics_collector.get_recent_metrics(
                f'circuit_breaker.{circuit_name}.failures', 60
            )
            assert len(failure_metrics) >= 3
            
        finally:
            monitoring.stop_monitoring()


def run_robustness_tests():
    """Run comprehensive robustness tests."""
    print("🔧 Starting Generation 2 Robustness Tests...")
    
    # Run pytest with detailed output
    test_args = [
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings'
    ]
    
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("✅ All robustness tests passed!")
        return True
    else:
        print("❌ Some robustness tests failed!")
        return False


def demonstrate_robustness_features():
    """Demonstrate key robustness features."""
    print("\n🚀 Demonstrating Generation 2 Robustness Features...")
    
    # 1. Error Handling Demonstration
    print("\n1. Advanced Error Handling:")
    error_handler = RobustErrorHandler()
    
    try:
        raise ValueError("Demonstration error")
    except Exception as e:
        error_context = error_handler.handle_error(
            e, 
            severity=ErrorSeverity.MEDIUM,
            context_data={'demo': 'error_handling'}
        )
        print(f"   Error ID: {error_context.error_id}")
        print(f"   Severity: {error_context.severity.value}")
        print(f"   Recovery Strategy: {error_context.recovery_strategy.value}")
    
    # 2. Monitoring Demonstration
    print("\n2. Advanced Monitoring:")
    monitoring = AdvancedMonitoringSystem({'console_alerts': False})
    monitoring.start_monitoring()
    
    # Record demo metrics
    for i in range(5):
        monitoring.record_metric('demo.latency', 100 + i * 10)
        monitoring.record_metric('demo.throughput', 1000 - i * 50)
    
    time.sleep(0.3)
    
    dashboard = monitoring.get_monitoring_dashboard()
    print(f"   Collected metrics: {dashboard['metrics_summary']['total_metrics_collected']}")
    print(f"   Health status: {dashboard['status']['status']}")
    print(f"   Active alerts: {len(dashboard['alert_summary']['by_severity'])}")
    
    monitoring.stop_monitoring()
    
    # 3. Circuit Breaker Demonstration
    print("\n3. Circuit Breaker Protection:")
    cb = CircuitBreaker("demo_service", failure_threshold=2)
    
    def demo_service():
        raise ConnectionError("Service down")
    
    try:
        # Trigger failures
        for i in range(3):
            try:
                cb.call(demo_service)
            except Exception:
                pass
    except Exception:
        pass
    
    state = cb.get_state()
    print(f"   Circuit state: {state.state}")
    print(f"   Failure count: {state.failure_count}")
    print(f"   Total calls: {state.total_calls}")
    
    print("\n✅ Generation 2 robustness features demonstrated successfully!")


if __name__ == "__main__":
    print("=" * 70)
    print("🔧 GENERATION 2 ROBUST TESTING SUITE")
    print("   Advanced Error Handling & Monitoring Validation")
    print("=" * 70)
    
    # Run comprehensive tests
    tests_passed = run_robustness_tests()
    
    # Demonstrate features
    demonstrate_robustness_features()
    
    print("\n" + "=" * 70)
    if tests_passed:
        print("🎉 Generation 2 Robustness Implementation: COMPLETE")
        print("   ✅ Advanced error handling operational")
        print("   ✅ Comprehensive monitoring active")  
        print("   ✅ Production reliability features verified")
    else:
        print("⚠️  Some tests failed - please review output above")
    print("=" * 70)