#!/usr/bin/env python3
"""
Health Check Script for Open MoE Trainer Lab

Comprehensive health monitoring for all system components including:
- Training services
- Inference endpoints
- Database connections
- Message queues
- Monitoring stack
- Resource utilization
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import aiohttp
import psutil
import requests
from dataclasses import dataclass, asdict
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service: str
    status: str  # "healthy", "unhealthy", "warning", "unknown"
    message: str
    details: Dict[str, Any]
    timestamp: str
    response_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthChecker:
    """Main health checker class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[HealthCheckResult] = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load health check configuration."""
        default_config = {
            "timeout": 30,
            "retry_count": 3,
            "retry_delay": 2,
            "services": {
                "training": {
                    "url": "http://localhost:8080/health",
                    "critical": True,
                    "timeout": 10
                },
                "inference": {
                    "url": "http://localhost:8000/health",
                    "critical": True,
                    "timeout": 5
                },
                "dashboard": {
                    "url": "http://localhost:8080/api/health",
                    "critical": False,
                    "timeout": 5
                },
                "prometheus": {
                    "url": "http://localhost:9090/-/healthy",
                    "critical": True,
                    "timeout": 5
                },
                "grafana": {
                    "url": "http://localhost:3001/api/health",
                    "critical": False,
                    "timeout": 5
                },
                "redis": {
                    "url": "redis://localhost:6379",
                    "critical": True,
                    "timeout": 3
                },
                "postgres": {
                    "url": "postgresql://moelab:moelab123@localhost:5432/moelab",
                    "critical": True,
                    "timeout": 5
                },
                "minio": {
                    "url": "http://localhost:9000/minio/health/live",
                    "critical": False,
                    "timeout": 5
                }
            },
            "system_checks": {
                "cpu_threshold": 90,
                "memory_threshold": 90,
                "disk_threshold": 90,
                "gpu_memory_threshold": 95
            },
            "alerts": {
                "webhook_url": os.getenv("HEALTH_CHECK_WEBHOOK"),
                "email_alerts": False,
                "slack_alerts": False
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config["timeout"])
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_http_service(self, name: str, config: Dict[str, Any]) -> HealthCheckResult:
        """Check HTTP service health."""
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=config.get("timeout", 5))
            async with self.session.get(config["url"], timeout=timeout) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        details = {"status_code": 200, "response_data": data}
                    except:
                        details = {"status_code": 200, "response_data": "OK"}
                    
                    return HealthCheckResult(
                        service=name,
                        status="healthy",
                        message="Service is healthy",
                        details=details,
                        timestamp=datetime.utcnow().isoformat(),
                        response_time_ms=response_time
                    )
                else:
                    return HealthCheckResult(
                        service=name,
                        status="unhealthy",
                        message=f"HTTP {response.status}",
                        details={"status_code": response.status},
                        timestamp=datetime.utcnow().isoformat(),
                        response_time_ms=response_time
                    )
                    
        except asyncio.TimeoutError:
            return HealthCheckResult(
                service=name,
                status="unhealthy",
                message="Timeout",
                details={"error": "Request timeout"},
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return HealthCheckResult(
                service=name,
                status="unhealthy",
                message=f"Connection error: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            )
    
    def check_system_resources(self) -> List[HealthCheckResult]:
        """Check system resource utilization."""
        results = []
        thresholds = self.config["system_checks"]
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = "healthy" if cpu_percent < thresholds["cpu_threshold"] else "warning"
        if cpu_percent > 95:
            cpu_status = "unhealthy"
        
        results.append(HealthCheckResult(
            service="system_cpu",
            status=cpu_status,
            message=f"CPU usage: {cpu_percent:.1f}%",
            details={"cpu_percent": cpu_percent, "threshold": thresholds["cpu_threshold"]},
            timestamp=datetime.utcnow().isoformat()
        ))
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_status = "healthy" if memory.percent < thresholds["memory_threshold"] else "warning"
        if memory.percent > 95:
            memory_status = "unhealthy"
        
        results.append(HealthCheckResult(
            service="system_memory",
            status=memory_status,
            message=f"Memory usage: {memory.percent:.1f}%",
            details={
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "threshold": thresholds["memory_threshold"]
            },
            timestamp=datetime.utcnow().isoformat()
        ))
        
        # Disk check
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_status = "healthy" if disk_percent < thresholds["disk_threshold"] else "warning"
        if disk_percent > 95:
            disk_status = "unhealthy"
        
        results.append(HealthCheckResult(
            service="system_disk",
            status=disk_status,
            message=f"Disk usage: {disk_percent:.1f}%",
            details={
                "disk_percent": disk_percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "threshold": thresholds["disk_threshold"]
            },
            timestamp=datetime.utcnow().isoformat()
        ))
        
        # GPU check (if available)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_percent = (memory_info.used / memory_info.total) * 100
                
                gpu_status = "healthy" if gpu_percent < thresholds["gpu_memory_threshold"] else "warning"
                if gpu_percent > 98:
                    gpu_status = "unhealthy"
                
                results.append(HealthCheckResult(
                    service=f"gpu_{i}_memory",
                    status=gpu_status,
                    message=f"GPU {i} memory: {gpu_percent:.1f}%",
                    details={
                        "gpu_id": i,
                        "memory_percent": gpu_percent,
                        "memory_used_mb": memory_info.used / (1024**2),
                        "memory_total_mb": memory_info.total / (1024**2),
                        "threshold": thresholds["gpu_memory_threshold"]
                    },
                    timestamp=datetime.utcnow().isoformat()
                ))
        except ImportError:
            results.append(HealthCheckResult(
                service="gpu",
                status="unknown",
                message="GPU monitoring unavailable (pynvml not installed)",
                details={"error": "pynvml not available"},
                timestamp=datetime.utcnow().isoformat()
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                service="gpu",
                status="unknown",
                message=f"GPU check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            ))
        
        return results
    
    def check_docker_services(self) -> List[HealthCheckResult]:
        """Check Docker services status."""
        results = []
        
        try:
            import docker
            client = docker.from_env()
            
            # List all containers
            containers = client.containers.list(all=True)
            
            for container in containers:
                if 'moe' in container.name.lower():  # MoE Lab related containers
                    status = "healthy" if container.status == "running" else "unhealthy"
                    
                    results.append(HealthCheckResult(
                        service=f"docker_{container.name}",
                        status=status,
                        message=f"Container {container.status}",
                        details={
                            "container_id": container.short_id,
                            "status": container.status,
                            "image": container.image.tags[0] if container.image.tags else "unknown"
                        },
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    
        except ImportError:
            results.append(HealthCheckResult(
                service="docker",
                status="unknown",
                message="Docker monitoring unavailable",
                details={"error": "docker library not available"},
                timestamp=datetime.utcnow().isoformat()
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                service="docker",
                status="unhealthy",
                message=f"Docker check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            ))
        
        return results
    
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        results = []
        
        # HTTP service checks
        service_tasks = []
        for name, config in self.config["services"].items():
            if config.get("url", "").startswith("http"):
                task = self.check_http_service(name, config)
                service_tasks.append(task)
        
        if service_tasks:
            service_results = await asyncio.gather(*service_tasks, return_exceptions=True)
            for result in service_results:
                if isinstance(result, HealthCheckResult):
                    results.append(result)
                else:
                    logger.error(f"Service check failed: {result}")
        
        # System resource checks
        results.extend(self.check_system_resources())
        
        # Docker service checks
        results.extend(self.check_docker_services())
        
        self.results = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate health check summary."""
        if not self.results:
            return {"status": "unknown", "message": "No health checks performed"}
        
        status_counts = {"healthy": 0, "unhealthy": 0, "warning": 0, "unknown": 0}
        critical_issues = []
        warnings = []
        
        for result in self.results:
            status_counts[result.status] += 1
            
            # Check if service is critical
            service_config = self.config["services"].get(result.service, {})
            is_critical = service_config.get("critical", False) or result.service.startswith("system_")
            
            if result.status == "unhealthy":
                if is_critical:
                    critical_issues.append(result.service)
                else:
                    warnings.append(result.service)
            elif result.status == "warning":
                warnings.append(result.service)
        
        # Determine overall status
        if critical_issues:
            overall_status = "unhealthy"
            message = f"Critical issues: {', '.join(critical_issues)}"
        elif warnings:
            overall_status = "warning"
            message = f"Warnings: {', '.join(warnings)}"
        else:
            overall_status = "healthy"
            message = "All systems operational"
        
        return {
            "status": overall_status,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_checks": len(self.results),
                "healthy": status_counts["healthy"],
                "unhealthy": status_counts["unhealthy"],
                "warning": status_counts["warning"],
                "unknown": status_counts["unknown"]
            },
            "critical_issues": critical_issues,
            "warnings": warnings
        }
    
    async def send_alerts(self, summary: Dict[str, Any]):
        """Send alerts if configured."""
        if summary["status"] in ["unhealthy", "warning"]:
            webhook_url = self.config["alerts"].get("webhook_url")
            if webhook_url:
                try:
                    payload = {
                        "text": f"MoE Lab Health Alert: {summary['message']}",
                        "status": summary["status"],
                        "details": summary
                    }
                    async with self.session.post(webhook_url, json=payload) as response:
                        if response.status == 200:
                            logger.info("Alert sent successfully")
                        else:
                            logger.error(f"Failed to send alert: {response.status}")
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
    
    def save_results(self, output_file: str):
        """Save results to file."""
        summary = self.generate_summary()
        
        output_data = {
            "summary": summary,
            "detailed_results": [result.to_dict() for result in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Health check results saved to {output_file}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MoE Lab Health Checker")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async with HealthChecker(args.config) as checker:
        if args.continuous:
            logger.info(f"Starting continuous health monitoring (interval: {args.interval}s)")
            while True:
                try:
                    await checker.run_all_checks()
                    summary = checker.generate_summary()
                    
                    if args.format == "json":
                        print(json.dumps(summary, indent=2))
                    else:
                        print(f"[{summary['timestamp']}] Status: {summary['status']} - {summary['message']}")
                    
                    await checker.send_alerts(summary)
                    
                    if args.output:
                        checker.save_results(args.output)
                    
                    await asyncio.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    logger.info("Health monitoring stopped")
                    break
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    await asyncio.sleep(args.interval)
        else:
            # Single run
            results = await checker.run_all_checks()
            summary = checker.generate_summary()
            
            if args.format == "json":
                output_data = {
                    "summary": summary,
                    "detailed_results": [result.to_dict() for result in results]
                }
                print(json.dumps(output_data, indent=2))
            else:
                print(f"Overall Status: {summary['status']}")
                print(f"Message: {summary['message']}")
                print(f"Checks: {summary['summary']['total_checks']} total, "
                      f"{summary['summary']['healthy']} healthy, "
                      f"{summary['summary']['unhealthy']} unhealthy, "
                      f"{summary['summary']['warning']} warnings")
                
                if args.verbose:
                    print("\nDetailed Results:")
                    for result in results:
                        print(f"  {result.service}: {result.status} - {result.message}")
            
            await checker.send_alerts(summary)
            
            if args.output:
                checker.save_results(args.output)
            
            # Exit with appropriate code
            if summary['status'] == "unhealthy":
                sys.exit(1)
            elif summary['status'] == "warning":
                sys.exit(2)
            else:
                sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())