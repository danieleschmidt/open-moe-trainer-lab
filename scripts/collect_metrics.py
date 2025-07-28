#!/usr/bin/env python3
"""
Metrics Collection Script for Open MoE Trainer Lab

Comprehensive metrics collection from various sources including:
- GitHub API (commits, PRs, issues, releases)
- CI/CD systems (build times, success rates)
- Code quality tools (coverage, complexity, security)
- Performance benchmarks (training, inference)
- Infrastructure monitoring (resource usage, uptime)
- User analytics and adoption metrics
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import aiohttp
import subprocess
import re
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a single metric value with metadata."""
    name: str
    value: Union[int, float, str, bool]
    unit: Optional[str] = None
    timestamp: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Main metrics collection orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics: List[MetricValue] = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        default_config = {
            "github": {
                "token": os.getenv("GITHUB_TOKEN"),
                "owner": os.getenv("GITHUB_REPOSITORY_OWNER", "your-org"),
                "repo": os.getenv("GITHUB_REPOSITORY", "open-moe-trainer-lab").split("/")[-1],
                "api_url": "https://api.github.com"
            },
            "prometheus": {
                "url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
                "timeout": 30
            },
            "grafana": {
                "url": os.getenv("GRAFANA_URL"),
                "token": os.getenv("GRAFANA_TOKEN")
            },
            "collection": {
                "lookback_days": 30,
                "batch_size": 100,
                "timeout": 60
            },
            "output": {
                "format": "json",
                "file": ".github/project-metrics.json",
                "pretty": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._deep_update(default_config, user_config)
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config["collection"]["timeout"])
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def collect_github_metrics(self) -> List[MetricValue]:
        """Collect metrics from GitHub API."""
        logger.info("Collecting GitHub metrics...")
        metrics = []
        
        if not self.config["github"]["token"]:
            logger.warning("GitHub token not provided, skipping GitHub metrics")
            return metrics
        
        headers = {
            "Authorization": f"token {self.config['github']['token']}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        base_url = self.config["github"]["api_url"]
        owner = self.config["github"]["owner"]
        repo = self.config["github"]["repo"]
        
        try:
            # Repository information
            repo_url = f"{base_url}/repos/{owner}/{repo}"
            async with self.session.get(repo_url, headers=headers) as response:
                if response.status == 200:
                    repo_data = await response.json()
                    
                    metrics.extend([
                        MetricValue("github_stars", repo_data["stargazers_count"], "count"),
                        MetricValue("github_forks", repo_data["forks_count"], "count"),
                        MetricValue("github_watchers", repo_data["watchers_count"], "count"),
                        MetricValue("github_open_issues", repo_data["open_issues_count"], "count"),
                        MetricValue("github_repo_size", repo_data["size"], "kb"),
                    ])
            
            # Recent commits
            since_date = (datetime.utcnow() - timedelta(days=self.config["collection"]["lookback_days"])).isoformat()
            commits_url = f"{base_url}/repos/{owner}/{repo}/commits"
            params = {"since": since_date, "per_page": 100}
            
            async with self.session.get(commits_url, headers=headers, params=params) as response:
                if response.status == 200:
                    commits = await response.json()
                    metrics.append(MetricValue("github_commits_30d", len(commits), "count"))
                    
                    # Analyze commit authors
                    authors = set()
                    for commit in commits:
                        if commit["author"]:
                            authors.add(commit["author"]["login"])
                    metrics.append(MetricValue("github_active_contributors_30d", len(authors), "count"))
            
            # Pull requests
            prs_url = f"{base_url}/repos/{owner}/{repo}/pulls"
            params = {"state": "all", "per_page": 100}
            
            async with self.session.get(prs_url, headers=headers, params=params) as response:
                if response.status == 200:
                    prs = await response.json()
                    
                    recent_prs = [pr for pr in prs if 
                                  datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00')) > 
                                  datetime.utcnow().replace(tzinfo=None) - timedelta(days=30)]
                    
                    metrics.append(MetricValue("github_prs_30d", len(recent_prs), "count"))
                    
                    # Calculate average merge time
                    merged_prs = [pr for pr in recent_prs if pr["merged_at"]]
                    if merged_prs:
                        total_time = sum([
                            (datetime.fromisoformat(pr["merged_at"].replace('Z', '+00:00')) - 
                             datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00'))).total_seconds()
                            for pr in merged_prs
                        ])
                        avg_merge_time = total_time / len(merged_prs) / 3600  # hours
                        metrics.append(MetricValue("github_avg_pr_merge_time", avg_merge_time, "hours"))
            
            # Issues
            issues_url = f"{base_url}/repos/{owner}/{repo}/issues"
            params = {"state": "all", "per_page": 100}
            
            async with self.session.get(issues_url, headers=headers, params=params) as response:
                if response.status == 200:
                    issues = await response.json()
                    
                    # Filter out PRs (GitHub includes PRs in issues endpoint)
                    actual_issues = [issue for issue in issues if "pull_request" not in issue]
                    
                    recent_issues = [issue for issue in actual_issues if 
                                    datetime.fromisoformat(issue["created_at"].replace('Z', '+00:00')) > 
                                    datetime.utcnow().replace(tzinfo=None) - timedelta(days=30)]
                    
                    metrics.append(MetricValue("github_issues_30d", len(recent_issues), "count"))
                    
                    # Calculate resolution time
                    closed_issues = [issue for issue in recent_issues if issue["closed_at"]]
                    if closed_issues:
                        total_time = sum([
                            (datetime.fromisoformat(issue["closed_at"].replace('Z', '+00:00')) - 
                             datetime.fromisoformat(issue["created_at"].replace('Z', '+00:00'))).total_seconds()
                            for issue in closed_issues
                        ])
                        avg_resolution_time = total_time / len(closed_issues) / 3600  # hours
                        metrics.append(MetricValue("github_avg_issue_resolution_time", avg_resolution_time, "hours"))
            
            # Releases
            releases_url = f"{base_url}/repos/{owner}/{repo}/releases"
            async with self.session.get(releases_url, headers=headers) as response:
                if response.status == 200:
                    releases = await response.json()
                    metrics.append(MetricValue("github_releases_total", len(releases), "count"))
                    
                    if releases:
                        latest_release = releases[0]
                        release_date = datetime.fromisoformat(latest_release["published_at"].replace('Z', '+00:00'))
                        days_since_release = (datetime.utcnow().replace(tzinfo=None) - release_date).days
                        metrics.append(MetricValue("github_days_since_last_release", days_since_release, "days"))
            
        except Exception as e:
            logger.error(f"Error collecting GitHub metrics: {e}")
        
        logger.info(f"Collected {len(metrics)} GitHub metrics")
        return metrics
    
    async def collect_prometheus_metrics(self) -> List[MetricValue]:
        """Collect metrics from Prometheus."""
        logger.info("Collecting Prometheus metrics...")
        metrics = []
        
        if not self.config["prometheus"]["url"]:
            logger.warning("Prometheus URL not configured, skipping Prometheus metrics")
            return metrics
        
        prometheus_url = self.config["prometheus"]["url"]
        
        # Define key metrics to collect
        queries = {
            "training_throughput": "moe_lab:training_throughput_tokens_per_second",
            "inference_latency_p99": "moe_lab:inference_latency_p99",
            "gpu_memory_utilization": "moe_lab:gpu_memory_utilization",
            "expert_utilization_variance": "moe_lab:expert_utilization_variance",
            "system_cpu_usage": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "system_memory_usage": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "uptime": "up"
        }
        
        try:
            for metric_name, query in queries.items():
                query_url = f"{prometheus_url}/api/v1/query"
                params = {"query": query}
                
                async with self.session.get(query_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data["status"] == "success" and data["data"]["result"]:
                            result = data["data"]["result"][0]
                            value = float(result["value"][1])
                            
                            metrics.append(MetricValue(
                                f"prometheus_{metric_name}",
                                value,
                                self._get_metric_unit(metric_name),
                                timestamp=datetime.utcnow().isoformat()
                            ))
                        
        except Exception as e:
            logger.error(f"Error collecting Prometheus metrics: {e}")
        
        logger.info(f"Collected {len(metrics)} Prometheus metrics")
        return metrics
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric."""
        unit_map = {
            "throughput": "tokens/sec",
            "latency": "ms",
            "utilization": "percent",
            "variance": "ratio",
            "usage": "percent",
            "uptime": "bool"
        }
        
        for key, unit in unit_map.items():
            if key in metric_name:
                return unit
        return "count"
    
    def collect_code_quality_metrics(self) -> List[MetricValue]:
        """Collect code quality metrics from local tools."""
        logger.info("Collecting code quality metrics...")
        metrics = []
        
        try:
            # Test coverage
            if Path("coverage.xml").exists():
                coverage_info = self._parse_coverage_xml("coverage.xml")
                if coverage_info:
                    metrics.append(MetricValue("test_coverage_percentage", coverage_info["line_rate"] * 100, "percent"))
                    metrics.append(MetricValue("test_lines_covered", coverage_info["lines_covered"], "count"))
                    metrics.append(MetricValue("test_lines_valid", coverage_info["lines_valid"], "count"))
            
            # Pytest results
            if Path("junit.xml").exists():
                test_results = self._parse_junit_xml("junit.xml")
                if test_results:
                    metrics.extend([
                        MetricValue("test_total", test_results["tests"], "count"),
                        MetricValue("test_failures", test_results["failures"], "count"),
                        MetricValue("test_errors", test_results["errors"], "count"),
                        MetricValue("test_time", test_results["time"], "seconds")
                    ])
            
            # Code complexity (if available)
            complexity_info = self._get_code_complexity()
            if complexity_info:
                metrics.extend([
                    MetricValue("code_complexity_avg", complexity_info["avg_complexity"], "score"),
                    MetricValue("code_complexity_max", complexity_info["max_complexity"], "score")
                ])
            
            # Lines of code
            loc_info = self._count_lines_of_code()
            if loc_info:
                metrics.extend([
                    MetricValue("lines_of_code_total", loc_info["total"], "lines"),
                    MetricValue("lines_of_code_python", loc_info["python"], "lines"),
                    MetricValue("lines_of_code_comments", loc_info["comments"], "lines")
                ])
                
        except Exception as e:
            logger.error(f"Error collecting code quality metrics: {e}")
        
        logger.info(f"Collected {len(metrics)} code quality metrics")
        return metrics
    
    def _parse_coverage_xml(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Parse coverage XML file."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            return {
                "line_rate": float(root.attrib["line-rate"]),
                "lines_covered": int(root.attrib["lines-covered"]),
                "lines_valid": int(root.attrib["lines-valid"])
            }
        except Exception as e:
            logger.warning(f"Failed to parse coverage XML: {e}")
            return None
    
    def _parse_junit_xml(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Parse JUnit XML file."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            return {
                "tests": int(root.attrib["tests"]),
                "failures": int(root.attrib["failures"]),
                "errors": int(root.attrib["errors"]),
                "time": float(root.attrib["time"])
            }
        except Exception as e:
            logger.warning(f"Failed to parse JUnit XML: {e}")
            return None
    
    def _get_code_complexity(self) -> Optional[Dict[str, float]]:
        """Get code complexity metrics using radon or similar tools."""
        try:
            result = subprocess.run(
                ["radon", "cc", "moe_lab/", "-a", "-j"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                complexities = []
                
                for file_path, functions in data.items():
                    for func in functions:
                        complexities.append(func["complexity"])
                
                if complexities:
                    return {
                        "avg_complexity": sum(complexities) / len(complexities),
                        "max_complexity": max(complexities)
                    }
        except Exception as e:
            logger.debug(f"Code complexity analysis failed: {e}")
        
        return None
    
    def _count_lines_of_code(self) -> Optional[Dict[str, int]]:
        """Count lines of code in the project."""
        try:
            # Use cloc if available
            result = subprocess.run(
                ["cloc", "moe_lab/", "--json"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                python_data = data.get("Python", {})
                total_data = data.get("SUM", {})
                
                return {
                    "total": total_data.get("code", 0),
                    "python": python_data.get("code", 0),
                    "comments": total_data.get("comment", 0)
                }
        except Exception:
            # Fallback to simple line counting
            try:
                python_files = list(Path("moe_lab").rglob("*.py"))
                total_lines = 0
                
                for file_path in python_files:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines += len(f.readlines())
                
                return {"total": total_lines, "python": total_lines, "comments": 0}
            except Exception as e:
                logger.debug(f"Line counting failed: {e}")
        
        return None
    
    def collect_performance_metrics(self) -> List[MetricValue]:
        """Collect performance benchmark metrics."""
        logger.info("Collecting performance metrics...")
        metrics = []
        
        # Check for benchmark results
        benchmark_files = [
            "benchmark-results.json",
            "performance-results.json",
            "training-benchmark.json",
            "inference-benchmark.json"
        ]
        
        for filename in benchmark_files:
            if Path(filename).exists():
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    
                    if "benchmarks" in data:
                        for benchmark in data["benchmarks"]:
                            name = benchmark["name"].replace("test_", "").replace("::", "_")
                            stats = benchmark["stats"]
                            
                            metrics.extend([
                                MetricValue(f"benchmark_{name}_mean", stats["mean"], "seconds"),
                                MetricValue(f"benchmark_{name}_min", stats["min"], "seconds"),
                                MetricValue(f"benchmark_{name}_max", stats["max"], "seconds"),
                                MetricValue(f"benchmark_{name}_stddev", stats["stddev"], "seconds")
                            ])
                            
                except Exception as e:
                    logger.warning(f"Failed to parse {filename}: {e}")
        
        logger.info(f"Collected {len(metrics)} performance metrics")
        return metrics
    
    def collect_security_metrics(self) -> List[MetricValue]:
        """Collect security-related metrics."""
        logger.info("Collecting security metrics...")
        metrics = []
        
        try:
            # Parse security scan results
            security_files = [
                "bandit-report.json",
                "safety-report.json",
                "trivy-results.json"
            ]
            
            vulnerability_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            
            for filename in security_files:
                if Path(filename).exists():
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    
                    # Parse different security tool formats
                    if "results" in data:  # Bandit format
                        for result in data["results"]:
                            severity = result.get("issue_severity", "low").lower()
                            if severity in vulnerability_counts:
                                vulnerability_counts[severity] += 1
                    
                    elif "vulnerabilities" in data:  # Safety format
                        for vuln in data["vulnerabilities"]:
                            # Safety typically reports as high priority
                            vulnerability_counts["high"] += 1
            
            # Add vulnerability metrics
            for severity, count in vulnerability_counts.items():
                metrics.append(MetricValue(f"security_vulnerabilities_{severity}", count, "count"))
            
            total_vulns = sum(vulnerability_counts.values())
            metrics.append(MetricValue("security_vulnerabilities_total", total_vulns, "count"))
            
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
        
        logger.info(f"Collected {len(metrics)} security metrics")
        return metrics
    
    async def collect_all_metrics(self) -> List[MetricValue]:
        """Collect all metrics from all sources."""
        logger.info("Starting comprehensive metrics collection...")
        
        all_metrics = []
        
        # Collect from different sources
        collectors = [
            self.collect_github_metrics(),
            self.collect_prometheus_metrics(),
        ]
        
        # Run async collectors
        results = await asyncio.gather(*collectors, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_metrics.extend(result)
            else:
                logger.error(f"Collector failed: {result}")
        
        # Run sync collectors
        all_metrics.extend(self.collect_code_quality_metrics())
        all_metrics.extend(self.collect_performance_metrics())
        all_metrics.extend(self.collect_security_metrics())
        
        # Add collection metadata
        all_metrics.append(MetricValue(
            "metrics_collection_total",
            len(all_metrics),
            "count",
            timestamp=datetime.utcnow().isoformat()
        ))
        
        self.metrics = all_metrics
        logger.info(f"Collected {len(all_metrics)} total metrics")
        return all_metrics
    
    def save_metrics(self, output_file: Optional[str] = None) -> None:
        """Save collected metrics to file."""
        if not self.metrics:
            logger.warning("No metrics to save")
            return
        
        output_file = output_file or self.config["output"]["file"]
        
        # Organize metrics by category
        organized_metrics = {}
        for metric in self.metrics:
            category = metric.name.split('_')[0]
            if category not in organized_metrics:
                organized_metrics[category] = {}
            
            organized_metrics[category][metric.name] = {
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp,
                "tags": metric.tags
            }
        
        # Create output structure
        output_data = {
            "schema_version": "1.0",
            "collection_timestamp": datetime.utcnow().isoformat(),
            "metrics": organized_metrics,
            "summary": {
                "total_metrics": len(self.metrics),
                "categories": list(organized_metrics.keys()),
                "collection_duration": "N/A"  # Could be calculated
            }
        }
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if self.config["output"]["pretty"]:
                json.dump(output_data, f, indent=2, default=str)
            else:
                json.dump(output_data, f, default=str)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print a summary of collected metrics."""
        if not self.metrics:
            print("No metrics collected")
            return
        
        # Group by category
        categories = {}
        for metric in self.metrics:
            category = metric.name.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(metric)
        
        print("\n📊 Metrics Collection Summary")
        print("=" * 50)
        print(f"Total metrics: {len(self.metrics)}")
        print(f"Categories: {len(categories)}")
        print(f"Collection time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        for category, metrics in categories.items():
            print(f"\n{category.title()}: {len(metrics)} metrics")
            for metric in metrics[:5]:  # Show first 5 metrics per category
                unit = f" {metric.unit}" if metric.unit else ""
                print(f"  • {metric.name}: {metric.value}{unit}")
            if len(metrics) > 5:
                print(f"  ... and {len(metrics) - 5} more")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument("--categories", nargs="+", help="Specific categories to collect",
                        choices=["github", "prometheus", "quality", "performance", "security"])
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async with MetricsCollector(args.config) as collector:
        # Override config with command line args
        if args.output:
            collector.config["output"]["file"] = args.output
        if args.pretty:
            collector.config["output"]["pretty"] = True
        
        # Collect metrics
        metrics = await collector.collect_all_metrics()
        
        # Save results
        collector.save_metrics()
        
        # Print summary if requested
        if args.summary:
            collector.print_summary()
        
        print(f"\n✅ Successfully collected {len(metrics)} metrics")


if __name__ == "__main__":
    asyncio.run(main())