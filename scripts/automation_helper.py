#!/usr/bin/env python3
"""
Automation Helper Script for Open MoE Trainer Lab

Provides utility functions and automation capabilities for:
- Repository maintenance tasks
- Continuous integration helpers
- Development workflow automation
- Performance monitoring automation
- Release preparation tasks
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import shutil
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutomationHelper:
    """Main automation helper class."""
    
    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = Path(repo_root or os.getcwd())
        self.ensure_repo_root()
    
    def ensure_repo_root(self):
        """Ensure we're operating from the repository root."""
        if not (self.repo_root / '.git').exists():
            # Try to find repo root
            current = self.repo_root
            while current.parent != current:
                if (current / '.git').exists():
                    self.repo_root = current
                    break
                current = current.parent
            else:
                raise RuntimeError(f"Not in a git repository: {self.repo_root}")
        
        os.chdir(self.repo_root)
        logger.info(f"Working from repository root: {self.repo_root}")
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   capture_output: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        cwd = cwd or self.repo_root
        
        logger.debug(f"Running command: {' '.join(cmd)} in {cwd}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=False
            )
            
            if result.returncode != 0:
                logger.warning(f"Command failed with code {result.returncode}: {' '.join(cmd)}")
                if result.stderr:
                    logger.warning(f"Error output: {result.stderr.strip()}")
            
            return result
        
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def cleanup_build_artifacts(self) -> bool:
        """Clean up build artifacts and temporary files."""
        logger.info("Cleaning up build artifacts...")
        
        cleanup_paths = [
            "build/", "dist/", "*.egg-info/",
            "__pycache__/", "*.pyc", "*.pyo",
            ".coverage", "coverage.xml", "junit.xml",
            ".pytest_cache/", ".mypy_cache/",
            "node_modules/", ".next/",
            "*.log", "*.tmp"
        ]
        
        cleaned_count = 0
        
        for pattern in cleanup_paths:
            if "*" in pattern or "?" in pattern:
                # Use glob for patterns
                for path in self.repo_root.rglob(pattern):
                    try:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Removed: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {path}: {e}")
            else:
                # Direct path
                path = self.repo_root / pattern
                if path.exists():
                    try:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Removed: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} artifacts")
        return True
    
    def update_dependencies(self, upgrade: bool = False) -> bool:
        """Update project dependencies."""
        logger.info("Updating dependencies...")
        
        # Python dependencies
        if (self.repo_root / "requirements.txt").exists():
            cmd = ["pip", "install", "-r", "requirements.txt"]
            if upgrade:
                cmd.append("--upgrade")
            
            result = self.run_command(cmd)
            if result.returncode != 0:
                logger.error("Failed to update Python dependencies")
                return False
        
        # Development dependencies
        if (self.repo_root / "pyproject.toml").exists() or (self.repo_root / "setup.py").exists():
            cmd = ["pip", "install", "-e", ".[dev]"]
            if upgrade:
                cmd.append("--upgrade")
            
            result = self.run_command(cmd)
            if result.returncode != 0:
                logger.error("Failed to install development dependencies")
                return False
        
        logger.info("Dependencies updated successfully")
        return True
    
    def run_quality_checks(self) -> Dict[str, bool]:
        """Run all code quality checks."""
        logger.info("Running code quality checks...")
        
        results = {}
        
        # Code formatting
        if shutil.which("black"):
            result = self.run_command(["black", "--check", "moe_lab/", "tests/"])
            results["formatting"] = result.returncode == 0
        
        # Import sorting
        if shutil.which("isort"):
            result = self.run_command(["isort", "--check-only", "moe_lab/", "tests/"])
            results["imports"] = result.returncode == 0
        
        # Linting
        if shutil.which("ruff"):
            result = self.run_command(["ruff", "check", "moe_lab/", "tests/"])
            results["linting"] = result.returncode == 0
        
        # Type checking
        if shutil.which("mypy"):
            result = self.run_command(["mypy", "moe_lab/"])
            results["typing"] = result.returncode == 0
        
        # Security linting
        if shutil.which("bandit"):
            result = self.run_command(["bandit", "-r", "moe_lab/", "-f", "json", "-o", "bandit-report.json"])
            results["security"] = result.returncode == 0
        
        # Report results
        passed = sum(results.values())
        total = len(results)
        logger.info(f"Quality checks: {passed}/{total} passed")
        
        for check, passed in results.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check}")
        
        return results
    
    def run_tests(self, test_type: str = "all", coverage: bool = True) -> bool:
        """Run tests with optional coverage."""
        logger.info(f"Running {test_type} tests...")
        
        cmd = ["pytest"]
        
        # Test selection
        if test_type == "unit":
            cmd.append("tests/unit/")
        elif test_type == "integration":
            cmd.append("tests/integration/")
        elif test_type == "e2e":
            cmd.append("tests/e2e/")
        elif test_type == "all":
            cmd.append("tests/")
        else:
            cmd.append(f"tests/{test_type}/")
        
        # Coverage options
        if coverage:
            cmd.extend([
                "--cov=moe_lab",
                "--cov-report=xml",
                "--cov-report=term-missing"
            ])
        
        # Output options
        cmd.extend([
            "-v",
            "--junitxml=junit.xml"
        ])
        
        result = self.run_command(cmd)
        
        if result.returncode == 0:
            logger.info("Tests passed successfully")
            return True
        else:
            logger.error("Tests failed")
            return False
    
    def generate_changelog(self, version: str, output_file: str = "CHANGELOG.md") -> bool:
        """Generate changelog from git history."""
        logger.info(f"Generating changelog for version {version}...")
        
        try:
            # Get previous tag
            result = self.run_command(["git", "describe", "--tags", "--abbrev=0", "HEAD^"])
            previous_tag = result.stdout.strip() if result.returncode == 0 else None
            
            # Get commits since previous tag
            if previous_tag:
                result = self.run_command(["git", "log", f"{previous_tag}..HEAD", "--pretty=format:%s (%h)"])
            else:
                result = self.run_command(["git", "log", "--pretty=format:%s (%h)"])
            
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Read existing changelog
            changelog_path = self.repo_root / output_file
            existing_content = ""
            if changelog_path.exists():
                with open(changelog_path, 'r') as f:
                    existing_content = f.read()
            
            # Generate new entry
            new_entry = f"""# Changelog

## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

### Changed
"""
            
            for commit in commits:
                if commit.strip():
                    new_entry += f"- {commit}\n"
            
            new_entry += "\n"
            
            # Merge with existing content
            if existing_content and not existing_content.startswith("# Changelog"):
                new_content = new_entry + existing_content
            else:
                # Insert after header
                lines = existing_content.split('\n')
                if lines and lines[0].startswith("# Changelog"):
                    new_content = lines[0] + "\n\n" + new_entry[len("# Changelog\n\n"):] + "\n".join(lines[1:])
                else:
                    new_content = new_entry + existing_content
            
            # Write changelog
            with open(changelog_path, 'w') as f:
                f.write(new_content)
            
            logger.info(f"Changelog updated: {changelog_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate changelog: {e}")
            return False
    
    def check_release_readiness(self, version: str) -> Dict[str, bool]:
        """Check if repository is ready for release."""
        logger.info(f"Checking release readiness for version {version}...")
        
        checks = {}
        
        # Check git status
        result = self.run_command(["git", "status", "--porcelain"])
        checks["clean_working_tree"] = len(result.stdout.strip()) == 0
        
        # Check if on main branch
        result = self.run_command(["git", "branch", "--show-current"])
        checks["on_main_branch"] = result.stdout.strip() in ["main", "master"]
        
        # Check if tests pass
        checks["tests_pass"] = self.run_tests(coverage=False)
        
        # Check if version is valid
        import re
        checks["valid_version"] = bool(re.match(r'^\d+\.\d+\.\d+(-\w+)?$', version))
        
        # Check if tag doesn't exist
        result = self.run_command(["git", "tag", "-l", f"v{version}"])
        checks["tag_available"] = len(result.stdout.strip()) == 0
        
        # Check required files
        required_files = ["README.md", "LICENSE", "pyproject.toml"]
        for file in required_files:
            checks[f"has_{file.lower().replace('.', '_')}"] = (self.repo_root / file).exists()
        
        # Report results
        passed = sum(checks.values())
        total = len(checks)
        logger.info(f"Release readiness: {passed}/{total} checks passed")
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check}")
        
        return checks
    
    def create_performance_baseline(self, output_file: str = "performance-baseline.json") -> bool:
        """Create performance baseline from current benchmarks."""
        logger.info("Creating performance baseline...")
        
        # Run benchmarks
        result = self.run_command([
            "pytest", "tests/benchmarks/",
            "--benchmark-only",
            "--benchmark-json=benchmark-results.json"
        ])
        
        if result.returncode != 0:
            logger.error("Failed to run benchmarks")
            return False
        
        # Process results
        try:
            with open("benchmark-results.json", 'r') as f:
                benchmark_data = json.load(f)
            
            baseline = {
                "created_at": datetime.now().isoformat(),
                "commit": self.run_command(["git", "rev-parse", "HEAD"]).stdout.strip(),
                "benchmarks": {}
            }
            
            for benchmark in benchmark_data.get("benchmarks", []):
                name = benchmark["name"]
                stats = benchmark["stats"]
                
                baseline["benchmarks"][name] = {
                    "mean": stats["mean"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "stddev": stats["stddev"]
                }
            
            # Save baseline
            with open(output_file, 'w') as f:
                json.dump(baseline, f, indent=2)
            
            logger.info(f"Performance baseline created: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create performance baseline: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive repository health check."""
        logger.info("Performing repository health check...")
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {}
        }
        
        # Git repository health
        result = self.run_command(["git", "fsck"])
        health["checks"]["git_integrity"] = result.returncode == 0
        
        # Dependencies health
        if (self.repo_root / "pyproject.toml").exists():
            result = self.run_command(["pip", "check"])
            health["checks"]["dependencies"] = result.returncode == 0
        
        # Code quality
        quality_results = self.run_quality_checks()
        health["checks"]["code_quality"] = all(quality_results.values())
        health["checks"]["quality_details"] = quality_results
        
        # Test health
        health["checks"]["tests"] = self.run_tests(coverage=False)
        
        # File permissions
        executable_files = [
            "scripts/collect_metrics.py",
            "scripts/build.sh",
            "scripts/deploy.sh"
        ]
        
        permission_issues = 0
        for file in executable_files:
            path = self.repo_root / file
            if path.exists() and not os.access(path, os.X_OK):
                permission_issues += 1
        
        health["checks"]["file_permissions"] = permission_issues == 0
        
        # Calculate overall status
        passed_checks = sum(
            1 for check, result in health["checks"].items()
            if isinstance(result, bool) and result
        )
        total_checks = sum(
            1 for check, result in health["checks"].items()
            if isinstance(result, bool)
        )
        
        if passed_checks == total_checks:
            health["overall_status"] = "healthy"
        elif passed_checks >= total_checks * 0.8:
            health["overall_status"] = "warning"
        else:
            health["overall_status"] = "critical"
        
        health["checks"]["summary"] = f"{passed_checks}/{total_checks} checks passed"
        
        logger.info(f"Repository health: {health['overall_status']} ({health['checks']['summary']})")
        return health


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Repository automation helper")
    parser.add_argument("--repo-root", help="Repository root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up build artifacts")
    
    # Dependencies command
    deps_parser = subparsers.add_parser("deps", help="Update dependencies")
    deps_parser.add_argument("--upgrade", action="store_true", help="Upgrade to latest versions")
    
    # Quality command
    quality_parser = subparsers.add_parser("quality", help="Run quality checks")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--type", choices=["unit", "integration", "e2e", "all"], 
                           default="all", help="Type of tests to run")
    test_parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    
    # Changelog command
    changelog_parser = subparsers.add_parser("changelog", help="Generate changelog")
    changelog_parser.add_argument("version", help="Version for changelog entry")
    changelog_parser.add_argument("--output", default="CHANGELOG.md", help="Output file")
    
    # Release readiness command
    release_parser = subparsers.add_parser("release-check", help="Check release readiness")
    release_parser.add_argument("version", help="Version to check for release")
    
    # Performance baseline command
    perf_parser = subparsers.add_parser("perf-baseline", help="Create performance baseline")
    perf_parser.add_argument("--output", default="performance-baseline.json", help="Output file")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Repository health check")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    helper = AutomationHelper(args.repo_root)
    
    try:
        if args.command == "cleanup":
            success = helper.cleanup_build_artifacts()
            return 0 if success else 1
            
        elif args.command == "deps":
            success = helper.update_dependencies(upgrade=args.upgrade)
            return 0 if success else 1
            
        elif args.command == "quality":
            results = helper.run_quality_checks()
            return 0 if all(results.values()) else 1
            
        elif args.command == "test":
            success = helper.run_tests(
                test_type=args.type,
                coverage=not args.no_coverage
            )
            return 0 if success else 1
            
        elif args.command == "changelog":
            success = helper.generate_changelog(args.version, args.output)
            return 0 if success else 1
            
        elif args.command == "release-check":
            results = helper.check_release_readiness(args.version)
            return 0 if all(results.values()) else 1
            
        elif args.command == "perf-baseline":
            success = helper.create_performance_baseline(args.output)
            return 0 if success else 1
            
        elif args.command == "health":
            health = helper.health_check()
            print(json.dumps(health, indent=2))
            return 0 if health["overall_status"] == "healthy" else 1
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())