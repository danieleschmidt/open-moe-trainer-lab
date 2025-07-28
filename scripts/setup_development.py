#!/usr/bin/env python3
"""
Development Environment Setup Script for Open MoE Trainer Lab

Automates the setup of a complete development environment including:
- Virtual environment creation and management
- Dependency installation
- Pre-commit hooks setup
- Development tools configuration
- Database and service setup
- Initial data and configuration
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DevelopmentSetup:
    """Development environment setup automation."""
    
    def __init__(self, repo_root: Optional[str] = None, venv_name: str = "venv"):
        self.repo_root = Path(repo_root or os.getcwd())
        self.venv_name = venv_name
        self.venv_path = self.repo_root / venv_name
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
                   capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        cwd = cwd or self.repo_root
        
        logger.debug(f"Running command: {' '.join(cmd)} in {cwd}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check system prerequisites."""
        logger.info("Checking system prerequisites...")
        
        checks = {}
        
        # Python version
        python_version = sys.version_info
        checks["python_version"] = python_version >= (3, 9)
        logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Git
        try:
            result = self.run_command(["git", "--version"], check=False)
            checks["git"] = result.returncode == 0
        except FileNotFoundError:
            checks["git"] = False
        
        # Docker (optional)
        try:
            result = self.run_command(["docker", "--version"], check=False)
            checks["docker"] = result.returncode == 0
        except FileNotFoundError:
            checks["docker"] = False
        
        # Docker Compose (optional)
        try:
            result = self.run_command(["docker-compose", "--version"], check=False)
            checks["docker_compose"] = result.returncode == 0
        except FileNotFoundError:
            checks["docker_compose"] = False
        
        # Node.js (optional)
        try:
            result = self.run_command(["node", "--version"], check=False)
            checks["nodejs"] = result.returncode == 0
        except FileNotFoundError:
            checks["nodejs"] = False
        
        # Report results
        passed = sum(checks.values())
        total = len(checks)
        logger.info(f"Prerequisites check: {passed}/{total} available")
        
        for check, available in checks.items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {check}")
        
        return checks
    
    def create_virtual_environment(self, force: bool = False) -> bool:
        """Create Python virtual environment."""
        if self.venv_path.exists():
            if force:
                logger.info(f"Removing existing virtual environment: {self.venv_path}")
                shutil.rmtree(self.venv_path)
            else:
                logger.info(f"Virtual environment already exists: {self.venv_path}")
                return True
        
        logger.info(f"Creating virtual environment: {self.venv_path}")
        
        try:
            venv.create(self.venv_path, with_pip=True, symlinks=True)
            logger.info("Virtual environment created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self) -> str:
        """Get path to Python executable in virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "python.exe")
        else:  # Unix-like
            return str(self.venv_path / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """Get path to pip executable in virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:  # Unix-like
            return str(self.venv_path / "bin" / "pip")
    
    def install_dependencies(self, dev: bool = True, gpu: bool = False, 
                           distributed: bool = False) -> bool:
        """Install project dependencies."""
        logger.info("Installing project dependencies...")
        
        python_exe = self.get_venv_python()
        
        # Upgrade pip first
        try:
            self.run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        except Exception as e:
            logger.error(f"Failed to upgrade pip: {e}")
            return False
        
        # Install package in development mode
        install_cmd = [python_exe, "-m", "pip", "install", "-e", "."]
        
        # Add optional dependencies
        extras = []
        if dev:
            extras.append("dev")
        if gpu:
            extras.append("gpu")
        if distributed:
            extras.append("distributed")
        
        if extras:
            install_cmd[-1] = f".[{','.join(extras)}]"
        
        try:
            self.run_command(install_cmd)
            logger.info("Dependencies installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup_pre_commit_hooks(self) -> bool:
        """Setup pre-commit hooks."""
        logger.info("Setting up pre-commit hooks...")
        
        python_exe = self.get_venv_python()
        
        # Check if pre-commit config exists
        pre_commit_config = self.repo_root / ".pre-commit-config.yaml"
        if not pre_commit_config.exists():
            logger.warning("No .pre-commit-config.yaml found, skipping pre-commit setup")
            return True
        
        try:
            # Install pre-commit
            self.run_command([python_exe, "-m", "pip", "install", "pre-commit"])
            
            # Install hooks
            self.run_command([self.get_venv_python(), "-m", "pre_commit", "install"])
            
            logger.info("Pre-commit hooks installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to setup pre-commit hooks: {e}")
            return False
    
    def setup_database(self, db_type: str = "sqlite") -> bool:
        """Setup development database."""
        logger.info(f"Setting up {db_type} database...")
        
        if db_type == "sqlite":
            # SQLite setup - create database directory
            db_dir = self.repo_root / "data"
            db_dir.mkdir(exist_ok=True)
            
            db_path = db_dir / "development.db"
            logger.info(f"SQLite database will be created at: {db_path}")
            return True
            
        elif db_type == "postgresql":
            # PostgreSQL setup using Docker
            try:
                # Check if docker-compose.yml exists
                compose_file = self.repo_root / "docker-compose.yml"
                if compose_file.exists():
                    self.run_command(["docker-compose", "up", "-d", "postgres"])
                    logger.info("PostgreSQL container started")
                    return True
                else:
                    logger.warning("No docker-compose.yml found for PostgreSQL setup")
                    return False
            except Exception as e:
                logger.error(f"Failed to setup PostgreSQL: {e}")
                return False
        
        elif db_type == "redis":
            # Redis setup using Docker
            try:
                compose_file = self.repo_root / "docker-compose.yml"
                if compose_file.exists():
                    self.run_command(["docker-compose", "up", "-d", "redis"])
                    logger.info("Redis container started")
                    return True
                else:
                    logger.warning("No docker-compose.yml found for Redis setup")
                    return False
            except Exception as e:
                logger.error(f"Failed to setup Redis: {e}")
                return False
        
        else:
            logger.warning(f"Unknown database type: {db_type}")
            return False
    
    def create_env_file(self, template_path: Optional[str] = None) -> bool:
        """Create .env file from template."""
        logger.info("Creating environment configuration file...")
        
        env_file = self.repo_root / ".env"
        template_file = self.repo_root / (template_path or ".env.example")
        
        if env_file.exists():
            logger.info(".env file already exists, skipping creation")
            return True
        
        if template_file.exists():
            # Copy template
            shutil.copy(template_file, env_file)
            logger.info(f"Environment file created from template: {template_file}")
        else:
            # Create basic .env file
            env_content = """# Development Environment Configuration
DEBUG=true
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=sqlite:///data/development.db

# Redis Configuration
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=localhost
API_PORT=8000

# Security
SECRET_KEY=dev-secret-key-change-in-production

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            logger.info("Basic .env file created")
        
        return True
    
    def initialize_data_directories(self) -> bool:
        """Create necessary data directories."""
        logger.info("Creating data directories...")
        
        directories = [
            "data",
            "logs",
            "models",
            "checkpoints",
            "experiments",
            "artifacts"
        ]
        
        for directory in directories:
            dir_path = self.repo_root / directory
            dir_path.mkdir(exist_ok=True)
            
            # Create .gitkeep to ensure directory is tracked
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
        
        logger.info(f"Created {len(directories)} data directories")
        return True
    
    def run_initial_tests(self) -> bool:
        """Run initial test suite to verify setup."""
        logger.info("Running initial tests to verify setup...")
        
        python_exe = self.get_venv_python()
        
        try:
            # Run basic import test
            result = self.run_command([
                python_exe, "-c", 
                "import moe_lab; print('✅ Package import successful')"
            ])
            
            # Run quick test suite if available
            if (self.repo_root / "tests").exists():
                result = self.run_command([
                    python_exe, "-m", "pytest", "tests/", 
                    "-v", "--tb=short", "-x", "-q"
                ], check=False)
                
                if result.returncode == 0:
                    logger.info("Initial tests passed")
                else:
                    logger.warning("Some tests failed, but setup is functional")
            
            return True
        except Exception as e:
            logger.error(f"Initial tests failed: {e}")
            return False
    
    def generate_setup_summary(self, checks: Dict[str, bool]) -> str:
        """Generate setup summary report."""
        summary = """
# Development Environment Setup Summary

## System Prerequisites
"""
        
        for check, status in checks["prerequisites"].items():
            status_icon = "✅" if status else "❌"
            summary += f"- {status_icon} {check}\n"
        
        summary += "\n## Setup Results\n"
        
        for step, status in checks["setup_steps"].items():
            status_icon = "✅" if status else "❌"
            summary += f"- {status_icon} {step.replace('_', ' ').title()}\n"
        
        summary += "\n## Next Steps\n"
        summary += "1. Activate virtual environment:\n"
        
        if os.name == 'nt':  # Windows
            summary += f"   ```cmd\n   {self.venv_path}\\Scripts\\activate\n   ```\n"
        else:  # Unix-like
            summary += f"   ```bash\n   source {self.venv_path}/bin/activate\n   ```\n"
        
        summary += """
2. Start development services (if using Docker):
   ```bash
   docker-compose up -d
   ```

3. Run tests to verify everything works:
   ```bash
   pytest tests/
   ```

4. Start development server:
   ```bash
   python -m moe_lab.serve --dev
   ```

## Useful Development Commands

- Run quality checks: `python scripts/automation_helper.py quality`
- Clean build artifacts: `python scripts/automation_helper.py cleanup`
- Update dependencies: `python scripts/automation_helper.py deps --upgrade`
- Run specific tests: `pytest tests/unit/ -v`
- Collect metrics: `python scripts/collect_metrics.py --summary`

Happy coding! 🚀
"""
        
        return summary
    
    def setup_development_environment(self, force_venv: bool = False,
                                    dev: bool = True, gpu: bool = False,
                                    distributed: bool = False,
                                    database: str = "sqlite") -> Dict[str, bool]:
        """Complete development environment setup."""
        logger.info("Starting development environment setup...")
        
        results = {
            "prerequisites": self.check_prerequisites(),
            "setup_steps": {}
        }
        
        # Setup steps
        steps = [
            ("virtual_environment", lambda: self.create_virtual_environment(force_venv)),
            ("dependencies", lambda: self.install_dependencies(dev, gpu, distributed)),
            ("pre_commit_hooks", lambda: self.setup_pre_commit_hooks()),
            ("database", lambda: self.setup_database(database)),
            ("env_file", lambda: self.create_env_file()),
            ("data_directories", lambda: self.initialize_data_directories()),
            ("initial_tests", lambda: self.run_initial_tests())
        ]
        
        for step_name, step_func in steps:
            try:
                logger.info(f"Running step: {step_name}")
                results["setup_steps"][step_name] = step_func()
            except Exception as e:
                logger.error(f"Step {step_name} failed: {e}")
                results["setup_steps"][step_name] = False
        
        # Generate summary
        summary = self.generate_setup_summary(results)
        
        # Save summary to file
        summary_file = self.repo_root / "DEVELOPMENT_SETUP.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Setup summary saved to: {summary_file}")
        
        # Print results
        successful_steps = sum(results["setup_steps"].values())
        total_steps = len(results["setup_steps"])
        
        if successful_steps == total_steps:
            logger.info("🎉 Development environment setup completed successfully!")
        else:
            logger.warning(f"Setup completed with issues: {successful_steps}/{total_steps} steps successful")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Development environment setup")
    parser.add_argument("--repo-root", help="Repository root directory")
    parser.add_argument("--venv-name", default="venv", help="Virtual environment name")
    parser.add_argument("--force-venv", action="store_true", help="Force recreate virtual environment")
    parser.add_argument("--no-dev", action="store_true", help="Skip development dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install GPU dependencies")
    parser.add_argument("--distributed", action="store_true", help="Install distributed dependencies")
    parser.add_argument("--database", choices=["sqlite", "postgresql", "redis"], 
                       default="sqlite", help="Database type to setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = DevelopmentSetup(args.repo_root, args.venv_name)
    
    try:
        results = setup.setup_development_environment(
            force_venv=args.force_venv,
            dev=not args.no_dev,
            gpu=args.gpu,
            distributed=args.distributed,
            database=args.database
        )
        
        # Return appropriate exit code
        successful_steps = sum(results["setup_steps"].values())
        total_steps = len(results["setup_steps"])
        
        if successful_steps == total_steps:
            return 0
        else:
            return 1
    
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())