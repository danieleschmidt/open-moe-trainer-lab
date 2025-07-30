#!/usr/bin/env python3
"""
Advanced Security Scanning for Open MoE Trainer Lab

This script provides comprehensive security scanning capabilities including:
- SLSA compliance verification
- Supply chain security analysis
- Container vulnerability scanning
- Code security analysis
- Dependency vulnerability assessment
- Model security validation
- Runtime security monitoring
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class AdvancedSecurityScanner:
    """Advanced security scanning and compliance verification."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "security/security-config.yaml"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "scan_results": {},
            "compliance_status": {},
            "recommendations": [],
            "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('security/security-scan.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_comprehensive_scan(self) -> Dict:
        """Run comprehensive security scanning."""
        self.logger.info("Starting comprehensive security scan...")
        
        # Code security analysis
        self.results["scan_results"]["code_security"] = self._scan_code_security()
        
        # Dependency vulnerabilities
        self.results["scan_results"]["dependencies"] = self._scan_dependencies()
        
        # Container security
        self.results["scan_results"]["containers"] = self._scan_containers()
        
        # SLSA compliance
        self.results["compliance_status"]["slsa"] = self._verify_slsa_compliance()
        
        # Supply chain security
        self.results["scan_results"]["supply_chain"] = self._analyze_supply_chain()
        
        # Model security
        self.results["scan_results"]["model_security"] = self._scan_model_security()
        
        # Configuration security
        self.results["scan_results"]["configuration"] = self._scan_configuration_security()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Calculate overall security score
        self.results["security_score"] = self._calculate_security_score()
        
        self.logger.info(f"Security scan completed. Score: {self.results['security_score']}/100")
        return self.results
    
    def _scan_code_security(self) -> Dict:
        """Scan source code for security vulnerabilities."""
        self.logger.info("Scanning code security...")
        
        results = {
            "bandit": self._run_bandit_scan(),
            "semgrep": self._run_semgrep_scan(),
            "secrets": self._scan_for_secrets(),
            "code_quality": self._analyze_code_quality()
        }
        
        return results
    
    def _run_bandit_scan(self) -> Dict:
        """Run Bandit security scanner."""
        try:
            cmd = ["bandit", "-r", "moe_lab/", "-f", "json", "-o", "security/bandit-results.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if os.path.exists("security/bandit-results.json"):
                with open("security/bandit-results.json") as f:
                    bandit_results = json.load(f)
                
                return {
                    "status": "completed",
                    "issues_found": len(bandit_results.get("results", [])),
                    "high_severity": len([r for r in bandit_results.get("results", []) 
                                        if r.get("issue_severity") == "HIGH"]),
                    "medium_severity": len([r for r in bandit_results.get("results", []) 
                                          if r.get("issue_severity") == "MEDIUM"]),
                    "results_file": "security/bandit-results.json"
                }
            else:
                return {"status": "failed", "error": "Results file not generated"}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Bandit scan timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _run_semgrep_scan(self) -> Dict:
        """Run Semgrep security scanner."""
        try:
            # Use Semgrep with security rulesets
            cmd = [
                "semgrep", "--config=auto", "--json", "--output=security/semgrep-results.json",
                "--exclude=tests/", "--exclude=docs/", "."
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if os.path.exists("security/semgrep-results.json"):
                with open("security/semgrep-results.json") as f:
                    semgrep_results = json.load(f)
                
                findings = semgrep_results.get("results", [])
                
                return {
                    "status": "completed",
                    "total_findings": len(findings),
                    "critical": len([f for f in findings if f.get("extra", {}).get("severity") == "ERROR"]),
                    "high": len([f for f in findings if f.get("extra", {}).get("severity") == "WARNING"]),
                    "results_file": "security/semgrep-results.json"
                }
            else:
                return {"status": "failed", "error": "Results file not generated"}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Semgrep scan timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _scan_for_secrets(self) -> Dict:
        """Scan for exposed secrets and credentials."""
        try:
            # Use detect-secrets if available
            cmd = ["detect-secrets", "scan", "--all-files", "--baseline", ".secrets.baseline"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse baseline file to count secrets
                if os.path.exists(".secrets.baseline"):
                    with open(".secrets.baseline") as f:
                        baseline = json.load(f)
                    
                    total_secrets = sum(len(files) for files in baseline.get("results", {}).values())
                else:
                    total_secrets = 0
                
                return {
                    "status": "completed",
                    "secrets_found": total_secrets,
                    "baseline_updated": True
                }
            else:
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _analyze_code_quality(self) -> Dict:
        """Analyze code quality from security perspective."""
        try:
            # Run ruff with security focus
            cmd = ["ruff", "check", "--select", "S", "--format", "json", "moe_lab/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                ruff_results = json.loads(result.stdout)
                
                return {
                    "status": "completed",
                    "security_issues": len(ruff_results),
                    "issues": ruff_results[:10]  # First 10 issues
                }
            else:
                return {"status": "completed", "security_issues": 0}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _scan_dependencies(self) -> Dict:
        """Scan dependencies for known vulnerabilities."""
        self.logger.info("Scanning dependencies...")
        
        results = {
            "safety": self._run_safety_scan(),
            "audit": self._run_pip_audit(),
            "outdated": self._check_outdated_packages(),
            "licenses": self._check_licenses()
        }
        
        return results
    
    def _run_safety_scan(self) -> Dict:
        """Run Safety dependency vulnerability scanner."""
        try:
            cmd = ["safety", "check", "--json", "--output", "security/safety-results.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if os.path.exists("security/safety-results.json"):
                with open("security/safety-results.json") as f:
                    safety_results = json.load(f)
                
                return {
                    "status": "completed",
                    "vulnerabilities_found": len(safety_results),
                    "critical": len([v for v in safety_results if "critical" in v.get("vulnerability_id", "").lower()]),
                    "results_file": "security/safety-results.json"
                }
            else:
                return {"status": "completed", "vulnerabilities_found": 0}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Safety scan timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _run_pip_audit(self) -> Dict:
        """Run pip-audit for dependency vulnerabilities."""
        try:
            cmd = ["pip-audit", "--format=json", "--output=security/pip-audit-results.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if os.path.exists("security/pip-audit-results.json"):
                with open("security/pip-audit-results.json") as f:
                    audit_results = json.load(f)
                
                vulnerabilities = audit_results.get("vulnerabilities", [])
                
                return {
                    "status": "completed",
                    "vulnerabilities_found": len(vulnerabilities),
                    "high_severity": len([v for v in vulnerabilities if "high" in str(v).lower()]),
                    "results_file": "security/pip-audit-results.json"
                }
            else:
                return {"status": "completed", "vulnerabilities_found": 0}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_outdated_packages(self) -> Dict:
        """Check for outdated packages."""
        try:
            cmd = ["pip", "list", "--outdated", "--format=json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                
                return {
                    "status": "completed",
                    "outdated_packages": len(outdated),
                    "packages": [{"name": p["name"], "current": p["version"], 
                                "latest": p["latest_version"]} for p in outdated[:10]]
                }
            else:
                return {"status": "completed", "outdated_packages": 0}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_licenses(self) -> Dict:
        """Check package licenses for compliance."""
        try:
            cmd = ["pip-licenses", "--format=json", "--output-file=security/licenses.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if os.path.exists("security/licenses.json"):
                with open("security/licenses.json") as f:
                    licenses = json.load(f)
                
                # Check for problematic licenses
                problematic = ["GPL-3.0", "AGPL-3.0", "GPL-2.0"]
                issues = [pkg for pkg in licenses 
                         if pkg.get("License") in problematic]
                
                return {
                    "status": "completed",
                    "total_packages": len(licenses),
                    "license_issues": len(issues),
                    "problematic_licenses": issues
                }
            else:
                return {"status": "failed", "error": "License check failed"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _scan_containers(self) -> Dict:
        """Scan container images for vulnerabilities."""
        self.logger.info("Scanning containers...")
        
        results = {
            "trivy": self._run_trivy_scan(),
            "dockerfile": self._analyze_dockerfile_security(),
            "runtime": self._check_runtime_security()
        }
        
        return results
    
    def _run_trivy_scan(self) -> Dict:
        """Run Trivy container vulnerability scanner."""
        try:
            # Scan the built image if it exists
            image_name = "open-moe-trainer-lab:latest"
            
            cmd = [
                "trivy", "image", "--format", "json", 
                "--output", "security/trivy-results.json", image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if os.path.exists("security/trivy-results.json"):
                with open("security/trivy-results.json") as f:
                    trivy_results = json.load(f)
                
                vulnerabilities = []
                for result in trivy_results.get("Results", []):
                    vulnerabilities.extend(result.get("Vulnerabilities", []))
                
                critical = len([v for v in vulnerabilities if v.get("Severity") == "CRITICAL"])
                high = len([v for v in vulnerabilities if v.get("Severity") == "HIGH"])
                
                return {
                    "status": "completed",
                    "total_vulnerabilities": len(vulnerabilities),
                    "critical": critical,
                    "high": high,
                    "results_file": "security/trivy-results.json"
                }
            else:
                return {"status": "image_not_found", "error": f"Image {image_name} not found"}
                
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Trivy scan timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _analyze_dockerfile_security(self) -> Dict:
        """Analyze Dockerfile for security best practices."""
        dockerfile_path = "Dockerfile"
        
        if not os.path.exists(dockerfile_path):
            return {"status": "not_found", "error": "Dockerfile not found"}
        
        issues = []
        recommendations = []
        
        with open(dockerfile_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.strip().upper()
            
            # Check for root user
            if line.startswith("USER ROOT") or "USER 0" in line:
                issues.append(f"Line {i}: Running as root user")
                recommendations.append("Use non-root user for security")
            
            # Check for latest tags
            if "FROM" in line and ":LATEST" in line:
                issues.append(f"Line {i}: Using 'latest' tag")
                recommendations.append("Pin specific version tags")
            
            # Check for ADD vs COPY
            if line.startswith("ADD ") and not ("http://" in line or "https://" in line):
                issues.append(f"Line {i}: Using ADD instead of COPY")
                recommendations.append("Use COPY instead of ADD for local files")
        
        return {
            "status": "completed",
            "issues_found": len(issues),
            "issues": issues,
            "recommendations": recommendations
        }
    
    def _check_runtime_security(self) -> Dict:
        """Check runtime security configurations."""
        security_configs = {
            "capabilities": [],
            "security_opts": [],
            "read_only": False,
            "user": None
        }
        
        # Check docker-compose for security settings
        compose_files = ["docker-compose.yml", "docker-compose.override.yml"]
        
        for compose_file in compose_files:
            if os.path.exists(compose_file):
                try:
                    with open(compose_file, 'r') as f:
                        compose_data = yaml.safe_load(f)
                    
                    services = compose_data.get("services", {})
                    for service_name, service_config in services.items():
                        # Check security configurations
                        if "cap_drop" in service_config:
                            security_configs["capabilities"].extend(service_config["cap_drop"])
                        if "security_opt" in service_config:
                            security_configs["security_opts"].extend(service_config["security_opt"])
                        if service_config.get("read_only"):
                            security_configs["read_only"] = True
                        if "user" in service_config:
                            security_configs["user"] = service_config["user"]
                            
                except Exception as e:
                    pass
        
        # Score the runtime security
        score = 0
        if security_configs["capabilities"]:
            score += 25
        if security_configs["security_opts"]:
            score += 25
        if security_configs["read_only"]:
            score += 25
        if security_configs["user"] and security_configs["user"] != "root":
            score += 25
        
        return {
            "status": "completed",
            "security_score": score,
            "configurations": security_configs,
            "recommendations": self._get_runtime_recommendations(security_configs)
        }
    
    def _get_runtime_recommendations(self, configs: Dict) -> List[str]:
        """Get runtime security recommendations."""
        recommendations = []
        
        if not configs["capabilities"]:
            recommendations.append("Drop unnecessary capabilities (e.g., cap_drop: [ALL])")
        
        if not configs["security_opts"]:
            recommendations.append("Enable security options (e.g., no-new-privileges:true)")
        
        if not configs["read_only"]:
            recommendations.append("Consider read-only root filesystem where possible")
        
        if not configs["user"] or configs["user"] == "root":
            recommendations.append("Run containers with non-root user")
        
        return recommendations
    
    def _verify_slsa_compliance(self) -> Dict:
        """Verify SLSA (Supply-chain Levels for Software Artifacts) compliance."""
        self.logger.info("Verifying SLSA compliance...")
        
        compliance = {
            "level": 0,
            "requirements": {
                "version_controlled": self._check_version_control(),
                "build_service": self._check_build_service(),
                "build_integrity": self._check_build_integrity(),
                "signed_provenance": self._check_signed_provenance(),
                "isolated_builds": self._check_isolated_builds(),
                "ephemeral_environment": self._check_ephemeral_environment(),
                "isolated_dependencies": self._check_isolated_dependencies(),
                "hermetic_builds": self._check_hermetic_builds()
            }
        }
        
        # Calculate SLSA level
        if all([compliance["requirements"]["version_controlled"],
                compliance["requirements"]["build_service"]]):
            compliance["level"] = 1
            
        if compliance["level"] == 1 and compliance["requirements"]["build_integrity"]:
            compliance["level"] = 2
            
        if (compliance["level"] == 2 and 
            all([compliance["requirements"]["signed_provenance"],
                 compliance["requirements"]["isolated_builds"]])):
            compliance["level"] = 3
            
        if (compliance["level"] == 3 and 
            all([compliance["requirements"]["ephemeral_environment"],
                 compliance["requirements"]["isolated_dependencies"],
                 compliance["requirements"]["hermetic_builds"]])):
            compliance["level"] = 4
        
        return compliance
    
    def _check_version_control(self) -> bool:
        """Check if source is version controlled."""
        return os.path.exists(".git")
    
    def _check_build_service(self) -> bool:
        """Check if using hosted build service."""
        github_workflows = Path(".github/workflows")
        return github_workflows.exists() and any(github_workflows.glob("*.yml"))
    
    def _check_build_integrity(self) -> bool:
        """Check build integrity measures."""
        # Look for integrity checks in workflows
        workflow_files = Path(".github/workflows").glob("*.yml") if Path(".github/workflows").exists() else []
        
        for workflow_file in workflow_files:
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read().lower()
                    if "sha256" in content or "checksum" in content or "hash" in content:
                        return True
            except:
                pass
        
        return False
    
    def _check_signed_provenance(self) -> bool:
        """Check for signed build provenance."""
        # Look for signing in workflows or SBOM generation
        return os.path.exists("security/generate-sbom.py") or self._check_signing_workflow()
    
    def _check_signing_workflow(self) -> bool:
        """Check if workflows include artifact signing."""
        workflow_files = Path(".github/workflows").glob("*.yml") if Path(".github/workflows").exists() else []
        
        for workflow_file in workflow_files:
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in ["cosign", "sigstore", "signing", "attestation"]):
                        return True
            except:
                pass
        
        return False
    
    def _check_isolated_builds(self) -> bool:
        """Check for build isolation."""
        # Check for containerized builds or clean environments
        return os.path.exists("Dockerfile") or os.path.exists(".devcontainer/devcontainer.json")
    
    def _check_ephemeral_environment(self) -> bool:
        """Check for ephemeral build environments."""
        # Check for GitHub Actions or similar ephemeral environments
        return self._check_build_service()
    
    def _check_isolated_dependencies(self) -> bool:
        """Check for dependency isolation."""
        # Check for lockfiles or pinned dependencies
        lockfiles = ["requirements.txt", "pyproject.toml", "poetry.lock", "Pipfile.lock"]
        return any(os.path.exists(f) for f in lockfiles)
    
    def _check_hermetic_builds(self) -> bool:
        """Check for hermetic builds."""
        # Check for reproducible build configuration
        return os.path.exists("Dockerfile") and self._check_reproducible_builds()
    
    def _check_reproducible_builds(self) -> bool:
        """Check for reproducible build measures."""
        if os.path.exists("Dockerfile"):
            with open("Dockerfile", 'r') as f:
                content = f.read()
                # Look for version pinning and deterministic practices
                return "FROM" in content and ":" in content and "latest" not in content.lower()
        return False
    
    def _analyze_supply_chain(self) -> Dict:
        """Analyze supply chain security."""
        self.logger.info("Analyzing supply chain security...")
        
        return {
            "dependency_analysis": self._analyze_dependency_chain(),
            "build_provenance": self._check_build_provenance(),
            "package_integrity": self._verify_package_integrity(),
            "update_policy": self._analyze_update_policy()
        }
    
    def _analyze_dependency_chain(self) -> Dict:
        """Analyze the dependency chain for security risks."""
        try:
            # Get dependency tree
            cmd = ["pip", "list", "--format=json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                packages = json.loads(result.stdout)
                
                # Analyze package sources and maintainers
                analysis = {
                    "total_dependencies": len(packages),
                    "direct_dependencies": 0,  # Would need parsing requirements.txt
                    "transitive_dependencies": 0,
                    "high_risk_packages": []
                }
                
                # Check for potentially risky packages
                risk_indicators = ["crypto", "ssl", "auth", "security", "network"]
                for package in packages:
                    package_name = package["name"].lower()
                    if any(indicator in package_name for indicator in risk_indicators):
                        analysis["high_risk_packages"].append(package)
                
                return analysis
            else:
                return {"status": "failed", "error": "Could not retrieve package list"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_build_provenance(self) -> Dict:
        """Check build provenance and attestation."""
        provenance_files = [
            "security/generate-sbom.py",
            ".github/workflows/release.yml",
            "SECURITY.md"
        ]
        
        found_files = [f for f in provenance_files if os.path.exists(f)]
        
        return {
            "provenance_files": found_files,
            "provenance_score": len(found_files) * 25,  # Max 100 for all files
            "recommendations": [] if found_files else ["Implement build provenance tracking"]
        }
    
    def _verify_package_integrity(self) -> Dict:
        """Verify package integrity and checksums."""
        try:
            # Check if pip-audit or similar tools verify integrity
            cmd = ["pip", "check"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return {
                "integrity_check": result.returncode == 0,
                "issues": result.stdout.split("\\n") if result.stdout else [],
                "status": "passed" if result.returncode == 0 else "failed"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _analyze_update_policy(self) -> Dict:
        """Analyze dependency update policies."""
        update_tools = []
        
        if os.path.exists(".github/dependabot.yml"):
            update_tools.append("dependabot")
        
        if os.path.exists("renovate.json") or os.path.exists(".renovaterc"):
            update_tools.append("renovate")
        
        # Check for automated updates in workflows
        workflow_files = Path(".github/workflows").glob("*.yml") if Path(".github/workflows").exists() else []
        for workflow_file in workflow_files:
            try:
                with open(workflow_file, 'r') as f:
                    content = f.read().lower()
                    if "update" in content and ("dependenc" in content or "package" in content):
                        update_tools.append("custom_workflow")
                        break
            except:
                pass
        
        return {
            "automated_updates": len(update_tools) > 0,
            "update_tools": update_tools,
            "policy_score": len(update_tools) * 25,
            "recommendations": [] if update_tools else ["Implement automated dependency updates"]
        }
    
    def _scan_model_security(self) -> Dict:
        """Scan for model-specific security issues."""
        self.logger.info("Scanning model security...")
        
        return {
            "model_validation": self._validate_model_integrity(),
            "input_validation": self._check_input_validation(),
            "output_sanitization": self._check_output_sanitization(),
            "adversarial_protection": self._check_adversarial_protection()
        }
    
    def _validate_model_integrity(self) -> Dict:
        """Validate model file integrity."""
        model_files = []
        
        # Look for model files in common directories
        search_dirs = ["models/", "checkpoints/", "weights/", "moe_lab/"]
        model_extensions = [".pth", ".pt", ".ckpt", ".safetensors", ".pkl"]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in model_extensions:
                    model_files.extend(Path(search_dir).glob(f"**/*{ext}"))
        
        return {
            "model_files_found": len(model_files),
            "security_recommendations": [
                "Verify model checksums before loading",
                "Use safetensors format when possible",
                "Implement model signature verification"
            ] if model_files else ["No model files found to validate"]
        }
    
    def _check_input_validation(self) -> Dict:
        """Check for input validation in model code."""
        validation_patterns = [
            "assert", "validate", "check", "sanitiz", "clean",
            "isinstance", "len(", "shape", "dtype"
        ]
        
        python_files = list(Path("moe_lab").glob("**/*.py"))
        files_with_validation = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in validation_patterns):
                        files_with_validation += 1
            except:
                pass
        
        validation_ratio = files_with_validation / len(python_files) if python_files else 0
        
        return {
            "files_checked": len(python_files),
            "files_with_validation": files_with_validation,
            "validation_ratio": validation_ratio,
            "security_score": validation_ratio * 100,
            "recommendations": [
                "Add input shape validation",
                "Implement data type checking",
                "Add bounds checking for numerical inputs"
            ] if validation_ratio < 0.8 else []
        }
    
    def _check_output_sanitization(self) -> Dict:
        """Check for output sanitization."""
        # Look for output processing and sanitization
        sanitization_files = []
        
        python_files = list(Path("moe_lab").glob("**/*.py"))
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(keyword in content for keyword in ["sanitiz", "clean", "filter", "escape"]):
                        sanitization_files.append(str(py_file))
            except:
                pass
        
        return {
            "files_with_sanitization": len(sanitization_files),
            "sanitization_files": sanitization_files,
            "recommendations": [
                "Implement output sanitization for generated text",
                "Add filtering for potentially harmful content",
                "Implement rate limiting for inference endpoints"
            ] if not sanitization_files else []
        }
    
    def _check_adversarial_protection(self) -> Dict:
        """Check for adversarial attack protection."""
        protection_indicators = [
            "adversarial", "robust", "defense", "attack", "perturbation",
            "clamp", "clip", "bound", "limit"
        ]
        
        python_files = list(Path("moe_lab").glob("**/*.py"))
        protection_files = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(indicator in content for indicator in protection_indicators):
                        protection_files.append(str(py_file))
            except:
                pass
        
        return {
            "adversarial_protection_files": len(protection_files),
            "protection_files": protection_files,
            "recommendations": [
                "Implement input preprocessing to detect adversarial examples",
                "Add robustness testing for model inputs",
                "Consider adversarial training techniques",
                "Implement uncertainty quantification"
            ] if not protection_files else []
        }
    
    def _scan_configuration_security(self) -> Dict:
        """Scan configuration files for security issues."""
        self.logger.info("Scanning configuration security...")
        
        config_files = [
            "docker-compose.yml", "docker-compose.override.yml",
            ".env.example", "pyproject.toml", "requirements.txt"
        ]
        
        issues = []
        recommendations = []
        
        for config_file in config_files:
            if os.path.exists(config_file):
                issues_found, recs = self._analyze_config_file(config_file)
                issues.extend(issues_found)
                recommendations.extend(recs)
        
        return {
            "files_scanned": len([f for f in config_files if os.path.exists(f)]),
            "issues_found": len(issues),
            "issues": issues,
            "recommendations": list(set(recommendations))  # Remove duplicates
        }
    
    def _analyze_config_file(self, config_file: str) -> Tuple[List[str], List[str]]:
        """Analyze a configuration file for security issues."""
        issues = []
        recommendations = []
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check for common security issues
            if "password" in content.lower() and ":" in content:
                issues.append(f"{config_file}: Potential password in configuration")
                recommendations.append("Use environment variables for sensitive data")
            
            if "secret" in content.lower() and not content.lower().startswith("#"):
                issues.append(f"{config_file}: Potential secret in configuration")
                recommendations.append("Use secrets management system")
            
            if config_file.endswith(".yml") or config_file.endswith(".yaml"):
                # Docker compose specific checks
                if "privileged: true" in content:
                    issues.append(f"{config_file}: Privileged mode enabled")
                    recommendations.append("Avoid privileged mode in containers")
                
                if "network_mode: host" in content:
                    issues.append(f"{config_file}: Host networking enabled")
                    recommendations.append("Use bridge networking instead of host")
        
        except Exception as e:
            issues.append(f"Error analyzing {config_file}: {str(e)}")
        
        return issues, recommendations
    
    def _generate_recommendations(self):
        """Generate security recommendations based on scan results."""
        self.logger.info("Generating security recommendations...")
        
        recommendations = set()
        
        # Add recommendations from scan results
        for scan_type, results in self.results["scan_results"].items():
            if isinstance(results, dict):
                self._extract_recommendations(results, recommendations)
        
        # Add SLSA compliance recommendations
        slsa_results = self.results["compliance_status"].get("slsa", {})
        if slsa_results.get("level", 0) < 3:
            recommendations.add("Improve SLSA compliance to level 3 or higher")
            
        for requirement, status in slsa_results.get("requirements", {}).items():
            if not status:
                recommendations.add(f"Implement {requirement.replace('_', ' ')} for SLSA compliance")
        
        # Convert to list and add priority scores
        self.results["recommendations"] = [
            {"recommendation": rec, "priority": self._get_recommendation_priority(rec)}
            for rec in sorted(recommendations)
        ]
    
    def _extract_recommendations(self, results: Dict, recommendations: set):
        """Extract recommendations from scan results."""
        if "recommendations" in results:
            if isinstance(results["recommendations"], list):
                recommendations.update(results["recommendations"])
            elif isinstance(results["recommendations"], str):
                recommendations.add(results["recommendations"])
        
        # Recursively check nested dictionaries
        for value in results.values():
            if isinstance(value, dict):
                self._extract_recommendations(value, recommendations)
    
    def _get_recommendation_priority(self, recommendation: str) -> str:
        """Assign priority to recommendations."""
        high_priority_keywords = ["critical", "vulnerability", "secret", "password", "root", "privileged"]
        medium_priority_keywords = ["update", "outdated", "license", "compliance"]
        
        recommendation_lower = recommendation.lower()
        
        if any(keyword in recommendation_lower for keyword in high_priority_keywords):
            return "high"
        elif any(keyword in recommendation_lower for keyword in medium_priority_keywords):
            return "medium"
        else:
            return "low"
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security score."""
        scores = []
        
        # Code security score
        code_results = self.results["scan_results"].get("code_security", {})
        bandit_issues = code_results.get("bandit", {}).get("issues_found", 0)
        semgrep_issues = code_results.get("semgrep", {}).get("total_findings", 0)
        secrets_found = code_results.get("secrets", {}).get("secrets_found", 0)
        
        code_score = max(0, 100 - (bandit_issues * 5 + semgrep_issues * 3 + secrets_found * 10))
        scores.append(code_score)
        
        # Dependency security score
        dep_results = self.results["scan_results"].get("dependencies", {})
        safety_vulns = dep_results.get("safety", {}).get("vulnerabilities_found", 0)
        outdated_count = dep_results.get("outdated", {}).get("outdated_packages", 0)
        
        dep_score = max(0, 100 - (safety_vulns * 10 + outdated_count * 2))
        scores.append(dep_score)
        
        # Container security score
        container_results = self.results["scan_results"].get("containers", {})
        runtime_score = container_results.get("runtime", {}).get("security_score", 50)
        scores.append(runtime_score)
        
        # SLSA compliance score
        slsa_level = self.results["compliance_status"].get("slsa", {}).get("level", 0)
        slsa_score = slsa_level * 25  # 0-100 scale
        scores.append(slsa_score)
        
        # Calculate weighted average
        if scores:
            return int(sum(scores) / len(scores))
        else:
            return 0
    
    def save_results(self, output_path: str = "security/security-scan-results.json"):
        """Save scan results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Security scan results saved to {output_path}")
    
    def generate_report(self, output_path: str = "security/security-report.md"):
        """Generate human-readable security report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = f"""# Security Scan Report
        
## Executive Summary

**Security Score: {self.results['security_score']}/100**
**Scan Date: {self.results['timestamp']}**

### Key Findings

- **SLSA Compliance Level**: {self.results['compliance_status'].get('slsa', {}).get('level', 0)}/4
- **Code Security Issues**: {sum(self.results['severity_counts'].values())} total issues
- **Critical Vulnerabilities**: {self.results['severity_counts']['critical']}
- **High Priority Issues**: {self.results['severity_counts']['high']}

## Detailed Results

### Code Security Analysis
{self._format_section_results(self.results['scan_results'].get('code_security', {}))}

### Dependency Security
{self._format_section_results(self.results['scan_results'].get('dependencies', {}))}

### Container Security
{self._format_section_results(self.results['scan_results'].get('containers', {}))}

### Supply Chain Security
{self._format_section_results(self.results['scan_results'].get('supply_chain', {}))}

### Model Security
{self._format_section_results(self.results['scan_results'].get('model_security', {}))}

## Recommendations

### High Priority
{self._format_recommendations('high')}

### Medium Priority
{self._format_recommendations('medium')}

### Low Priority
{self._format_recommendations('low')}

## Next Steps

1. Address critical and high-priority vulnerabilities immediately
2. Implement missing SLSA compliance requirements
3. Establish regular security scanning schedule
4. Update security policies and procedures
5. Consider security training for development team

---
*Report generated by Advanced Security Scanner v1.0*
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Security report generated: {output_path}")
    
    def _format_section_results(self, section_results: Dict) -> str:
        """Format section results for report."""
        if not section_results:
            return "No data available"
        
        formatted = []
        for key, value in section_results.items():
            if isinstance(value, dict) and "status" in value:
                status = value["status"]
                if status == "completed":
                    issues = value.get("issues_found", value.get("total_findings", 0))
                    formatted.append(f"- {key.replace('_', ' ').title()}: {issues} issues found")
                else:
                    formatted.append(f"- {key.replace('_', ' ').title()}: {status}")
            elif isinstance(value, (int, float)):
                formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\\n".join(formatted) if formatted else "No issues found"
    
    def _format_recommendations(self, priority: str) -> str:
        """Format recommendations by priority."""
        recs = [r["recommendation"] for r in self.results.get("recommendations", []) 
                if r.get("priority") == priority]
        
        if recs:
            return "\\n".join(f"- {rec}" for rec in recs)
        else:
            return f"No {priority} priority recommendations"


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Advanced Security Scanner for Open MoE Trainer Lab")
    
    parser.add_argument("--scan", action="store_true", default=True,
                       help="Run comprehensive security scan")
    parser.add_argument("--config", type=str, default="security/security-config.yaml",
                       help="Security configuration file")
    parser.add_argument("--output", type=str, default="security/security-scan-results.json",
                       help="Output file for scan results")
    parser.add_argument("--report", type=str, default="security/security-report.md",
                       help="Output file for human-readable report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    scanner = AdvancedSecurityScanner(args.config)
    
    if args.scan:
        results = scanner.run_comprehensive_scan()
        scanner.save_results(args.output)
        scanner.generate_report(args.report)
        
        print(f"\\nSecurity Scan Complete!")
        print(f"Security Score: {results['security_score']}/100")
        print(f"SLSA Compliance Level: {results['compliance_status'].get('slsa', {}).get('level', 0)}/4")
        print(f"Total Issues: {sum(results['severity_counts'].values())}")
        print(f"\\nResults saved to: {args.output}")
        print(f"Report generated: {args.report}")
        
        # Exit with non-zero code if critical issues found
        if results['severity_counts']['critical'] > 0:
            print("\\n⚠️  Critical security issues found!")
            sys.exit(1)
        elif results['severity_counts']['high'] > 0:
            print("\\n⚠️  High priority security issues found!")
            sys.exit(1)


if __name__ == "__main__":
    main()