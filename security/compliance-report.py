#!/usr/bin/env python3
"""
Security Compliance Report Generator for Open MoE Trainer Lab
Generates comprehensive security compliance reports for audit purposes.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceReporter:
    """Generate comprehensive security compliance reports."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "repository": str(repo_path.absolute()),
            "compliance_frameworks": ["OWASP", "NIST", "SOC2"],
            "checks": {}
        }
    
    def run_full_report(self) -> Dict[str, Any]:
        """Run all compliance checks and generate full report."""
        logger.info("🔍 Starting security compliance assessment...")
        
        checks = [
            self.check_dependency_vulnerabilities,
            self.check_code_security,
            self.check_secrets_detection,
            self.check_container_security,
            self.check_configuration_security,
            self.check_documentation_compliance,
            self.check_ci_cd_security,
            self.generate_sbom
        ]
        
        for check in checks:
            try:
                check_name = check.__name__.replace("check_", "").replace("_", " ").title()
                logger.info(f"Running {check_name}...")
                result = check()
                self.report_data["checks"][check.__name__] = result
                status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
                logger.info(f"{status} {check_name}")
            except Exception as e:
                logger.error(f"Error in {check.__name__}: {e}")
                self.report_data["checks"][check.__name__] = {
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return self.report_data
    
    def check_dependency_vulnerabilities(self) -> Dict[str, Any]:
        """Check for known vulnerabilities in dependencies."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "vulnerabilities": [],
            "tools_used": ["safety", "pip-audit"]
        }
        
        try:
            # Run safety check
            safety_result = subprocess.run(
                ["python", "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if safety_result.returncode == 0:
                safety_data = json.loads(safety_result.stdout) if safety_result.stdout else []
                result["safety_vulnerabilities"] = safety_data
                if safety_data:
                    result["passed"] = False
                    result["vulnerabilities"].extend(safety_data)
            
            # Run pip-audit if available
            try:
                audit_result = subprocess.run(
                    ["pip-audit", "--format=json"],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                
                if audit_result.returncode == 0:
                    audit_data = json.loads(audit_result.stdout) if audit_result.stdout else {}
                    result["pip_audit_results"] = audit_data
                    if audit_data.get("vulnerabilities"):
                        result["passed"] = False
                        result["vulnerabilities"].extend(audit_data["vulnerabilities"])
            except FileNotFoundError:
                result["pip_audit_results"] = "Tool not available"
        
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
        
        result["vulnerability_count"] = len(result["vulnerabilities"])
        return result
    
    def check_code_security(self) -> Dict[str, Any]:
        """Run static code analysis for security issues."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "issues": [],
            "tools_used": ["bandit"]
        }
        
        try:
            # Run Bandit security linter
            bandit_result = subprocess.run(
                ["python", "-m", "bandit", "-r", ".", "-f", "json", 
                 "--exclude", "./tests,./venv,./build"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if bandit_result.stdout:
                bandit_data = json.loads(bandit_result.stdout)
                result["bandit_results"] = bandit_data
                
                high_severity = [r for r in bandit_data.get("results", []) 
                               if r.get("issue_severity") in ["HIGH", "MEDIUM"]]
                result["issues"] = high_severity
                result["high_severity_count"] = len(high_severity)
                
                if high_severity:
                    result["passed"] = False
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
        
        return result
    
    def check_secrets_detection(self) -> Dict[str, Any]:
        """Check for accidentally committed secrets."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "secrets_found": [],
            "tools_used": ["detect-secrets"]
        }
        
        try:
            # Check if .secrets.baseline exists
            baseline_path = self.repo_path / ".secrets.baseline"
            if baseline_path.exists():
                # Run detect-secrets scan
                secrets_result = subprocess.run(
                    ["detect-secrets", "scan", "--baseline", str(baseline_path)],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path
                )
                
                if secrets_result.returncode != 0:
                    result["passed"] = False
                    result["secrets_found"] = secrets_result.stdout.split('\n')
                
                result["baseline_exists"] = True
            else:
                result["baseline_exists"] = False
                result["recommendation"] = "Create .secrets.baseline file"
        
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
        
        return result
    
    def check_container_security(self) -> Dict[str, Any]:
        """Check Docker container security configuration."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "issues": [],
            "tools_used": ["hadolint"]
        }
        
        dockerfile_paths = list(self.repo_path.glob("**/Dockerfile*"))
        result["dockerfiles_found"] = len(dockerfile_paths)
        
        for dockerfile in dockerfile_paths:
            try:
                # Run Hadolint on Dockerfile
                hadolint_result = subprocess.run(
                    ["hadolint", "--format", "json", str(dockerfile)],
                    capture_output=True,
                    text=True
                )
                
                if hadolint_result.stdout:
                    issues = json.loads(hadolint_result.stdout)
                    error_issues = [i for i in issues if i.get("level") == "error"]
                    
                    if error_issues:
                        result["passed"] = False
                        result["issues"].extend(error_issues)
                    
                    result[f"hadolint_{dockerfile.name}"] = issues
            
            except Exception as e:
                result["issues"].append(f"Error checking {dockerfile}: {e}")
        
        return result
    
    def check_configuration_security(self) -> Dict[str, Any]:
        """Check security of configuration files."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "issues": [],
            "files_checked": []
        }
        
        # Check for common security misconfigurations
        security_files = [
            ".env",
            ".env.example", 
            "docker-compose.yml",
            "docker-compose.override.yml",
            "pyproject.toml",
            "requirements.txt"
        ]
        
        for file_name in security_files:
            file_path = self.repo_path / file_name
            if file_path.exists():
                result["files_checked"].append(file_name)
                
                # Check for potential security issues
                content = file_path.read_text()
                
                # Look for potential hardcoded secrets
                suspicious_patterns = [
                    "password=",
                    "secret=",
                    "key=",
                    "token=",
                    "api_key="
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in content.lower():
                        # Check if it's just an example or comment
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line.lower() and not line.strip().startswith('#'):
                                if "example" not in line.lower() and "your_" not in line.lower():
                                    result["issues"].append({
                                        "file": file_name,
                                        "line": i + 1,
                                        "issue": f"Potential hardcoded {pattern[:-1]}",
                                        "content": line.strip()
                                    })
                                    result["passed"] = False
        
        return result
    
    def check_documentation_compliance(self) -> Dict[str, Any]:
        """Check if required security documentation exists."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "missing_docs": [],
            "found_docs": []
        }
        
        required_docs = [
            "SECURITY.md",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            ".github/SECURITY.md"
        ]
        
        for doc in required_docs:
            doc_path = self.repo_path / doc
            if doc_path.exists():
                result["found_docs"].append(doc)
            else:
                result["missing_docs"].append(doc)
                result["passed"] = False
        
        # Check for security section in README
        readme_path = self.repo_path / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text().lower()
            if "security" not in readme_content:
                result["missing_docs"].append("Security section in README.md")
                result["passed"] = False
        
        return result
    
    def check_ci_cd_security(self) -> Dict[str, Any]:
        """Check CI/CD security configuration."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "issues": [],
            "workflows_checked": []
        }
        
        workflows_dir = self.repo_path / ".github" / "workflows"
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob("*.yml"):
                result["workflows_checked"].append(workflow_file.name)
                
                try:
                    import yaml
                    workflow_content = yaml.safe_load(workflow_file.read_text())
                    
                    # Check for security best practices
                    if "permissions" not in workflow_content:
                        result["issues"].append({
                            "file": workflow_file.name,
                            "issue": "Missing permissions declaration"
                        })
                        result["passed"] = False
                    
                    # Check for hardcoded secrets
                    workflow_text = workflow_file.read_text()
                    if any(pattern in workflow_text for pattern in ["password:", "token:", "key:"]):
                        result["issues"].append({
                            "file": workflow_file.name,
                            "issue": "Potential hardcoded secrets"
                        })
                        result["passed"] = False
                
                except Exception as e:
                    result["issues"].append({
                        "file": workflow_file.name,
                        "error": str(e)
                    })
        
        return result
    
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate Software Bill of Materials."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "passed": True,
            "sbom_generated": False
        }
        
        try:
            # Try to generate SBOM using cyclonedx or similar tool
            sbom_result = subprocess.run(
                ["python", "-m", "cyclonedx.cli", "requirements", 
                 "-o", "sbom.json", "--format", "json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if sbom_result.returncode == 0:
                result["sbom_generated"] = True
                result["sbom_location"] = "sbom.json"
            else:
                # Fallback: generate simple dependency list
                pip_result = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True,
                    text=True
                )
                
                if pip_result.returncode == 0:
                    dependencies = json.loads(pip_result.stdout)
                    with open(self.repo_path / "dependencies.json", "w") as f:
                        json.dump(dependencies, f, indent=2)
                    result["dependencies_list"] = "dependencies.json"
        
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
        
        return result
    
    def generate_report(self, output_file: Optional[str] = None) -> None:
        """Generate and save the compliance report."""
        report = self.run_full_report()
        
        # Calculate overall compliance score
        total_checks = len(report["checks"])
        passed_checks = sum(1 for check in report["checks"].values() 
                          if check.get("passed", False))
        compliance_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        
        report["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "compliance_score": round(compliance_score, 2),
            "overall_status": "COMPLIANT" if compliance_score >= 80 else "NON_COMPLIANT"
        }
        
        # Save report
        if output_file is None:
            output_file = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.repo_path / output_file
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Compliance report saved to: {output_path}")
        logger.info(f"🎯 Compliance Score: {compliance_score:.1f}%")
        logger.info(f"📈 Status: {report['summary']['overall_status']}")
        
        return report


def main():
    """Main function to run compliance reporting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate security compliance report")
    parser.add_argument("--output", "-o", help="Output file name")
    parser.add_argument("--repo-path", "-r", default=".", help="Repository path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    reporter = ComplianceReporter(Path(args.repo_path))
    report = reporter.generate_report(args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("🛡️  SECURITY COMPLIANCE REPORT SUMMARY")
    print("="*60)
    print(f"Repository: {report['repository']}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Compliance Score: {report['summary']['compliance_score']}%")
    print(f"Status: {report['summary']['overall_status']}")
    print(f"Checks Passed: {report['summary']['passed_checks']}/{report['summary']['total_checks']}")
    
    if report['summary']['failed_checks'] > 0:
        print(f"\n❌ Failed Checks: {report['summary']['failed_checks']}")
        for check_name, check_result in report['checks'].items():
            if not check_result.get('passed', False):
                print(f"  - {check_name.replace('_', ' ').title()}")
    
    print("\n" + "="*60)
    
    return 0 if report['summary']['overall_status'] == 'COMPLIANT' else 1


if __name__ == "__main__":
    sys.exit(main())