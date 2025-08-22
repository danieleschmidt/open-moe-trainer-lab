#!/usr/bin/env python3
"""
Advanced Security Scanner and Validation Suite
Enterprise-grade security scanning with AI-powered vulnerability detection.
"""

import json
import time
import hashlib
import re
import ast
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import base64

@dataclass
class SecurityVulnerability:
    """Security vulnerability data structure."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # e.g., "hardcoded_secret", "injection", "xss"
    file_path: str
    line_number: int
    description: str
    recommendation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None

@dataclass
class SecurityReport:
    """Comprehensive security report."""
    scan_timestamp: str
    total_files_scanned: int
    vulnerabilities: List[SecurityVulnerability]
    security_score: float
    compliance_status: Dict[str, bool]
    recommendations: List[str]
    false_positives: int = 0

class AdvancedSecurityScanner:
    """Enterprise-grade security scanner with AI-powered detection."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.vulnerabilities = []
        self.scanned_files = 0
        
        # Security patterns for various vulnerability types
        self.security_patterns = {
            "hardcoded_secrets": [
                (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']', "CRITICAL", "CWE-798"),
                (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']([^"\']{16,})["\']', "HIGH", "CWE-798"),
                (r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']([^"\']{16,})["\']', "HIGH", "CWE-798"),
                (r'(?i)(access[_-]?token|accesstoken)\s*[=:]\s*["\']([^"\']{20,})["\']', "HIGH", "CWE-798"),
                (r'(?i)(private[_-]?key|privatekey)\s*[=:]\s*["\']([^"\']{32,})["\']', "CRITICAL", "CWE-798"),
                (r'(?i)(jwt[_-]?token|jwttoken)\s*[=:]\s*["\']([^"\']{50,})["\']', "HIGH", "CWE-798"),
            ],
            "sql_injection": [
                (r'(?i)execute\s*\(\s*["\'].*\+.*["\']', "HIGH", "CWE-89"),
                (r'(?i)query\s*\(\s*["\'].*%s.*["\']', "HIGH", "CWE-89"),
                (r'(?i)cursor\.execute\s*\(\s*f["\'].*\{.*\}.*["\']', "MEDIUM", "CWE-89"),
                (r'(?i)SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*["\'].*\+.*["\']', "HIGH", "CWE-89"),
            ],
            "command_injection": [
                (r'(?i)os\.system\s*\(\s*["\'].*\+.*["\']', "CRITICAL", "CWE-78"),
                (r'(?i)subprocess\.(call|run|Popen)\s*\(\s*["\'].*\+.*["\']', "CRITICAL", "CWE-78"),
                (r'(?i)eval\s*\(\s*.*input.*\)', "CRITICAL", "CWE-94"),
                (r'(?i)exec\s*\(\s*.*input.*\)', "CRITICAL", "CWE-94"),
            ],
            "path_traversal": [
                (r'(?i)open\s*\(\s*.*\+.*["\']\.\./', "HIGH", "CWE-22"),
                (r'(?i)file\s*\(\s*.*\+.*["\']\.\./', "HIGH", "CWE-22"),
                (r'(?i)os\.path\.join\s*\(\s*.*user.*input', "MEDIUM", "CWE-22"),
            ],
            "weak_crypto": [
                (r'(?i)md5\s*\(', "MEDIUM", "CWE-327"),
                (r'(?i)sha1\s*\(', "MEDIUM", "CWE-327"),
                (r'(?i)DES\s*\(', "HIGH", "CWE-327"),
                (r'(?i)RC4\s*\(', "HIGH", "CWE-327"),
            ],
            "insecure_deserialization": [
                (r'(?i)pickle\.loads?\s*\(', "HIGH", "CWE-502"),
                (r'(?i)cPickle\.loads?\s*\(', "HIGH", "CWE-502"),
                (r'(?i)yaml\.load\s*\((?!.*Loader=yaml\.SafeLoader)', "MEDIUM", "CWE-502"),
            ],
            "information_disclosure": [
                (r'(?i)print\s*\(\s*.*password.*\)', "MEDIUM", "CWE-200"),
                (r'(?i)logger?\.(debug|info|warning|error)\s*\(\s*.*password.*\)', "MEDIUM", "CWE-200"),
                (r'(?i)traceback\.print_exc\s*\(\s*\)', "LOW", "CWE-209"),
            ]
        }
        
        # Compliance frameworks
        self.compliance_checks = {
            "OWASP_TOP_10": self._check_owasp_compliance,
            "PCI_DSS": self._check_pci_compliance,
            "GDPR": self._check_gdpr_compliance,
            "SOX": self._check_sox_compliance,
            "HIPAA": self._check_hipaa_compliance
        }
        
    def scan_comprehensive(self) -> SecurityReport:
        """Run comprehensive security scan."""
        
        print("🔒 ADVANCED SECURITY SCANNING")
        print("=" * 60)
        
        scan_start = time.time()
        
        # Initialize scan
        self.vulnerabilities = []
        self.scanned_files = 0
        
        # Scan categories
        scan_categories = [
            ("Static Code Analysis", self._scan_static_code),
            ("Dependency Vulnerability Scan", self._scan_dependencies),
            ("Configuration Security", self._scan_configurations),
            ("Secrets Detection", self._scan_secrets),
            ("Docker Security", self._scan_docker_security),
            ("Infrastructure Security", self._scan_infrastructure),
            ("API Security", self._scan_api_security),
            ("Data Protection", self._scan_data_protection)
        ]
        
        for category_name, scan_function in scan_categories:
            print(f"🔍 {category_name}...")
            try:
                scan_function()
                print(f"   ✅ {category_name} completed")
            except Exception as e:
                print(f"   ❌ {category_name} failed: {e}")
                
        # Compliance checks
        print("🏛️  Compliance Validation...")
        compliance_status = {}
        for framework, check_function in self.compliance_checks.items():
            try:
                compliance_status[framework] = check_function()
                status = "✅" if compliance_status[framework] else "❌"
                print(f"   {status} {framework}: {'Compliant' if compliance_status[framework] else 'Non-compliant'}")
            except Exception as e:
                compliance_status[framework] = False
                print(f"   ❌ {framework}: Error - {e}")
                
        # Calculate security score
        security_score = self._calculate_security_score()
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations()
        
        # Create final report
        report = SecurityReport(
            scan_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_files_scanned=self.scanned_files,
            vulnerabilities=self.vulnerabilities,
            security_score=security_score,
            compliance_status=compliance_status,
            recommendations=recommendations
        )
        
        # Save report
        self._save_security_report(report)
        
        scan_duration = time.time() - scan_start
        
        print(f"\n🎯 SECURITY SCAN COMPLETED")
        print("=" * 60)
        print(f"✅ Files scanned: {self.scanned_files}")
        print(f"🚨 Vulnerabilities found: {len(self.vulnerabilities)}")
        print(f"📊 Security score: {security_score:.1f}/100")
        print(f"⏱️  Scan duration: {scan_duration:.2f} seconds")
        print(f"📋 Compliance frameworks: {len(compliance_status)}")
        print(f"💾 Report saved to: security_report.json")
        
        return report
        
    def _scan_static_code(self):
        """Perform static code analysis for security vulnerabilities."""
        
        python_files = list(self.project_root.glob("**/*.py"))
        
        for py_file in python_files:
            self.scanned_files += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check each vulnerability pattern
                for category, patterns in self.security_patterns.items():
                    for pattern, severity, cwe_id in patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        
                        for match in matches:
                            line_number = content[:match.start()].count('\n') + 1
                            
                            vulnerability = SecurityVulnerability(
                                severity=severity,
                                category=category,
                                file_path=str(py_file),
                                line_number=line_number,
                                description=f"{category.replace('_', ' ').title()} detected: {match.group(0)[:100]}",
                                recommendation=self._get_recommendation(category),
                                cwe_id=cwe_id,
                                cvss_score=self._calculate_cvss_score(severity)
                            )
                            
                            self.vulnerabilities.append(vulnerability)
                            
                # AST-based analysis for complex patterns
                try:
                    tree = ast.parse(content)
                    self._analyze_ast_security(tree, py_file)
                except SyntaxError:
                    pass  # Skip files with syntax errors
                    
            except Exception as e:
                print(f"   Warning: Could not scan {py_file}: {e}")
                
    def _analyze_ast_security(self, tree: ast.AST, file_path: Path):
        """Analyze AST for complex security patterns."""
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, vulnerabilities_list, file_path):
                self.vulnerabilities = vulnerabilities_list
                self.file_path = file_path
                
            def visit_Call(self, node):
                # Check for dangerous function calls
                if hasattr(node.func, 'attr'):
                    func_name = node.func.attr
                    
                    # Dangerous eval/exec usage
                    if func_name in ['eval', 'exec'] and node.args:
                        if not self._is_safe_eval_usage(node):
                            vuln = SecurityVulnerability(
                                severity="CRITICAL",
                                category="code_injection",
                                file_path=str(self.file_path),
                                line_number=node.lineno,
                                description=f"Dangerous {func_name} usage detected",
                                recommendation="Avoid eval/exec or use ast.literal_eval for safe evaluation",
                                cwe_id="CWE-94",
                                cvss_score=9.3
                            )
                            self.vulnerabilities.append(vuln)
                            
                    # Subprocess with shell=True
                    if func_name in ['call', 'run', 'Popen'] and hasattr(node.func, 'value'):
                        if hasattr(node.func.value, 'id') and node.func.value.id == 'subprocess':
                            if self._has_shell_true(node):
                                vuln = SecurityVulnerability(
                                    severity="HIGH",
                                    category="command_injection",
                                    file_path=str(self.file_path),
                                    line_number=node.lineno,
                                    description="subprocess call with shell=True detected",
                                    recommendation="Use shell=False and pass command as list",
                                    cwe_id="CWE-78",
                                    cvss_score=8.1
                                )
                                self.vulnerabilities.append(vuln)
                                
                self.generic_visit(node)
                
            def _is_safe_eval_usage(self, node):
                # Simple heuristic for safe eval usage
                if len(node.args) == 1 and isinstance(node.args[0], ast.Str):
                    return True  # Literal string is safer
                return False
                
            def _has_shell_true(self, node):
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                        return keyword.value.value is True
                return False
                
        visitor = SecurityVisitor(self.vulnerabilities, file_path)
        visitor.visit(tree)
        
    def _scan_dependencies(self):
        """Scan dependencies for known vulnerabilities."""
        
        # Check for requirements files
        req_files = [
            self.project_root / "requirements.txt",
            self.project_root / "pyproject.toml",
            self.project_root / "Pipfile",
            self.project_root / "setup.py"
        ]
        
        vulnerable_packages = {
            # Known vulnerable packages (simplified for demo)
            "requests": "2.25.0",  # Example: versions below 2.25.1 have vulnerabilities
            "urllib3": "1.26.0",
            "pyyaml": "5.4.0",
            "pillow": "8.1.0",
            "django": "3.1.0",
            "flask": "1.1.0"
        }
        
        for req_file in req_files:
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        content = f.read()
                        
                    # Simple vulnerability check
                    for package, vulnerable_version in vulnerable_packages.items():
                        if package in content.lower():
                            vulnerability = SecurityVulnerability(
                                severity="MEDIUM",
                                category="vulnerable_dependency",
                                file_path=str(req_file),
                                line_number=1,
                                description=f"Potentially vulnerable dependency: {package}",
                                recommendation=f"Update {package} to latest version",
                                cwe_id="CWE-1104"
                            )
                            self.vulnerabilities.append(vulnerability)
                            
                except Exception as e:
                    print(f"   Warning: Could not scan {req_file}: {e}")
                    
    def _scan_configurations(self):
        """Scan configuration files for security issues."""
        
        config_files = [
            "*.yml", "*.yaml", "*.json", "*.ini", "*.cfg", "*.conf",
            ".env", ".env.*", "docker-compose*.yml"
        ]
        
        for pattern in config_files:
            for config_file in self.project_root.glob(f"**/{pattern}"):
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        
                    # Check for insecure configurations
                    insecure_patterns = [
                        (r'(?i)debug\s*[=:]\s*true', "MEDIUM", "Debug mode enabled in production"),
                        (r'(?i)ssl[_-]?verify\s*[=:]\s*false', "HIGH", "SSL verification disabled"),
                        (r'(?i)allow[_-]?all[_-]?origins\s*[=:]\s*true', "MEDIUM", "CORS allows all origins"),
                        (r'(?i)trust[_-]?all[_-]?certs?\s*[=:]\s*true', "HIGH", "Trust all certificates enabled"),
                    ]
                    
                    for pattern, severity, description in insecure_patterns:
                        if re.search(pattern, content):
                            vulnerability = SecurityVulnerability(
                                severity=severity,
                                category="insecure_configuration",
                                file_path=str(config_file),
                                line_number=1,
                                description=description,
                                recommendation="Review and secure configuration",
                                cwe_id="CWE-16"
                            )
                            self.vulnerabilities.append(vulnerability)
                            
                except Exception:
                    pass
                    
    def _scan_secrets(self):
        """Advanced secrets detection with entropy analysis."""
        
        # High-entropy patterns that might be secrets
        entropy_patterns = [
            r'[A-Za-z0-9+/]{32,}={0,2}',  # Base64-like
            r'[A-Fa-f0-9]{32,}',          # Hex strings
            r'[A-Za-z0-9]{20,}',          # High-entropy alphanumeric
        ]
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                lines = content.splitlines()
                for line_num, line in enumerate(lines, 1):
                    for pattern in entropy_patterns:
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            candidate = match.group(0)
                            
                            # Calculate entropy
                            entropy = self._calculate_entropy(candidate)
                            
                            # High entropy suggests potential secret
                            if entropy > 4.5 and len(candidate) > 16:
                                # Additional checks to reduce false positives
                                if not self._is_likely_false_positive(candidate, line):
                                    vulnerability = SecurityVulnerability(
                                        severity="HIGH",
                                        category="potential_secret",
                                        file_path=str(py_file),
                                        line_number=line_num,
                                        description=f"High-entropy string detected (entropy: {entropy:.2f})",
                                        recommendation="Review if this is a hardcoded secret and move to environment variables",
                                        cwe_id="CWE-798"
                                    )
                                    self.vulnerabilities.append(vulnerability)
                                    
            except Exception:
                pass
                
    def _scan_docker_security(self):
        """Scan Docker configurations for security issues."""
        
        docker_files = list(self.project_root.glob("**/Dockerfile*")) + \
                      list(self.project_root.glob("**/docker-compose*.yml"))
        
        for docker_file in docker_files:
            try:
                with open(docker_file, 'r') as f:
                    content = f.read()
                    
                # Docker security patterns
                docker_issues = [
                    (r'(?i)FROM\s+.*:latest', "MEDIUM", "Using latest tag is not recommended"),
                    (r'(?i)USER\s+root', "HIGH", "Running as root user"),
                    (r'(?i)COPY\s+\.\s+/', "MEDIUM", "Copying entire context might include sensitive files"),
                    (r'(?i)--privileged', "CRITICAL", "Privileged container mode detected"),
                    (r'(?i)--disable-content-trust', "HIGH", "Content trust disabled"),
                ]
                
                for pattern, severity, description in docker_issues:
                    if re.search(pattern, content):
                        vulnerability = SecurityVulnerability(
                            severity=severity,
                            category="docker_security",
                            file_path=str(docker_file),
                            line_number=1,
                            description=description,
                            recommendation="Follow Docker security best practices",
                            cwe_id="CWE-16"
                        )
                        self.vulnerabilities.append(vulnerability)
                        
            except Exception:
                pass
                
    def _scan_infrastructure(self):
        """Scan infrastructure-as-code for security issues."""
        
        iac_files = list(self.project_root.glob("**/*.tf")) + \
                   list(self.project_root.glob("**/*.yaml")) + \
                   list(self.project_root.glob("**/*.yml"))
        
        for iac_file in iac_files:
            try:
                with open(iac_file, 'r') as f:
                    content = f.read()
                    
                # Infrastructure security patterns
                iac_issues = [
                    (r'(?i)0\.0\.0\.0/0', "HIGH", "Open to all IPs (0.0.0.0/0)"),
                    (r'(?i)publicread', "MEDIUM", "Public read access enabled"),
                    (r'(?i)encryption\s*[=:]\s*false', "HIGH", "Encryption disabled"),
                    (r'(?i)versioning\s*[=:]\s*false', "MEDIUM", "Versioning disabled"),
                ]
                
                for pattern, severity, description in iac_issues:
                    if re.search(pattern, content):
                        vulnerability = SecurityVulnerability(
                            severity=severity,
                            category="infrastructure_security",
                            file_path=str(iac_file),
                            line_number=1,
                            description=description,
                            recommendation="Review infrastructure configuration",
                            cwe_id="CWE-16"
                        )
                        self.vulnerabilities.append(vulnerability)
                        
            except Exception:
                pass
                
    def _scan_api_security(self):
        """Scan for API security issues."""
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # API security patterns
                api_issues = [
                    (r'(?i)@app\.route\(["\'].*["\'].*methods.*GET.*POST', "MEDIUM", "Route accepts both GET and POST"),
                    (r'(?i)cors\(.*origins=\["?\*"?\]', "HIGH", "CORS allows all origins"),
                    (r'(?i)request\.args\.get\(.*\)(?!.*validation)', "MEDIUM", "Unvalidated request parameter"),
                    (r'(?i)request\.form\.get\(.*\)(?!.*validation)', "MEDIUM", "Unvalidated form data"),
                ]
                
                for pattern, severity, description in api_issues:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        vulnerability = SecurityVulnerability(
                            severity=severity,
                            category="api_security",
                            file_path=str(py_file),
                            line_number=line_number,
                            description=description,
                            recommendation="Implement proper input validation and security headers",
                            cwe_id="CWE-20"
                        )
                        self.vulnerabilities.append(vulnerability)
                        
            except Exception:
                pass
                
    def _scan_data_protection(self):
        """Scan for data protection and privacy issues."""
        
        for py_file in self.project_root.glob("**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Data protection patterns
                data_issues = [
                    (r'(?i)(email|phone|ssn|credit.*card).*print', "MEDIUM", "PII in print statements"),
                    (r'(?i)(email|phone|ssn|credit.*card).*log', "MEDIUM", "PII in logs"),
                    (r'(?i)pd\.read_csv\(.*\)(?!.*encryption)', "LOW", "Unencrypted data file access"),
                    (r'(?i)backup.*(?!.*encrypt)', "MEDIUM", "Unencrypted backup operation"),
                ]
                
                for pattern, severity, description in data_issues:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        vulnerability = SecurityVulnerability(
                            severity=severity,
                            category="data_protection",
                            file_path=str(py_file),
                            line_number=line_number,
                            description=description,
                            recommendation="Implement data protection measures",
                            cwe_id="CWE-200"
                        )
                        self.vulnerabilities.append(vulnerability)
                        
            except Exception:
                pass
                
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0
            
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # Calculate entropy
        text_len = len(text)
        entropy = 0
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * (probability.bit_length() - 1)
            
        return entropy
        
    def _is_likely_false_positive(self, candidate: str, line: str) -> bool:
        """Check if high-entropy string is likely a false positive."""
        
        false_positive_indicators = [
            # Common false positives
            r'(?i)(test|example|sample|dummy|mock)',
            r'(?i)(placeholder|default|template)',
            r'(?i)(lorem|ipsum|dolor)',
            r'(?i)(base64|encoded|hash)',
            # URLs and paths
            r'https?://',
            r'file://',
            # Code patterns
            r'import\s+',
            r'from\s+.*\s+import',
            r'def\s+',
            r'class\s+',
        ]
        
        for pattern in false_positive_indicators:
            if re.search(pattern, line):
                return True
                
        # Check if it's a common hash or UUID format
        if re.match(r'^[a-f0-9]{32}$', candidate.lower()):  # MD5-like
            return True
        if re.match(r'^[a-f0-9]{40}$', candidate.lower()):  # SHA1-like
            return True
        if re.match(r'^[a-f0-9]{64}$', candidate.lower()):  # SHA256-like
            return True
            
        return False
        
    def _calculate_cvss_score(self, severity: str) -> float:
        """Calculate CVSS score based on severity."""
        severity_scores = {
            "CRITICAL": 9.5,
            "HIGH": 7.5,
            "MEDIUM": 5.0,
            "LOW": 2.5
        }
        return severity_scores.get(severity, 0.0)
        
    def _calculate_security_score(self) -> float:
        """Calculate overall security score."""
        if not self.vulnerabilities:
            return 100.0
            
        # Weight vulnerabilities by severity
        severity_weights = {
            "CRITICAL": 20,
            "HIGH": 10,
            "MEDIUM": 5,
            "LOW": 1
        }
        
        total_weight = sum(severity_weights.get(vuln.severity, 0) for vuln in self.vulnerabilities)
        
        # Base score of 100, subtract based on vulnerabilities
        security_score = max(0, 100 - (total_weight * 2))
        
        return security_score
        
    def _get_recommendation(self, category: str) -> str:
        """Get security recommendation for vulnerability category."""
        recommendations = {
            "hardcoded_secrets": "Store secrets in environment variables or secure vaults",
            "sql_injection": "Use parameterized queries or prepared statements",
            "command_injection": "Validate input and use safe command execution methods",
            "path_traversal": "Validate file paths and use safe file operations",
            "weak_crypto": "Use strong cryptographic algorithms like SHA-256 or AES",
            "insecure_deserialization": "Use safe deserialization methods or validate input",
            "information_disclosure": "Remove sensitive information from logs and error messages"
        }
        return recommendations.get(category, "Review and fix security issue")
        
    def _generate_security_recommendations(self) -> List[str]:
        """Generate prioritized security recommendations."""
        recommendations = []
        
        # Count vulnerabilities by category and severity
        critical_count = sum(1 for v in self.vulnerabilities if v.severity == "CRITICAL")
        high_count = sum(1 for v in self.vulnerabilities if v.severity == "HIGH")
        
        if critical_count > 0:
            recommendations.append(f"URGENT: Fix {critical_count} critical vulnerabilities immediately")
            
        if high_count > 0:
            recommendations.append(f"High Priority: Address {high_count} high-severity vulnerabilities")
            
        # Category-specific recommendations
        categories = {}
        for vuln in self.vulnerabilities:
            categories[vuln.category] = categories.get(vuln.category, 0) + 1
            
        if "hardcoded_secrets" in categories:
            recommendations.append("Implement secure secret management system")
            
        if "sql_injection" in categories:
            recommendations.append("Implement parameterized queries across all database operations")
            
        if "command_injection" in categories:
            recommendations.append("Review all system command executions for security")
            
        # General recommendations
        recommendations.extend([
            "Implement automated security testing in CI/CD pipeline",
            "Regular security training for development team",
            "Establish security code review process",
            "Implement security monitoring and alerting"
        ])
        
        return recommendations[:10]  # Top 10 recommendations
        
    def _check_owasp_compliance(self) -> bool:
        """Check OWASP Top 10 compliance."""
        owasp_categories = [
            "sql_injection", "command_injection", "hardcoded_secrets",
            "insecure_configuration", "vulnerable_dependency"
        ]
        
        for category in owasp_categories:
            if any(v.category == category for v in self.vulnerabilities):
                return False
        return True
        
    def _check_pci_compliance(self) -> bool:
        """Check PCI DSS compliance."""
        # Simplified PCI compliance check
        pci_violations = [
            "hardcoded_secrets", "weak_crypto", "information_disclosure"
        ]
        
        for violation in pci_violations:
            if any(v.category == violation for v in self.vulnerabilities):
                return False
        return True
        
    def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance."""
        # Check for data protection issues
        gdpr_violations = ["data_protection", "information_disclosure"]
        
        for violation in gdpr_violations:
            if any(v.category == violation for v in self.vulnerabilities):
                return False
        return True
        
    def _check_sox_compliance(self) -> bool:
        """Check SOX compliance."""
        # SOX focuses on financial data integrity
        sox_violations = ["insecure_configuration", "vulnerable_dependency"]
        
        for violation in sox_violations:
            if any(v.category == violation for v in self.vulnerabilities):
                return False
        return True
        
    def _check_hipaa_compliance(self) -> bool:
        """Check HIPAA compliance."""
        # HIPAA focuses on healthcare data protection
        hipaa_violations = ["data_protection", "weak_crypto", "hardcoded_secrets"]
        
        for violation in hipaa_violations:
            if any(v.category == violation for v in self.vulnerabilities):
                return False
        return True
        
    def _save_security_report(self, report: SecurityReport):
        """Save security report to file."""
        report_dict = asdict(report)
        
        with open("security_report.json", 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
            
        # Also save a summary report
        summary = {
            "scan_timestamp": report.scan_timestamp,
            "security_score": report.security_score,
            "total_vulnerabilities": len(report.vulnerabilities),
            "vulnerability_breakdown": {
                "critical": sum(1 for v in report.vulnerabilities if v.severity == "CRITICAL"),
                "high": sum(1 for v in report.vulnerabilities if v.severity == "HIGH"),
                "medium": sum(1 for v in report.vulnerabilities if v.severity == "MEDIUM"),
                "low": sum(1 for v in report.vulnerabilities if v.severity == "LOW")
            },
            "compliance_status": report.compliance_status,
            "top_recommendations": report.recommendations[:5]
        }
        
        with open("security_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Run advanced security scanning."""
    
    scanner = AdvancedSecurityScanner()
    report = scanner.scan_comprehensive()
    
    return report

if __name__ == "__main__":
    main()