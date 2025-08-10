#!/usr/bin/env python3
"""
Security Scan for MoE Trainer Lab
Comprehensive security analysis and vulnerability assessment.
"""

import os
import sys
import json
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set
import ast
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SecurityScanner:
    """Comprehensive security scanner for MoE codebase."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.vulnerabilities = []
        self.security_issues = []
        self.suspicious_patterns = []
        
        # Security patterns to detect
        self.dangerous_patterns = [
            # Code injection patterns
            (r'eval\s*\(', 'eval() function usage - code injection risk'),
            (r'exec\s*\(', 'exec() function usage - code execution risk'),
            (r'__import__\s*\(', 'dynamic import - potential code injection'),
            (r'compile\s*\(', 'compile() function - dynamic code compilation'),
            
            # File system risks
            (r'open\s*\([^)]*[\'\"]\.\.[\/\\]', 'Path traversal vulnerability'),
            (r'os\.system\s*\(', 'os.system() usage - command injection risk'),
            (r'subprocess\.[a-zA-Z_]+\([^)]*shell\s*=\s*True', 'shell=True subprocess - injection risk'),
            
            # Network security
            (r'urllib\.request\.urlopen\s*\([^)]*[\'\"](http[^s])', 'Insecure HTTP connection'),
            (r'requests\.[a-z]+\s*\([^)]*verify\s*=\s*False', 'SSL verification disabled'),
            
            # Secret patterns
            (r'password\s*=\s*[\'\"]\w+[\'\"]+', 'Hardcoded password'),
            (r'secret\s*=\s*[\'\"]\w+[\'\"]+', 'Hardcoded secret'),
            (r'api[_-]?key\s*=\s*[\'\"]\w+[\'\"]+', 'Hardcoded API key'),
            (r'token\s*=\s*[\'\"]\w+[\'\"]+', 'Hardcoded token'),
            
            # SQL injection patterns
            (r'SELECT.*\+.*[\'\"]+', 'Potential SQL injection'),
            (r'INSERT.*\+.*[\'\"]+', 'Potential SQL injection'),
            (r'UPDATE.*\+.*[\'\"]+', 'Potential SQL injection'),
            (r'DELETE.*\+.*[\'\"]+', 'Potential SQL injection'),
        ]
        
        # File extensions to scan
        self.scannable_extensions = {'.py', '.yml', '.yaml', '.json', '.sh', '.md', '.txt'}
        
        # Excluded directories
        self.excluded_dirs = {'__pycache__', '.git', '.pytest_cache', 'htmlcov', 'node_modules', 'venv', '.venv'}
    
    def scan_all(self) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        logger.info(f"Starting security scan of {self.project_root}")
        
        results = {
            'project_root': str(self.project_root),
            'timestamp': __import__('time').time(),
            'file_scans': [],
            'vulnerability_summary': {},
            'security_score': 0,
            'recommendations': [],
            'total_files_scanned': 0
        }
        
        # Scan all files
        for file_path in self._get_scannable_files():
            file_results = self._scan_file(file_path)
            if file_results:
                results['file_scans'].append(file_results)
            results['total_files_scanned'] += 1
        
        # Generate summary
        results['vulnerability_summary'] = self._generate_vulnerability_summary()
        results['security_score'] = self._calculate_security_score()
        results['recommendations'] = self._generate_recommendations()
        
        return results
    
    def _get_scannable_files(self) -> List[Path]:
        """Get all files that should be scanned."""
        files = []
        
        for root, dirs, filenames in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in self.scannable_extensions:
                    files.append(file_path)
        
        return files
    
    def _scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan individual file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return None
        
        file_issues = []
        
        # Pattern-based scanning
        for line_num, line in enumerate(content.split('\n'), 1):
            for pattern, description in self.dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    file_issues.append({
                        'line': line_num,
                        'issue': description,
                        'pattern': pattern,
                        'code_snippet': line.strip()[:100],
                        'severity': self._assess_severity(description)
                    })
        
        # Python-specific AST analysis
        if file_path.suffix == '.py':
            ast_issues = self._analyze_python_ast(file_path, content)
            file_issues.extend(ast_issues)
        
        # File permissions check
        permission_issues = self._check_file_permissions(file_path)
        file_issues.extend(permission_issues)
        
        if file_issues:
            return {
                'file_path': str(file_path.relative_to(self.project_root)),
                'issues': file_issues,
                'issue_count': len(file_issues),
                'file_hash': hashlib.md5(content.encode()).hexdigest()[:8]
            }
        
        return None
    
    def _analyze_python_ast(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Analyze Python file using AST for deeper security analysis."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        
                        if func_name in ['eval', 'exec', 'compile']:
                            issues.append({
                                'line': getattr(node, 'lineno', 0),
                                'issue': f'Dangerous function: {func_name}()',
                                'pattern': 'AST_ANALYSIS',
                                'code_snippet': f'{func_name}() call detected',
                                'severity': 'HIGH'
                            })
                
                # Check for hardcoded strings that might be secrets
                if isinstance(node, ast.Str):
                    if len(node.s) > 20 and re.match(r'^[a-zA-Z0-9+/=]{20,}$', node.s):
                        issues.append({
                            'line': getattr(node, 'lineno', 0),
                            'issue': 'Potential hardcoded secret/token',
                            'pattern': 'AST_ANALYSIS',
                            'code_snippet': f'String: {node.s[:30]}...',
                            'severity': 'MEDIUM'
                        })
                
                # Check for subprocess calls with shell=True
                if (isinstance(node, ast.Call) and 
                    isinstance(node.func, ast.Attribute) and
                    getattr(node.func.value, 'id', None) == 'subprocess'):
                    
                    for keyword in node.keywords:
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                            if keyword.value.value is True:
                                issues.append({
                                    'line': getattr(node, 'lineno', 0),
                                    'issue': 'subprocess call with shell=True',
                                    'pattern': 'AST_ANALYSIS', 
                                    'code_snippet': 'subprocess call with shell=True',
                                    'severity': 'HIGH'
                                })
        
        except SyntaxError:
            # File has syntax errors, skip AST analysis
            pass
        except Exception as e:
            logger.debug(f"AST analysis failed for {file_path}: {e}")
        
        return issues
    
    def _check_file_permissions(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check file permissions for security issues."""
        issues = []
        
        try:
            stat_info = file_path.stat()
            mode = stat_info.st_mode
            
            # Check if file is world-writable
            if mode & 0o002:
                issues.append({
                    'line': 0,
                    'issue': 'World-writable file',
                    'pattern': 'FILE_PERMISSIONS',
                    'code_snippet': f'Mode: {oct(mode)}',
                    'severity': 'MEDIUM'
                })
            
            # Check executable files
            if mode & 0o111 and file_path.suffix not in {'.sh', '.py'}:
                issues.append({
                    'line': 0,
                    'issue': 'Unexpected executable file',
                    'pattern': 'FILE_PERMISSIONS',
                    'code_snippet': f'Executable: {file_path.name}',
                    'severity': 'LOW'
                })
                
        except Exception:
            pass
        
        return issues
    
    def _assess_severity(self, description: str) -> str:
        """Assess severity level of security issue."""
        high_risk_keywords = ['injection', 'eval', 'exec', 'system', 'shell=True']
        medium_risk_keywords = ['password', 'secret', 'token', 'api_key', 'traversal']
        
        description_lower = description.lower()
        
        for keyword in high_risk_keywords:
            if keyword in description_lower:
                return 'HIGH'
        
        for keyword in medium_risk_keywords:
            if keyword in description_lower:
                return 'MEDIUM'
        
        return 'LOW'
    
    def _generate_vulnerability_summary(self) -> Dict[str, Any]:
        """Generate summary of all vulnerabilities found."""
        summary = {
            'total_issues': 0,
            'severity_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'issue_types': {},
            'most_vulnerable_files': []
        }
        
        all_issues = []
        file_issue_counts = []
        
        for scan_result in getattr(self, '_scan_results', []):
            if 'issues' in scan_result:
                file_issue_counts.append((scan_result['file_path'], scan_result['issue_count']))
                
                for issue in scan_result['issues']:
                    all_issues.append(issue)
                    summary['total_issues'] += 1
                    
                    # Count by severity
                    severity = issue.get('severity', 'LOW')
                    summary['severity_breakdown'][severity] += 1
                    
                    # Count by issue type
                    issue_type = issue['issue']
                    summary['issue_types'][issue_type] = summary['issue_types'].get(issue_type, 0) + 1
        
        # Most vulnerable files
        file_issue_counts.sort(key=lambda x: x[1], reverse=True)
        summary['most_vulnerable_files'] = file_issue_counts[:5]
        
        return summary
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security score (0-100)."""
        if not hasattr(self, '_scan_results'):
            return 100
        
        total_issues = 0
        severity_weights = {'HIGH': 10, 'MEDIUM': 5, 'LOW': 1}
        weighted_issues = 0
        
        for scan_result in self._scan_results:
            if 'issues' in scan_result:
                for issue in scan_result['issues']:
                    total_issues += 1
                    severity = issue.get('severity', 'LOW')
                    weighted_issues += severity_weights.get(severity, 1)
        
        if total_issues == 0:
            return 100
        
        # Score calculation: start from 100, subtract points based on issues
        score = 100 - min(90, weighted_issues * 2)  # Cap at 10 minimum score
        return max(10, score)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if not hasattr(self, '_scan_results'):
            return ["Security scan completed successfully with no issues found."]
        
        # Count issue types
        issue_counts = {}
        for scan_result in self._scan_results:
            if 'issues' in scan_result:
                for issue in scan_result['issues']:
                    issue_type = issue['issue']
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Generate specific recommendations
        if any('eval' in issue_type.lower() or 'exec' in issue_type.lower() for issue_type in issue_counts):
            recommendations.append("Remove eval() and exec() function calls - use safer alternatives like ast.literal_eval()")
        
        if any('password' in issue_type.lower() or 'secret' in issue_type.lower() for issue_type in issue_counts):
            recommendations.append("Move hardcoded secrets to environment variables or secure configuration files")
        
        if any('shell=true' in issue_type.lower() for issue_type in issue_counts):
            recommendations.append("Avoid shell=True in subprocess calls - use array arguments instead")
        
        if any('injection' in issue_type.lower() for issue_type in issue_counts):
            recommendations.append("Use parameterized queries and input validation to prevent injection attacks")
        
        # General recommendations
        recommendations.extend([
            "Implement input validation and sanitization for all user inputs",
            "Use secure coding practices and regular security reviews",
            "Keep dependencies updated to patch known vulnerabilities",
            "Implement proper authentication and authorization mechanisms",
            "Use HTTPS for all network communications",
            "Implement logging and monitoring for security events"
        ])
        
        return recommendations[:10]  # Limit to top 10


def run_security_scan() -> Dict[str, Any]:
    """Run comprehensive security scan and return results."""
    print("🛡️  Security Scan - MoE Trainer Lab")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    scanner = SecurityScanner(str(project_root))
    
    # Run the scan
    results = scanner.scan_all()
    
    # Store results for internal access
    scanner._scan_results = results['file_scans']
    
    # Regenerate summary with stored results
    results['vulnerability_summary'] = scanner._generate_vulnerability_summary()
    results['security_score'] = scanner._calculate_security_score()
    results['recommendations'] = scanner._generate_recommendations()
    
    # Display results
    print(f"\n📊 Security Scan Results:")
    print(f"  Files scanned: {results['total_files_scanned']}")
    print(f"  Total security issues: {results['vulnerability_summary']['total_issues']}")
    print(f"  Security score: {results['security_score']}/100")
    
    severity_breakdown = results['vulnerability_summary']['severity_breakdown']
    if severity_breakdown['HIGH'] + severity_breakdown['MEDIUM'] + severity_breakdown['LOW'] > 0:
        print(f"\n🚨 Issues by severity:")
        print(f"  HIGH: {severity_breakdown['HIGH']}")
        print(f"  MEDIUM: {severity_breakdown['MEDIUM']}")
        print(f"  LOW: {severity_breakdown['LOW']}")
    
    if results['vulnerability_summary']['most_vulnerable_files']:
        print(f"\n📁 Files with most issues:")
        for file_path, issue_count in results['vulnerability_summary']['most_vulnerable_files']:
            print(f"  {file_path}: {issue_count} issues")
    
    print(f"\n💡 Security Recommendations:")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    # Security score assessment
    score = results['security_score']
    if score >= 90:
        print(f"\n✅ Excellent security posture!")
    elif score >= 75:
        print(f"\n🟡 Good security with room for improvement")
    elif score >= 50:
        print(f"\n🟠 Security needs attention")
    else:
        print(f"\n🔴 Critical security issues found")
    
    return results


if __name__ == "__main__":
    # Run security scan
    scan_results = run_security_scan()
    
    # Save results to file
    output_file = "security_scan_results.json"
    with open(output_file, 'w') as f:
        json.dump(scan_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to {output_file}")
    
    # Exit with appropriate code
    exit_code = 0 if scan_results['security_score'] >= 75 else 1
    print(f"\n🏁 Security scan complete (exit code: {exit_code})")
    sys.exit(exit_code)