#!/bin/bash

# Security scanning script for Open MoE Trainer Lab
# Runs comprehensive security checks on the codebase

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SECURITY_DIR="${PROJECT_ROOT}/security"
REPORTS_DIR="${SECURITY_DIR}/reports"

# Create reports directory
mkdir -p "${REPORTS_DIR}"

# Timestamp for reports
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# Exit codes
EXIT_CODE=0

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    print_status "$BLUE" "=== $1 ==="
}

print_success() {
    print_status "$GREEN" "✅ $1"
}

print_warning() {
    print_status "$YELLOW" "⚠️  $1"
}

print_error() {
    print_status "$RED" "❌ $1"
    EXIT_CODE=1
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python dependencies if needed
ensure_dependencies() {
    print_header "Checking Dependencies"
    
    local deps=(
        "bandit"
        "safety"
        "pip-audit"
        "semgrep"
        "checkov"
    )
    
    for dep in "${deps[@]}"; do
        if ! command_exists "$dep"; then
            print_warning "$dep not found, attempting to install..."
            pip install "$dep" || {
                print_error "Failed to install $dep"
                return 1
            }
        else
            print_success "$dep found"
        fi
    done
}

# Function to run bandit security linting
run_bandit() {
    print_header "Running Bandit Security Linting"
    
    local bandit_config="${SECURITY_DIR}/bandit.yml"
    local report_file="${REPORTS_DIR}/bandit_${TIMESTAMP}.json"
    
    if [[ -f "$bandit_config" ]]; then
        bandit_args="-c $bandit_config"
    else
        bandit_args=""
    fi
    
    if bandit -r "${PROJECT_ROOT}/moe_lab" \
        $bandit_args \
        -f json \
        -o "$report_file" \
        -ll; then
        print_success "Bandit scan completed successfully"
    else
        print_error "Bandit found security issues - check $report_file"
    fi
    
    # Generate human-readable report
    bandit -r "${PROJECT_ROOT}/moe_lab" $bandit_args -f txt || true
}

# Function to run safety check for dependencies
run_safety() {
    print_header "Running Safety Check for Dependencies"
    
    local report_file="${REPORTS_DIR}/safety_${TIMESTAMP}.json"
    
    cd "$PROJECT_ROOT"
    
    if safety check --json --output "$report_file"; then
        print_success "Safety check passed - no known vulnerabilities"
    else
        print_error "Safety found vulnerable dependencies - check $report_file"
    fi
    
    # Generate human-readable report
    safety check || true
}

# Function to run pip-audit
run_pip_audit() {
    print_header "Running pip-audit for Python Package Vulnerabilities"
    
    local report_file="${REPORTS_DIR}/pip_audit_${TIMESTAMP}.json"
    
    cd "$PROJECT_ROOT"
    
    if pip-audit --format=json --output="$report_file"; then
        print_success "pip-audit check passed"
    else
        print_error "pip-audit found vulnerabilities - check $report_file"
    fi
    
    # Generate human-readable report
    pip-audit || true
}

# Function to run semgrep static analysis
run_semgrep() {
    print_header "Running Semgrep Static Analysis"
    
    if ! command_exists semgrep; then
        print_warning "Semgrep not found, skipping..."
        return 0
    fi
    
    local report_file="${REPORTS_DIR}/semgrep_${TIMESTAMP}.json"
    
    cd "$PROJECT_ROOT"
    
    if semgrep --config=auto \
        --json \
        --output="$report_file" \
        --exclude="tests/" \
        --exclude="docs/" \
        moe_lab/; then
        print_success "Semgrep scan completed successfully"
    else
        print_error "Semgrep found issues - check $report_file"
    fi
    
    # Generate human-readable report
    semgrep --config=auto moe_lab/ || true
}

# Function to run secret scanning
run_secret_scan() {
    print_header "Running Secret Scanning"
    
    local report_file="${REPORTS_DIR}/secrets_${TIMESTAMP}.txt"
    
    # Check for common secret patterns
    local secret_patterns=(
        "password\s*=\s*['\"][^'\"]*['\"]"
        "api_key\s*=\s*['\"][^'\"]*['\"]"
        "secret\s*=\s*['\"][^'\"]*['\"]"
        "token\s*=\s*['\"][^'\"]*['\"]"
        "-----BEGIN.*PRIVATE KEY-----"
        "AKIA[0-9A-Z]{16}"  # AWS Access Key
        "sk-[a-zA-Z0-9]{48}"  # OpenAI API Key
    )
    
    echo "Secret scan results:" > "$report_file"
    
    local secrets_found=false
    for pattern in "${secret_patterns[@]}"; do
        if grep -r -E "$pattern" "${PROJECT_ROOT}/moe_lab" 2>/dev/null; then
            secrets_found=true
            echo "Found potential secret: $pattern" >> "$report_file"
        fi
    done
    
    if [[ "$secrets_found" == true ]]; then
        print_error "Potential secrets found - check $report_file"
    else
        print_success "No obvious secrets detected"
    fi
}

# Function to check file permissions
check_file_permissions() {
    print_header "Checking File Permissions"
    
    local report_file="${REPORTS_DIR}/permissions_${TIMESTAMP}.txt"
    
    echo "File permission check results:" > "$report_file"
    
    # Check for overly permissive files
    local issues_found=false
    
    # Find world-writable files
    if find "${PROJECT_ROOT}" -type f -perm -002 -not -path "*/.*" 2>/dev/null | head -10 | tee -a "$report_file"; then
        issues_found=true
        print_warning "Found world-writable files"
    fi
    
    # Find files with execute permissions that shouldn't have them
    if find "${PROJECT_ROOT}" -name "*.py" -perm -111 -not -path "*/scripts/*" -not -path "*/bin/*" 2>/dev/null | head -10 | tee -a "$report_file"; then
        issues_found=true
        print_warning "Found Python files with execute permissions"
    fi
    
    if [[ "$issues_found" == false ]]; then
        print_success "File permissions look good"
    fi
}

# Function to check Docker security
check_docker_security() {
    print_header "Checking Docker Security"
    
    if ! command_exists docker; then
        print_warning "Docker not found, skipping Docker security checks"
        return 0
    fi
    
    local report_file="${REPORTS_DIR}/docker_${TIMESTAMP}.txt"
    
    echo "Docker security check results:" > "$report_file"
    
    # Check Dockerfile security
    local dockerfiles=(
        "${PROJECT_ROOT}/Dockerfile"
        "${PROJECT_ROOT}/Dockerfile.dev"
        "${PROJECT_ROOT}/Dockerfile.prod"
    )
    
    for dockerfile in "${dockerfiles[@]}"; do
        if [[ -f "$dockerfile" ]]; then
            echo "Checking $dockerfile..." >> "$report_file"
            
            # Check for running as root
            if grep -q "USER root" "$dockerfile" 2>/dev/null; then
                print_warning "Dockerfile runs as root: $dockerfile"
                echo "WARNING: Running as root in $dockerfile" >> "$report_file"
            fi
            
            # Check for latest tags
            if grep -q ":latest" "$dockerfile" 2>/dev/null; then
                print_warning "Dockerfile uses :latest tag: $dockerfile"
                echo "WARNING: Using :latest tag in $dockerfile" >> "$report_file"
            fi
            
            # Check for COPY without chown
            if grep -E "^COPY.*[^[:space:]]" "$dockerfile" | grep -v "chown" >/dev/null 2>&1; then
                print_warning "COPY without explicit chown in: $dockerfile"
                echo "WARNING: COPY without chown in $dockerfile" >> "$report_file"
            fi
        fi
    done
    
    print_success "Docker security check completed"
}

# Function to check dependencies for known vulnerabilities
check_dependency_licenses() {
    print_header "Checking Dependency Licenses"
    
    local report_file="${REPORTS_DIR}/licenses_${TIMESTAMP}.txt"
    
    cd "$PROJECT_ROOT"
    
    echo "Dependency license check results:" > "$report_file"
    
    if command_exists pip-licenses; then
        pip-licenses --format=plain >> "$report_file"
        
        # Check for potentially problematic licenses
        local problematic_licenses=("GPL-3.0" "AGPL-3.0" "SSPL")
        
        for license in "${problematic_licenses[@]}"; do
            if pip-licenses | grep -q "$license"; then
                print_warning "Found potentially problematic license: $license"
                echo "WARNING: Found $license license" >> "$report_file"
            fi
        done
        
        print_success "License check completed"
    else
        print_warning "pip-licenses not found, installing..."
        pip install pip-licenses
        pip-licenses --format=plain >> "$report_file"
    fi
}

# Function to generate summary report
generate_summary() {
    print_header "Generating Security Scan Summary"
    
    local summary_file="${REPORTS_DIR}/summary_${TIMESTAMP}.txt"
    
    cat > "$summary_file" << EOF
Security Scan Summary
====================
Date: $(date)
Project: Open MoE Trainer Lab
Scan ID: ${TIMESTAMP}

Reports Generated:
EOF
    
    # List all generated reports
    for report in "${REPORTS_DIR}"/*_"${TIMESTAMP}".*; do
        if [[ -f "$report" ]]; then
            echo "- $(basename "$report")" >> "$summary_file"
        fi
    done
    
    cat >> "$summary_file" << EOF

Exit Code: ${EXIT_CODE}

Recommendations:
- Review all reports for security findings
- Address any high-severity vulnerabilities immediately
- Update dependencies with known vulnerabilities
- Fix any configuration issues identified
- Re-run scan after remediation

Next Steps:
1. Prioritize findings by severity
2. Create tickets for remediation
3. Schedule follow-up scans
4. Update security policies if needed
EOF
    
    print_success "Summary report generated: $summary_file"
}

# Function to cleanup old reports
cleanup_old_reports() {
    print_header "Cleaning Up Old Reports"
    
    # Keep only the last 10 reports
    find "${REPORTS_DIR}" -name "*_*.txt" -o -name "*_*.json" | sort -r | tail -n +11 | xargs rm -f
    
    print_success "Old reports cleaned up"
}

# Main function
main() {
    print_header "Starting Security Scan for Open MoE Trainer Lab"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Ensure we have the required tools
    ensure_dependencies || {
        print_error "Failed to install required dependencies"
        exit 1
    }
    
    # Run all security checks
    run_bandit
    run_safety
    run_pip_audit
    run_semgrep
    run_secret_scan
    check_file_permissions
    check_docker_security
    check_dependency_licenses
    
    # Generate summary and cleanup
    generate_summary
    cleanup_old_reports
    
    print_header "Security Scan Completed"
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        print_success "No critical security issues found!"
    else
        print_error "Security issues detected - review reports in ${REPORTS_DIR}"
    fi
    
    print_status "$BLUE" "Reports available in: ${REPORTS_DIR}"
    
    exit $EXIT_CODE
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi