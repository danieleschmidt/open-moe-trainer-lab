# Repository Configuration Guide

This document provides comprehensive guidance for configuring the Open MoE Trainer Lab repository for optimal development, CI/CD, and project management workflows.

## Table of Contents

- [Branch Protection Rules](#branch-protection-rules)
- [Required Status Checks](#required-status-checks)
- [GitHub Actions Secrets](#github-actions-secrets)
- [Repository Settings](#repository-settings)
- [Team Permissions](#team-permissions)
- [Issue and PR Templates](#issue-and-pr-templates)
- [Automation Configuration](#automation-configuration)
- [Security Configuration](#security-configuration)
- [Monitoring Setup](#monitoring-setup)

## Branch Protection Rules

### Main Branch Protection

Configure the `main` branch with the following protection rules:

```yaml
# .github/branch-protection-config.yml (documentation only - must be configured via GitHub UI)
main:
  required_status_checks:
    strict: true
    contexts:
      - "ci-success"
      - "quality / Code Quality"
      - "test-unit (3.11)"
      - "build / Build & Package"
      - "docker / Docker Build"
  enforce_admins: false
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
    restrict_pushes: true
  restrictions:
    users: []
    teams: ["core-maintainers"]
  allow_force_pushes: false
  allow_deletions: false
```

### Develop Branch Protection

```yaml
develop:
  required_status_checks:
    strict: false
    contexts:
      - "quality / Code Quality"
      - "test-unit (3.11)"
  required_pull_request_reviews:
    required_approving_review_count: 1
    dismiss_stale_reviews: false
    require_code_owner_reviews: false
  allow_force_pushes: false
  allow_deletions: false
```

### Release Branch Protection

```yaml
"release/*":
  required_status_checks:
    strict: true
    contexts:
      - "ci-success"
      - "security-scan"
      - "performance-tests"
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
  allow_force_pushes: false
  allow_deletions: false
```

## Required Status Checks

The following status checks should be required for the `main` branch:

### Core CI Checks
- `quality / Code Quality` - Code formatting, linting, type checking
- `test-unit (3.11)` - Unit tests on Python 3.11
- `test-integration / Integration Tests` - Integration test suite
- `build / Build & Package` - Package building and verification
- `docker / Docker Build` - Docker image building and testing

### Security Checks
- `static-analysis / Static Analysis` - Security linting (Bandit, Semgrep)
- `container-scan / Container Security` - Container vulnerability scanning
- `dependency-scan / Dependency Security` - Dependency vulnerability scanning

### Optional Checks (for release branches)
- `performance / Performance Tests` - Performance regression testing
- `test-e2e / E2E Tests` - End-to-end testing
- `security-scan / Security Scan` - Comprehensive security scanning

## GitHub Actions Secrets

Configure the following secrets in repository settings:

### Required Secrets

```bash
# GitHub token for API access (auto-generated)
GITHUB_TOKEN  # Automatically provided by GitHub Actions

# Docker registry credentials
DOCKER_USERNAME  # Docker Hub or GHCR username
DOCKER_PASSWORD  # Docker Hub or GHCR password/token

# PyPI publishing (for releases)
PYPI_API_TOKEN  # PyPI API token for package publishing
```

### Optional Secrets

```bash
# Monitoring and observability
PROMETHEUS_URL     # Prometheus server URL
GRAFANA_URL       # Grafana dashboard URL
GRAFANA_TOKEN     # Grafana API token

# Notification services
SLACK_WEBHOOK_URL          # General notifications
SECURITY_SLACK_WEBHOOK_URL # Security-specific notifications
METRICS_ENDPOINT          # Custom metrics endpoint

# External services
CODECOV_TOKEN     # Codecov integration token
SONARCLOUD_TOKEN  # SonarCloud analysis token

# Security scanning
GITLEAKS_LICENSE  # GitLeaks license (if using premium)
```

### Environment-Specific Secrets

```bash
# Development environment
DEV_DATABASE_URL
DEV_REDIS_URL
DEV_API_KEY

# Staging environment
STAGING_DATABASE_URL
STAGING_REDIS_URL
STAGING_API_KEY
STAGING_DEPLOY_KEY

# Production environment
PROD_DATABASE_URL
PROD_REDIS_URL
PROD_API_KEY
PROD_DEPLOY_KEY
```

## Repository Settings

### General Settings

```yaml
Repository Name: open-moe-trainer-lab
Description: "Advanced training laboratory for Mixture of Experts (MoE) models with distributed computing capabilities"
Website: https://moe-lab.example.com
Topics: 
  - machine-learning
  - mixture-of-experts
  - distributed-training
  - pytorch
  - transformer
  - ai
  - deep-learning
  - moe
  - model-training
  - gpu-computing

# Features
Issues: Enabled
Projects: Enabled
Wiki: Disabled
Discussions: Enabled
Security: Enabled

# Pull Requests
Allow merge commits: false
Allow squash merging: true (default)
Allow rebase merging: true
Always suggest updating pull request branches: true
Allow auto-merge: true
Automatically delete head branches: true
```

### Security Settings

```yaml
# Vulnerability reporting
Security advisories: Enabled
Private vulnerability reporting: Enabled

# Dependency graph
Dependency graph: Enabled
Dependabot alerts: Enabled
Dependabot security updates: Enabled
Dependabot version updates: Enabled

# Code scanning
Code scanning: Enabled
Secret scanning: Enabled
Secret scanning push protection: Enabled
```

### Pages Settings (if using GitHub Pages)

```yaml
Source: GitHub Actions
Custom domain: docs.moe-lab.example.com
Enforce HTTPS: true
```

## Team Permissions

### Recommended Team Structure

```yaml
teams:
  core-maintainers:
    permission: admin
    members:
      - lead-architect
      - senior-engineers
    
  maintainers:
    permission: maintain
    members:
      - experienced-contributors
      - domain-experts
    
  contributors:
    permission: write
    members:
      - regular-contributors
      - trusted-developers
    
  reviewers:
    permission: triage
    members:
      - code-reviewers
      - community-moderators
    
  external-contributors:
    permission: read
    members:
      - occasional-contributors
      - external-researchers
```

### CODEOWNERS Configuration

Create `.github/CODEOWNERS`:

```bash
# Global owners
* @core-maintainers

# Core library code
/moe_lab/ @core-maintainers @maintainers

# Configuration and infrastructure
/.github/ @core-maintainers
/docker/ @core-maintainers @infrastructure-team
/scripts/ @core-maintainers @maintainers

# Documentation
/docs/ @maintainers @documentation-team
*.md @maintainers @documentation-team

# Tests
/tests/ @maintainers @qa-team

# Security-sensitive files
/security/ @core-maintainers @security-team
/.github/workflows/security.yml @core-maintainers @security-team
/scripts/collect_metrics.py @core-maintainers @maintainers

# Build and deployment
/Dockerfile @core-maintainers @infrastructure-team
/docker-compose.yml @core-maintainers @infrastructure-team
/pyproject.toml @core-maintainers @maintainers
```

## Issue and PR Templates

### Issue Template Configuration

Create `.github/ISSUE_TEMPLATE/config.yml`:

```yaml
blank_issues_enabled: false
contact_links:
  - name: Security Vulnerability
    url: https://github.com/your-org/open-moe-trainer-lab/security/advisories/new
    about: Report security vulnerabilities privately
  - name: Discussion Forum
    url: https://github.com/your-org/open-moe-trainer-lab/discussions
    about: Ask questions and discuss ideas
```

Templates should be created for:
- Bug reports (`.github/ISSUE_TEMPLATE/bug_report.yml`)
- Feature requests (`.github/ISSUE_TEMPLATE/feature_request.yml`)
- Performance issues (`.github/ISSUE_TEMPLATE/performance.yml`)
- Documentation improvements (`.github/ISSUE_TEMPLATE/documentation.yml`)

### Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Test improvements

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)
- [ ] Manual testing completed

## Checklist
- [ ] Code follows the style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented appropriately
- [ ] Documentation updated (if needed)
- [ ] Tests added/updated for changes
- [ ] All CI checks pass
- [ ] No sensitive information committed

## Related Issues
Closes #(issue_number)

## Performance Impact
Describe any performance implications of the changes.

## Breaking Changes
List any breaking changes and migration steps needed.
```

## Automation Configuration

### Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
      time: "04:00"
    open-pull-requests-limit: 10
    reviewers:
      - "maintainers"
    assignees:
      - "core-maintainers"
    commit-message:
      prefix: "deps"
      include: "scope"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    reviewers:
      - "core-maintainers"
    commit-message:
      prefix: "ci"

  # Docker
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "infrastructure-team"
```

### Auto-merge Configuration

Create `.github/workflows/auto-merge.yml`:

```yaml
name: Auto-merge
on:
  pull_request:
    types: [labeled, unlabeled, synchronize, opened, edited, ready_for_review, reopened]
  pull_request_review:
    types: [submitted]
  check_suite:
    types: [completed]
  status: {}

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]' || contains(github.event.pull_request.labels.*.name, 'auto-merge')
    steps:
      - name: Auto-merge
        uses: pascalgn/auto-merge-action@v0.15.6
        with:
          merge_method: "squash"
          github_token: "${{ secrets.GITHUB_TOKEN }}"
```

## Security Configuration

### Security Policy

Create `SECURITY.md`:

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| 0.x.x   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please report security vulnerabilities through GitHub's private vulnerability reporting feature or email security@moe-lab.example.com.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- Initial response: Within 48 hours
- Status update: Within 1 week
- Resolution: Depends on severity and complexity
```

### Security Scanning Configuration

Ensure the following security tools are configured:

1. **CodeQL Analysis** - Enabled in repository settings
2. **Dependabot Security Updates** - Configured in `.github/dependabot.yml`
3. **Secret Scanning** - Enabled with push protection
4. **Container Scanning** - Integrated in CI/CD workflows
5. **License Compliance** - Monitored through workflows

## Monitoring Setup

### Repository Insights

Configure the following monitoring:

1. **Pulse Dashboard** - Track activity and contributions
2. **Code Frequency** - Monitor development velocity
3. **Contributors** - Track contributor engagement
4. **Traffic** - Monitor repository usage
5. **Community Profile** - Ensure completeness

### Custom Metrics Collection

The repository includes automated metrics collection via `scripts/collect_metrics.py`. Configure the following:

```bash
# Environment variables for metrics collection
export GITHUB_TOKEN="your-github-token"
export PROMETHEUS_URL="http://prometheus.example.com:9090"
export GRAFANA_URL="http://grafana.example.com:3000"
export GRAFANA_TOKEN="your-grafana-token"
```

### Alert Configuration

Set up alerts for:

- Failed CI/CD workflows
- Security vulnerabilities
- Performance regressions
- High error rates
- Dependency updates
- Release notifications

## Integration Checklist

After configuring the repository, verify the following:

### Repository Settings
- [ ] Branch protection rules configured
- [ ] Required status checks enabled
- [ ] Team permissions assigned
- [ ] Security features enabled
- [ ] Issue/PR templates created

### Secrets Configuration
- [ ] Required secrets added
- [ ] Environment-specific secrets configured
- [ ] Monitoring service tokens added
- [ ] Notification webhooks configured

### Automation Setup
- [ ] Dependabot configuration active
- [ ] Auto-merge rules working
- [ ] CI/CD workflows passing
- [ ] Security scans running
- [ ] Metrics collection operational

### Documentation
- [ ] README updated with setup instructions
- [ ] CONTRIBUTING guide available
- [ ] Security policy published
- [ ] Code of conduct established
- [ ] License file present

### Testing
- [ ] All CI workflows passing
- [ ] Branch protection enforced
- [ ] Required reviews working
- [ ] Auto-merge functioning (for dependabot)
- [ ] Security scans completing
- [ ] Metrics collection successful

## Maintenance

### Regular Tasks

1. **Weekly**: Review dependency updates
2. **Monthly**: Update security configurations
3. **Quarterly**: Review team permissions
4. **Annually**: Audit repository settings and configurations

### Configuration Updates

When updating repository configuration:

1. Test changes in a fork or test repository first
2. Document changes in this file
3. Notify team members of significant changes
4. Monitor for any issues after implementation

### Backup and Recovery

Ensure the following are backed up:

- Repository settings export
- Team and permission configurations
- Secrets list (names only, not values)
- Workflow configurations
- Custom automation scripts

---

For questions about repository configuration, please contact the core maintainers or create an issue with the `repository-config` label.