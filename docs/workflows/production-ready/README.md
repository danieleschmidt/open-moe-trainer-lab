# Production-Ready GitHub Actions Workflows

This directory contains enterprise-grade GitHub Actions workflows designed specifically for the Open MoE Trainer Lab ML/AI development workflow.

## 🚀 Workflow Overview

### [`ci.yml`](./ci.yml) - Comprehensive CI Pipeline
**Purpose**: Multi-platform testing with ML/AI optimizations

**Key Features:**
- **Matrix Testing**: Python 3.9-3.12 × PyTorch 2.0-2.1 × OS (Ubuntu/macOS/Windows)
- **GPU Support**: CUDA testing with distributed training validation
- **Performance Benchmarking**: Automated regression detection with trend analysis
- **Security Integration**: Bandit, Safety, and CodeQL scanning
- **Coverage Reporting**: Automated test coverage with Codecov integration
- **Docker Builds**: Multi-stage containerization with caching

**Triggers:**
- Push to `main`, `develop` branches
- Pull requests to `main`, `develop`
- Weekly scheduled runs (Mondays 2 AM UTC)

### [`release.yml`](./release.yml) - Automated Release Management
**Purpose**: Zero-touch releases with comprehensive validation

**Key Features:**
- **Semantic Versioning**: Automated version validation and tagging
- **Multi-Platform Builds**: Docker images for CPU and GPU variants
- **PyPI Publishing**: Automated package publishing with test environment validation
- **SBOM Generation**: Supply chain security with software bill of materials
- **Documentation Deployment**: Automated docs updates to GitHub Pages
- **Changelog Generation**: Automated release notes from commit history

**Triggers:**
- Git tags matching `v*` pattern
- Manual workflow dispatch with version input

### [`security.yml`](./security.yml) - Multi-Layer Security Scanning
**Purpose**: Comprehensive security posture management

**Key Features:**
- **Secret Detection**: TruffleHog for credential scanning
- **Dependency Scanning**: Safety and pip-audit for vulnerability detection
- **Static Analysis**: Bandit, CodeQL, and Semgrep for code security
- **Container Security**: Trivy for Docker image vulnerability scanning
- **License Compliance**: Automated license compatibility verification
- **OSSF Scorecard**: Security posture scoring and recommendations

**Triggers:**
- Push and pull request events
- Weekly security scans (Mondays 6 AM UTC)

## 🛠️ Setup Instructions

### 1. Activate Workflows

```bash
# From repository root
mkdir -p .github/workflows
cp docs/workflows/production-ready/*.yml .github/workflows/
git add .github/workflows/
git commit -m "feat: activate production-ready GitHub Actions workflows"
git push origin main
```

### 2. Configure Repository Secrets

Navigate to **Settings → Secrets and variables → Actions**:

```
# Required for releases
PYPI_API_TOKEN          # Production PyPI publishing token
TEST_PYPI_API_TOKEN     # Test PyPI publishing token

# Optional for enhanced features
SEMGREP_APP_TOKEN       # Semgrep security analysis token
CODECOV_TOKEN           # Code coverage reporting token
```

### 3. Repository Settings Configuration

#### Actions Permissions
**Settings → Actions → General**:
- ✅ Allow GitHub Actions to create and approve pull requests
- ✅ Read and write permissions
- ✅ Allow GitHub Actions to write to security events

#### Branch Protection
**Settings → Branches → Add rule** for `main`:
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators
- **Required status checks**:
  - `Pre-commit Checks`
  - `Test Suite (ubuntu-latest, 3.11, 2.1.0)`
  - `Security Scanning / Code Security Analysis`
  - `Quality Gate`

### 4. Self-Hosted Runner Setup (Optional)

For GPU testing, configure runners with:
- NVIDIA GPU support (CUDA 11.8+)
- Docker with GPU passthrough
- Labels: `[self-hosted, gpu]`

If unavailable, modify `ci.yml`:
```yaml
gpu-tests:
  runs-on: ubuntu-latest  # Change from [self-hosted, gpu]
  if: false               # Disable GPU tests
```

## 📊 Workflow Behavior

### CI Pipeline Flow
1. **Pre-commit Checks**: Code quality validation
2. **Matrix Testing**: Multi-platform and version compatibility
3. **GPU Tests**: CUDA and distributed training validation (if runners available)
4. **Performance Tests**: Benchmark execution and regression detection
5. **Security Scanning**: Vulnerability and compliance verification
6. **Documentation Build**: Docs generation and validation
7. **Docker Build**: Container image creation and registry push
8. **Quality Gate**: Final validation before merge approval

### Release Pipeline Flow
1. **Version Validation**: Semantic version format and uniqueness
2. **Comprehensive Testing**: Full test suite execution
3. **Artifact Building**: Python packages and Docker images
4. **Changelog Generation**: Automated release notes
5. **GitHub Release**: Tag creation and asset upload
6. **PyPI Publishing**: Package distribution (Test → Production)
7. **Documentation Update**: Version-specific docs deployment
8. **Notification**: Success/failure alerts

### Security Pipeline Flow
1. **Secret Scanning**: Credential and key detection
2. **Dependency Analysis**: Vulnerability assessment
3. **Static Code Analysis**: Security pattern detection
4. **Container Scanning**: Docker image vulnerability assessment
5. **License Compliance**: Legal compatibility verification
6. **SBOM Generation**: Supply chain transparency
7. **Compliance Reporting**: Automated security posture reports

## 🎯 Performance Optimizations

### Caching Strategy
- **pip cache**: Python package installations
- **Docker buildx cache**: Multi-layer container builds
- **GitHub Actions cache**: Workflow artifacts and dependencies

### Parallel Execution
- **Matrix jobs**: Parallel testing across configurations
- **Conditional execution**: Smart job skipping based on changes
- **Artifact sharing**: Efficient build artifact distribution

### Resource Optimization
- **Concurrency groups**: Prevent redundant workflow runs
- **Selective triggers**: Context-aware execution
- **Timeout management**: Prevent resource waste

## 🔍 Monitoring and Alerts

### Success Metrics
- **Test Coverage**: Minimum 80% enforcement
- **Build Time**: Performance trend monitoring
- **Security Score**: OSSF Scorecard tracking
- **Release Frequency**: Deployment velocity metrics

### Alert Configuration
- **Failed Tests**: Immediate notification
- **Security Vulnerabilities**: High/Critical findings
- **Performance Regression**: Benchmark threshold violations
- **Release Issues**: Deployment failure alerts

## 🚀 Advanced Features

### AI/ML Specific Optimizations
- **GPU Memory Management**: CUDA error handling and cleanup
- **Model Artifact Handling**: Large file exclusion and management
- **Distributed Testing**: Multi-node training validation
- **Performance Profiling**: GPU utilization and memory analysis

### Enterprise Integration
- **SLSA Compliance**: Supply chain security framework
- **FIPS Validation**: Federal cryptographic standards
- **SOC 2 Alignment**: Security control framework compliance
- **Audit Trail**: Comprehensive activity logging

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyTorch CI Best Practices](https://pytorch.org/docs/stable/notes/ci.html)
- [Docker Multi-Stage Builds](https://docs.docker.com/develop/dev-best-practices/dockerfile_best-practices/)
- [OSSF Scorecard](https://github.com/ossf/scorecard)
- [SLSA Framework](https://slsa.dev/)

---

**Note**: These workflows are designed for production environments and include comprehensive security, testing, and monitoring capabilities optimized for ML/AI development workflows.