# GitHub Actions Workflows Setup Required

Due to GitHub App permissions, the workflow files cannot be committed directly. Please manually copy the workflow files from the temporary location to `.github/workflows/`.

## Required Actions

### 1. Copy Workflow Files

Copy these files from the repository root to `.github/workflows/`:

```bash
# Navigate to repository root
cd /path/to/open-moe-trainer-lab

# Create workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy the workflow files (they are currently in .github/workflows/ but not committed)
# You'll need to manually create these files with the content provided below
```

### 2. Workflow Files to Create

#### `.github/workflows/ci.yml`
- **Purpose**: Comprehensive CI pipeline with matrix testing
- **Features**: Multi-platform testing, GPU support, performance benchmarks, security scanning
- **Triggers**: Push to main/develop, PRs, weekly schedule

#### `.github/workflows/release.yml`  
- **Purpose**: Automated release management
- **Features**: Version validation, Docker builds, PyPI publishing, SBOM generation
- **Triggers**: Git tags (v*), manual workflow dispatch

#### `.github/workflows/security.yml`
- **Purpose**: Multi-layer security scanning  
- **Features**: Secret detection, dependency scanning, CodeQL, container security, OSSF Scorecard
- **Triggers**: Push, PRs, weekly security scans

### 3. Repository Secrets Configuration

Configure these secrets in GitHub repository settings:

```
# Required for releases
PYPI_API_TOKEN           # PyPI publishing
TEST_PYPI_API_TOKEN      # Test PyPI publishing  

# Optional for enhanced security scanning  
SEMGREP_APP_TOKEN        # Semgrep security analysis
```

### 4. GitHub Actions Permissions

Enable the following in repository settings → Actions → General:
- ✅ **Allow GitHub Actions to create and approve pull requests**
- ✅ **Allow GitHub Actions to approve pull requests**  
- ✅ **Read and write permissions**
- ✅ **Allow GitHub Actions to write to security events**

### 5. Branch Protection Rules

Configure branch protection for `main`:
- ✅ **Require status checks to pass before merging**
- ✅ **Require up-to-date branches before merging**
- ✅ **Include administrators**
- ✅ **Required status checks**:
  - `Pre-commit Checks`
  - `Test Suite`
  - `Security Scanning` 
  - `Quality Gate`

### 6. Workflow File Contents

The workflow files are comprehensive and include:

- **Matrix testing** across Python 3.9-3.12 and PyTorch versions
- **GPU testing** with CUDA support (requires self-hosted runners)
- **Security scanning** with multiple tools (Bandit, Safety, CodeQL, Trivy)
- **Performance benchmarking** with trend analysis
- **Automated releases** with semantic versioning
- **Docker multi-arch builds** for CPU and GPU variants
- **PyPI publishing** with test environment validation

### 7. Self-Hosted Runner Setup (Optional)

For GPU testing, configure self-hosted runners:

```yaml
runs-on: [self-hosted, gpu]
```

If no GPU runners available, modify the workflow to use:

```yaml  
runs-on: ubuntu-latest
# Remove GPU-specific tests or mark as conditional
```

### 8. Verification Steps

After copying workflows:

1. **Check workflow syntax**: GitHub will validate YAML syntax automatically
2. **Test with PR**: Create a test PR to verify workflows execute  
3. **Monitor Actions tab**: Check for any configuration issues
4. **Review security alerts**: Verify security scanning is working

### 9. Expected Workflow Behavior

- **CI Pipeline**: Runs on every push/PR with comprehensive testing
- **Security Scanning**: Weekly scans + on-demand for PRs  
- **Release Process**: Triggered by git tags or manual dispatch
- **Quality Gates**: Must pass before merge permissions

## Benefits After Setup

✅ **Automated Quality Assurance**: Comprehensive testing matrix  
✅ **Security Compliance**: Multi-layer vulnerability detection  
✅ **Zero-Touch Releases**: Automated version management and publishing  
✅ **Performance Monitoring**: Regression detection and benchmarking  
✅ **Developer Experience**: Pre-commit hooks and quality feedback

---

**Note**: The workflow files contain production-ready configurations tailored for the Open MoE Trainer Lab's ML/AI development workflow with PyTorch, CUDA, and distributed training support.