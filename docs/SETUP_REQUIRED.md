# Manual Setup Required

## Overview
Due to GitHub App limitations, certain repository configurations require manual setup by maintainers.

## Required Manual Actions

### 1. GitHub Actions Workflows
```bash
# Copy templates to .github/workflows/
mkdir -p .github/workflows
cp docs/workflows/examples/*.yml .github/workflows/
```

### 2. Repository Settings
- **Actions**: Enable GitHub Actions in repository settings
- **Security**: Enable vulnerability alerts and security advisories
- **Pages**: Configure GitHub Pages for documentation (optional)

### 3. Branch Protection Rules
Configure protection for `main` branch:
- ✅ Require pull request reviews
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Include administrators

### 4. Repository Secrets
Add these secrets in Settings → Secrets and variables → Actions:
- `DOCKER_USERNAME` - Docker registry username
- `DOCKER_PASSWORD` - Docker registry password/token  
- `PYPI_TOKEN` - PyPI publishing token

### 5. Issue Templates
```bash
# Create GitHub issue templates directory
mkdir -p .github/ISSUE_TEMPLATE
# Copy templates from docs/templates/ if available
```

## Verification Steps
1. Create test pull request to verify CI workflows
2. Check branch protection prevents direct pushes to main
3. Verify secrets are accessible in workflow runs
4. Test issue template functionality

## External Resources
- [GitHub Repository Settings Guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)
- [Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository)
- [GitHub Actions Setup Guide](https://docs.github.com/en/actions/quickstart)