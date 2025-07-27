# GitHub Workflows Setup Guide

⚠️ **Important Note**: As Terry, I cannot directly create or modify GitHub workflows/actions. This guide helps you set up the CI/CD pipelines manually.

## 📋 Required Setup Steps

### 1. Copy Workflow Templates

The following template files have been created in `.github/workflows/`:

- `ci.yml.template` → Copy to `ci.yml`
- `release.yml.template` → Copy to `release.yml`

**Command to copy templates:**
```bash
cp .github/workflows/ci.yml.template .github/workflows/ci.yml
cp .github/workflows/release.yml.template .github/workflows/release.yml
```

### 2. Configure GitHub Secrets

Add the following secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

#### PyPI Publishing
- `PYPI_API_TOKEN` - PyPI API token for publishing packages
- `TEST_PYPI_API_TOKEN` - Test PyPI API token for pre-releases

#### Container Registry
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

#### Notifications
- `SLACK_WEBHOOK` - Slack webhook URL for notifications (optional)

#### Additional Secrets (if needed)
- `CODECOV_TOKEN` - Codecov token for coverage reporting
- Custom deployment secrets based on your infrastructure

### 3. Configure Self-Hosted Runners (Optional)

For GPU and distributed testing, you may need self-hosted runners:

#### GPU Runner Requirements
```yaml
labels: [self-hosted, gpu]
requirements:
  - NVIDIA GPU with CUDA support
  - Docker with NVIDIA runtime
  - Minimum 16GB RAM
  - PyTorch with GPU support
```

#### Multi-GPU Runner Requirements
```yaml
labels: [self-hosted, multi-gpu]
requirements:
  - Multiple NVIDIA GPUs
  - NCCL support for distributed training
  - Docker with NVIDIA runtime
  - Minimum 32GB RAM
```

### 4. Repository Settings

#### Branch Protection Rules
Configure branch protection for `main` and `develop` branches:

```yaml
main branch protection:
  - Require pull request reviews: 2
  - Require status checks to pass:
    - code-quality
    - unit-tests
    - integration-tests
    - docker-build
  - Require branches to be up to date
  - Restrict pushes to matching branches
  - Allow force pushes: false
  - Allow deletions: false

develop branch protection:
  - Require pull request reviews: 1
  - Require status checks to pass:
    - code-quality
    - unit-tests
  - Allow force pushes: false
```

#### Environments
Create the following environments in repository settings:

1. **staging**
   - Deployment to staging infrastructure
   - Require reviewers: 1
   - Deployment protection rules as needed

2. **production**
   - Deployment to production infrastructure
   - Require reviewers: 2
   - Wait timer: 5 minutes
   - Deployment protection rules

3. **pypi**
   - For PyPI package publishing
   - Require reviewers: 1
   - Environment secrets: `PYPI_API_TOKEN`, `TEST_PYPI_API_TOKEN`

### 5. Additional Workflow Files

You may want to create additional workflows:

#### Dependency Updates (`dependabot.yml`)
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "python"
  
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies" 
      - "docker"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "ci"
```

#### Issue Labeler (`labeler.yml`)
```yaml
# .github/labeler.yml
"area:core":
  - moe_lab/core/**/*
  - moe_lab/models/**/*

"area:training":
  - moe_lab/training/**/*
  - moe_lab/distributed/**/*

"area:inference":
  - moe_lab/inference/**/*
  - moe_lab/serving/**/*

"area:docs":
  - docs/**/*
  - "*.md"

"area:tests":
  - tests/**/*

"area:ci":
  - .github/**/*
  - Dockerfile*
  - docker-compose*.yml
```

### 6. Workflow Customization

#### Modify for Your Infrastructure

1. **Container Registry**: Update registry URLs if not using GitHub Container Registry
2. **GPU Requirements**: Adjust GPU tests based on available hardware
3. **Notification Channels**: Configure Slack/Teams/Email notifications
4. **Deployment Targets**: Customize deployment steps for your infrastructure

#### Performance Tuning

1. **Parallel Jobs**: Adjust `strategy.matrix` for faster builds
2. **Caching**: Optimize pip/Docker layer caching
3. **Conditional Execution**: Use `if` conditions to skip unnecessary jobs

### 7. Security Considerations

#### Repository Security
- Enable vulnerability alerts
- Configure security advisories
- Set up code scanning (CodeQL)
- Enable secret scanning

#### Workflow Security
- Use specific action versions (not `@main`)
- Pin Docker image tags
- Validate inputs in workflow_dispatch
- Use least-privilege principles for tokens

### 8. Monitoring and Observability

#### Workflow Monitoring
- Set up workflow failure notifications
- Monitor workflow execution times
- Track resource usage for self-hosted runners

#### Quality Metrics
- Code coverage trending
- Performance benchmark tracking
- Security vulnerability tracking

## 🚀 Quick Start Commands

After copying the templates:

```bash
# 1. Copy workflow templates
cp .github/workflows/*.template .github/workflows/
rename 's/\.template$//' .github/workflows/*.template

# 2. Validate workflow syntax
gh workflow list  # Requires GitHub CLI

# 3. Test workflows locally (requires act)
act -j code-quality  # Test code quality job
act -j unit-tests    # Test unit tests job

# 4. Enable workflows
git add .github/workflows/
git commit -m "feat: add CI/CD workflows"
git push origin main
```

## 📞 Support

If you need help setting up the workflows:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow logs for debugging
3. Test workflows on feature branches first
4. Consider starting with a subset of jobs and expanding gradually

## ✅ Verification Checklist

After setup, verify:

- [ ] Workflows appear in Actions tab
- [ ] All required secrets are configured
- [ ] Branch protection rules are active
- [ ] Self-hosted runners are connected (if applicable)
- [ ] Test PR triggers appropriate workflows
- [ ] Release process works end-to-end
- [ ] Notifications are delivered correctly

---

**Remember**: Start with basic workflows and gradually add complexity as your team becomes comfortable with the CI/CD pipeline.