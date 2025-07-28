# Claude Instructions for Open MoE Trainer Lab

This file contains instructions and context for Claude Code when working with the Open MoE Trainer Lab repository.

## Repository Overview

The Open MoE Trainer Lab is an advanced training laboratory for Mixture of Experts (MoE) models with distributed computing capabilities. It provides a comprehensive platform for training, evaluating, and serving large-scale MoE models with focus on efficiency, scalability, and reproducibility.

## Project Structure

```
open-moe-trainer-lab/
├── moe_lab/                    # Core library code
│   ├── models/                 # MoE model implementations
│   ├── training/               # Training infrastructure
│   ├── inference/              # Inference serving
│   └── utils/                  # Common utilities
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── benchmarks/             # Performance benchmarks
│   └── e2e/                    # End-to-end tests
├── scripts/                    # Automation scripts
├── docs/                       # Documentation
├── monitoring/                 # Monitoring configurations
├── .github/                    # GitHub workflows and templates
└── examples/                   # Usage examples
```

## Key Technologies

- **Python 3.9+**: Primary programming language
- **PyTorch**: Deep learning framework
- **Docker**: Containerization
- **Kubernetes**: Orchestration (for production)
- **Prometheus/Grafana**: Monitoring and observability
- **PostgreSQL**: Primary database
- **Redis**: Caching and job queuing
- **FastAPI**: API framework
- **pytest**: Testing framework

## Development Workflow

### Setting Up Development Environment

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd open-moe-trainer-lab
   python scripts/setup_development.py
   ```

2. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Run Tests**:
   ```bash
   pytest tests/ -v
   ```

### Code Quality Standards

- **Formatting**: Use `black` for code formatting
- **Import Sorting**: Use `isort` for import organization  
- **Linting**: Use `ruff` and `pylint` for code quality
- **Type Checking**: Use `mypy` for static type checking
- **Security**: Use `bandit` for security linting
- **Testing**: Maintain 85%+ test coverage

### Common Commands

```bash
# Quality checks
python scripts/automation_helper.py quality

# Run specific tests
pytest tests/unit/ -v
pytest tests/integration/ -v --cov=moe_lab

# Clean build artifacts
python scripts/automation_helper.py cleanup

# Update dependencies
python scripts/automation_helper.py deps --upgrade

# Collect metrics
python scripts/collect_metrics.py --summary

# Health check
python scripts/automation_helper.py health
```

## Important Files and Directories

### Configuration Files
- `pyproject.toml`: Python package configuration and dependencies
- `pytest.ini`: Test configuration
- `docker-compose.yml`: Local development environment
- `.pre-commit-config.yaml`: Pre-commit hooks (if present)
- `.env.example`: Environment variable template

### Key Scripts
- `scripts/collect_metrics.py`: Comprehensive metrics collection
- `scripts/automation_helper.py`: Repository automation utilities
- `scripts/setup_development.py`: Development environment setup
- `scripts/build.sh`: Docker build automation
- `scripts/deploy.sh`: Deployment automation

### Documentation
- `docs/guides/`: User and developer guides
- `docs/workflows/examples/`: GitHub Actions workflow templates
- `DEPLOYMENT_GUIDE.md`: Comprehensive deployment documentation
- `REPOSITORY_CONFIGURATION.md`: Repository setup and configuration guide

### Monitoring
- `monitoring/health-check.py`: Health monitoring script
- `monitoring/prometheus.yml`: Prometheus configuration
- `.github/project-metrics.json`: Metrics schema and configuration

## Architecture Patterns

### MoE Model Architecture
- Expert networks with routing mechanisms
- Load balancing across experts
- Sparsity-aware training and inference
- Distributed training support

### Training Infrastructure
- Multi-GPU training support
- Distributed training across nodes
- Checkpointing and resumption
- Performance monitoring and optimization

### Serving Infrastructure
- FastAPI-based REST API
- Model versioning and deployment
- Load balancing and scaling
- Health checks and monitoring

## Testing Strategy

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Service integration testing
- **End-to-End Tests**: Full workflow testing
- **Performance Tests**: Benchmarking and regression testing
- **GPU Tests**: CUDA-specific functionality (marked with `@pytest.mark.gpu`)

### Test Markers
```python
@pytest.mark.slow         # Long-running tests
@pytest.mark.gpu          # Requires GPU
@pytest.mark.distributed  # Requires multiple processes/nodes
@pytest.mark.integration  # Integration tests
@pytest.mark.e2e          # End-to-end tests
```

## Troubleshooting Common Issues

### Development Environment
- **Virtual environment issues**: Use `scripts/setup_development.py --force-venv`
- **Dependency conflicts**: Try `pip install --force-reinstall`
- **Import errors**: Ensure package is installed in development mode: `pip install -e .`

### Testing Issues
- **GPU tests failing**: Ensure CUDA is available and properly configured
- **Slow tests**: Use `pytest -m "not slow"` to skip long-running tests
- **Coverage issues**: Run with `--cov-report=html` for detailed coverage report

### Docker Issues
- **Build failures**: Check Docker daemon is running and buildkit is enabled
- **Permission issues**: Ensure user has proper Docker permissions
- **Resource constraints**: Increase Docker memory/CPU limits

## CI/CD Integration

### GitHub Actions Workflows
The repository uses a comprehensive CI/CD pipeline with the following workflows:

- **CI Pipeline** (`ci.yml`): Code quality, testing, and building
- **Security Scan** (`security.yml`): Security vulnerability scanning
- **Performance Tests** (`performance.yml`): Performance benchmarking
- **Release** (`release.yml`): Automated releases and deployments

### Workflow Templates
Pre-configured workflow templates are available in `docs/workflows/examples/` for:
- Continuous Integration
- Security Scanning
- Performance Testing
- Release Management

### Deployment
- **Development**: Docker Compose locally
- **Staging**: Kubernetes with Helm charts
- **Production**: Cloud deployment with monitoring

## Metrics and Monitoring

### Automated Metrics Collection
The repository includes comprehensive metrics collection via `scripts/collect_metrics.py`:

- **GitHub Metrics**: Stars, forks, PRs, issues, contributors
- **Code Quality**: Test coverage, complexity, lines of code
- **Performance**: Training throughput, inference latency, resource utilization
- **Security**: Vulnerability counts, compliance status

### Monitoring Stack
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboards and visualization
- **Health Checks**: Automated service monitoring
- **Alerting**: Notification systems for critical issues

## Security Considerations

### Security Practices
- No secrets or credentials in code
- Regular dependency vulnerability scanning
- Container security scanning
- Static code analysis for security issues
- Encrypted communications and storage

### Security Tools
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability checking
- **Trivy**: Container vulnerability scanning
- **GitLeaks**: Secret detection in git history

## Performance Optimization

### Training Performance
- Mixed precision training (FP16/BF16)
- Gradient accumulation and checkpointing
- Expert parallelism and load balancing
- Efficient data loading and preprocessing

### Inference Performance
- Model quantization and pruning
- Dynamic batching and caching
- GPU memory optimization
- Load balancing across instances

### Monitoring Performance
- Training throughput (tokens/second)
- Expert utilization distribution
- GPU memory usage and efficiency
- Network communication overhead

## Contributing Guidelines

### Code Contributions
1. Fork the repository and create a feature branch
2. Implement changes with appropriate tests
3. Ensure all quality checks pass
4. Submit pull request with clear description
5. Address review feedback

### Documentation
- Update relevant documentation for any changes
- Include docstrings for all public functions/classes
- Add examples for new features
- Update changelog for significant changes

### Issue Reporting
Use appropriate issue templates for:
- Bug reports
- Feature requests
- Performance issues
- Documentation improvements

## Support and Resources

### Getting Help
- Check existing documentation first
- Search through closed issues
- Use GitHub Discussions for questions
- Create issues with appropriate labels

### External Resources
- PyTorch Documentation: https://pytorch.org/docs/
- Kubernetes Documentation: https://kubernetes.io/docs/
- Prometheus Documentation: https://prometheus.io/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/

## Environment Variables

### Required Environment Variables
```bash
# Database Configuration
DATABASE_URL="postgresql://user:pass@host:5432/dbname"
REDIS_URL="redis://host:6379"

# API Configuration
SECRET_KEY="your-secure-secret-key"
API_HOST="0.0.0.0"
API_PORT="8000"

# Training Configuration
MODEL_CACHE_DIR="/app/models"
DATA_DIR="/app/data"
CHECKPOINT_DIR="/app/checkpoints"
```

### Optional Environment Variables
```bash
# Monitoring
PROMETHEUS_URL="http://localhost:9090"
GRAFANA_URL="http://localhost:3000"
METRICS_EXPORT_INTERVAL="30"

# Development
DEBUG="false"
LOG_LEVEL="INFO"
PROFILING_ENABLED="false"

# GPU Configuration
CUDA_VISIBLE_DEVICES="0,1,2,3"
NCCL_DEBUG="WARN"
```

## Recent Updates and Changes

This repository has been enhanced with a comprehensive SDLC automation system following Terragon's checkpoint methodology:

### Major Components Added
1. **Project Foundation**: Architecture docs, community files, project charter
2. **Development Environment**: DevContainer, linting configs, tooling setup
3. **Testing Infrastructure**: Comprehensive test framework with coverage requirements
4. **Build & Containerization**: Multi-stage Dockerfiles, build automation
5. **Monitoring & Observability**: Health checks, metrics collection, Prometheus/Grafana setup
6. **Workflow Documentation**: CI/CD templates and comprehensive guides
7. **Metrics & Automation**: Automated metrics collection and repository maintenance tools
8. **Integration & Configuration**: Repository configuration and deployment guides

### New Automation Scripts
- **Metrics Collection**: `scripts/collect_metrics.py` - Comprehensive metrics from multiple sources
- **Automation Helper**: `scripts/automation_helper.py` - Repository maintenance and quality checks
- **Development Setup**: `scripts/setup_development.py` - Automated environment configuration

The repository now follows enterprise-grade practices with full observability, comprehensive testing, and automated quality assurance.

---

**Note**: This file should be kept up-to-date as the repository evolves. When making significant changes, please update the relevant sections to maintain accuracy.