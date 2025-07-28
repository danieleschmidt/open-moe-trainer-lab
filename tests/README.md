# Testing Guide - Open MoE Trainer Lab

This directory contains the comprehensive test suite for the Open MoE Trainer Lab project. Our testing strategy ensures reliability, performance, and correctness across all components.

## Test Structure

```
tests/
├── unit/                    # Fast, isolated unit tests
├── integration/             # Integration tests for component interaction
├── e2e/                    # End-to-end workflow tests
├── distributed/            # Multi-GPU and multi-node tests
├── benchmarks/             # Performance benchmarks
├── performance/            # Performance regression tests
├── fixtures/               # Shared test data and utilities
├── conftest.py            # Pytest configuration and fixtures
└── README.md              # This file
```

## Test Categories

### 🧪 Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Functions, classes, and modules
- **Speed**: Fast (< 1 second per test)
- **Coverage**: Aim for >95% code coverage

**Example**:
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific unit test
pytest tests/unit/test_routing.py::TestTopKRouter::test_routing_logic -v
```

### 🔗 Integration Tests (`integration/`)
- **Purpose**: Test component interactions
- **Scope**: Multiple modules working together
- **Speed**: Medium (1-10 seconds per test)
- **Coverage**: Critical integration paths

**Example**:
```bash
# Run integration tests
pytest tests/integration/ -v -m "not slow"

# Run training pipeline integration
pytest tests/integration/test_training_pipeline.py -v
```

### 🎯 End-to-End Tests (`e2e/`)
- **Purpose**: Test complete workflows
- **Scope**: Full training/inference pipelines
- **Speed**: Slow (10+ seconds per test)
- **Coverage**: User-facing workflows

**Example**:
```bash
# Run E2E tests (requires GPU)
pytest tests/e2e/ -v -m "gpu"

# Run full training pipeline test
pytest tests/e2e/test_full_training_pipeline.py -v --capture=no
```

### 🌐 Distributed Tests (`distributed/`)
- **Purpose**: Test multi-GPU and multi-node training
- **Scope**: Distributed training workflows
- **Speed**: Slow (requires multiple GPUs)
- **Coverage**: Parallel training scenarios

**Example**:
```bash
# Run distributed tests (requires multiple GPUs)
pytest tests/distributed/ -v -m "distributed"

# Run with specific number of processes
pytest tests/distributed/ -v -n 4
```

### 📊 Benchmarks (`benchmarks/`)
- **Purpose**: Performance measurement and optimization
- **Scope**: Critical performance paths
- **Speed**: Variable (includes warm-up)
- **Coverage**: Performance-critical components

**Example**:
```bash
# Run all benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# Run with performance comparison
pytest tests/benchmarks/ -v --benchmark-compare=0001

# Generate performance report
pytest tests/benchmarks/ --benchmark-json=performance.json
```

### ⚡ Performance Tests (`performance/`)
- **Purpose**: Performance regression detection
- **Scope**: Training and inference speed
- **Speed**: Medium to slow
- **Coverage**: Performance baselines

## Test Configuration

### Pytest Configuration
The main pytest configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "--strict-markers",
    "--verbose",
    "--tb=short",
    "--cov=moe_lab",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov"
]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "gpu: marks tests as requiring GPU",
    "distributed: marks tests as requiring distributed setup",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests"
]
```

### Test Markers
Use markers to categorize and selectively run tests:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (>10 seconds)
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.distributed` - Distributed training tests
- `@pytest.mark.benchmark` - Performance benchmarks

### Fixtures and Utilities
Common test utilities are provided in `conftest.py`:

- `torch_device` - Best available device (GPU/CPU)
- `temp_dir` - Temporary directory for test files
- `mock_config` - Mock configuration for testing
- `sample_batch` - Sample training batch
- `mock_tokenizer` - Mock tokenizer
- `checkpoint_dir` - Temporary checkpoint directory

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=moe_lab --cov-report=html

# Run specific test categories
pytest -m "unit"                    # Unit tests only
pytest -m "not slow"                # Skip slow tests
pytest -m "gpu and not distributed" # GPU tests, not distributed
```

### Advanced Test Execution
```bash
# Parallel execution
pytest -n auto                      # Auto-detect CPU cores
pytest -n 4                         # Use 4 workers

# Output control
pytest -v                           # Verbose output
pytest -s                           # Don't capture output
pytest --tb=long                    # Detailed tracebacks

# Debugging
pytest --pdb                        # Drop into debugger on failure
pytest --lf                         # Run last failed tests only
pytest --ff                         # Run failed tests first
```

### Performance Testing
```bash
# Run benchmarks
pytest tests/benchmarks/ --benchmark-only

# Compare with baseline
pytest tests/benchmarks/ --benchmark-compare=baseline.json

# Memory profiling
pytest tests/ --profile --profile-svg

# Generate performance report
pytest tests/performance/ --benchmark-json=perf-report.json
```

## Writing Tests

### Test Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use descriptive names: `test_router_selects_top_k_experts()`

### Test Structure
Follow the Arrange-Act-Assert pattern:

```python
def test_moe_model_forward_pass(mock_config, sample_batch, torch_device):
    # Arrange
    model = MoEModel(**mock_config["model"]).to(torch_device)
    
    # Act
    outputs = model(**sample_batch)
    
    # Assert
    assert outputs.logits.shape == (4, 64, 1000)  # batch, seq, vocab
    assert outputs.router_info is not None
    assert len(outputs.router_info.expert_indices) == mock_config["model"]["num_layers"]
```

### Testing Best Practices

1. **Deterministic Tests**: Use fixed seeds for reproducibility
2. **Isolated Tests**: Each test should be independent
3. **Fast Unit Tests**: Keep unit tests under 1 second
4. **Mock External Dependencies**: Use mocks for APIs, file I/O
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Descriptive Assertions**: Use meaningful assert messages

### Example Test Class
```python
class TestMoERouter:
    """Test suite for MoE routing algorithms."""
    
    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        self.config = mock_config["model"]
        self.router = TopKRouter(
            hidden_size=self.config["hidden_size"],
            num_experts=self.config["num_experts"],
            top_k=self.config["experts_per_token"]
        )
    
    def test_router_initialization(self):
        """Test router is properly initialized."""
        assert self.router.num_experts == self.config["num_experts"]
        assert self.router.top_k == self.config["experts_per_token"]
    
    @pytest.mark.gpu
    def test_router_forward_gpu(self, sample_batch, torch_device):
        """Test router forward pass on GPU."""
        self.router = self.router.to(torch_device)
        hidden_states = torch.randn(4, 64, 512, device=torch_device)
        
        expert_indices, expert_weights, routing_info = self.router(hidden_states)
        
        assert expert_indices.device == torch_device
        assert expert_weights.device == torch_device
    
    @pytest.mark.slow
    def test_router_load_balancing(self):
        """Test router load balancing over many iterations."""
        # Test that router balances load over many forward passes
        pass
```

## Continuous Integration

### Pre-commit Hooks
Tests are automatically run via pre-commit hooks:
- Fast unit tests on every commit
- Full test suite on push
- Coverage checks before merge

### GitHub Actions
Our CI pipeline runs:
1. **Unit Tests**: On every PR
2. **Integration Tests**: On every PR
3. **E2E Tests**: On main branch
4. **Performance Tests**: Nightly
5. **Distributed Tests**: Weekly

### Test Reports
Test results are published to:
- GitHub PR comments (coverage, failures)
- Test artifacts (HTML reports, logs)
- Performance dashboard (benchmark results)

## Test Data and Fixtures

### Sample Data
Located in `tests/fixtures/`:
- `sample_data.py` - Mock datasets and tokenizers
- Model checkpoints for testing
- Reference outputs for regression testing

### Environment Setup
Tests automatically configure:
- Deterministic random seeds
- Appropriate device selection
- Environment variable mocking
- Temporary directories

## Debugging Failed Tests

### Common Issues
1. **CUDA OOM**: Reduce batch sizes in test configs
2. **Timing Issues**: Use fixtures with proper setup/teardown
3. **Flaky Tests**: Check for race conditions in distributed tests
4. **Import Errors**: Ensure PYTHONPATH includes project root

### Debug Commands
```bash
# Run single test with debugging
pytest tests/unit/test_routing.py::test_specific_function -vvv -s --pdb

# Check test discovery
pytest --collect-only

# Run with warnings
pytest -W error::DeprecationWarning

# Memory profiling
pytest --profile-svg
```

## Contributing Tests

When contributing new features:

1. **Write tests first** (TDD approach)
2. **Maintain >90% coverage** for new code
3. **Add integration tests** for new workflows
4. **Include benchmarks** for performance-critical code
5. **Update documentation** for new test categories

### Test Review Checklist
- [ ] Tests are deterministic and reproducible
- [ ] Edge cases and error conditions covered
- [ ] Performance impact considered
- [ ] Appropriate test markers used
- [ ] Documentation updated if needed

---

For questions about testing, see our [Contributing Guide](../CONTRIBUTING.md) or open an issue on GitHub.