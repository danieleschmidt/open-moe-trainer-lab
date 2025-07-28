# Development Guide

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker and Docker Compose
- Git
- CUDA 11.8+ (for GPU support)

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/open-moe-trainer-lab.git
   cd open-moe-trainer-lab
   ```

2. **Setup development environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -e ".[dev,gpu,distributed]"
   
   # Setup pre-commit hooks
   pre-commit install
   ```

3. **Start development services**:
   ```bash
   # Start monitoring stack
   docker-compose -f monitoring/docker-compose.monitoring.yml up -d
   
   # Start development database
   docker-compose up -d postgres redis
   ```

4. **Verify installation**:
   ```bash
   # Run tests
   python -m pytest tests/unit/ -v
   
   # Start development server
   python -m moe_lab.dashboard --dev
   ```

### Development with Docker

1. **Build development image**:
   ```bash
   docker build --target development -t moe-lab:dev .
   ```

2. **Run development container**:
   ```bash
   docker run -it --rm \
     --gpus all \
     -v $(pwd):/workspace \
     -p 8080:8080 \
     -p 8888:8888 \
     moe-lab:dev
   ```

3. **Use VS Code Dev Containers**:
   - Install the "Remote - Containers" extension
   - Open the project in VS Code
   - Click "Reopen in Container" when prompted

## Project Structure

```
open-moe-trainer-lab/
├── moe_lab/                 # Main package
│   ├── models/              # Model definitions
│   ├── training/            # Training logic
│   ├── inference/           # Inference engine
│   ├── routing/             # Expert routing
│   ├── analytics/           # Analytics and visualization
│   ├── dashboard/           # Web dashboard
│   ├── cli/                 # Command-line interface
│   └── utils/               # Utilities
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── e2e/                 # End-to-end tests
│   ├── performance/         # Performance tests
│   └── fixtures/            # Test fixtures
├── docs/                    # Documentation
│   ├── guides/              # User guides
│   ├── api/                 # API documentation
│   ├── runbooks/            # Operational runbooks
│   └── adr/                 # Architecture decisions
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
├── monitoring/              # Monitoring configuration
└── security/                # Security tools and configs
```

## Development Workflow

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **pylint**: Code linting
- **mypy**: Type checking
- **ruff**: Fast Python linter

```bash
# Format code
npm run format

# Check code quality
npm run lint
npm run typecheck

# Run all checks
pre-commit run --all-files
```

### Testing

#### Running Tests

```bash
# Run all tests
npm test

# Run specific test categories
npm run test:unit
npm run test:integration
npm run test:e2e

# Run with coverage
npm run test:coverage

# Run performance tests
npm run test:performance
```

#### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark performance metrics

#### Writing Tests

```python
# Unit test example
import pytest
from moe_lab.routing import TopKRouter

def test_top_k_router():
    router = TopKRouter(num_experts=4, top_k=2)
    logits = torch.randn(8, 32, 4)  # batch, seq_len, experts
    
    indices, weights = router(logits)
    
    assert indices.shape == (8, 32, 2)
    assert weights.shape == (8, 32, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8, 32))

# Integration test example
@pytest.mark.integration
def test_training_pipeline(mock_dataset, temp_dir):
    trainer = MoETrainer(
        model_config=small_model_config,
        training_config=fast_training_config,
        output_dir=temp_dir
    )
    
    result = trainer.train(mock_dataset)
    
    assert result.final_loss < result.initial_loss
    assert (temp_dir / "checkpoints").exists()
```

### Adding New Features

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement the feature**:
   - Follow the existing code patterns
   - Add comprehensive tests
   - Update documentation
   - Add type hints

3. **Test your changes**:
   ```bash
   npm test
   npm run lint
   npm run typecheck
   ```

4. **Create a pull request**:
   - Follow the PR template
   - Ensure all CI checks pass
   - Request review from maintainers

### Working with Models

#### Creating Custom Expert Types

```python
from moe_lab.experts import Expert
import torch.nn as nn

class CustomExpert(Expert):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

#### Implementing Custom Routers

```python
from moe_lab.routing import Router
import torch
import torch.nn as nn

class CustomRouter(Router):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)
        
    def forward(self, hidden_states: torch.Tensor) -> tuple:
        # Compute routing logits
        logits = self.gate(hidden_states)
        
        # Apply custom routing logic
        probabilities = torch.softmax(logits, dim=-1)
        
        # Return expert indices and weights
        return self.select_experts(probabilities)
```

### Configuration Management

#### Training Configuration

```yaml
# configs/training.yaml
model:
  hidden_size: 768
  num_experts: 8
  experts_per_token: 2
  num_layers: 12

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 1000
  
load_balancing:
  loss_coef: 0.01
  router_z_loss_coef: 0.001
```

#### Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
vim .env
```

### Dashboard Development

The dashboard is built with React and provides real-time monitoring:

```bash
# Start dashboard in development mode
cd dashboard
npm install
npm start

# Build for production
npm run build
```

#### Adding Dashboard Components

```javascript
// dashboard/src/components/ExpertUtilization.jsx
import React from 'react';
import { Line } from 'react-chartjs-2';

const ExpertUtilization = ({ data }) => {
  const chartData = {
    labels: data.timestamps,
    datasets: data.experts.map((expert, index) => ({
      label: `Expert ${index}`,
      data: expert.utilization,
      borderColor: `hsl(${index * 360 / data.experts.length}, 70%, 50%)`,
      fill: false,
    }))
  };
  
  return (
    <div className="expert-utilization">
      <h3>Expert Utilization</h3>
      <Line data={chartData} />
    </div>
  );
};

export default ExpertUtilization;
```

### Distributed Training

#### Single Node, Multiple GPUs

```bash
# Launch training with torchrun
torchrun --nproc_per_node=4 \
  -m moe_lab.train \
  --config configs/distributed.yaml
```

#### Multiple Nodes

```bash
# Node 0 (master)
torchrun --nnodes=2 --node_rank=0 \
  --master_addr=192.168.1.100 --master_port=29500 \
  --nproc_per_node=4 \
  -m moe_lab.train --config configs/distributed.yaml

# Node 1
torchrun --nnodes=2 --node_rank=1 \
  --master_addr=192.168.1.100 --master_port=29500 \
  --nproc_per_node=4 \
  -m moe_lab.train --config configs/distributed.yaml
```

### Debugging

#### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   trainer.config.batch_size = 16
   
   # Enable gradient checkpointing
   trainer.config.gradient_checkpointing = True
   
   # Clear GPU cache
   torch.cuda.empty_cache()
   ```

2. **Expert Load Imbalance**:
   ```python
   # Increase load balancing coefficient
   trainer.config.load_balance_loss_coef = 0.02
   
   # Add router z-loss
   trainer.config.router_z_loss_coef = 0.001
   ```

3. **Slow Training**:
   ```python
   # Enable mixed precision
   trainer.config.mixed_precision = True
   
   # Increase expert parallelism
   trainer.config.expert_parallel_size = 4
   ```

#### Profiling

```bash
# Profile training performance
python -m moe_lab.profiler \
  --config configs/training.yaml \
  --profile-memory \
  --profile-compute

# Generate profiling report
python -m moe_lab.profiler \
  --report profiling_results/ \
  --format html
```

### Monitoring and Logging

#### Accessing Dashboards

- **Training Dashboard**: http://localhost:8080
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jupyter Lab**: http://localhost:8888

#### Log Locations

```bash
# Application logs
tail -f logs/training.log

# Container logs
docker logs -f moe-trainer

# Kubernetes logs
kubectl logs -f deployment/moe-trainer
```

### Performance Optimization

#### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use 16-bit precision
trainer = MoETrainer(
    model=model,
    args=TrainingArguments(
        fp16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True
    )
)
```

#### Compute Optimization

```python
# Compile model for faster inference
model = torch.compile(model, mode="reduce-overhead")

# Enable expert caching
inference_engine = MoEInferenceEngine(
    model=model,
    expert_cache_size_gb=8,
    cache_policy="lru"
)
```

## Contributing Guidelines

### Code Review Process

1. All changes must be reviewed by at least one maintainer
2. CI checks must pass
3. Test coverage must not decrease
4. Documentation must be updated for new features

### Commit Message Format

We use [Conventional Commits](https://conventionalcommits.org/):

```
type(scope): description

feat(routing): add expert choice routing algorithm
fix(training): resolve memory leak in expert computation
docs(api): update routing API documentation
test(integration): add distributed training tests
```

### Release Process

1. Features are merged to `main` branch
2. Releases are created from `main` using semantic versioning
3. Release notes are generated automatically
4. Packages are published to PyPI automatically

## Getting Help

- **Documentation**: Check this guide and API docs
- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our community server (link in README)
- **Email**: Contact the maintainers directly

## Resources

- [Architecture Overview](ARCHITECTURE.md)
- [API Documentation](docs/api/)
- [Deployment Guide](docs/guides/deployment.md)
- [Performance Tuning](docs/guides/performance.md)
- [Troubleshooting](docs/guides/troubleshooting.md)
