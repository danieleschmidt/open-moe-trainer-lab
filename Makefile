# Makefile for Open MoE Trainer Lab
# Provides convenient commands for development and deployment

.PHONY: help build dev test lint format clean docker-* docs install

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := open-moe-trainer-lab
IMAGE_NAME := open-moe-trainer
REGISTRY := your-registry.com

# Help target
help: ## Show this help message
	@echo "Open MoE Trainer Lab - Development Commands"
	@echo "=========================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Development Environment
# =============================================================================

install: ## Install package in development mode
	$(PIP) install -e .[dev]

install-all: ## Install package with all optional dependencies
	$(PIP) install -e .[all]

setup-dev: ## Set up development environment
	$(PIP) install -e .[dev]
	pre-commit install
	@echo "Development environment set up successfully!"

setup-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# =============================================================================
# Code Quality
# =============================================================================

format: ## Format code with black and isort
	black moe_lab/ tests/ examples/
	isort moe_lab/ tests/ examples/

format-check: ## Check code formatting
	black --check moe_lab/ tests/ examples/
	isort --check-only moe_lab/ tests/ examples/

lint: ## Run linting with pylint
	pylint moe_lab/ tests/

lint-fix: ## Auto-fix linting issues where possible
	autopep8 --in-place --recursive moe_lab/ tests/
	isort moe_lab/ tests/ examples/
	black moe_lab/ tests/ examples/

typecheck: ## Run type checking with mypy
	mypy moe_lab/

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

security-scan: ## Run security scanning
	safety check
	bandit -r moe_lab/

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-distributed: ## Run distributed tests
	pytest tests/distributed/ -v -m distributed

test-slow: ## Run slow tests (benchmarks)
	pytest tests/ -v -m slow

test-coverage: ## Run tests with coverage
	pytest tests/ --cov=moe_lab --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	pytest tests/ -v --tb=short -x --looponfail

# =============================================================================
# Docker Commands
# =============================================================================

docker-build: ## Build all Docker images
	$(DOCKER_COMPOSE) build

docker-build-dev: ## Build development image
	$(DOCKER) build --target development -t $(IMAGE_NAME):dev .

docker-build-prod: ## Build production image
	$(DOCKER) build --target production -t $(IMAGE_NAME):latest .

docker-build-train: ## Build training image
	$(DOCKER) build --target training -t $(IMAGE_NAME):train .

docker-build-serve: ## Build inference image
	$(DOCKER) build --target inference -t $(IMAGE_NAME):serve .

docker-dev: ## Start development environment
	$(DOCKER_COMPOSE) up -d dev
	@echo "Development environment started. Access:"
	@echo "  Jupyter Lab: http://localhost:8888"
	@echo "  Dashboard:   http://localhost:8080"
	@echo "  TensorBoard: http://localhost:6006"

docker-train: ## Start training service
	$(DOCKER_COMPOSE) up train

docker-serve: ## Start inference service
	$(DOCKER_COMPOSE) up -d serve
	@echo "Inference service started at http://localhost:8000"

docker-distributed: ## Start distributed training
	$(DOCKER_COMPOSE) up coordinator worker

docker-full: ## Start full stack (all services)
	$(DOCKER_COMPOSE) up -d
	@echo "Full stack started. Access:"
	@echo "  Dashboard:    http://localhost:8080"
	@echo "  Jupyter Lab:  http://localhost:8888"
	@echo "  Grafana:      http://localhost:3001"
	@echo "  Prometheus:   http://localhost:9090"
	@echo "  MinIO:        http://localhost:9001"

docker-logs: ## View logs from all services
	$(DOCKER_COMPOSE) logs -f

docker-logs-train: ## View training logs
	$(DOCKER_COMPOSE) logs -f train

docker-down: ## Stop all services
	$(DOCKER_COMPOSE) down

docker-clean: ## Clean up containers and volumes
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(DOCKER) system prune -f

docker-shell: ## Open shell in development container
	$(DOCKER_COMPOSE) exec dev bash

docker-test: ## Run tests in Docker
	$(DOCKER_COMPOSE) run --rm test

docker-benchmark: ## Run benchmarks in Docker
	$(DOCKER_COMPOSE) run --rm benchmark

# =============================================================================
# Training and Evaluation
# =============================================================================

train: ## Train a model locally
	$(PYTHON) -m moe_lab.train --config configs/training.yaml

train-debug: ## Train with debug settings
	$(PYTHON) -m moe_lab.train --config configs/debug.yaml --debug

train-distributed: ## Train with distributed setup
	torchrun --nproc_per_node=2 -m moe_lab.train --config configs/distributed.yaml

evaluate: ## Evaluate a trained model
	$(PYTHON) -m moe_lab.evaluate --model-path checkpoints/latest --config configs/eval.yaml

benchmark: ## Run performance benchmarks
	$(PYTHON) -m moe_lab.benchmark --config benchmarks/config.yaml

dashboard: ## Start training dashboard
	$(PYTHON) -m moe_lab.dashboard --port 8080

# =============================================================================
# Data and Model Management
# =============================================================================

download-data: ## Download sample datasets
	$(PYTHON) scripts/download_data.py --dataset wikitext-103

prepare-data: ## Prepare training data
	$(PYTHON) scripts/prepare_data.py --input data/raw --output data/processed

export-model: ## Export model for deployment
	$(PYTHON) -m moe_lab.export --model-path checkpoints/latest --format onnx --output models/

convert-checkpoint: ## Convert checkpoint format
	$(PYTHON) scripts/convert_checkpoint.py --input checkpoints/pytorch_model.bin --output checkpoints/converted/

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	cd docs && $(PYTHON) -m sphinx -b html source build

docs-serve: ## Serve documentation locally
	cd docs/build && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && rm -rf build/

docs-auto: ## Auto-build documentation on changes
	cd docs && sphinx-autobuild source build --host 0.0.0.0 --port 8000

# =============================================================================
# Release and Deployment
# =============================================================================

build-package: ## Build Python package
	$(PYTHON) -m build

upload-test: ## Upload to test PyPI
	twine upload --repository testpypi dist/*

upload-prod: ## Upload to production PyPI
	twine upload dist/*

tag-release: ## Tag a new release
	@read -p "Enter version (e.g., v0.1.0): " version; \
	git tag -a $$version -m "Release $$version"; \
	git push origin $$version

docker-tag: ## Tag Docker images for release
	@read -p "Enter version (e.g., v0.1.0): " version; \
	$(DOCKER) tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):$$version; \
	$(DOCKER) tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):latest

docker-push: ## Push Docker images to registry
	@read -p "Enter version (e.g., v0.1.0): " version; \
	$(DOCKER) push $(REGISTRY)/$(IMAGE_NAME):$$version; \
	$(DOCKER) push $(REGISTRY)/$(IMAGE_NAME):latest

# =============================================================================
# Monitoring and Maintenance
# =============================================================================

logs: ## View application logs
	tail -f logs/training.log

monitor: ## Start monitoring dashboard
	$(DOCKER_COMPOSE) up -d monitoring grafana
	@echo "Monitoring started:"
	@echo "  Grafana:    http://localhost:3001 (admin/moelab)"
	@echo "  Prometheus: http://localhost:9090"

health-check: ## Check system health
	$(PYTHON) scripts/health_check.py

update-deps: ## Update Python dependencies
	pip-review --auto
	pip-audit

check-security: ## Run security checks
	safety check
	bandit -r moe_lab/

profile: ## Profile model performance
	$(PYTHON) -m moe_lab.profiler --model-config configs/model.yaml

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean up temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*.pyd' -delete
	find . -name '.coverage' -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

clean-data: ## Clean up data files (use with caution)
	@read -p "This will delete all data files. Continue? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf data/processed/*; \
		rm -rf checkpoints/*; \
		rm -rf logs/*; \
		rm -rf wandb/*; \
		echo "Data files cleaned"; \
	else \
		echo "Cancelled"; \
	fi

clean-all: clean clean-data docker-clean ## Clean everything

# =============================================================================
# Development Utilities
# =============================================================================

jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard: ## Start TensorBoard
	tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006

wandb-sync: ## Sync offline wandb runs
	wandb sync wandb/offline-*

git-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

env-info: ## Show environment information
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "PyTorch version: $$($(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU count: $$($(PYTHON) -c 'import torch; print(torch.cuda.device_count())')"
	@echo "Working directory: $$(pwd)"

# =============================================================================
# Quick Start Commands
# =============================================================================

quickstart: ## Quick start for new developers
	@echo "🚀 Open MoE Trainer Lab Quick Start"
	@echo "=================================="
	make setup-dev
	make docker-build-dev
	@echo ""
	@echo "✅ Setup complete! Next steps:"
	@echo "  1. Start development environment: make docker-dev"
	@echo "  2. Run tests: make test"
	@echo "  3. Open Jupyter Lab: http://localhost:8888"
	@echo "  4. View documentation: make docs-serve"

demo: ## Run a quick demo
	@echo "🎯 Running MoE Trainer Lab Demo"
	$(PYTHON) -m moe_lab.demo --quick

# =============================================================================
# CI/CD Commands
# =============================================================================

ci-test: ## Run full CI test suite
	make format-check
	make lint
	make typecheck
	make security-scan
	make test-coverage

ci-build: ## Build all images for CI
	make docker-build
	make build-package

ci-deploy: ## Deploy for CI (staging)
	make docker-tag
	make docker-push