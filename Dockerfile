# Multi-stage Dockerfile for Open MoE Trainer Lab
# Optimized for both development and production use

ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=11.8
ARG PYTHON_VERSION=3.9

# Base stage with PyTorch and CUDA
FROM nvcr.io/nvidia/pytorch:24.01-py3 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    curl \
    wget \
    unzip \
    htop \
    nvtop \
    tree \
    jq \
    vim \
    tmux \
    rsync \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for dashboard development
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=moeuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better layer caching
COPY pyproject.toml ./
COPY requirements*.txt* ./

# Development stage - includes all development tools
FROM base as development

# Install Python development dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install build \
    && pip install -e .[dev,gpu,distributed,visualization,benchmarking,cloud]

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    notebook \
    pre-commit \
    black \
    isort \
    pylint \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    sphinx \
    sphinx-rtd-theme

# Install monitoring and profiling tools
RUN pip install \
    wandb \
    tensorboard \
    mlflow \
    prometheus-client \
    memory-profiler \
    line-profiler \
    py-spy

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set up git hooks if .git exists
RUN if [ -d ".git" ]; then pre-commit install; fi

# Create necessary directories
RUN mkdir -p /workspace/{data,checkpoints,logs,wandb,outputs,cache} \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to non-root user
USER $USERNAME

# Set up Jupyter Lab
RUN jupyter lab --generate-config \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Expose ports
EXPOSE 8080 8888 6006 3000

# Development entrypoint
CMD ["bash"]

# Production stage - minimal runtime dependencies
FROM base as production

# Install only runtime dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e .[gpu,distributed]

# Copy only necessary files
COPY moe_lab/ ./moe_lab/
COPY pyproject.toml setup.py README.md LICENSE ./
COPY configs/ ./configs/

# Install the package
RUN pip install .

# Create directories for runtime
RUN mkdir -p /workspace/{data,checkpoints,logs,outputs,cache} \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to non-root user
USER $USERNAME

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import moe_lab; print('OK')" || exit 1

# Production entrypoint
ENTRYPOINT ["python", "-m", "moe_lab.cli"]
CMD ["--help"]

# Training stage - optimized for training workloads
FROM production as training

# Install additional training dependencies
USER root
RUN pip install \
    wandb \
    tensorboard \
    deepspeed \
    fairscale

# Copy training scripts and configs
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Set training-specific environment variables
ENV WANDB_PROJECT=open-moe-trainer-lab \
    NCCL_DEBUG=INFO \
    CUDA_LAUNCH_BLOCKING=0

USER $USERNAME

# Training entrypoint
ENTRYPOINT ["python", "-m", "moe_lab.train"]

# Inference stage - optimized for serving
FROM production as inference

# Install inference-specific dependencies
USER root
RUN pip install \
    fastapi \
    uvicorn \
    onnx \
    onnxruntime-gpu

# Copy inference scripts
COPY scripts/serve.py ./scripts/

# Set inference-specific environment variables
ENV INFERENCE_MODE=true \
    MODEL_CACHE_DIR=/workspace/cache \
    MAX_BATCH_SIZE=32

USER $USERNAME

# Inference entrypoint
EXPOSE 8000
ENTRYPOINT ["python", "scripts/serve.py"]

# Benchmark stage - for performance testing
FROM development as benchmark

# Install benchmarking dependencies
USER root
RUN pip install \
    pytest-benchmark \
    memory-profiler \
    py-spy \
    nvitop

# Copy benchmark scripts
COPY tests/benchmarks/ ./tests/benchmarks/
COPY benchmarks/ ./benchmarks/

USER $USERNAME

# Benchmark entrypoint
ENTRYPOINT ["python", "-m", "pytest", "tests/benchmarks/"]

# CI stage - for continuous integration
FROM development as ci

# Install CI-specific tools
USER root
RUN pip install \
    safety \
    bandit \
    coverage \
    codecov

# Set CI environment variables
ENV CI=true \
    PYTHONPATH=/workspace

USER $USERNAME

# CI entrypoint for running tests
ENTRYPOINT ["python", "-m", "pytest"]