#!/bin/bash
set -e

echo "🚀 Setting up Open MoE Trainer Lab development environment..."

# Update system packages
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git-lfs \
    htop \
    nvtop \
    tree \
    jq \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
pip install --upgrade pip setuptools wheel

# Install core ML dependencies
pip install \
    torch torchvision torchaudio \
    transformers \
    datasets \
    accelerate \
    deepspeed \
    fairscale \
    wandb \
    tensorboard \
    jupyterlab \
    ipywidgets

# Install development tools
pip install \
    black \
    isort \
    pylint \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pre-commit \
    sphinx \
    sphinx-rtd-theme \
    nbsphinx

# Install visualization and analysis tools
pip install \
    matplotlib \
    seaborn \
    plotly \
    dash \
    streamlit \
    pandas \
    numpy \
    scikit-learn

# Install project in development mode if setup.py exists
if [ -f "setup.py" ]; then
    pip install -e .
fi

# Initialize git hooks
git config --global --add safe.directory /workspace
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
fi

# Create necessary directories
mkdir -p data checkpoints logs wandb

# Set up Jupyter Lab configuration
jupyter lab --generate-config
echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Initialize wandb (will require user login)
echo "📊 Initialize W&B for experiment tracking:"
echo "Run: wandb login"

# Display GPU information
echo "🔧 GPU Information:"
nvidia-smi

echo "✅ Development environment setup complete!"
echo "🎯 Next steps:"
echo "   1. Run: wandb login"
echo "   2. Start Jupyter: jupyter lab --allow-root"
echo "   3. Run tests: pytest"
echo "   4. Format code: black ."