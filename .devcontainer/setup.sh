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

# Install project in development mode with all extras
if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev,gpu,distributed,visualization,benchmarking,cloud,all]"
elif [ -f "setup.py" ]; then
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

# Create development shortcuts
cat > ~/.bash_aliases << 'EOF'
# MoE Lab Development Shortcuts
alias moe-test='pytest tests/ -v'
alias moe-quality='python scripts/automation_helper.py quality'
alias moe-metrics='python scripts/collect_metrics.py --summary'
alias moe-health='python scripts/automation_helper.py health'
alias moe-dashboard='python -c "from moe_lab.dashboard import main; main()"'
alias moe-train='python -m moe_lab.cli train'
alias moe-eval='python -m moe_lab.cli evaluate'
alias gs='git status'
alias gd='git diff'
alias dcup='docker-compose up -d'
alias dcdown='docker-compose down'
EOF

echo "✅ Development environment setup complete!"
echo "🎯 Available shortcuts:"
echo "   moe-test       # Run test suite"
echo "   moe-quality    # Check code quality"
echo "   moe-metrics    # View project metrics"
echo "   moe-dashboard  # Launch training dashboard"
echo "   moe-train      # Start model training"
echo "   moe-eval       # Evaluate models"
echo ""
echo "🔧 Next steps:"
echo "   1. Run: wandb login"
echo "   2. Start development: moe-dashboard"
echo "   3. Run tests: moe-test"
echo "   4. Check quality: moe-quality"