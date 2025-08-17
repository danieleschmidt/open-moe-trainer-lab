#!/bin/bash
set -e

echo "🚀 Building MoE Trainer Lab for production..."

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.production -t moe-trainer-lab:latest .

# Tag for registry
if [ ! -z "$REGISTRY" ]; then
    echo "Tagging for registry: $REGISTRY"
    docker tag moe-trainer-lab:latest $REGISTRY/moe-trainer-lab:latest
    docker tag moe-trainer-lab:latest $REGISTRY/moe-trainer-lab:$(git rev-parse --short HEAD)
fi

echo "✅ Build completed successfully"
