#!/bin/bash
set -e

echo "🚀 Deploying MoE Trainer Lab to production..."

# Check if Docker Compose or Kubernetes
if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    echo "Deploying to Kubernetes..."
    kubectl apply -f kubernetes/
    kubectl rollout status deployment/moe-trainer-deployment -n moe-trainer-lab
elif [ "$DEPLOYMENT_TYPE" = "compose" ]; then
    echo "Deploying with Docker Compose..."
    docker-compose -f docker-compose.production.yml up -d
else
    echo "Please set DEPLOYMENT_TYPE to 'k8s' or 'compose'"
    exit 1
fi

echo "✅ Deployment completed successfully"
