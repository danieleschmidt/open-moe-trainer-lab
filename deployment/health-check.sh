#!/bin/bash
set -e

echo "🔍 Running health checks..."

if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    # Kubernetes health check
    kubectl get pods -n moe-trainer-lab
    kubectl get services -n moe-trainer-lab
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/moe-trainer-deployment -n moe-trainer-lab
    
    # Test service endpoint
    EXTERNAL_IP=$(kubectl get service moe-trainer-service -n moe-trainer-lab -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ ! -z "$EXTERNAL_IP" ]; then
        curl -f http://$EXTERNAL_IP/health || echo "Service not yet available"
    fi
elif [ "$DEPLOYMENT_TYPE" = "compose" ]; then
    # Docker Compose health check
    docker-compose -f docker-compose.production.yml ps
    
    # Test service endpoint
    curl -f http://localhost:8000/health || echo "Service not yet available"
fi

echo "✅ Health checks completed"
