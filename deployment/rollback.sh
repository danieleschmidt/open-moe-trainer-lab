#!/bin/bash
set -e

echo "🔄 Rolling back deployment..."

if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
    kubectl rollout undo deployment/moe-trainer-deployment -n moe-trainer-lab
    kubectl rollout status deployment/moe-trainer-deployment -n moe-trainer-lab
elif [ "$DEPLOYMENT_TYPE" = "compose" ]; then
    docker-compose -f docker-compose.production.yml down
    # Restore previous version (would need to be implemented based on your backup strategy)
    echo "Manual intervention required for Docker Compose rollback"
fi

echo "✅ Rollback completed"
