#!/bin/bash

# Production Deployment Script for MoE Lab
# Supports Docker Compose and Kubernetes deployment modes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOYMENT_MODE="${1:-docker-compose}"
ENVIRONMENT="${2:-production}"
PROFILE="${3:-default}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    case "$DEPLOYMENT_MODE" in
        "docker-compose")
            if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
                log_error "Docker and docker-compose are required for Docker Compose deployment"
                exit 1
            fi
            ;;
        "kubernetes"|"k8s")
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is required for Kubernetes deployment"
                exit 1
            fi
            if ! kubectl cluster-info &> /dev/null; then
                log_error "kubectl cannot connect to a Kubernetes cluster"
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported deployment mode: $DEPLOYMENT_MODE"
            echo "Supported modes: docker-compose, kubernetes, k8s"
            exit 1
            ;;
    esac
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main image
    docker build -f deployment/production/Dockerfile.optimized \
        --target production \
        -t moe-lab:latest \
        -t moe-lab:$ENVIRONMENT \
        .
    
    # Build GPU image
    docker build -f deployment/production/Dockerfile.optimized \
        --target gpu-production \
        -t moe-lab-gpu:latest \
        -t moe-lab-gpu:$ENVIRONMENT \
        .
    
    log_info "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT/deployment/production"
    
    # Create necessary directories
    mkdir -p ssl logs
    
    # Generate environment file if it doesn't exist
    if [[ ! -f .env ]]; then
        log_info "Creating environment file..."
        cat > .env << EOF
# MoE Lab Production Environment
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
SECRET_KEY=$(openssl rand -base64 64)

# Service URLs
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://moe_user:\${POSTGRES_PASSWORD}@postgres:5432/moe_lab

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Deployment
ENVIRONMENT=$ENVIRONMENT
PROFILE=$PROFILE
EOF
        log_warn "Generated .env file with random passwords. Please review and update as needed."
    fi
    
    # Deploy based on profile
    case "$PROFILE" in
        "gpu")
            docker-compose -f docker-compose.production.yml --profile gpu up -d
            ;;
        "distributed")
            docker-compose -f docker-compose.production.yml --profile distributed up -d
            ;;
        "full")
            docker-compose -f docker-compose.production.yml --profile gpu --profile distributed up -d
            ;;
        *)
            docker-compose -f docker-compose.production.yml up -d
            ;;
    esac
    
    log_info "Docker Compose deployment completed"
    
    # Show service status
    docker-compose -f docker-compose.production.yml ps
    
    # Show access URLs
    echo ""
    log_info "Service URLs:"
    echo "  API Server: http://localhost:8000"
    echo "  Grafana: http://localhost:3000 (admin/admin123)"
    echo "  Prometheus: http://localhost:9090"
    
    if [[ "$PROFILE" == "gpu" || "$PROFILE" == "full" ]]; then
        echo "  GPU Server: http://localhost:8001"
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT/deployment/production/kubernetes"
    
    # Create namespace and basic resources
    kubectl apply -f namespace.yaml
    
    # Create secrets
    create_kubernetes_secrets
    
    # Create storage resources
    kubectl apply -f storage.yaml
    
    # Deploy core services
    kubectl apply -f moe-api-deployment.yaml
    
    # Deploy GPU services if requested
    if [[ "$PROFILE" == "gpu" || "$PROFILE" == "full" ]]; then
        kubectl apply -f moe-gpu-deployment.yaml
    fi
    
    # Deploy monitoring (assumed to be available)
    deploy_monitoring_kubernetes
    
    # Create ingress
    kubectl apply -f ingress.yaml
    
    # Set up autoscaling
    kubectl apply -f hpa.yaml
    
    log_info "Kubernetes deployment completed"
    
    # Show deployment status
    kubectl get pods -n moe-lab
    kubectl get services -n moe-lab
    kubectl get ingress -n moe-lab
}

# Create Kubernetes secrets
create_kubernetes_secrets() {
    log_info "Creating Kubernetes secrets..."
    
    # Check if secrets already exist
    if kubectl get secret moe-lab-secrets -n moe-lab &> /dev/null; then
        log_warn "Secrets already exist, skipping creation"
        return
    fi
    
    # Generate random passwords
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    SECRET_KEY=$(openssl rand -base64 64)
    
    # Create secret
    kubectl create secret generic moe-lab-secrets -n moe-lab \
        --from-literal=postgres-url="postgresql://moe_user:$POSTGRES_PASSWORD@postgres:5432/moe_lab" \
        --from-literal=redis-url="redis://:$REDIS_PASSWORD@redis:6379" \
        --from-literal=secret-key="$SECRET_KEY"
    
    log_info "Secrets created successfully"
}

# Deploy monitoring to Kubernetes
deploy_monitoring_kubernetes() {
    log_info "Deploying monitoring stack..."
    
    # This assumes you have Prometheus Operator or similar monitoring setup
    # You would typically use Helm charts for this
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: moe-lab-config
  namespace: moe-lab
data:
  config.yaml: |
    logging:
      level: INFO
    monitoring:
      enabled: true
      prometheus_url: http://prometheus:9090
    model:
      cache_dir: /app/models
      checkpoint_dir: /app/checkpoints
EOF
    
    log_info "Monitoring configuration applied"
}

# Health check
health_check() {
    log_info "Running health checks..."
    
    case "$DEPLOYMENT_MODE" in
        "docker-compose")
            # Wait for services to be ready
            sleep 30
            
            # Check API health
            if curl -f http://localhost:8000/health &> /dev/null; then
                log_info "✅ API server is healthy"
            else
                log_error "❌ API server health check failed"
                return 1
            fi
            
            # Check Grafana
            if curl -f http://localhost:3000/api/health &> /dev/null; then
                log_info "✅ Grafana is healthy"
            else
                log_warn "⚠️ Grafana health check failed"
            fi
            ;;
        "kubernetes"|"k8s")
            # Wait for pods to be ready
            kubectl wait --for=condition=ready pod -l app=moe-api -n moe-lab --timeout=300s
            
            # Check if services are running
            if kubectl get pods -n moe-lab | grep -q "moe-api.*Running"; then
                log_info "✅ MoE API pods are running"
            else
                log_error "❌ MoE API pods are not running"
                return 1
            fi
            ;;
    esac
    
    log_info "Health checks completed"
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    
    case "$DEPLOYMENT_MODE" in
        "docker-compose")
            cd "$PROJECT_ROOT/deployment/production"
            docker-compose -f docker-compose.production.yml down
            ;;
        "kubernetes"|"k8s")
            kubectl delete namespace moe-lab
            ;;
    esac
    
    log_info "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Main deployment function
main() {
    log_info "Starting MoE Lab production deployment"
    log_info "Mode: $DEPLOYMENT_MODE, Environment: $ENVIRONMENT, Profile: $PROFILE"
    
    # Set up trap for cleanup
    trap cleanup EXIT
    trap rollback ERR
    
    # Run deployment steps
    check_prerequisites
    build_images
    
    case "$DEPLOYMENT_MODE" in
        "docker-compose")
            deploy_docker_compose
            ;;
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
    esac
    
    health_check
    
    log_info "🎉 MoE Lab deployment completed successfully!"
    log_info "Check the logs and monitoring dashboards to ensure everything is working correctly."
}

# Help function
show_help() {
    echo "MoE Lab Production Deployment Script"
    echo ""
    echo "Usage: $0 [MODE] [ENVIRONMENT] [PROFILE]"
    echo ""
    echo "Parameters:"
    echo "  MODE:        Deployment mode (docker-compose|kubernetes|k8s) [default: docker-compose]"
    echo "  ENVIRONMENT: Environment name (production|staging|dev) [default: production]"
    echo "  PROFILE:     Deployment profile (default|gpu|distributed|full) [default: default]"
    echo ""
    echo "Examples:"
    echo "  $0 docker-compose production gpu      # Docker Compose with GPU support"
    echo "  $0 kubernetes production full         # Kubernetes with all features"
    echo "  $0 k8s staging default               # Kubernetes staging deployment"
    echo ""
    echo "Profiles:"
    echo "  default:     Basic CPU-only deployment"
    echo "  gpu:         Include GPU-accelerated inference"
    echo "  distributed: Include distributed training workers"
    echo "  full:        Include all features (GPU + distributed)"
}

# Handle command line arguments
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"