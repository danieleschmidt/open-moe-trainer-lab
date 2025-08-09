#!/bin/bash

# Production Deployment Script for MoE Trainer Lab
# Usage: ./scripts/deploy_production.sh [docker|kubernetes] [environment]

set -e

DEPLOY_TYPE=${1:-docker}
ENVIRONMENT=${2:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose for Docker deployment
    if [[ "$DEPLOY_TYPE" == "docker" ]] && ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check kubectl for Kubernetes deployment
    if [[ "$DEPLOY_TYPE" == "kubernetes" ]] && ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker for GPU support
    if ! docker info | grep -q nvidia; then
        log_warning "NVIDIA Docker runtime not detected. GPU support may not work."
    fi
    
    log_success "Prerequisites check completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build production image
    docker build -t moe-trainer:latest --target production .
    
    # Tag with environment and timestamp
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    docker tag moe-trainer:latest "moe-trainer:${ENVIRONMENT}-${TIMESTAMP}"
    docker tag moe-trainer:latest "moe-trainer:${ENVIRONMENT}-latest"
    
    log_success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_DIR"
    
    # Create necessary directories
    mkdir -p data models checkpoints logs outputs monitoring/grafana/{dashboards,datasources}
    
    # Generate monitoring configurations if they don't exist
    if [[ ! -f monitoring/prometheus.yml ]]; then
        log_info "Generating Prometheus configuration..."
        cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'moe-trainer'
    static_configs:
      - targets: ['moe-trainer:9090']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF
    fi
    
    # Deploy services
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    check_service_health_docker
    
    log_success "Docker deployment completed"
}

# Deploy with Kubernetes
deploy_kubernetes() {
    log_info "Deploying with Kubernetes..."
    
    cd "$PROJECT_DIR"
    
    # Create namespace if it doesn't exist
    kubectl create namespace moe-trainer --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/deployment.yaml -n moe-trainer
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/moe-trainer -n moe-trainer
    
    # Get service information
    log_info "Getting service information..."
    kubectl get services -n moe-trainer
    
    log_success "Kubernetes deployment completed"
}

# Check service health for Docker deployment
check_service_health_docker() {
    log_info "Checking service health..."
    
    # Check MoE Trainer
    if curl -f -s http://localhost:8000/health > /dev/null; then
        log_success "MoE Trainer is healthy"
    else
        log_error "MoE Trainer is not responding"
    fi
    
    # Check Redis
    if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli ping | grep -q PONG; then
        log_success "Redis is healthy"
    else
        log_error "Redis is not responding"
    fi
    
    # Check Prometheus
    if curl -f -s http://localhost:9091/-/healthy > /dev/null; then
        log_success "Prometheus is healthy"
    else
        log_error "Prometheus is not responding"
    fi
    
    # Check Grafana
    if curl -f -s http://localhost:3000/api/health > /dev/null; then
        log_success "Grafana is healthy"
    else
        log_error "Grafana is not responding"
    fi
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring and observability..."
    
    if [[ "$DEPLOY_TYPE" == "docker" ]]; then
        # Configure Grafana dashboards
        log_info "Configuring Grafana dashboards..."
        
        # Wait for Grafana to be ready
        sleep 30
        
        # Create datasource
        curl -X POST \
            http://admin:moe_admin_2024@localhost:3000/api/datasources \
            -H 'Content-Type: application/json' \
            -d '{
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "access": "proxy",
                "isDefault": true
            }' || true
    fi
    
    log_success "Monitoring setup completed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    if [[ "$DEPLOY_TYPE" == "docker" ]]; then
        # Test API endpoint
        if curl -f -s http://localhost:8000/health | grep -q "healthy"; then
            log_success "API health check passed"
        else
            log_error "API health check failed"
            return 1
        fi
        
        # Test dashboard
        if curl -f -s http://localhost:8080 > /dev/null; then
            log_success "Dashboard accessibility check passed"
        else
            log_error "Dashboard accessibility check failed"
            return 1
        fi
        
        # Test metrics endpoint
        if curl -f -s http://localhost:9090/metrics > /dev/null; then
            log_success "Metrics endpoint check passed"
        else
            log_error "Metrics endpoint check failed"
            return 1
        fi
        
    elif [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        # Test Kubernetes deployment
        if kubectl get pods -n moe-trainer | grep -q "Running"; then
            log_success "Kubernetes pods are running"
        else
            log_error "Kubernetes pods are not running properly"
            return 1
        fi
        
        # Port forward for testing
        kubectl port-forward -n moe-trainer service/moe-trainer-service 8000:8000 &
        PF_PID=$!
        sleep 10
        
        if curl -f -s http://localhost:8000/health | grep -q "healthy"; then
            log_success "Kubernetes API health check passed"
        else
            log_error "Kubernetes API health check failed"
            kill $PF_PID || true
            return 1
        fi
        
        kill $PF_PID || true
    fi
    
    log_success "Smoke tests completed successfully"
}

# Display deployment information
show_deployment_info() {
    log_info "Deployment Information:"
    echo "=========================="
    echo "Deployment Type: $DEPLOY_TYPE"
    echo "Environment: $ENVIRONMENT"
    echo ""
    
    if [[ "$DEPLOY_TYPE" == "docker" ]]; then
        echo "Service URLs:"
        echo "  - API: http://localhost:8000"
        echo "  - Dashboard: http://localhost:8080" 
        echo "  - Grafana: http://localhost:3000 (admin/moe_admin_2024)"
        echo "  - Prometheus: http://localhost:9091"
        echo ""
        echo "Useful Commands:"
        echo "  - View logs: docker-compose -f docker-compose.prod.yml logs -f"
        echo "  - Stop services: docker-compose -f docker-compose.prod.yml down"
        echo "  - Scale services: docker-compose -f docker-compose.prod.yml up -d --scale moe-trainer=2"
        
    elif [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        echo "Kubernetes Resources:"
        kubectl get all -n moe-trainer
        echo ""
        echo "Service URLs (after port-forward):"
        echo "  - API: kubectl port-forward -n moe-trainer service/moe-trainer-service 8000:8000"
        echo "  - Dashboard: kubectl port-forward -n moe-trainer service/moe-trainer-service 8080:8080"
        echo ""
        echo "Useful Commands:"
        echo "  - View logs: kubectl logs -f deployment/moe-trainer -n moe-trainer"
        echo "  - Scale deployment: kubectl scale deployment/moe-trainer --replicas=3 -n moe-trainer"
        echo "  - Delete deployment: kubectl delete -f kubernetes/deployment.yaml -n moe-trainer"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary resources..."
    
    if [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        # Kill any remaining port-forward processes
        pkill -f "kubectl port-forward" || true
    fi
}

# Main deployment workflow
main() {
    log_info "Starting MoE Trainer Lab production deployment..."
    log_info "Deploy Type: $DEPLOY_TYPE, Environment: $ENVIRONMENT"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    build_images
    
    if [[ "$DEPLOY_TYPE" == "docker" ]]; then
        deploy_docker
        setup_monitoring
    elif [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        deploy_kubernetes
    else
        log_error "Unknown deployment type: $DEPLOY_TYPE"
        exit 1
    fi
    
    # Run verification
    run_smoke_tests
    
    # Show deployment info
    show_deployment_info
    
    log_success "Production deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    -h|--help)
        echo "Usage: $0 [docker|kubernetes] [environment]"
        echo ""
        echo "Arguments:"
        echo "  deployment_type   Docker or Kubernetes deployment (default: docker)"
        echo "  environment       Environment name (default: production)"
        echo ""
        echo "Examples:"
        echo "  $0 docker production"
        echo "  $0 kubernetes staging"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac