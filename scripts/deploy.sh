#!/bin/bash
set -e

# Open MoE Trainer Lab - Deployment Script
# Automated deployment for various environments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="open-moe-trainer"
REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
VERSION="${DEPLOY_VERSION:-latest}"
ENVIRONMENT="${DEPLOY_ENV:-development}"
NAMESPACE="${K8S_NAMESPACE:-moe-lab}"

# Deployment environments
ENVIRONMENTS=("development" "staging" "production" "distributed" "inference")

# Functions
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

show_help() {
    cat << EOF
Open MoE Trainer Lab - Deployment Script

USAGE:
    $0 [OPTIONS] [ENVIRONMENT]

OPTIONS:
    -h, --help              Show this help message
    -v, --version VERSION   Set deployment version (default: latest)
    -r, --registry URL      Set Docker registry URL
    -n, --namespace NS      Set Kubernetes namespace
    -c, --config FILE       Use custom configuration file
    -d, --dry-run           Show what would be deployed without doing it
    -w, --wait              Wait for deployment to be ready
    --rollback              Rollback to previous deployment
    --scale REPLICAS        Scale deployment to specified replicas
    --status                Show deployment status

ENVIRONMENTS:
    development             Local development deployment
    staging                 Staging environment
    production              Production environment  
    distributed             Multi-node training setup
    inference               Model serving deployment

EXAMPLES:
    $0 development          # Deploy to development
    $0 --version v1.0.0 production  # Deploy specific version to production
    $0 --dry-run staging    # Preview staging deployment
    $0 --rollback production  # Rollback production deployment
    $0 --scale 3 inference  # Scale inference to 3 replicas

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY         Docker registry URL
    DEPLOY_VERSION          Deployment version
    DEPLOY_ENV              Target environment
    K8S_NAMESPACE           Kubernetes namespace
    KUBECONFIG             Kubernetes config file

EOF
}

check_requirements() {
    log_info "Checking deployment requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose not found. Some deployments may not work."
    fi
    
    # Check kubectl for Kubernetes deployments
    if [[ "$ENVIRONMENT" == "production" ]] || [[ "$ENVIRONMENT" == "staging" ]]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is required for $ENVIRONMENT deployment"
            exit 1
        fi
        
        # Check cluster connectivity
        if ! kubectl cluster-info &>/dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            exit 1
        fi
    fi
    
    log_success "Requirements check passed"
}

deploy_development() {
    log_info "Deploying to development environment..."
    
    # Stop existing containers
    docker-compose down 2>/dev/null || true
    
    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose pull
    
    # Start development stack
    log_info "Starting development stack..."
    docker-compose up -d dev monitoring grafana
    
    # Wait for services to be ready
    if [[ "$WAIT" == "true" ]]; then
        wait_for_service "http://localhost:8080" "Dashboard"
        wait_for_service "http://localhost:3001" "Grafana"
    fi
    
    log_success "Development environment deployed"
    show_service_urls_dev
}

deploy_staging() {
    log_info "Deploying to staging environment..."
    
    # Check if staging namespace exists
    if ! kubectl get namespace "$NAMESPACE-staging" &>/dev/null; then
        log_info "Creating staging namespace..."
        kubectl create namespace "$NAMESPACE-staging"
    fi
    
    # Apply staging configurations
    log_info "Applying staging configurations..."
    sed "s/{{VERSION}}/$VERSION/g; s/{{ENVIRONMENT}}/staging/g" k8s/staging.yaml | \
        kubectl apply -n "$NAMESPACE-staging" -f -
    
    # Wait for deployment if requested
    if [[ "$WAIT" == "true" ]]; then
        log_info "Waiting for staging deployment to be ready..."
        kubectl rollout status deployment/moe-trainer -n "$NAMESPACE-staging" --timeout=300s
    fi
    
    log_success "Staging environment deployed"
}

deploy_production() {
    log_info "Deploying to production environment..."
    
    # Safety check
    if [[ "$VERSION" == "latest" ]]; then
        log_error "Cannot deploy 'latest' tag to production. Please specify a version."
        exit 1
    fi
    
    # Confirmation prompt
    read -p "Are you sure you want to deploy $VERSION to production? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Production deployment cancelled"
        exit 0
    fi
    
    # Check if production namespace exists
    if ! kubectl get namespace "$NAMESPACE-prod" &>/dev/null; then
        log_info "Creating production namespace..."
        kubectl create namespace "$NAMESPACE-prod"
    fi
    
    # Apply production configurations
    log_info "Applying production configurations..."
    sed "s/{{VERSION}}/$VERSION/g; s/{{ENVIRONMENT}}/production/g" k8s/production.yaml | \
        kubectl apply -n "$NAMESPACE-prod" -f -
    
    # Wait for deployment
    log_info "Waiting for production deployment to be ready..."
    kubectl rollout status deployment/moe-trainer -n "$NAMESPACE-prod" --timeout=600s
    
    # Run post-deployment health checks
    run_health_checks "$NAMESPACE-prod"
    
    log_success "Production environment deployed"
}

deploy_distributed() {
    log_info "Deploying distributed training environment..."
    
    # Start coordinator and workers
    docker-compose up -d coordinator worker
    
    # Scale workers if specified
    if [[ -n "$SCALE_REPLICAS" ]]; then
        log_info "Scaling workers to $SCALE_REPLICAS replicas..."
        docker-compose up -d --scale worker="$SCALE_REPLICAS" worker
    fi
    
    # Wait for services to be ready
    if [[ "$WAIT" == "true" ]]; then
        wait_for_distributed_training
    fi
    
    log_success "Distributed training environment deployed"
}

deploy_inference() {
    log_info "Deploying inference environment..."
    
    # Start inference service
    docker-compose up -d serve
    
    # Scale if specified
    if [[ -n "$SCALE_REPLICAS" ]]; then
        log_info "Scaling inference service to $SCALE_REPLICAS replicas..."
        docker-compose up -d --scale serve="$SCALE_REPLICAS" serve
    fi
    
    # Wait for service to be ready
    if [[ "$WAIT" == "true" ]]; then
        wait_for_service "http://localhost:8000/health" "Inference API"
    fi
    
    log_success "Inference environment deployed"
    show_inference_info
}

wait_for_service() {
    local url="$1"
    local name="$2"
    local timeout=60
    local count=0
    
    log_info "Waiting for $name to be ready at $url..."
    
    while [[ $count -lt $timeout ]]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            log_success "$name is ready"
            return 0
        fi
        sleep 2
        ((count+=2))
    done
    
    log_warning "$name did not become ready within ${timeout}s"
    return 1
}

wait_for_distributed_training() {
    log_info "Waiting for distributed training setup..."
    
    # Check coordinator
    if docker-compose logs coordinator | grep -q "Initialized process group"; then
        log_success "Distributed training coordinator ready"
    else
        log_warning "Coordinator may not be ready yet"
    fi
    
    # Check workers
    local worker_count=$(docker-compose ps -q worker | wc -l)
    log_info "Found $worker_count worker(s)"
}

run_health_checks() {
    local namespace="$1"
    
    log_info "Running health checks..."
    
    # Check deployment status
    local ready_replicas=$(kubectl get deployment moe-trainer -n "$namespace" -o jsonpath='{.status.readyReplicas}')
    local desired_replicas=$(kubectl get deployment moe-trainer -n "$namespace" -o jsonpath='{.spec.replicas}')
    
    if [[ "$ready_replicas" == "$desired_replicas" ]]; then
        log_success "All replicas are ready ($ready_replicas/$desired_replicas)"
    else
        log_warning "Only $ready_replicas/$desired_replicas replicas are ready"
    fi
    
    # Check service endpoints
    local service_ip=$(kubectl get service moe-trainer -n "$namespace" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -n "$service_ip" ]]; then
        log_success "Service accessible at $service_ip"
    else
        log_info "Service IP not yet assigned"
    fi
}

rollback_deployment() {
    log_info "Rolling back deployment in $ENVIRONMENT..."
    
    case "$ENVIRONMENT" in
        development)
            log_info "Stopping current containers..."
            docker-compose down
            log_info "Starting previous version..."
            docker-compose up -d
            ;;
        staging|production)
            local ns="$NAMESPACE-$ENVIRONMENT"
            log_info "Rolling back deployment in $ns..."
            kubectl rollout undo deployment/moe-trainer -n "$ns"
            kubectl rollout status deployment/moe-trainer -n "$ns"
            ;;
        *)
            log_error "Rollback not supported for $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    log_success "Rollback completed"
}

scale_deployment() {
    local replicas="$1"
    
    log_info "Scaling $ENVIRONMENT deployment to $replicas replicas..."
    
    case "$ENVIRONMENT" in
        development|distributed|inference)
            local service="serve"
            [[ "$ENVIRONMENT" == "distributed" ]] && service="worker"
            docker-compose up -d --scale "$service=$replicas" "$service"
            ;;
        staging|production)
            local ns="$NAMESPACE-$ENVIRONMENT"
            kubectl scale deployment moe-trainer --replicas="$replicas" -n "$ns"
            kubectl rollout status deployment/moe-trainer -n "$ns"
            ;;
        *)
            log_error "Scaling not supported for $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    log_success "Scaled to $replicas replicas"
}

show_deployment_status() {
    log_info "Deployment status for $ENVIRONMENT:"
    
    case "$ENVIRONMENT" in
        development|distributed|inference)
            docker-compose ps
            ;;
        staging|production)
            local ns="$NAMESPACE-$ENVIRONMENT"
            echo "Namespace: $ns"
            kubectl get deployments,services,pods -n "$ns"
            ;;
        *)
            log_error "Status not available for $ENVIRONMENT"
            exit 1
            ;;
    esac
}

show_service_urls_dev() {
    echo
    log_info "Development services available at:"
    echo "  🎯 Dashboard:    http://localhost:8080"
    echo "  📊 Jupyter Lab: http://localhost:8888"
    echo "  📈 Grafana:     http://localhost:3001 (admin/moelab)"
    echo "  🔍 Prometheus:  http://localhost:9090"
    echo "  📋 TensorBoard: http://localhost:6006"
}

show_inference_info() {
    echo
    log_info "Inference API available at:"
    echo "  🚀 API Endpoint: http://localhost:8000"
    echo "  📖 API Docs:    http://localhost:8000/docs"
    echo "  ❤️  Health:      http://localhost:8000/health"
}

# Parse command line arguments
DRY_RUN=false
WAIT=false
ROLLBACK=false
SCALE_REPLICAS=""
SHOW_STATUS=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -w|--wait)
            WAIT=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --scale)
            SCALE_REPLICAS="$2"
            shift 2
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        development|staging|production|distributed|inference)
            ENVIRONMENT="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting deployment process..."
    echo "  Environment: $ENVIRONMENT"
    echo "  Version: $VERSION"
    echo "  Registry: $REGISTRY"
    
    # Check requirements
    check_requirements
    
    # Handle special operations
    if [[ "$SHOW_STATUS" == "true" ]]; then
        show_deployment_status
        exit 0
    fi
    
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        exit 0
    fi
    
    if [[ -n "$SCALE_REPLICAS" ]]; then
        scale_deployment "$SCALE_REPLICAS"
        exit 0
    fi
    
    # Dry run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy $VERSION to $ENVIRONMENT"
        exit 0
    fi
    
    # Deploy to specified environment
    case "$ENVIRONMENT" in
        development)
            deploy_development
            ;;
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
        distributed)
            deploy_distributed
            ;;
        inference)
            deploy_inference
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"