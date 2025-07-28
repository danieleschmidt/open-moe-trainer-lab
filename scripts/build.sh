#!/bin/bash
set -e

# Open MoE Trainer Lab - Build Script
# Comprehensive build automation for all deployment targets

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
VERSION="${BUILD_VERSION:-$(git describe --tags --always 2>/dev/null || echo 'dev')}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')

# Build targets
TARGETS=("development" "production" "training" "inference" "benchmark" "ci")

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
Open MoE Trainer Lab - Build Script

USAGE:
    $0 [OPTIONS] [TARGETS...]

OPTIONS:
    -h, --help              Show this help message
    -v, --version VERSION   Set build version (default: git describe)
    -r, --registry URL      Set Docker registry URL
    -p, --push              Push images to registry after build
    -c, --clean             Clean build cache before building
    -l, --list              List available build targets
    --no-cache              Build without using Docker cache
    --parallel              Build targets in parallel
    --test                  Run tests after building

TARGETS:
    all                     Build all targets (default)
    development             Development image with all tools
    production              Minimal production runtime
    training                Optimized for training workloads
    inference               Optimized for model serving
    benchmark               Performance testing image
    ci                      Continuous integration image

EXAMPLES:
    $0                      # Build all targets
    $0 development          # Build only development image
    $0 --push production    # Build and push production image
    $0 --clean --no-cache   # Clean build without cache
    $0 --parallel training inference  # Build multiple targets in parallel

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY         Docker registry URL
    BUILD_VERSION           Build version tag
    DOCKER_BUILDKIT         Enable BuildKit (recommended: 1)
    BUILD_ARGS              Additional build arguments

EOF
}

list_targets() {
    log_info "Available build targets:"
    for target in "${TARGETS[@]}"; do
        echo "  - $target"
    done
}

check_requirements() {
    log_info "Checking build requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check BuildKit support
    if [[ "${DOCKER_BUILDKIT:-}" != "1" ]]; then
        log_warning "DOCKER_BUILDKIT not enabled. Consider setting DOCKER_BUILDKIT=1 for better performance"
    fi
    
    # Check git (for version info)
    if ! command -v git &> /dev/null; then
        log_warning "Git not found. Using default version info"
    fi
    
    log_success "Requirements check passed"
}

clean_build_cache() {
    log_info "Cleaning Docker build cache..."
    docker builder prune -f
    docker system prune -f
    log_success "Build cache cleaned"
}

build_target() {
    local target="$1"
    local push="${2:-false}"
    
    log_info "Building target: $target"
    
    # Build arguments
    local build_args=(
        --target "$target"
        --tag "$IMAGE_NAME:$target"
        --tag "$IMAGE_NAME:$target-$VERSION"
        --build-arg "BUILD_DATE=$BUILD_DATE"
        --build-arg "VERSION=$VERSION"
        --build-arg "COMMIT_SHA=$COMMIT_SHA"
    )
    
    # Add registry tags if pushing
    if [[ "$push" == "true" ]]; then
        build_args+=(
            --tag "$REGISTRY/$IMAGE_NAME:$target"
            --tag "$REGISTRY/$IMAGE_NAME:$target-$VERSION"
        )
    fi
    
    # Add no-cache flag if requested
    if [[ "${NO_CACHE:-false}" == "true" ]]; then
        build_args+=(--no-cache)
    fi
    
    # Add custom build args
    if [[ -n "${BUILD_ARGS:-}" ]]; then
        IFS=' ' read -ra ADDR <<< "$BUILD_ARGS"
        for arg in "${ADDR[@]}"; do
            build_args+=(--build-arg "$arg")
        done
    fi
    
    # Build the image
    if docker build "${build_args[@]}" .; then
        log_success "Built $target successfully"
        
        # Push if requested
        if [[ "$push" == "true" ]]; then
            log_info "Pushing $target to registry..."
            docker push "$REGISTRY/$IMAGE_NAME:$target"
            docker push "$REGISTRY/$IMAGE_NAME:$target-$VERSION"
            log_success "Pushed $target to registry"
        fi
        
        return 0
    else
        log_error "Failed to build $target"
        return 1
    fi
}

build_parallel() {
    local targets=("$@")
    local pids=()
    local failed=()
    
    log_info "Building ${#targets[@]} targets in parallel..."
    
    # Start builds in parallel
    for target in "${targets[@]}"; do
        build_target "$target" "$PUSH" &
        pids+=($!)
    done
    
    # Wait for all builds to complete
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            failed+=("${targets[$i]}")
        fi
    done
    
    # Report results
    if [[ ${#failed[@]} -eq 0 ]]; then
        log_success "All parallel builds completed successfully"
        return 0
    else
        log_error "Failed builds: ${failed[*]}"
        return 1
    fi
}

build_sequential() {
    local targets=("$@")
    local failed=()
    
    log_info "Building ${#targets[@]} targets sequentially..."
    
    for target in "${targets[@]}"; do
        if ! build_target "$target" "$PUSH"; then
            failed+=("$target")
        fi
    done
    
    # Report results
    if [[ ${#failed[@]} -eq 0 ]]; then
        log_success "All sequential builds completed successfully"
        return 0
    else
        log_error "Failed builds: ${failed[*]}"
        return 1
    fi
}

run_tests() {
    log_info "Running post-build tests..."
    
    # Test development image
    if docker run --rm "$IMAGE_NAME:development" python -c "import moe_lab; print('Development image OK')"; then
        log_success "Development image test passed"
    else
        log_error "Development image test failed"
        return 1
    fi
    
    # Test production image
    if docker run --rm "$IMAGE_NAME:production" python -c "import moe_lab; print('Production image OK')"; then
        log_success "Production image test passed"
    else
        log_error "Production image test failed"
        return 1
    fi
    
    log_success "All post-build tests passed"
}

show_build_info() {
    log_info "Build Information:"
    echo "  Version: $VERSION"
    echo "  Commit: $COMMIT_SHA"
    echo "  Build Date: $BUILD_DATE"
    echo "  Registry: $REGISTRY"
    echo "  Image Name: $IMAGE_NAME"
    echo "  Docker Version: $(docker --version)"
    if [[ "${DOCKER_BUILDKIT:-}" == "1" ]]; then
        echo "  BuildKit: Enabled"
    else
        echo "  BuildKit: Disabled"
    fi
}

# Parse command line arguments
PUSH=false
CLEAN=false
NO_CACHE=false
PARALLEL=false
RUN_TESTS=false
BUILD_TARGETS=()

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
        -p|--push)
            PUSH=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -l|--list)
            list_targets
            exit 0
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        all)
            BUILD_TARGETS=("${TARGETS[@]}")
            shift
            ;;
        development|production|training|inference|benchmark|ci)
            BUILD_TARGETS+=("$1")
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default to all targets if none specified
if [[ ${#BUILD_TARGETS[@]} -eq 0 ]]; then
    BUILD_TARGETS=("${TARGETS[@]}")
fi

# Main execution
main() {
    log_info "Starting Open MoE Trainer Lab build process..."
    show_build_info
    
    # Check requirements
    check_requirements
    
    # Clean if requested
    if [[ "$CLEAN" == "true" ]]; then
        clean_build_cache
    fi
    
    # Build targets
    if [[ "$PARALLEL" == "true" ]] && [[ ${#BUILD_TARGETS[@]} -gt 1 ]]; then
        build_parallel "${BUILD_TARGETS[@]}"
    else
        build_sequential "${BUILD_TARGETS[@]}"
    fi
    
    # Run tests if requested
    if [[ "$RUN_TESTS" == "true" ]]; then
        run_tests
    fi
    
    log_success "Build process completed successfully!"
    
    # Show final summary
    echo
    log_info "Built images:"
    for target in "${BUILD_TARGETS[@]}"; do
        if docker image inspect "$IMAGE_NAME:$target" &> /dev/null; then
            size=$(docker image inspect "$IMAGE_NAME:$target" --format '{{.Size}}' | numfmt --to=iec)
            echo "  - $IMAGE_NAME:$target ($size)"
        fi
    done
}

# Run main function
main "$@"