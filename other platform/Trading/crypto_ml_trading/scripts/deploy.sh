#!/bin/bash

# Crypto ML Trading System Deployment Script
# Supports development, staging, and production deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENVIRONMENT="development"
DEFAULT_CONFIG_FILE="config/system_config.yaml"

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
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
Crypto ML Trading System Deployment Script

USAGE:
    $0 [OPTIONS] COMMAND

COMMANDS:
    build           Build Docker images
    deploy          Deploy the system
    start           Start the system
    stop            Stop the system
    restart         Restart the system
    status          Show system status
    logs            Show logs
    backup          Create system backup
    restore         Restore from backup
    test            Run tests
    cleanup         Clean up resources

OPTIONS:
    -e, --environment   Target environment (development|staging|production)
    -c, --config        Configuration file path
    -f, --force         Force operation without confirmation
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -d, --dry-run       Show what would be done without executing

EXAMPLES:
    $0 deploy --environment production
    $0 start --config config/custom_config.yaml
    $0 backup --environment production
    $0 logs --follow

ENVIRONMENT VARIABLES:
    ENVIRONMENT         Target deployment environment
    CONFIG_FILE         Configuration file path
    DOCKER_REGISTRY     Docker registry for images
    BACKUP_LOCATION     Backup storage location

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="${ENVIRONMENT:-$DEFAULT_ENVIRONMENT}"
    CONFIG_FILE="${CONFIG_FILE:-$DEFAULT_CONFIG_FILE}"
    FORCE="false"
    VERBOSE="false"
    DRY_RUN="false"
    COMMAND=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -f|--force)
                FORCE="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                DEBUG="true"
                shift
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            build|deploy|start|stop|restart|status|logs|backup|restore|test|cleanup)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "$COMMAND" ]]; then
        log_error "No command specified"
        show_help
        exit 1
    fi

    # Validate environment
    case "$ENVIRONMENT" in
        development|staging|production)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check configuration file
    if [[ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi

    # Check .env file
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            log_warn ".env file not found, copying from .env.example"
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        else
            log_error ".env file not found and no .env.example available"
            exit 1
        fi
    fi

    log_info "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images for $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build Docker images"
        return
    fi

    # Set build target based on environment
    local build_target
    case "$ENVIRONMENT" in
        development)
            build_target="development"
            ;;
        staging|production)
            build_target="production"
            ;;
    esac

    # Build images
    DOCKER_BUILDKIT=1 docker-compose build \
        --build-arg BUILD_TARGET="$build_target" \
        --parallel \
        --progress=plain \
        crypto-ml-trading

    log_info "Docker images built successfully"
}

# Deploy system
deploy_system() {
    log_info "Deploying system in $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy system"
        return
    fi

    # Create necessary directories
    mkdir -p data/historical logs saved_models

    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    export BUILD_TARGET="$([ "$ENVIRONMENT" = "development" ] && echo "development" || echo "production")"

    # Deploy based on environment
    case "$ENVIRONMENT" in
        development)
            docker-compose --profile development up -d
            ;;
        staging)
            docker-compose up -d
            ;;
        production)
            # Production deployment with monitoring
            docker-compose --profile monitoring up -d
            ;;
    esac

    # Wait for services to be healthy
    wait_for_services

    log_info "System deployed successfully"
}

# Start system
start_system() {
    log_info "Starting system..."
    
    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would start system"
        return
    fi

    docker-compose start
    wait_for_services
    
    log_info "System started successfully"
}

# Stop system
stop_system() {
    log_info "Stopping system..."
    
    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would stop system"
        return
    fi

    docker-compose stop
    
    log_info "System stopped successfully"
}

# Restart system
restart_system() {
    log_info "Restarting system..."
    stop_system
    start_system
    log_info "System restarted successfully"
}

# Show system status
show_status() {
    log_info "System status:"
    
    cd "$PROJECT_ROOT"
    docker-compose ps
    
    # Show resource usage
    echo
    log_info "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
}

# Show logs
show_logs() {
    log_info "System logs:"
    
    cd "$PROJECT_ROOT"
    
    if [[ "${1:-}" == "--follow" ]]; then
        docker-compose logs -f
    else
        docker-compose logs --tail=100
    fi
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        local healthy_services=0
        local total_services=0
        
        # Check each service
        for container in $(docker-compose ps -q); do
            ((total_services++))
            
            local health_status
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
            
            case "$health_status" in
                healthy)
                    ((healthy_services++))
                    ;;
                starting)
                    log_debug "Service $container is starting..."
                    ;;
                unhealthy)
                    log_warn "Service $container is unhealthy"
                    ;;
                unknown)
                    # Service might not have health check, assume healthy if running
                    local running
                    running=$(docker inspect --format='{{.State.Running}}' "$container" 2>/dev/null || echo "false")
                    if [[ "$running" == "true" ]]; then
                        ((healthy_services++))
                    fi
                    ;;
            esac
        done
        
        if [[ $healthy_services -eq $total_services ]] && [[ $total_services -gt 0 ]]; then
            log_info "All services are healthy"
            return 0
        fi
        
        log_debug "Attempt $attempt/$max_attempts: $healthy_services/$total_services services healthy"
        sleep 5
        ((attempt++))
    done
    
    log_warn "Timeout waiting for services to be healthy"
    return 1
}

# Create backup
create_backup() {
    log_info "Creating system backup..."
    
    local backup_date=$(date +"%Y%m%d_%H%M%S")
    local backup_dir="${BACKUP_LOCATION:-./backups}/backup_${ENVIRONMENT}_${backup_date}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create backup at $backup_dir"
        return
    fi

    mkdir -p "$backup_dir"
    
    cd "$PROJECT_ROOT"
    
    # Backup configuration
    cp -r config "$backup_dir/"
    
    # Backup data
    if [[ -d "data" ]]; then
        cp -r data "$backup_dir/"
    fi
    
    # Backup models
    if [[ -d "saved_models" ]]; then
        cp -r saved_models "$backup_dir/"
    fi
    
    # Backup logs (recent only)
    if [[ -d "logs" ]]; then
        mkdir -p "$backup_dir/logs"
        find logs -name "*.log" -mtime -7 -exec cp {} "$backup_dir/logs/" \;
    fi
    
    # Backup database if running
    if docker-compose ps postgres | grep -q "Up"; then
        log_info "Backing up database..."
        docker-compose exec -T postgres pg_dump -U "${POSTGRES_USER:-trading_user}" "${POSTGRES_DB:-trading_db}" > "$backup_dir/database_backup.sql"
    fi
    
    # Create archive
    tar -czf "${backup_dir}.tar.gz" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"
    rm -rf "$backup_dir"
    
    log_info "Backup created: ${backup_dir}.tar.gz"
}

# Restore from backup
restore_backup() {
    local backup_file="$1"
    
    if [[ -z "$backup_file" ]]; then
        log_error "Backup file not specified"
        exit 1
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    if [[ "$FORCE" != "true" ]]; then
        echo -n "This will overwrite existing data. Continue? (y/N): "
        read -r response
        if [[ "$response" != "y" ]]; then
            log_info "Restore cancelled"
            exit 0
        fi
    fi
    
    log_info "Restoring from backup: $backup_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would restore from backup"
        return
    fi
    
    # Extract backup
    local temp_dir=$(mktemp -d)
    tar -xzf "$backup_file" -C "$temp_dir"
    local backup_dir=$(find "$temp_dir" -maxdepth 1 -type d -name "backup_*" | head -1)
    
    cd "$PROJECT_ROOT"
    
    # Stop system
    docker-compose down
    
    # Restore files
    if [[ -d "$backup_dir/config" ]]; then
        cp -r "$backup_dir/config"/* config/
    fi
    
    if [[ -d "$backup_dir/data" ]]; then
        rm -rf data
        cp -r "$backup_dir/data" .
    fi
    
    if [[ -d "$backup_dir/saved_models" ]]; then
        rm -rf saved_models
        cp -r "$backup_dir/saved_models" .
    fi
    
    # Start system
    docker-compose up -d
    wait_for_services
    
    # Restore database
    if [[ -f "$backup_dir/database_backup.sql" ]]; then
        log_info "Restoring database..."
        docker-compose exec -T postgres psql -U "${POSTGRES_USER:-trading_user}" -d "${POSTGRES_DB:-trading_db}" < "$backup_dir/database_backup.sql"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_info "Restore completed successfully"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run tests"
        return
    fi

    # Run tests in container
    docker-compose run --rm crypto-ml-trading python -m pytest tests/ -v --cov=. --cov-report=html
    
    log_info "Tests completed"
}

# Cleanup resources
cleanup_resources() {
    log_info "Cleaning up resources..."
    
    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would cleanup resources"
        return
    fi

    # Stop and remove containers
    docker-compose down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (with confirmation)
    if [[ "$FORCE" == "true" ]]; then
        docker volume prune -f
    else
        docker volume prune
    fi
    
    log_info "Cleanup completed"
}

# Main execution
main() {
    parse_args "$@"
    
    log_info "Starting deployment script..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Config file: $CONFIG_FILE"
    log_info "Command: $COMMAND"
    
    check_prerequisites
    
    case "$COMMAND" in
        build)
            build_images
            ;;
        deploy)
            build_images
            deploy_system
            ;;
        start)
            start_system
            ;;
        stop)
            stop_system
            ;;
        restart)
            restart_system
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$@"
            ;;
        backup)
            create_backup
            ;;
        restore)
            if [[ $# -gt 0 ]]; then
                restore_backup "$1"
            else
                log_error "Backup file not specified for restore"
                exit 1
            fi
            ;;
        test)
            run_tests
            ;;
        cleanup)
            cleanup_resources
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
    
    log_info "Deployment script completed successfully"
}

# Execute main function with all arguments
main "$@"