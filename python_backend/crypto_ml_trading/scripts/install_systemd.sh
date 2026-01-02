#!/bin/bash

# Install systemd service for Crypto ML Trading System
# This script sets up the trading system as a systemd service

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="crypto-ml-trading"
SERVICE_USER="trading"
INSTALL_DIR="/opt/crypto-ml-trading"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Create service user
create_user() {
    log_info "Creating service user: $SERVICE_USER"
    
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd --system --home-dir "$INSTALL_DIR" --shell /bin/bash "$SERVICE_USER"
        log_info "User $SERVICE_USER created"
    else
        log_info "User $SERVICE_USER already exists"
    fi
}

# Install application files
install_application() {
    log_info "Installing application to $INSTALL_DIR"
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy application files
    cp -r "$PROJECT_ROOT"/* "$INSTALL_DIR/"
    
    # Create necessary directories
    mkdir -p "$INSTALL_DIR"/{data/historical,logs,saved_models}
    
    # Set ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    
    # Set permissions
    chmod -R 750 "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR/main.py"
    chmod +x "$INSTALL_DIR/scripts/"*.sh
    
    log_info "Application installed successfully"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies"
    
    # Install system packages
    apt-get update
    apt-get install -y python3 python3-pip python3-venv
    
    # Create virtual environment
    sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
    
    # Install Python packages
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
    sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
    
    log_info "Dependencies installed successfully"
}

# Install systemd service
install_service() {
    log_info "Installing systemd service"
    
    # Update service file with correct paths
    sed -i "s|/opt/crypto-ml-trading|$INSTALL_DIR|g" "$PROJECT_ROOT/systemd/$SERVICE_NAME.service"
    sed -i "s|/usr/bin/python3|$INSTALL_DIR/venv/bin/python|g" "$PROJECT_ROOT/systemd/$SERVICE_NAME.service"
    sed -i "s|User=trading|User=$SERVICE_USER|g" "$PROJECT_ROOT/systemd/$SERVICE_NAME.service"
    sed -i "s|Group=trading|Group=$SERVICE_USER|g" "$PROJECT_ROOT/systemd/$SERVICE_NAME.service"
    
    # Copy service file
    cp "$PROJECT_ROOT/systemd/$SERVICE_NAME.service" "/etc/systemd/system/"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable "$SERVICE_NAME"
    
    log_info "Systemd service installed and enabled"
}

# Configure logging
configure_logging() {
    log_info "Configuring logging"
    
    # Create log rotation config
    cat > "/etc/logrotate.d/$SERVICE_NAME" << EOF
$INSTALL_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload $SERVICE_NAME || true
    endscript
}
EOF
    
    # Create rsyslog config for structured logs
    cat > "/etc/rsyslog.d/49-$SERVICE_NAME.conf" << EOF
# Crypto ML Trading System logs
if \$programname == '$SERVICE_NAME' then {
    action(type="omfile" file="/var/log/$SERVICE_NAME.log")
    stop
}
EOF
    
    # Restart rsyslog
    systemctl restart rsyslog
    
    log_info "Logging configured successfully"
}

# Configure firewall (if ufw is installed)
configure_firewall() {
    if command -v ufw &> /dev/null; then
        log_info "Configuring firewall"
        
        # Allow API port if enabled
        ufw allow 8000/tcp comment "Crypto ML Trading API"
        
        log_info "Firewall configured"
    else
        log_warn "UFW firewall not installed, skipping firewall configuration"
    fi
}

# Create configuration files
create_config() {
    log_info "Creating configuration files"
    
    # Copy environment file
    if [[ ! -f "$INSTALL_DIR/.env" ]]; then
        cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
        
        # Update for production
        sed -i 's/ENVIRONMENT=development/ENVIRONMENT=production/' "$INSTALL_DIR/.env"
        sed -i 's/DEBUG_MODE=true/DEBUG_MODE=false/' "$INSTALL_DIR/.env"
        sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=WARNING/' "$INSTALL_DIR/.env"
        
        log_warn "Default .env file created. Please review and update: $INSTALL_DIR/.env"
    fi
    
    # Set secure permissions
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
}

# Install monitoring (optional)
install_monitoring() {
    read -p "Install monitoring tools (Prometheus node exporter)? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installing Prometheus node exporter"
        
        # Download and install node exporter
        NODEEXP_VERSION="1.6.0"
        cd /tmp
        wget "https://github.com/prometheus/node_exporter/releases/download/v${NODEEXP_VERSION}/node_exporter-${NODEEXP_VERSION}.linux-amd64.tar.gz"
        tar -xzf "node_exporter-${NODEEXP_VERSION}.linux-amd64.tar.gz"
        
        # Install binary
        cp "node_exporter-${NODEEXP_VERSION}.linux-amd64/node_exporter" /usr/local/bin/
        chown root:root /usr/local/bin/node_exporter
        chmod +x /usr/local/bin/node_exporter
        
        # Create systemd service for node exporter
        cat > "/etc/systemd/system/node_exporter.service" << EOF
[Unit]
Description=Prometheus Node Exporter
After=network.target

[Service]
Type=simple
User=nobody
Group=nogroup
ExecStart=/usr/local/bin/node_exporter
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
        
        # Enable and start node exporter
        systemctl daemon-reload
        systemctl enable node_exporter
        systemctl start node_exporter
        
        # Configure firewall
        if command -v ufw &> /dev/null; then
            ufw allow 9100/tcp comment "Prometheus Node Exporter"
        fi
        
        log_info "Node exporter installed and started"
        rm -rf /tmp/node_exporter-*
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation"
    
    # Check service status
    if systemctl is-enabled "$SERVICE_NAME" &>/dev/null; then
        log_info "✓ Service is enabled"
    else
        log_error "✗ Service is not enabled"
    fi
    
    # Check user
    if id "$SERVICE_USER" &>/dev/null; then
        log_info "✓ Service user exists"
    else
        log_error "✗ Service user does not exist"
    fi
    
    # Check installation directory
    if [[ -d "$INSTALL_DIR" ]]; then
        log_info "✓ Installation directory exists"
    else
        log_error "✗ Installation directory does not exist"
    fi
    
    # Check Python environment
    if sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/python" -c "import numpy, pandas, scipy" &>/dev/null; then
        log_info "✓ Python dependencies installed"
    else
        log_error "✗ Python dependencies missing"
    fi
    
    log_info "Installation verification completed"
}

# Start service
start_service() {
    read -p "Start the service now? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Starting $SERVICE_NAME service"
        systemctl start "$SERVICE_NAME"
        
        # Wait a moment and check status
        sleep 5
        if systemctl is-active "$SERVICE_NAME" &>/dev/null; then
            log_info "✓ Service started successfully"
            systemctl status "$SERVICE_NAME" --no-pager -l
        else
            log_error "✗ Service failed to start"
            systemctl status "$SERVICE_NAME" --no-pager -l
            journalctl -u "$SERVICE_NAME" --no-pager -l
        fi
    fi
}

# Show post-installation information
show_info() {
    cat << EOF

========================================================================
                    INSTALLATION COMPLETED
========================================================================

Service Name:       $SERVICE_NAME
Installation Dir:   $INSTALL_DIR
Service User:       $SERVICE_USER
Configuration:      $INSTALL_DIR/.env

MANAGEMENT COMMANDS:
    Start service:      systemctl start $SERVICE_NAME
    Stop service:       systemctl stop $SERVICE_NAME
    Restart service:    systemctl restart $SERVICE_NAME
    Check status:       systemctl status $SERVICE_NAME
    View logs:          journalctl -u $SERVICE_NAME -f
    
IMPORTANT NEXT STEPS:
1. Review and update configuration: $INSTALL_DIR/.env
2. Configure your trading API keys
3. Set up monitoring and alerting
4. Test the service: systemctl start $SERVICE_NAME

LOGS LOCATION:
    System logs:        journalctl -u $SERVICE_NAME
    Application logs:   $INSTALL_DIR/logs/
    
SECURITY:
    - Service runs as user: $SERVICE_USER
    - Configuration files have secure permissions
    - Firewall rules configured (if UFW is enabled)

========================================================================

EOF
}

# Main installation function
main() {
    log_info "Starting Crypto ML Trading System installation"
    
    check_root
    create_user
    install_application
    install_dependencies
    install_service
    configure_logging
    configure_firewall
    create_config
    install_monitoring
    verify_installation
    start_service
    show_info
    
    log_info "Installation completed successfully!"
}

# Execute main function
main "$@"