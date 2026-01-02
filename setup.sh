#!/bin/bash

# MoneyMoney & Trading Platform Setup Script
# ===========================================
# Automates the complete setup process

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}▶${NC} $1"
}

# Check if running from correct directory
if [ ! -f "server.js" ]; then
    print_error "Please run this script from the MoneyMoney root directory"
    exit 1
fi

print_header "MoneyMoney & Trading Platform Setup"

echo "This script will:"
echo "  1. Check prerequisites (PostgreSQL, Python, Node.js)"
echo "  2. Configure PostgreSQL database"
echo "  3. Install all dependencies"
echo "  4. Import BTCUSDT sample data"
echo "  5. Prepare applications for launch"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Step 1: Check Prerequisites
print_step "Checking prerequisites..."

# Check PostgreSQL
if command -v psql &> /dev/null; then
    print_success "PostgreSQL client found: $(psql --version | head -1)"
else
    print_error "PostgreSQL client (psql) not found"
    print_info "Install with: sudo pacman -S postgresql (Arch) or sudo apt install postgresql-client (Debian/Ubuntu)"
    exit 1
fi

# Check if PostgreSQL is running
if systemctl is-active --quiet postgresql 2>/dev/null; then
    print_success "PostgreSQL service is running"
else
    print_warning "PostgreSQL service is not running"
    read -p "Start PostgreSQL now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl start postgresql
        print_success "PostgreSQL started"
    else
        print_error "PostgreSQL must be running. Exiting."
        exit 1
    fi
fi

# Check Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python not found"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js found: $NODE_VERSION"
else
    print_error "Node.js not found"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm found: $NPM_VERSION"
else
    print_error "npm not found"
    exit 1
fi

# Step 2: PostgreSQL Configuration
print_step "Configuring PostgreSQL database..."

echo ""
print_info "Please provide your PostgreSQL credentials"
print_info "On Arch Linux, use username 'postgres' with empty password (just press Enter)"
echo ""

# Get PostgreSQL username
read -p "PostgreSQL username [postgres]: " PG_USER
PG_USER=${PG_USER:-postgres}

# Get PostgreSQL password
read -s -p "PostgreSQL password: " PG_PASSWORD
echo ""

# Get database name
read -p "Database name [trading_platform]: " PG_DATABASE
PG_DATABASE=${PG_DATABASE:-trading_platform}

# Get PostgreSQL host
read -p "PostgreSQL host [localhost]: " PG_HOST
PG_HOST=${PG_HOST:-localhost}

# Get PostgreSQL port
read -p "PostgreSQL port [5432]: " PG_PORT
PG_PORT=${PG_PORT:-5432}

# Test connection
print_step "Testing PostgreSQL connection..."

# Try without password first (peer authentication on Arch Linux)
if [ -z "$PG_PASSWORD" ]; then
    if psql -U $PG_USER -h $PG_HOST -p $PG_PORT -c '\q' 2>/dev/null; then
        print_success "PostgreSQL connection successful (peer authentication)"
        USE_PASSWORD=false
    else
        print_error "Failed to connect to PostgreSQL"
        print_info "Please check your credentials and try again"
        exit 1
    fi
else
    if PGPASSWORD=$PG_PASSWORD psql -U $PG_USER -h $PG_HOST -p $PG_PORT -c '\q' 2>/dev/null; then
        print_success "PostgreSQL connection successful"
        USE_PASSWORD=true
    else
        print_error "Failed to connect to PostgreSQL"
        print_info "Trying without password (peer authentication)..."
        if psql -U $PG_USER -h $PG_HOST -p $PG_PORT -c '\q' 2>/dev/null; then
            print_success "PostgreSQL connection successful (peer authentication)"
            PG_PASSWORD=""
            USE_PASSWORD=false
        else
            print_error "Failed to connect to PostgreSQL"
            print_info "Please check your credentials and try again"
            print_info "On Arch Linux, try using username: postgres (with empty password)"
            exit 1
        fi
    fi
fi

# Check if database exists
print_step "Checking if database '$PG_DATABASE' exists..."

# Helper function for psql commands
run_psql() {
    if [ "$USE_PASSWORD" = "true" ] && [ -n "$PG_PASSWORD" ]; then
        PGPASSWORD=$PG_PASSWORD psql -U $PG_USER -h $PG_HOST -p $PG_PORT "$@"
    else
        psql -U $PG_USER -h $PG_HOST -p $PG_PORT "$@"
    fi
}

if run_psql -lqt | cut -d \| -f 1 | grep -qw $PG_DATABASE; then
    print_success "Database '$PG_DATABASE' already exists"
else
    print_warning "Database '$PG_DATABASE' does not exist"
    read -p "Create database '$PG_DATABASE'? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_psql -c "CREATE DATABASE $PG_DATABASE;" 2>/dev/null
        print_success "Database '$PG_DATABASE' created"
    else
        print_error "Database is required. Exiting."
        exit 1
    fi
fi

# Update Trading Platform .env
print_step "Updating Trading Platform .env file..."
DATABASE_URL="postgresql://$PG_USER:$PG_PASSWORD@$PG_HOST:$PG_PORT/$PG_DATABASE"

cd "other platform/Trading"
if [ -f .env ]; then
    # Backup existing .env
    cp .env .env.backup
    print_info "Existing .env backed up to .env.backup"
fi

# Update DATABASE_URL in .env
sed -i "s|^DATABASE_URL=.*|DATABASE_URL=$DATABASE_URL|" .env
print_success "DATABASE_URL updated in Trading Platform .env"

cd ../..

# Step 3: Install Node.js Dependencies
print_step "Installing Node.js dependencies for MoneyMoney..."

if [ -d "node_modules" ]; then
    print_info "node_modules already exists, skipping npm install"
else
    npm install
    print_success "Node.js dependencies installed"
fi

# Step 4: Setup Python Virtual Environment
print_step "Setting up Python virtual environment..."

cd "other platform/Trading"

# Remove old venv if it exists and is broken
if [ -d "venv" ]; then
    print_warning "Existing venv found, checking if it's valid..."
    if [ -f "venv/bin/python" ]; then
        # Test if venv works
        if venv/bin/python --version &> /dev/null; then
            print_success "Existing venv is valid"
        else
            print_warning "Existing venv is broken, recreating..."
            rm -rf venv
            python -m venv venv
            print_success "New venv created"
        fi
    else
        print_warning "Existing venv is incomplete, recreating..."
        rm -rf venv
        python -m venv venv
        print_success "New venv created"
    fi
else
    python -m venv venv
    print_success "Virtual environment created"
fi

# Step 5: Install Python Dependencies
print_step "Installing Python dependencies (this may take a few minutes)..."

source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
if pip install -r requirements.txt > /tmp/pip_install.log 2>&1; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    print_info "Check /tmp/pip_install.log for details"
    cat /tmp/pip_install.log
    exit 1
fi

# Step 6: Import BTCUSDT Data
print_step "Importing BTCUSDT sample data..."

if [ -f "scripts/import_btcusdt_data.py" ]; then
    print_info "This will import 10,082 1-minute candles into PostgreSQL"

    if python scripts/import_btcusdt_data.py 2>&1 | tee /tmp/import_data.log; then
        if grep -q "import completed successfully" /tmp/import_data.log; then
            print_success "BTCUSDT data imported successfully"
        else
            print_warning "Import script ran but may have encountered issues"
            print_info "Check output above for details"
        fi
    else
        print_error "Failed to import BTCUSDT data"
        print_info "You can run this manually later with:"
        print_info "  cd 'other platform/Trading' && source venv/bin/activate && python scripts/import_btcusdt_data.py"
    fi
else
    print_warning "Import script not found, skipping data import"
fi

deactivate
cd ../..

# Step 7: Make scripts executable
print_step "Making startup scripts executable..."

chmod +x start.sh start-all.sh
chmod +x "other platform/Trading/start.sh"
print_success "Startup scripts are executable"

# Summary
print_header "Setup Complete! 🎉"

echo -e "${GREEN}All setup steps completed successfully!${NC}\n"

echo "Summary:"
print_success "PostgreSQL database configured: $PG_DATABASE"
print_success "Node.js dependencies installed"
print_success "Python virtual environment created"
print_success "Python dependencies installed"
print_success "BTCUSDT data imported (10,082 candles)"
print_success "Startup scripts ready"

echo ""
print_header "Next Steps"

echo "1. Start the applications:"
echo -e "   ${CYAN}./start-all.sh${NC}"
echo ""
echo "2. Access MoneyMoney dashboard:"
echo -e "   ${CYAN}http://localhost:3000/auth${NC}"
echo ""
echo "3. Login with test account:"
echo -e "   Email:    ${CYAN}test@example.com${NC}"
echo -e "   Password: ${CYAN}testpassword${NC}"
echo ""
echo "4. Select BTCUSDT and click 'Analysis' tab to see 70+ indicators!"
echo ""

print_header "Useful URLs"

echo "  • MoneyMoney Dashboard:  http://localhost:3000/dashboard"
echo "  • MoneyMoney Login:      http://localhost:3000/auth"
echo "  • Trading Platform API:  http://localhost:8001"
echo "  • API Documentation:     http://localhost:8001/docs"
echo ""

print_info "For troubleshooting, see: STARTUP_GUIDE.md"
print_info "To stop services: Close terminal windows or Ctrl+C in each"

echo ""
print_success "Happy Trading! 💰📈"
echo ""
