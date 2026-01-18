#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Windows Setup Script for TradingDashboard Platform

.DESCRIPTION
    Complete environment setup for the TradingDashboard project including:
    - Chocolatey package manager
    - Python 3.11, Node.js LTS, PostgreSQL 15, Git
    - CUDA 12.1 for RTX 3060 GPU support
    - Python virtual environment with all dependencies
    - Node.js dependencies
    - PostgreSQL database configuration
    - Environment file generation

.NOTES
    Target GPU: NVIDIA RTX 3060
    Requires: Administrator privileges
#>

param(
    [switch]$SkipChocolatey,
    [switch]$SkipCuda,
    [switch]$SkipPostgresSetup,
    [switch]$SkipDatabaseInit,
    [string]$PostgresPassword = ""
)

# ============================================================================
# Configuration
# ============================================================================
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptRoot = $PSScriptRoot
$TradingPlatformPath = Join-Path $ScriptRoot "other platform\Trading"
$VenvPath = Join-Path $TradingPlatformPath "venv"

$DatabaseName = "trading_platform"
$DatabaseUser = "postgres"
$DefaultJwtSecret = "tradingdashboard_jwt_secret_key_2024_change_in_production"

# Colors for output
function Write-Step { param($Message) Write-Host "`n[STEP] $Message" -ForegroundColor Cyan }
function Write-Success { param($Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor White }

# ============================================================================
# Phase 1: Admin Check & Chocolatey
# ============================================================================
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  TradingDashboard Windows Setup Script" -ForegroundColor Magenta
Write-Host "  Target: NVIDIA RTX 3060 + CUDA 12.1" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta

Write-Step "Phase 1: Checking prerequisites..."

# Verify admin rights
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator!"
    Write-Info "Right-click PowerShell and select 'Run as Administrator'"
    exit 1
}
Write-Success "Running with Administrator privileges"

# Install Chocolatey if not present
if (-not $SkipChocolatey) {
    Write-Step "Checking Chocolatey installation..."

    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Write-Info "Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

        # Refresh environment
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        Write-Success "Chocolatey installed successfully"
    } else {
        Write-Success "Chocolatey is already installed"
    }
}

# ============================================================================
# Phase 2: Install System Prerequisites via Chocolatey
# ============================================================================
Write-Step "Phase 2: Installing system prerequisites..."

$packages = @(
    @{ Name = "python311"; DisplayName = "Python 3.11" },
    @{ Name = "nodejs-lts"; DisplayName = "Node.js LTS" },
    @{ Name = "postgresql15"; DisplayName = "PostgreSQL 15" },
    @{ Name = "git"; DisplayName = "Git" }
)

foreach ($pkg in $packages) {
    Write-Info "Checking $($pkg.DisplayName)..."
    $installed = choco list --local-only $pkg.Name 2>$null | Select-String $pkg.Name

    if (-not $installed) {
        Write-Info "Installing $($pkg.DisplayName)..."
        choco install $pkg.Name -y --no-progress
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to install $($pkg.DisplayName). You may need to install it manually."
        } else {
            Write-Success "$($pkg.DisplayName) installed"
        }
    } else {
        Write-Success "$($pkg.DisplayName) is already installed"
    }
}

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# ============================================================================
# Phase 3: Install CUDA Toolkit
# ============================================================================
if (-not $SkipCuda) {
    Write-Step "Phase 3: Installing CUDA Toolkit 12.1..."

    # Check if CUDA is already installed
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    if (Test-Path $cudaPath) {
        Write-Success "CUDA 12.1 is already installed"
    } else {
        Write-Info "Installing CUDA 12.1 (this may take a while)..."
        choco install cuda --version=12.1.0 -y --no-progress

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "CUDA installation via Chocolatey failed."
            Write-Info "Please install CUDA 12.1 manually from: https://developer.nvidia.com/cuda-12-1-0-download-archive"
        } else {
            Write-Success "CUDA 12.1 installed successfully"
        }
    }

    # Verify NVIDIA driver
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        Write-Info "GPU Status:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    } else {
        Write-Warning "nvidia-smi not found. Ensure NVIDIA drivers are installed."
    }
} else {
    Write-Info "Skipping CUDA installation (--SkipCuda flag set)"
}

# ============================================================================
# Phase 4: Create Python Virtual Environment
# ============================================================================
Write-Step "Phase 4: Setting up Python virtual environment..."

# Find Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    # Try Python 3.11 specific paths
    $possiblePaths = @(
        "C:\Python311\python.exe",
        "C:\Program Files\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    )
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $pythonCmd = $path
            break
        }
    }
}

if (-not $pythonCmd) {
    Write-Error "Python not found! Please ensure Python 3.11 is installed and in PATH."
    exit 1
}

Write-Info "Using Python: $pythonCmd"

# Create virtual environment
Push-Location $TradingPlatformPath
try {
    if (-not (Test-Path $VenvPath)) {
        Write-Info "Creating virtual environment..."
        & $pythonCmd -m venv venv
        Write-Success "Virtual environment created at $VenvPath"
    } else {
        Write-Success "Virtual environment already exists"
    }
} finally {
    Pop-Location
}

# ============================================================================
# Phase 5: Install Python Dependencies
# ============================================================================
Write-Step "Phase 5: Installing Python dependencies..."

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
$venvPip = Join-Path $VenvPath "Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment Python not found at $venvPython"
    exit 1
}

Push-Location $TradingPlatformPath
try {
    # Upgrade pip first
    Write-Info "Upgrading pip..."
    & $venvPython -m pip install --upgrade pip

    # Install PyTorch with CUDA 12.1 support (for RTX 3060)
    Write-Info "Installing PyTorch with CUDA 12.1 support..."
    & $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install requirements from main requirements.txt
    if (Test-Path "requirements.txt") {
        Write-Info "Installing from requirements.txt..."
        & $venvPip install -r requirements.txt
    }

    # Install requirements from crypto_ml_trading
    $mlRequirements = Join-Path $TradingPlatformPath "crypto_ml_trading\requirements.txt"
    if (Test-Path $mlRequirements) {
        Write-Info "Installing from crypto_ml_trading/requirements.txt..."
        & $venvPip install -r $mlRequirements
    }

    # Install additional ML/DL dependencies not in requirements.txt
    Write-Info "Installing additional ML dependencies..."
    $additionalPackages = @(
        "gym",           # Reinforcement learning environments
        "h5py",          # HDF5 large dataset support
        "GPUtil",        # GPU monitoring
        "scikit-learn"   # ML utilities/metrics
    )

    foreach ($pkg in $additionalPackages) {
        Write-Info "  Installing $pkg..."
        & $venvPip install $pkg
    }

    Write-Success "All Python dependencies installed"

    # Verify PyTorch CUDA
    Write-Info "Verifying PyTorch CUDA support..."
    $cudaCheck = & $venvPython -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
    Write-Info $cudaCheck

} finally {
    Pop-Location
}

# ============================================================================
# Phase 6: Install Node.js Dependencies
# ============================================================================
Write-Step "Phase 6: Installing Node.js dependencies..."

Push-Location $ScriptRoot
try {
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        Write-Warning "npm not found. Please ensure Node.js is installed and in PATH."
    } else {
        Write-Info "Running npm install..."
        npm install

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Node.js dependencies installed"
        } else {
            Write-Warning "npm install reported errors. Check the output above."
        }
    }
} finally {
    Pop-Location
}

# ============================================================================
# Phase 7: Configure PostgreSQL
# ============================================================================
if (-not $SkipPostgresSetup) {
    Write-Step "Phase 7: Configuring PostgreSQL..."

    # Start PostgreSQL service
    $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pgService) {
        if ($pgService.Status -ne "Running") {
            Write-Info "Starting PostgreSQL service..."
            Start-Service $pgService.Name
            Start-Sleep -Seconds 3
        }
        Write-Success "PostgreSQL service is running"
    } else {
        Write-Warning "PostgreSQL service not found. You may need to start it manually."
    }

    # Find psql
    $psqlPath = Get-Command psql -ErrorAction SilentlyContinue
    if (-not $psqlPath) {
        $possiblePsqlPaths = @(
            "C:\Program Files\PostgreSQL\15\bin\psql.exe",
            "C:\Program Files\PostgreSQL\14\bin\psql.exe"
        )
        foreach ($path in $possiblePsqlPaths) {
            if (Test-Path $path) {
                $psqlPath = $path
                break
            }
        }
    }

    if ($psqlPath) {
        Write-Info "Creating database '$DatabaseName'..."

        # Create database (ignore error if exists)
        $createDbResult = & $psqlPath -U $DatabaseUser -c "CREATE DATABASE $DatabaseName;" 2>&1
        if ($createDbResult -match "already exists") {
            Write-Success "Database '$DatabaseName' already exists"
        } elseif ($LASTEXITCODE -eq 0) {
            Write-Success "Database '$DatabaseName' created"
        } else {
            Write-Warning "Could not create database. You may need to create it manually."
            Write-Info "Run: CREATE DATABASE $DatabaseName;"
        }
    } else {
        Write-Warning "psql not found. Please configure PostgreSQL manually."
    }
} else {
    Write-Info "Skipping PostgreSQL setup (--SkipPostgresSetup flag set)"
}

# ============================================================================
# Phase 8: Generate .env Files
# ============================================================================
Write-Step "Phase 8: Generating .env files..."

# Generate a random JWT secret for production (optional)
function New-JwtSecret {
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return [Convert]::ToBase64String($bytes)
}

# Root .env file (Node.js MoneyMoney frontend)
$rootEnvPath = Join-Path $ScriptRoot ".env"
$rootEnvExamplePath = Join-Path $ScriptRoot ".env.example"

if (-not (Test-Path $rootEnvPath)) {
    if (Test-Path $rootEnvExamplePath) {
        Write-Info "Creating .env from .env.example (root)..."
        Copy-Item $rootEnvExamplePath $rootEnvPath
        Write-Success "Created $rootEnvPath"
    } else {
        Write-Warning ".env.example not found at root. Skipping."
    }
} else {
    Write-Success ".env already exists at root"
}

# Trading Platform .env file
$tradingEnvPath = Join-Path $TradingPlatformPath ".env"
$tradingEnvExamplePath = Join-Path $TradingPlatformPath ".env.example"

if (-not (Test-Path $tradingEnvPath)) {
    if (Test-Path $tradingEnvExamplePath) {
        Write-Info "Creating .env from .env.example (Trading Platform)..."
        Copy-Item $tradingEnvExamplePath $tradingEnvPath
        Write-Success "Created $tradingEnvPath"
    } else {
        Write-Warning ".env.example not found for Trading Platform. Skipping."
    }
} else {
    Write-Success ".env already exists for Trading Platform"
}

Write-Info "IMPORTANT: Review and update your .env files with production values!"

# ============================================================================
# Phase 9: Initialize Database
# ============================================================================
if (-not $SkipDatabaseInit) {
    Write-Step "Phase 9: Initializing database..."

    # Check for initialization scripts
    $initScriptPath = Join-Path $TradingPlatformPath "database\init.sql"
    $alembicPath = Join-Path $TradingPlatformPath "alembic.ini"

    if (Test-Path $alembicPath) {
        Write-Info "Running Alembic migrations..."
        Push-Location $TradingPlatformPath
        try {
            & $venvPython -m alembic upgrade head
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Database migrations complete"
            } else {
                Write-Warning "Alembic migration failed. You may need to run migrations manually."
            }
        } finally {
            Pop-Location
        }
    } elseif (Test-Path $initScriptPath) {
        Write-Info "Running database initialization script..."
        if ($psqlPath) {
            & $psqlPath -U $DatabaseUser -d $DatabaseName -f $initScriptPath
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Database initialized"
            }
        }
    } else {
        Write-Info "No database initialization scripts found. The application will create tables on first run."
    }
} else {
    Write-Info "Skipping database initialization (--SkipDatabaseInit flag set)"
}

# ============================================================================
# Summary
# ============================================================================
Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Magenta

Write-Host "`nInstalled Components:" -ForegroundColor Cyan
Write-Host "  - Python 3.11 with virtual environment"
Write-Host "  - PyTorch with CUDA 12.1 (RTX 3060 support)"
Write-Host "  - Node.js LTS with npm packages"
Write-Host "  - PostgreSQL 15"
Write-Host "  - CUDA Toolkit 12.1"

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "  1. Review and update .env files with your configuration"
Write-Host "  2. Run .\start-dev.ps1 to start the development servers"
Write-Host "  3. Open http://localhost:3000 for the frontend"
Write-Host "  4. Open http://localhost:8001/docs for API documentation"

Write-Host "`nTo verify GPU support:" -ForegroundColor Cyan
Write-Host "  cd `"$TradingPlatformPath`""
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python -c `"import torch; print(torch.cuda.is_available())`""

Write-Host ""
