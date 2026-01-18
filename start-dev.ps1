<#
.SYNOPSIS
    Start Development Servers for TradingDashboard

.DESCRIPTION
    Starts all services required for local development:
    - PostgreSQL database service
    - Trading Platform FastAPI backend (port 8001)
    - MoneyMoney Node.js frontend (port 3000)

.PARAMETER NoFrontend
    Skip starting the frontend server

.PARAMETER NoBackend
    Skip starting the backend server

.PARAMETER Attach
    Run servers in foreground (no new windows)

.EXAMPLE
    .\start-dev.ps1
    Starts all services in separate terminal windows.

.EXAMPLE
    .\start-dev.ps1 -NoFrontend
    Starts only the backend API server.
#>

param(
    [switch]$NoFrontend,
    [switch]$NoBackend,
    [switch]$Attach
)

# ============================================================================
# Configuration
# ============================================================================
$ErrorActionPreference = "Continue"

$ScriptRoot = $PSScriptRoot
$TradingPlatformPath = Join-Path $ScriptRoot "other platform\Trading"
$VenvPath = Join-Path $TradingPlatformPath "venv"
$VenvActivate = Join-Path $VenvPath "Scripts\Activate.ps1"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

$FrontendPort = 3000
$BackendPort = 8001

# Colors for output
function Write-Header { param($Message) Write-Host "`n$Message" -ForegroundColor Cyan }
function Write-Success { param($Message) Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor White }

# ============================================================================
# Banner
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  TradingDashboard Development Servers" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

# ============================================================================
# Pre-flight Checks
# ============================================================================
Write-Header "Running pre-flight checks..."

# Check .env files
$rootEnvPath = Join-Path $ScriptRoot ".env"
$tradingEnvPath = Join-Path $TradingPlatformPath ".env"

$envMissing = $false
if (-not (Test-Path $rootEnvPath)) {
    Write-Warning "Missing .env file at project root"
    Write-Info "  Run: Copy-Item .env.example .env"
    $envMissing = $true
}
if (-not (Test-Path $tradingEnvPath)) {
    Write-Warning "Missing .env file in Trading Platform"
    Write-Info "  Run: Copy-Item `"$TradingPlatformPath\.env.example`" `"$tradingEnvPath`""
    $envMissing = $true
}

if ($envMissing) {
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Info "Please create .env files from .env.example templates first."
        exit 1
    }
}

# Check virtual environment
if (-not (Test-Path $VenvPython)) {
    Write-Error "Python virtual environment not found at $VenvPath"
    Write-Info "Run .\setup-windows.ps1 first to set up the environment."
    exit 1
}
Write-Success "Virtual environment found"

# Check Node.js
$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCmd -and -not $NoFrontend) {
    Write-Error "npm not found. Please install Node.js or use -NoFrontend flag."
    exit 1
}
if (-not $NoFrontend) {
    Write-Success "Node.js found"
}

# ============================================================================
# Check PostgreSQL Service
# ============================================================================
Write-Header "Checking PostgreSQL..."

$pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pgService) {
    if ($pgService.Status -ne "Running") {
        Write-Info "Starting PostgreSQL service..."
        try {
            Start-Service $pgService.Name -ErrorAction Stop
            Start-Sleep -Seconds 2
            Write-Success "PostgreSQL service started"
        } catch {
            Write-Warning "Could not start PostgreSQL service. You may need admin rights."
            Write-Info "Run: Start-Service $($pgService.Name)"
        }
    } else {
        Write-Success "PostgreSQL is running"
    }
} else {
    Write-Warning "PostgreSQL service not found. Ensure PostgreSQL is installed and running."
}

# ============================================================================
# Check for running processes on ports
# ============================================================================
function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

if (-not $NoBackend -and (Test-PortInUse -Port $BackendPort)) {
    Write-Warning "Port $BackendPort is already in use"
    $kill = Read-Host "Kill existing process? (y/N)"
    if ($kill -eq "y" -or $kill -eq "Y") {
        $proc = Get-NetTCPConnection -LocalPort $BackendPort | Select-Object -First 1
        Stop-Process -Id $proc.OwningProcess -Force
        Start-Sleep -Seconds 1
    }
}

if (-not $NoFrontend -and (Test-PortInUse -Port $FrontendPort)) {
    Write-Warning "Port $FrontendPort is already in use"
    $kill = Read-Host "Kill existing process? (y/N)"
    if ($kill -eq "y" -or $kill -eq "Y") {
        $proc = Get-NetTCPConnection -LocalPort $FrontendPort | Select-Object -First 1
        Stop-Process -Id $proc.OwningProcess -Force
        Start-Sleep -Seconds 1
    }
}

# ============================================================================
# Start Backend Server
# ============================================================================
if (-not $NoBackend) {
    Write-Header "Starting Trading Platform API (Backend)..."

    $backendCommand = @"
Set-Location '$TradingPlatformPath'
Write-Host 'Activating virtual environment...' -ForegroundColor Cyan
& '$VenvActivate'
Write-Host 'Starting FastAPI server on port $BackendPort...' -ForegroundColor Cyan
Write-Host 'API Docs: http://localhost:$BackendPort/docs' -ForegroundColor Green
Write-Host ''
python -m uvicorn api.main_simple:app --host 0.0.0.0 --port $BackendPort --reload
"@

    if ($Attach) {
        Write-Info "Backend will start in this terminal (use Ctrl+C to stop)"
    } else {
        # Start in new terminal window
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand -WindowStyle Normal
        Write-Success "Backend server starting in new window (port $BackendPort)"
        Start-Sleep -Seconds 2
    }
}

# ============================================================================
# Start Frontend Server
# ============================================================================
if (-not $NoFrontend) {
    Write-Header "Starting MoneyMoney Frontend..."

    $frontendCommand = @"
Set-Location '$ScriptRoot'
Write-Host 'Starting Node.js server on port $FrontendPort...' -ForegroundColor Cyan
Write-Host 'Frontend: http://localhost:$FrontendPort' -ForegroundColor Green
Write-Host ''
npm run dev
"@

    if ($Attach) {
        Write-Info "Use a separate terminal for frontend, or run with -NoBackend flag"
    } else {
        # Start in new terminal window
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand -WindowStyle Normal
        Write-Success "Frontend server starting in new window (port $FrontendPort)"
        Start-Sleep -Seconds 2
    }
}

# ============================================================================
# Summary
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  Development Servers Started" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Magenta

Write-Host ""
Write-Host "Available URLs:" -ForegroundColor Cyan

if (-not $NoFrontend) {
    Write-Host "  Frontend:     http://localhost:$FrontendPort" -ForegroundColor White
    Write-Host "  Admin Panel:  http://localhost:$FrontendPort/admin" -ForegroundColor White
}

if (-not $NoBackend) {
    Write-Host "  API Docs:     http://localhost:$BackendPort/docs" -ForegroundColor White
    Write-Host "  API Base:     http://localhost:$BackendPort" -ForegroundColor White
}

Write-Host ""
Write-Host "Tips:" -ForegroundColor Yellow
Write-Host "  - Close the terminal windows to stop the servers"
Write-Host "  - Check server logs in their respective terminal windows"
Write-Host "  - Backend auto-reloads on Python file changes"
Write-Host "  - Frontend auto-reloads via nodemon"

if ($Attach -and -not $NoBackend) {
    Write-Host ""
    Write-Host "Starting backend in foreground..." -ForegroundColor Cyan
    Write-Host ""

    Push-Location $TradingPlatformPath
    try {
        & $VenvActivate
        python -m uvicorn api.main_simple:app --host 0.0.0.0 --port $BackendPort --reload
    } finally {
        Pop-Location
    }
}

Write-Host ""
