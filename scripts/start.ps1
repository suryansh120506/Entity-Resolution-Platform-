# scripts/start.ps1 — start everything the platform needs, in one command.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start.ps1
#
# Starts, in order, and skips anything already running:
#   1. Docker Desktop (if its engine is not up) and the Postgres + Redis containers
#   2. Database migrations
#   3. The API (port 8000), the pipeline worker, and the web app (port 3000),
#      each in its own window so its log is visible and it can be stopped alone
# then opens http://localhost:3000/dashboard.
#
# Stop everything with scripts\stop.ps1.

# "Continue", not "Stop": in Windows PowerShell 5.1, redirecting a native
# command's stderr (`docker info *> $null`) under "Stop" throws even when the
# command succeeds. Failures are checked explicitly and `throw`n instead.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "    $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    $text" -ForegroundColor Yellow }

function Test-Port($port) {
    [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Until($seconds, [scriptblock]$condition) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $condition) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

# --- Environment -----------------------------------------------------------------
# Alembic reads MIGRATION_DATABASE_URL from the process environment, and every
# window started below inherits this one's, so .env is loaded once, here.
if (-not (Test-Path ".env")) { throw ".env not found in $root. Copy .env.example to .env first." }
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim().Trim('"'), "Process")
    }
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python venv not found at .venv. Run scripts\dev_setup.ps1 first." }

# --- 1. Docker -------------------------------------------------------------------
Step "Docker"
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    )
    $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exe) { throw "Docker Desktop is not running and could not be found. Start it manually." }
    Warn "Docker engine not running - starting Docker Desktop (this can take a minute)..."
    Start-Process -FilePath $exe
    $up = Wait-Until 180 { docker info *> $null; $LASTEXITCODE -eq 0 }
    if (-not $up) { throw "Docker Desktop did not start within 3 minutes." }
}
Ok "Docker engine running"

docker compose up -d | Out-Null
$ready = Wait-Until 60 { docker exec er-postgres pg_isready *> $null; $LASTEXITCODE -eq 0 }
if (-not $ready) { throw "Postgres did not become ready. Check: docker compose logs postgres" }
Ok "Postgres and Redis running"

# --- 2. Migrations ---------------------------------------------------------------
Step "Database migrations"
& $python -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { throw "Migrations failed." }
Ok "Schema up to date"

# --- 3. Services -----------------------------------------------------------------
function Start-Service-Window($title, $command) {
    # -NoExit keeps the window (and its log) open if the service stops.
    $full = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location '$root'; $command"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $full | Out-Null
}

Step "API (port 8000)"
if (Test-Port 8000) {
    Ok "already running"
} else {
    # SelectorEventLoop: psycopg's async driver cannot run on Windows'
    # default Proactor loop, and uvicorn ignores the loop *policy* setting.
    Start-Service-Window "ER API" "& '$python' -m uvicorn app.main:app --port 8000 --log-level warning --loop asyncio:SelectorEventLoop"
    if (Wait-Until 60 { Test-Port 8000 }) { Ok "started" } else { Warn "not listening yet - check the 'ER API' window" }
}

Step "Pipeline worker"
$workerRunning = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*app.queue.worker*" }
if ($workerRunning) {
    Ok "already running"
} else {
    # The Postgres-backed queue worker (Celery and Redis were removed).
    Start-Service-Window "ER Worker" "& '$python' -m app.queue.worker --concurrency 4"
    Ok "started"
}

Step "Web app (port 3000)"
if (Test-Port 3000) {
    Ok "already running"
} else {
    Start-Service-Window "ER Web" "Set-Location '$root\frontend'; npm run dev"
    if (Wait-Until 90 { Test-Port 3000 }) { Ok "started" } else { Warn "not listening yet - check the 'ER Web' window" }
}

Step "Opening http://localhost:3000/dashboard"
Start-Process "http://localhost:3000/dashboard"
Write-Host ""
Write-Host "All set. The status indicator at the top of the page shows if anything stops." -ForegroundColor Green
