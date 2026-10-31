# scripts/dev_setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "=== 1. Checking Infrastructure (Docker & WSL) ===" -ForegroundColor Cyan
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker Desktop is not running or WSL2 is not enabled. Start Docker Desktop and re-run."
}

Write-Host "=== 2. Starting Postgres (pgvector) & Redis ===" -ForegroundColor Cyan
docker compose up -d

Write-Host "Waiting for Postgres to accept connections..." -ForegroundColor Yellow
$retries = 30
while ($retries -gt 0) {
    docker exec er-postgres pg_isready -U postgres > $null 2>&1
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 1
    $retries--
}
if ($retries -eq 0) { Write-Error "Postgres failed to initialize in time." }

Write-Host "=== 3. Checking Python Virtual Environment ===" -ForegroundColor Cyan
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

Write-Host "=== 4. Syncing Dependencies ===" -ForegroundColor Cyan
pip install -e ".[dev,ml]"

Write-Host "=== 5. Running Database Migrations ===" -ForegroundColor Cyan
alembic upgrade head

Write-Host "=== 6. Seeding Reference Pincodes ===" -ForegroundColor Cyan
$pincodeCsv = "data/all_india_pincodes.csv"
if (Test-Path $pincodeCsv) {
    python scripts/seed_pincodes.py --csv-path $pincodeCsv
} else {
    Write-Host "Pincode CSV not found at $pincodeCsv. Skipping seed." -ForegroundColor Yellow
}

Write-Host "=== 7. Executing Full Test Suite ===" -ForegroundColor Cyan
pytest -v

Write-Host "=== Pipeline Ready! ===" -ForegroundColor Green