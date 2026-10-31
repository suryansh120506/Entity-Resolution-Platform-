# scripts/stop.ps1 — stop the API, the pipeline worker and the web app.
#
#   powershell -ExecutionPolicy Bypass -File scripts\stop.ps1            # leaves Postgres/Redis running
#   powershell -ExecutionPolicy Bypass -File scripts\stop.ps1 -Containers  # stops them too (data is kept)
#
# Only processes that belong to this project are stopped: whatever listens on
# ports 8000 and 3000, and python processes running this project's Celery app.

param([switch]$Containers)

$root = Split-Path -Parent $PSScriptRoot

function Stop-Port($port, $label) {
    $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    if (-not $owners) { Write-Host "$label : not running"; return }
    foreach ($processId in $owners) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "$label : stopped" -ForegroundColor Green
}

Stop-Port 8000 "API"
Stop-Port 3000 "Web app"

$workers = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*celery*app.workers.celery_app*worker*" }
if ($workers) {
    $workers | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "Worker : stopped" -ForegroundColor Green
} else {
    Write-Host "Worker : not running"
}

if ($Containers) {
    Set-Location $root
    docker compose stop
    Write-Host "Postgres and Redis : stopped (data kept)" -ForegroundColor Green
}
