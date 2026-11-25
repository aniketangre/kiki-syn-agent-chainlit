# start.ps1 — Start the PostgreSQL container then launch the Chainlit app.
# Usage: .\start.ps1

$projectDir = $PSScriptRoot

# Step 1: Ensure the rag_postgres container is running.
# Use 'docker ps -q' to check — returns the container ID if running, empty if not.
$runningId = docker ps -q --filter "name=kiki_postgres" --filter "status=running" 2>$null

if ($runningId) {
    Write-Host "kiki_postgres is already running." -ForegroundColor Green
} else {
    Write-Host "Starting kiki_postgres container..." -ForegroundColor Cyan
    docker compose -f "$projectDir\docker-compose.yml" up -d

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to start Docker container. Is Docker Desktop running?" -ForegroundColor Red
        exit 1
    }

    Write-Host "kiki_postgres is ready." -ForegroundColor Green
}

# Step 2: Check if the KIKI API is reachable.
# Reads KIKI_API_URL from .env if present; falls back to the default.
$envFile = "$projectDir\.env"
$kikiapiUrl = "http://127.0.0.1:8000"
if (Test-Path $envFile) {
    $match = Select-String -Path $envFile -Pattern "^\s*KIKI_API_URL\s*=\s*(.+)" | Select-Object -First 1
    if ($match) {
        $kikiapiUrl = $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'")
    }
}

Write-Host "Checking KIKI API at $kikiapiUrl ..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "$kikiapiUrl/docs" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "KIKI API is reachable (HTTP $($response.StatusCode))." -ForegroundColor Green
} catch {
    Write-Host "WARNING: KIKI API is NOT reachable at $kikiapiUrl" -ForegroundColor Yellow
    Write-Host "         The agent will start, but kiki_recommend calls will fail until the API is running." -ForegroundColor Yellow
}

# Step 3: Launch the Chainlit app
Write-Host "Launching Chainlit app..." -ForegroundColor Cyan
Set-Location $projectDir
& ".\.venv\Scripts\chainlit.exe" run app.py --port 8080
