# Zora One-Click Sharing Script
# Starts the FastAPI backend and the Ngrok tunnel.

Write-Host "--- ZORA SHARED BACKEND STARTUP ---" -ForegroundColor Cyan

# 1. Kill any existing processes on port 8000
Write-Host "[1/3] Clearing port 8000..."
$proc = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($proc) {
    Stop-Process -Id $proc.OwningProcess -Force -ErrorAction SilentlyContinue
}

# 1.5 Data Integrity Check
Write-Host "`n[1.5/3] Checking Multi-Modal Datasets..." -ForegroundColor Cyan
$missingData = $false
if (-not (Test-Path "c:\Users\Gaman a rai\Desktop\zora ml\zora\data\networks\protein_links.txt.gz")) { 
    Write-Host "[!] WARNING: Local STRING-DB network file missing. Falling back to API mode." -ForegroundColor Yellow
    $missingData = $true
}
if (-not (Test-Path "c:\Users\Gaman a rai\Desktop\zora ml\zora\data\structures")) { 
    Write-Host "[!] WARNING: Local AlphaFold PDB folder missing. Falling back to API mode." -ForegroundColor Yellow
    $missingData = $true
}
if ($missingData) {
    Write-Host "Hybrid-Local intelligence is PARTIAL. System will still function via external APIs.`n" -ForegroundColor DarkGray
} else {
    Write-Host "Success: All local biological intelligence datasets detected (Hybrid-Local ACTIVE).`n" -ForegroundColor Green
}

# 2. Start FastAPI Backend in the background
Write-Host "[2/3] Starting Zora Backend..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    cd "c:\Users\Gaman a rai\Desktop\zora ml\zora"
    $env:PYTHONPATH = ".;.."
    .\venv311\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
}

# Wait for backend to warm up
Start-Sleep -Seconds 5

# 3. Start Ngrok Tunnel
Write-Host "[3/3] Launching Ngrok Tunnel..." -ForegroundColor Yellow
$ngrokProc = Start-Process -FilePath "c:\Users\Gaman a rai\Desktop\zora ml\ngrok\ngrok.exe" -ArgumentList "http 8000 --log=stdout" -NoNewWindow -PassThru

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "ZORA IS LIVE!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "1. Wait a few seconds for the tunnel to initialize."
Write-Host "2. Go to: https://dashboard.ngrok.com/tunnels/agents"
Write-Host "3. Copy the URL and send it to your friend!"
Write-Host "------------------------------------------------`n"

Write-Host "Press Ctrl+C to stop sharing."
try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    Write-Host "`nStopping Zora Sharing..." -ForegroundColor Red
    Stop-Process -Id $ngrokProc.Id -Force -ErrorAction SilentlyContinue
    Stop-Job $backendJob -ErrorAction SilentlyContinue
}
