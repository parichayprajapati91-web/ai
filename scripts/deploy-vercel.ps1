# Vercel Deployment Build Script (Windows)
Write-Host "Building Voice Authenticator for Vercel deployment..." -ForegroundColor Cyan

# 1. Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\Activate.ps1"
pip install --upgrade pip
pip install scipy numpy scikit-learn librosa soundfile

# 2. Install Node.js dependencies
Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
npm install

# 3. Build frontend
Write-Host "Building frontend (Vite)..." -ForegroundColor Yellow
npm run build

# 4. Run TypeScript check
Write-Host "Running TypeScript type check..." -ForegroundColor Yellow
npm run check

# 5. Setup database
Write-Host "Setting up database..." -ForegroundColor Yellow
sqlite3 sqlite.db "CREATE TABLE IF NOT EXISTS api_keys (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE, owner TEXT NOT NULL, is_active BOOLEAN DEFAULT 1, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
sqlite3 sqlite.db "CREATE TABLE IF NOT EXISTS request_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, api_key_id INTEGER REFERENCES api_keys(id), language TEXT NOT NULL, classification TEXT NOT NULL, confidence_score REAL NOT NULL, explanation TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, client_ip TEXT);"

Write-Host "Setting up test API key..." -ForegroundColor Yellow
$pythonExe = ".\.venv\Scripts\python.exe"
& $pythonExe setup_api_key.py

Write-Host "Build complete! Ready for Vercel deployment." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. npm i -g vercel" -ForegroundColor White
Write-Host "  2. vercel link" -ForegroundColor White
Write-Host "  3. Set environment variables" -ForegroundColor White
Write-Host "  4. vercel --prod" -ForegroundColor White
