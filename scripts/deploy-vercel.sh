#!/bin/bash
# Vercel Deployment Build Script
# This script prepares the Voice Authenticator application for deployment on Vercel

set -e

echo "🔨 Building Voice Authenticator for Vercel deployment..."

# 1. Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install scipy numpy scikit-learn librosa soundfile

# 2. Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# 3. Build frontend
echo "🏗️ Building frontend (Vite)..."
npm run build

# 4. Run TypeScript check
echo "✅ Running TypeScript type check..."
npm run check

# 5. Create database if needed
echo "🗄️ Setting up database..."
sqlite3 sqlite.db "CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    owner TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"

sqlite3 sqlite.db "CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id INTEGER REFERENCES api_keys(id),
    language TEXT NOT NULL,
    classification TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    explanation TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    client_ip TEXT
);"

echo "✅ Setup test API key in database..."
python3 setup_api_key.py

echo "✅ Build complete! Ready for Vercel deployment."
echo ""
echo "📋 Next steps:"
echo "  1. Create Vercel project: vercel link"
echo "  2. Set environment variables (see .env.example)"
echo "  3. Deploy: vercel --prod"
