#!/bin/bash

# Trading Platform Startup Script
# ================================

echo "🚀 Starting Trading Platform..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env with your actual configuration before running again."
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if dependencies are installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "🌐 Starting FastAPI server on http://localhost:8001"
echo "📊 API Documentation available at http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run with uvicorn from project root for proper module resolution
python -m uvicorn api.main_simple:app --host 0.0.0.0 --port 8001
