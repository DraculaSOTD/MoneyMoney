#!/bin/bash

# MoneyMoney Startup Script
# ==========================
# Automatically kills existing processes on ports 3000 and 8001 before starting

echo "💰 Starting MoneyMoney..."
echo ""

# Function to kill process on a specific port
kill_port() {
    local port=$1
    local service_name=$2

    # Use ss to find PIDs (works on Arch Linux and most modern systems)
    local pids=$(ss -tulpn 2>/dev/null | grep ":$port" | grep -oP 'pid=\K[0-9]+' | sort -u)

    if [ -n "$pids" ]; then
        echo "⚠️  Port $port is in use by $service_name"
        for pid in $pids; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "   Killing process $pid..."
                kill $pid 2>/dev/null
                sleep 1

                # Check if still running, force kill if needed
                if ps -p $pid > /dev/null 2>&1; then
                    echo "   Force killing process $pid..."
                    kill -9 $pid 2>/dev/null
                    sleep 0.5
                fi
            fi
        done
        echo "   ✅ Port $port freed"
    fi
}

# Kill existing processes on required ports
echo "🧹 Cleaning up existing processes..."
kill_port 8001 "Python Backend"
kill_port 3000 "Node.js Frontend"
echo ""

# Give processes time to fully terminate
sleep 1

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✏️  Please edit .env with your actual configuration before running again."
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📥 Installing dependencies..."
    npm install
fi

# Check if database directory exists
if [ ! -d "database" ]; then
    echo "📁 Creating database directory..."
    mkdir -p database
fi

# Start the server
echo "🌐 Starting MoneyMoney server on http://localhost:3000"
echo "🏠 Landing page: http://localhost:3000"
echo "🔐 Login page: http://localhost:3000/auth"
echo "📊 Dashboard: http://localhost:3000/dashboard"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

node server.js
