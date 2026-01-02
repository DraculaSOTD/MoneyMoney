#!/bin/bash

# Start All Services Script
# ==========================
# Starts Trading Platform Backend, MoneyMoney Landing Page, and Admin Frontend
# Automatically kills any existing processes on required ports

echo "🚀 Starting All Services..."
echo ""

# Check if we're in the correct directory
if [ ! -f "server.js" ]; then
    echo "❌ Error: Please run this script from the MoneyMoney root directory"
    exit 1
fi

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

# Detect terminal emulator (prioritize modern terminals)
if command -v kitty > /dev/null; then
    TERMINAL="kitty"
elif command -v alacritty > /dev/null; then
    TERMINAL="alacritty"
elif command -v wezterm > /dev/null; then
    TERMINAL="wezterm"
elif command -v terminator > /dev/null; then
    TERMINAL="terminator"
elif command -v tilix > /dev/null; then
    TERMINAL="tilix"
elif command -v gnome-terminal > /dev/null; then
    TERMINAL="gnome-terminal"
elif command -v konsole > /dev/null; then
    TERMINAL="konsole"
elif command -v xterm > /dev/null; then
    TERMINAL="xterm"
else
    echo "❌ No supported terminal emulator found"
    echo "   Supported: kitty, alacritty, wezterm, terminator, tilix, gnome-terminal, konsole, xterm"
    echo ""
    echo "   Alternative: Run directly without multiple terminals:"
    echo "   node server.js"
    echo ""
    echo "   Or use tmux/screen for multiplexing"
    exit 1
fi

echo "📡 Starting Trading Platform (Backend)..."
if [ "$TERMINAL" = "kitty" ]; then
    kitty --title="Trading Platform" -e bash -c "cd 'other platform/Trading' && ./start.sh; exec bash" &
elif [ "$TERMINAL" = "alacritty" ]; then
    alacritty --title "Trading Platform" -e bash -c "cd 'other platform/Trading' && ./start.sh; exec bash" &
elif [ "$TERMINAL" = "wezterm" ]; then
    wezterm start --class "Trading Platform" -- bash -c "cd 'other platform/Trading' && ./start.sh; exec bash" &
elif [ "$TERMINAL" = "terminator" ]; then
    terminator --title="Trading Platform" -e "bash -c \"cd 'other platform/Trading' && ./start.sh; exec bash\"" &
elif [ "$TERMINAL" = "tilix" ]; then
    tilix --title="Trading Platform" -e "bash -c \"cd 'other platform/Trading' && ./start.sh; exec bash\"" &
elif [ "$TERMINAL" = "gnome-terminal" ]; then
    gnome-terminal --title="Trading Platform" -- bash -c "cd 'other platform/Trading' && ./start.sh; exec bash"
elif [ "$TERMINAL" = "konsole" ]; then
    konsole --title "Trading Platform" -e bash -c "cd 'other platform/Trading' && ./start.sh; exec bash" &
elif [ "$TERMINAL" = "xterm" ]; then
    xterm -title "Trading Platform" -e bash -c "cd 'other platform/Trading' && ./start.sh; exec bash" &
fi

sleep 3

echo "💰 Starting MoneyMoney (Landing Page)..."
if [ "$TERMINAL" = "kitty" ]; then
    kitty --title="MoneyMoney" -e bash -c "./start.sh; exec bash" &
elif [ "$TERMINAL" = "alacritty" ]; then
    alacritty --title "MoneyMoney" -e bash -c "./start.sh; exec bash" &
elif [ "$TERMINAL" = "wezterm" ]; then
    wezterm start --class "MoneyMoney" -- bash -c "./start.sh; exec bash" &
elif [ "$TERMINAL" = "terminator" ]; then
    terminator --title="MoneyMoney" -e "bash -c \"./start.sh; exec bash\"" &
elif [ "$TERMINAL" = "tilix" ]; then
    tilix --title="MoneyMoney" -e "bash -c \"./start.sh; exec bash\"" &
elif [ "$TERMINAL" = "gnome-terminal" ]; then
    gnome-terminal --title="MoneyMoney" -- bash -c "./start.sh; exec bash"
elif [ "$TERMINAL" = "konsole" ]; then
    konsole --title "MoneyMoney" -e bash -c "./start.sh; exec bash" &
elif [ "$TERMINAL" = "xterm" ]; then
    xterm -title "MoneyMoney" -e bash -c "./start.sh; exec bash" &
fi

sleep 3

echo ""
echo "✅ All services started!"
echo ""
echo "🌐 Services:"
echo "   - Trading Platform API: http://localhost:8001"
echo "   - Trading Platform Docs: http://localhost:8001/docs"
echo "   - MoneyMoney Landing: http://localhost:3000"
echo "   - Admin Dashboard: http://localhost:3000/admin/dashboard"
echo "   - User Dashboard: http://localhost:3000/dashboard"
echo ""
echo "🔑 Login at: http://localhost:3000/auth"
echo "   Use your email and password to login"
echo ""
echo "⏹️  To stop all services, close the terminal windows or press Ctrl+C in each"
