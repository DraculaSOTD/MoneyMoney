#!/bin/bash

# Start Services in Split Terminal
# ==================================
# Uses tmux to run both services in split panes (one terminal window)

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Trading Dashboard Services${NC}"
echo ""

# Check if running from correct directory
if [ ! -f "server.js" ]; then
    echo -e "${RED}❌ Error: Please run this script from the MoneyMoney root directory${NC}"
    exit 1
fi

# Check if ports are available
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

if check_port 8001; then
    echo -e "${YELLOW}⚠️  Port 8001 is already in use (Trading Platform)${NC}"
    echo -n "   Kill existing process? (y/n) "
    read -r response
    if [ "$response" = "y" ]; then
        kill $(lsof -t -i:8001) 2>/dev/null
        sleep 2
    else
        exit 1
    fi
fi

if check_port 3000; then
    echo -e "${YELLOW}⚠️  Port 3000 is already in use (MoneyMoney)${NC}"
    echo -n "   Kill existing process? (y/n) "
    read -r response
    if [ "$response" = "y" ]; then
        kill $(lsof -t -i:3000) 2>/dev/null
        sleep 2
    else
        exit 1
    fi
fi

# Check if tmux is available
if ! command -v tmux > /dev/null; then
    echo -e "${YELLOW}⚠️  tmux not found. Trying screen...${NC}"

    # Try screen as fallback
    if ! command -v screen > /dev/null; then
        echo -e "${RED}❌ Neither tmux nor screen found${NC}"
        echo ""
        echo "Install one of them:"
        echo "  Arch: sudo pacman -S tmux"
        echo "  Ubuntu/Debian: sudo apt install tmux"
        echo ""
        echo "Or use start-all.sh to launch in separate windows"
        exit 1
    fi

    USE_SCREEN=true
else
    USE_SCREEN=false
fi

SESSION_NAME="trading_dashboard"

if [ "$USE_SCREEN" = true ]; then
    # Using screen
    echo -e "${BLUE}📺 Starting services with screen...${NC}"

    # Kill existing session if it exists
    screen -S $SESSION_NAME -X quit 2>/dev/null || true

    # Create new session
    screen -dmS $SESSION_NAME

    # Split and run commands
    screen -S $SESSION_NAME -X screen -t "Trading Platform" bash -c "cd 'other platform/Trading' && ./start.sh"
    screen -S $SESSION_NAME -X screen -t "MoneyMoney" bash -c "./start.sh"

    echo ""
    echo -e "${GREEN}✅ Services started in screen session: $SESSION_NAME${NC}"
    echo ""
    echo "To attach: screen -r $SESSION_NAME"
    echo "To detach: Press Ctrl+A then D"
    echo "To switch windows: Ctrl+A then N (next) or P (previous)"
    echo "To stop: screen -S $SESSION_NAME -X quit"

else
    # Using tmux
    echo -e "${BLUE}📺 Starting services with tmux...${NC}"

    # Kill existing session if it exists
    tmux kill-session -t $SESSION_NAME 2>/dev/null || true

    # Create new session with first pane (Trading Platform)
    tmux new-session -d -s $SESSION_NAME -n "Services" \
        "cd 'other platform/Trading' && ./start.sh"

    # Split window horizontally and run MoneyMoney
    tmux split-window -h -t $SESSION_NAME \
        "cd '$PWD' && ./start.sh"

    # Adjust pane sizes (50/50 split)
    tmux select-layout -t $SESSION_NAME even-horizontal

    echo ""
    echo -e "${GREEN}✅ Services started in tmux session: $SESSION_NAME${NC}"
    echo ""
    echo "Commands:"
    echo "  Attach to session:  tmux attach -t $SESSION_NAME"
    echo "  Detach from tmux:   Press Ctrl+B then D"
    echo "  Switch panes:       Ctrl+B then arrow keys"
    echo "  Stop services:      tmux kill-session -t $SESSION_NAME"
    echo ""

    # Auto-attach option
    echo -n "Attach to tmux session now? (y/n) "
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        tmux attach -t $SESSION_NAME
    fi
fi

echo ""
echo -e "${BLUE}🌐 Services:${NC}"
echo "   - Trading Platform API: http://localhost:8001"
echo "   - Trading Platform Docs: http://localhost:8001/docs"
echo "   - MoneyMoney: http://localhost:3000"
echo "   - MoneyMoney Dashboard: http://localhost:3000/dashboard"
echo ""
