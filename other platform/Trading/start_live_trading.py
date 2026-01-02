#!/usr/bin/env python3

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
import uvicorn
from multiprocessing import Process
import signal
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from live_trading_integration import LiveTradingIntegration
from api.main import app

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(signum, frame):
    global shutdown_flag
    print("\nReceived shutdown signal. Stopping gracefully...")
    shutdown_flag = True

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server in a separate process"""
    uvicorn.run(app, host=host, port=port, log_level="info")

async def run_trading_system(config_path: str, demo_mode: bool = False):
    """Run the main trading system"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trading_system.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Create integration
    integration = LiveTradingIntegration(config_path)
    
    try:
        # Initialize
        logger.info("Initializing trading system...")
        await integration.initialize()
        
        # Print status
        status = integration.get_status()
        logger.info(f"System status: {status}")
        
        if demo_mode:
            logger.info("Running in DEMO mode - no real trades will be executed")
            
        # Start trading loop
        logger.info("Starting trading loop...")
        trading_task = asyncio.create_task(integration.start_trading_loop())
        
        # Monitor for shutdown
        while not shutdown_flag:
            await asyncio.sleep(1)
            
            # Periodic status update
            if int(time.time()) % 60 == 0:  # Every minute
                metrics = integration.execution_engine.get_metrics() if integration.execution_engine else {}
                if metrics:
                    logger.info(f"Trading metrics: {metrics}")
                    
    except Exception as e:
        logger.error(f"Trading system error: {e}", exc_info=True)
    finally:
        logger.info("Shutting down trading system...")
        await integration.shutdown()

def check_environment():
    """Check required environment variables and configuration"""
    required_env = {
        'BINANCE_API_KEY': 'Binance API key',
        'BINANCE_API_SECRET': 'Binance API secret',
        'API_TOKEN': 'API authentication token',
        'MASTER_PASSWORD': 'Master password for key encryption (if using file/keyring storage)'
    }
    
    optional_env = {
        'BINANCE_TESTNET': 'Use Binance testnet (true/false)',
        'REDIS_URL': 'Redis URL for caching',
        'DATABASE_URL': 'PostgreSQL database URL',
        'TWITTER_API_KEY': 'Twitter API key for sentiment',
        'NEWS_API_KEY': 'News API key'
    }
    
    missing_required = []
    for env_var, description in required_env.items():
        if not os.getenv(env_var):
            missing_required.append(f"{env_var} - {description}")
            
    if missing_required:
        print("Missing required environment variables:")
        for var in missing_required:
            print(f"  - {var}")
        print("\nPlease set these environment variables or create a .env file")
        return False
        
    print("Environment check passed!")
    print("\nOptional environment variables:")
    for env_var, description in optional_env.items():
        value = os.getenv(env_var)
        if value:
            if 'SECRET' in env_var or 'PASSWORD' in env_var:
                print(f"  ✓ {env_var} is set")
            else:
                print(f"  ✓ {env_var} = {value}")
        else:
            print(f"  ✗ {env_var} - {description}")
            
    return True

def create_example_env_file():
    """Create an example .env file"""
    example_content = """# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true  # Set to false for real trading

# API Authentication
API_TOKEN=your-secret-token-here

# Security
MASTER_PASSWORD=your_master_password_here

# Optional: Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading

# Optional: Redis
REDIS_URL=redis://localhost:6379/0

# Optional: Alternative Data Sources
TWITTER_API_KEY=
TWITTER_API_SECRET=
TWITTER_BEARER_TOKEN=
NEWS_API_KEY=

# Optional: Monitoring
SLACK_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
    
    with open('.env.example', 'w') as f:
        f.write(example_content)
        
    print("Created .env.example file. Copy it to .env and fill in your values.")

def main():
    parser = argparse.ArgumentParser(description='Crypto ML Trading System')
    parser.add_argument('--config', default='config/system_config.yaml',
                       help='Path to system configuration file')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (no real trades)')
    parser.add_argument('--no-api', action='store_true',
                       help='Do not start the API server')
    parser.add_argument('--api-host', default='0.0.0.0',
                       help='API server host')
    parser.add_argument('--api-port', type=int, default=8000,
                       help='API server port')
    parser.add_argument('--check-env', action='store_true',
                       help='Check environment variables and exit')
    parser.add_argument('--create-env', action='store_true',
                       help='Create example .env file')
    
    args = parser.parse_args()
    
    if args.create_env:
        create_example_env_file()
        return
        
    if args.check_env:
        check_environment()
        return
        
    # Load environment variables from .env file if it exists
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
        
    # Check environment
    if not check_environment():
        print("\nUse --create-env to create an example .env file")
        sys.exit(1)
        
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start API server in separate process if not disabled
    api_process = None
    if not args.no_api:
        print(f"Starting API server on {args.api_host}:{args.api_port}")
        api_process = Process(target=run_api_server, args=(args.api_host, args.api_port))
        api_process.start()
        time.sleep(2)  # Give API time to start
        
    try:
        # Run trading system
        print(f"Starting trading system with config: {args.config}")
        asyncio.run(run_trading_system(args.config, args.demo))
        
    finally:
        # Clean up API process
        if api_process:
            print("Stopping API server...")
            api_process.terminate()
            api_process.join(timeout=5)
            if api_process.is_alive():
                api_process.kill()

if __name__ == "__main__":
    main()