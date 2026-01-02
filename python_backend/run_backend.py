#!/usr/bin/env python3
"""
Run the simple backend API server
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env file before starting server

import uvicorn

if __name__ == "__main__":
    # Run the API with Socket.IO support
    uvicorn.run(
        "python_backend.main:socket_app",
        host="0.0.0.0",
        port=8002,
        reload=True
    )