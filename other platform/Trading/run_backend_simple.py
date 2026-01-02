#!/usr/bin/env python3
"""
Run the simple backend API server
"""
import uvicorn

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "api.main_simple:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )