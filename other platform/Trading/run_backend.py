#!/usr/bin/env python3
"""
Run the backend API server
"""
import uvicorn

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )