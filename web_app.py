#!/usr/bin/env python3
"""
REST API for Memory Agent - Refactored Modular Version

This is the new modular version of the Memory Agent API that uses
a clean separation of concerns with the api package structure.
"""

from api.app import create_app
from api.startup import startup

# Create the FastAPI application
app = create_app()

# Add startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    startup()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "web_app_new:app",
        host="0.0.0.0",
        port=5001,
        log_level="info",
        reload=False
    )
