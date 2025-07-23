"""
FastAPI application factory and configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import memory, klines, health, agent, config
from .core.config import app_config


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Memory Agent API",
        description="REST API for Memory Agent with LangGraph integration",
        version="1.0.0"
    )

    # Enable CORS middleware
    if app_config["web_server"]["cors_enabled"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include routers
    app.include_router(health.router)
    app.include_router(memory.router)
    app.include_router(klines.router)
    app.include_router(agent.router)
    app.include_router(config.router)

    return app
