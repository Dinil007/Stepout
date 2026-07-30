"""
StepOut AI Football Analytics Platform - Main Entry Point

Re-exports the FastAPI application instance from app.api.main to support
running the server via `uvicorn app.main:app --reload` as well as `uvicorn app.api.main:app --reload`.
"""

from app.api.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
