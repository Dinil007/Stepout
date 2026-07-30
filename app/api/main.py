"""
FastAPI Main Application
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.config import settings
from app.api.database import init_db
from app.api.logging_config import logger, get_logger
from app.api.error_handlers import (
    AppException,
    app_exception_handler,
    validation_exception_handler,
    sqlalchemy_exception_handler,
    not_found_handler,
    internal_error_handler,
)
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

# Import routers
from app.api.routers import auth, matches, players, teams, reports, season, admin

# Create logger
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Football Analytics API")
    init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Football Analytics API")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Enterprise Football Analytics REST API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Register exception handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(404, not_found_handler)
app.add_exception_handler(500, internal_error_handler)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(matches.router, prefix="/api/v1")
app.include_router(players.router, prefix="/api/v1")
app.include_router(teams.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(season.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": settings.VERSION
    }


@app.get("/health", tags=["Health"])
async def health_check_detailed():
    """Detailed health check."""
    from app.api.database import engine
    
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "database": db_status,
        "redis": "not_checked"  # TODO: Add Redis health check
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )