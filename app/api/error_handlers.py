"""
Global Error Handlers
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from app.api.logging_config import logger


class AppException(Exception):
    """Base application exception."""
    def __init__(self, status_code: int, detail: str, headers: dict = None):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers
        super().__init__(self.detail)


async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions."""
    logger.error(f"Application error: {exc.detail}", extra={"path": request.url.path})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "application_error"}},
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc.errors()}", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "type": "validation_error",
                "details": exc.errors(),
            }
        },
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle database errors."""
    logger.error(f"Database error: {str(exc)}", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": "Database error", "type": "database_error"}},
    )


async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    logger.warning(f"Not found: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": {"message": exc.detail, "type": "not_found"}},
    )


async def internal_error_handler(request: Request, exc: Exception):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(exc)}", extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": "Internal server error", "type": "internal_error"}},
    )