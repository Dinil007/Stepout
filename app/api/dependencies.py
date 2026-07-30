"""
FastAPI Dependencies for Authentication and Authorization
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.api.database import get_db
from app.api.auth import get_current_user
from app.api.models import User, UserRole
from app.api.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def admin_only(current_user: User = Depends(get_current_active_user)) -> User:
    """Require admin role."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def analyst_or_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Require analyst or admin role."""
    if current_user.role not in [UserRole.ADMIN, UserRole.ANALYST]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst or Admin access required"
        )
    return current_user


async def coach_only(current_user: User = Depends(get_current_active_user)) -> User:
    """Require coach role."""
    if current_user.role != UserRole.COACH:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Coach access required"
        )
    return current_user


async def scout_only(current_user: User = Depends(get_current_active_user)) -> User:
    """Require scout role."""
    if current_user.role != UserRole.SCOUT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Scout access required"
        )
    return current_user


def get_client_ip(request: Request) -> str:
    """Get client IP address."""
    if request.client:
        return request.client.host
    return "unknown"


def get_user_agent(request: Request) -> str:
    """Get user agent string."""
    return request.headers.get("user-agent", "unknown")