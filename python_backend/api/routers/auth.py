"""
User Authentication Module
===========================

Provides JWT authentication for user-facing endpoints (MoneyMoney integration).
Separate from admin authentication (admin_auth.py).
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()

# User JWT secret (different from admin JWT)
USER_JWT_SECRET = os.getenv("JWT_SECRET", "your_super_secret_jwt_key_change_this_in_production_2024")
USER_JWT_ALGORITHM = "HS256"


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency for user authentication.

    Validates JWT token from Authorization header and returns user info.
    Used by user-facing endpoints (MoneyMoney integration).

    Args:
        credentials: HTTP Bearer token from Authorization header

    Returns:
        dict: User information from JWT payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials

    try:
        # Decode and validate JWT token
        payload = jwt.decode(token, USER_JWT_SECRET, algorithms=[USER_JWT_ALGORITHM])

        user_id = payload.get("id")
        user_email = payload.get("email")

        if user_id is None or user_email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Return user info from token
        return {
            "id": user_id,
            "email": user_email
        }

    except JWTError as e:
        logger.error(f"JWT validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
