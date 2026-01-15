from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from jose import jwt
import bcrypt
import os
import logging
from database.models import AdminUser, SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])
security = HTTPBearer()

# Admin JWT secret - should match Node.js JWT_SECRET for cross-platform auth
# Priority: ADMIN_JWT_SECRET > JWT_SECRET > default (matching Node.js default)
ADMIN_JWT_SECRET = os.getenv("ADMIN_JWT_SECRET") or os.getenv("JWT_SECRET") or "tradingdashboard_jwt_secret_key_2024_change_in_production"
ADMIN_JWT_ALGORITHM = "HS256"
ADMIN_ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str
    is_superuser: bool
    expires_at: datetime

class AdminCreateRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    is_superuser: bool = False

class AdminInfoResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    last_login: Optional[datetime]
    created_at: datetime

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_admin_token(admin_id: int, username: str, is_superuser: bool) -> tuple[str, datetime]:
    """Create a JWT token for admin authentication"""
    expires_at = datetime.utcnow() + timedelta(minutes=ADMIN_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(admin_id),
        "username": username,
        "is_superuser": is_superuser,
        "exp": expires_at,
        "iat": datetime.utcnow(),
        "type": "admin"  # Important: mark as admin token
    }
    token = jwt.encode(payload, ADMIN_JWT_SECRET, algorithm=ADMIN_JWT_ALGORITHM)
    return token, expires_at

def verify_admin_token(token: str) -> dict:
    """Verify and decode admin JWT token"""
    try:
        # Try Python backend format first (ADMIN_JWT_SECRET)
        try:
            payload = jwt.decode(token, ADMIN_JWT_SECRET, algorithms=[ADMIN_JWT_ALGORITHM])
        except jwt.JWTError:
            # Try Node.js backend format (JWT_SECRET from .env)
            node_jwt_secret = os.getenv("JWT_SECRET", "tradingdashboard_jwt_secret_key_2024_change_in_production")
            payload = jwt.decode(token, node_jwt_secret, algorithms=[ADMIN_JWT_ALGORITHM])

        # Verify it's an admin token
        if payload.get("type") != "admin":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        # Normalize payload format: convert 'id' to 'sub' if needed (Node.js format)
        if "id" in payload and "sub" not in payload:
            payload["sub"] = str(payload["id"])

        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated admin"""
    token = credentials.credentials
    payload = verify_admin_token(token)

    # Get admin from database by email (works across Node.js and Python user tables)
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).filter(AdminUser.email == payload["email"]).first()
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin not found"
            )
        if not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is inactive"
            )

        # Check if account is locked
        if admin.locked_until and admin.locked_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is temporarily locked"
            )

        return admin
    finally:
        db.close()


async def get_current_admin_from_token(token: str):
    """Get admin from JWT token string (for WebSocket authentication)"""
    payload = verify_admin_token(token)

    # Get admin from database by email (works across Node.js and Python user tables)
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).filter(AdminUser.email == payload["email"]).first()
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin not found"
            )
        if not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is inactive"
            )

        # Check if account is locked
        if admin.locked_until and admin.locked_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is temporarily locked"
            )

        return admin
    finally:
        db.close()


async def get_superuser_admin(admin: AdminUser = Depends(get_current_admin)):
    """Dependency to ensure admin is a superuser"""
    if not admin.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required"
        )
    return admin

@router.post("/login", response_model=AdminLoginResponse)
async def admin_login(request: AdminLoginRequest):
    """Admin login endpoint"""
    db = SessionLocal()
    try:
        # Find admin by username
        admin = db.query(AdminUser).filter(AdminUser.username == request.username).first()

        if not admin:
            # Don't reveal whether username exists
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Check if account is locked
        if admin.locked_until and admin.locked_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is temporarily locked. Please try again later."
            )

        # Verify password
        if not verify_password(request.password, admin.password_hash):
            # Increment failed login attempts
            admin.failed_login_attempts += 1

            # Lock account after 5 failed attempts
            if admin.failed_login_attempts >= 5:
                admin.locked_until = datetime.utcnow() + timedelta(minutes=30)
                logger.warning(f"Admin account locked: {admin.username}")

            db.commit()

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Check if account is active
        if not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin account is inactive"
            )

        # Reset failed login attempts
        admin.failed_login_attempts = 0
        admin.last_login = datetime.utcnow()

        # Create token
        token, expires_at = create_admin_token(admin.id, admin.username, admin.is_superuser)

        # Store session token
        admin.session_token = token
        admin.session_expires = expires_at

        db.commit()

        logger.info(f"Admin logged in: {admin.username}")

        return AdminLoginResponse(
            access_token=token,
            username=admin.username,
            email=admin.email,
            is_superuser=admin.is_superuser,
            expires_at=expires_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )
    finally:
        db.close()

@router.post("/logout")
async def admin_logout(admin: AdminUser = Depends(get_current_admin)):
    """Admin logout endpoint"""
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).filter(AdminUser.id == admin.id).first()
        admin.session_token = None
        admin.session_expires = None
        db.commit()

        logger.info(f"Admin logged out: {admin.username}")

        return {"message": "Logged out successfully"}
    finally:
        db.close()

@router.get("/verify", response_model=AdminInfoResponse)
async def verify_admin(admin: AdminUser = Depends(get_current_admin)):
    """Verify admin token and return admin info"""
    return AdminInfoResponse(
        id=admin.id,
        username=admin.username,
        email=admin.email,
        full_name=admin.full_name,
        is_active=admin.is_active,
        is_superuser=admin.is_superuser,
        last_login=admin.last_login,
        created_at=admin.created_at
    )

@router.post("/create", response_model=AdminInfoResponse)
async def create_admin(
    request: AdminCreateRequest,
    current_admin: AdminUser = Depends(get_superuser_admin)
):
    """Create a new admin user (superuser only)"""
    db = SessionLocal()
    try:
        # Check if username or email already exists
        existing = db.query(AdminUser).filter(
            (AdminUser.username == request.username) | (AdminUser.email == request.email)
        ).first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already exists"
            )

        # Create new admin
        admin = AdminUser(
            username=request.username,
            email=request.email,
            password_hash=hash_password(request.password),
            full_name=request.full_name,
            is_superuser=request.is_superuser,
            is_active=True,
            created_by=current_admin.id
        )

        db.add(admin)
        db.commit()
        db.refresh(admin)

        logger.info(f"New admin created: {admin.username} by {current_admin.username}")

        return AdminInfoResponse(
            id=admin.id,
            username=admin.username,
            email=admin.email,
            full_name=admin.full_name,
            is_active=admin.is_active,
            is_superuser=admin.is_superuser,
            last_login=admin.last_login,
            created_at=admin.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin creation error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create admin"
        )
    finally:
        db.close()

@router.get("/list", response_model=list[AdminInfoResponse])
async def list_admins(current_admin: AdminUser = Depends(get_superuser_admin)):
    """List all admin users (superuser only)"""
    db = SessionLocal()
    try:
        admins = db.query(AdminUser).all()
        return [
            AdminInfoResponse(
                id=admin.id,
                username=admin.username,
                email=admin.email,
                full_name=admin.full_name,
                is_active=admin.is_active,
                is_superuser=admin.is_superuser,
                last_login=admin.last_login,
                created_at=admin.created_at
            )
            for admin in admins
        ]
    finally:
        db.close()
