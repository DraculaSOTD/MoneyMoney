from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from jose import jwt
from jose.exceptions import JWTError
import bcrypt
import os
import logging
from python_backend.database.models import AdminUser, SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/auth", tags=["admin-auth"])
security = HTTPBearer()

# Admin JWT secret - uses same secret as Node.js for unified auth
ADMIN_JWT_SECRET = os.getenv("JWT_SECRET") or os.getenv("ADMIN_JWT_SECRET", "admin_trading_platform_secret_2024_change_in_prod")
ADMIN_JWT_ALGORITHM = "HS256"
ADMIN_ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# DEBUG: Log the actual JWT secret being used
logger.warning("=" * 80)
logger.warning("JWT SECRET DIAGNOSTIC AT MODULE LOAD:")
logger.warning(f"  JWT_SECRET from env: {os.getenv('JWT_SECRET')[:20] if os.getenv('JWT_SECRET') else 'NOT SET'}...")
logger.warning(f"  ADMIN_JWT_SECRET from env: {os.getenv('ADMIN_JWT_SECRET')[:20] if os.getenv('ADMIN_JWT_SECRET') else 'NOT SET'}...")
logger.warning(f"  ADMIN_JWT_SECRET being used (first 20 chars): {ADMIN_JWT_SECRET[:20]}...")
logger.warning(f"  ADMIN_JWT_SECRET length: {len(ADMIN_JWT_SECRET)} chars")
logger.warning(f"  Expected: 'tradingdashboard_jwt...' (57 chars)")
logger.warning(f"  Using fallback secret: {ADMIN_JWT_SECRET == 'admin_trading_platform_secret_2024_change_in_prod'}")
logger.warning("=" * 80)

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
    """Create a JWT token for admin authentication using Node.js-compatible format"""
    expires_at = datetime.utcnow() + timedelta(minutes=ADMIN_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "id": admin_id,           # Node.js format: uses "id" not "sub"
        "email": username,        # Node.js format: uses "email" not "username"
        "role": "admin",          # Node.js format: uses "role" not "is_superuser"
        "type": "admin",          # Token type identifier
        "exp": expires_at,
        "iat": datetime.utcnow()
    }
    token = jwt.encode(payload, ADMIN_JWT_SECRET, algorithm=ADMIN_JWT_ALGORITHM)
    return token, expires_at

def verify_admin_token(token: str) -> dict:
    """Verify and decode admin JWT token"""
    try:
        # DEBUG: Log token verification attempt
        logger.info(f"[TOKEN-VERIFY] Attempting to verify token (first 30 chars): {token[:30]}...")
        logger.info(f"[TOKEN-VERIFY] Using secret (first 20 chars): {ADMIN_JWT_SECRET[:20]}...")
        logger.info(f"[TOKEN-VERIFY] Secret length: {len(ADMIN_JWT_SECRET)} chars")

        payload = jwt.decode(token, ADMIN_JWT_SECRET, algorithms=[ADMIN_JWT_ALGORITHM])

        logger.info(f"[TOKEN-VERIFY] Successfully decoded token payload: {payload}")

        # Verify it's an admin token
        if payload.get("type") != "admin":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        return payload
    except jwt.ExpiredSignatureError as e:
        logger.error(f"[TOKEN-VERIFY] Token expired: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except JWTError as e:
        logger.error(f"[TOKEN-VERIFY] JWT Error type: {type(e).__name__}")
        logger.error(f"[TOKEN-VERIFY] JWT Error message: {str(e)}")
        logger.error(f"[TOKEN-VERIFY] JWT Error args: {e.args}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated admin"""
    token = credentials.credentials
    payload = verify_admin_token(token)

    # Get admin from python_backend.database
    # Token uses Node.js format with "id" field
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).filter(AdminUser.id == int(payload["id"])).first()
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

    # Get admin from python_backend.database
    # Token uses Node.js format with "id" field
    db = SessionLocal()
    try:
        admin = db.query(AdminUser).filter(AdminUser.id == int(payload["id"])).first()
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
        # Find admin by username OR email (allows email-based login)
        admin = db.query(AdminUser).filter(
            (AdminUser.username == request.username) | (AdminUser.email == request.username)
        ).first()

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
