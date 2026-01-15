"""
Admin Management Module
Handles admin user operations, permissions, and security
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List
import bcrypt
from database.models import AdminUser, SessionLocal

logger = logging.getLogger(__name__)

class AdminManager:
    """Manage admin users and permissions"""

    @staticmethod
    def create_default_admin():
        """Create default admin user if none exists"""
        db = SessionLocal()
        try:
            # Check if any admin exists
            admin_count = db.query(AdminUser).count()

            if admin_count == 0:
                # Create default superuser
                default_admin = AdminUser(
                    username="admin",
                    email="admin@trading.com",
                    password_hash=AdminManager.hash_password("admin123"),  # CHANGE IN PRODUCTION!
                    full_name="Default Admin",
                    is_superuser=True,
                    is_active=True
                )

                db.add(default_admin)
                db.commit()

                logger.warning(
                    "DEFAULT ADMIN CREATED: email='admin@trading.com', password='admin123' - "
                    "CHANGE PASSWORD IMMEDIATELY IN PRODUCTION!"
                )

                return True
            return False
        except Exception as e:
            logger.error(f"Error creating default admin: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False

    @staticmethod
    def get_admin_by_username(username: str) -> Optional[AdminUser]:
        """Get admin by username"""
        db = SessionLocal()
        try:
            return db.query(AdminUser).filter(AdminUser.username == username).first()
        finally:
            db.close()

    @staticmethod
    def get_admin_by_email(email: str) -> Optional[AdminUser]:
        """Get admin by email"""
        db = SessionLocal()
        try:
            return db.query(AdminUser).filter(AdminUser.email == email).first()
        finally:
            db.close()

    @staticmethod
    def get_admin_by_id(admin_id: int) -> Optional[AdminUser]:
        """Get admin by ID"""
        db = SessionLocal()
        try:
            return db.query(AdminUser).filter(AdminUser.id == admin_id).first()
        finally:
            db.close()

    @staticmethod
    def update_admin_password(admin_id: int, new_password: str) -> bool:
        """Update admin password"""
        db = SessionLocal()
        try:
            admin = db.query(AdminUser).filter(AdminUser.id == admin_id).first()
            if not admin:
                return False

            admin.password_hash = AdminManager.hash_password(new_password)
            admin.updated_at = datetime.utcnow()

            db.commit()

            logger.info(f"Password updated for admin: {admin.username}")
            return True
        except Exception as e:
            logger.error(f"Error updating password: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    @staticmethod
    def deactivate_admin(admin_id: int) -> bool:
        """Deactivate an admin account"""
        db = SessionLocal()
        try:
            admin = db.query(AdminUser).filter(AdminUser.id == admin_id).first()
            if not admin:
                return False

            admin.is_active = False
            admin.session_token = None
            admin.session_expires = None
            admin.updated_at = datetime.utcnow()

            db.commit()

            logger.info(f"Admin deactivated: {admin.username}")
            return True
        except Exception as e:
            logger.error(f"Error deactivating admin: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    @staticmethod
    def reactivate_admin(admin_id: int) -> bool:
        """Reactivate an admin account"""
        db = SessionLocal()
        try:
            admin = db.query(AdminUser).filter(AdminUser.id == admin_id).first()
            if not admin:
                return False

            admin.is_active = True
            admin.failed_login_attempts = 0
            admin.locked_until = None
            admin.updated_at = datetime.utcnow()

            db.commit()

            logger.info(f"Admin reactivated: {admin.username}")
            return True
        except Exception as e:
            logger.error(f"Error reactivating admin: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    @staticmethod
    def unlock_admin(admin_id: int) -> bool:
        """Unlock a locked admin account"""
        db = SessionLocal()
        try:
            admin = db.query(AdminUser).filter(AdminUser.id == admin_id).first()
            if not admin:
                return False

            admin.locked_until = None
            admin.failed_login_attempts = 0
            admin.updated_at = datetime.utcnow()

            db.commit()

            logger.info(f"Admin unlocked: {admin.username}")
            return True
        except Exception as e:
            logger.error(f"Error unlocking admin: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    @staticmethod
    def get_active_admins() -> List[AdminUser]:
        """Get all active admin users"""
        db = SessionLocal()
        try:
            return db.query(AdminUser).filter(AdminUser.is_active == True).all()
        finally:
            db.close()

    @staticmethod
    def get_all_admins() -> List[AdminUser]:
        """Get all admin users"""
        db = SessionLocal()
        try:
            return db.query(AdminUser).all()
        finally:
            db.close()

    @staticmethod
    def check_admin_permissions(admin_id: int, required_superuser: bool = False) -> bool:
        """Check if admin has required permissions"""
        admin = AdminManager.get_admin_by_id(admin_id)

        if not admin:
            return False

        if not admin.is_active:
            return False

        if admin.locked_until and admin.locked_until > datetime.utcnow():
            return False

        if required_superuser and not admin.is_superuser:
            return False

        return True

    @staticmethod
    def log_admin_action(admin_id: int, action: str, details: str = None):
        """Log admin action for audit trail"""
        logger.info(f"Admin Action - ID: {admin_id}, Action: {action}, Details: {details}")

    @staticmethod
    def invalidate_all_sessions(admin_id: int) -> bool:
        """Invalidate all sessions for an admin (force logout)"""
        db = SessionLocal()
        try:
            admin = db.query(AdminUser).filter(AdminUser.id == admin_id).first()
            if not admin:
                return False

            admin.session_token = None
            admin.session_expires = None
            db.commit()

            logger.info(f"All sessions invalidated for admin: {admin.username}")
            return True
        except Exception as e:
            logger.error(f"Error invalidating sessions: {e}")
            db.rollback()
            return False
        finally:
            db.close()


# Initialize default admin on module import (for development)
def init_admin_system():
    """Initialize admin system - create default admin if needed"""
    try:
        AdminManager.create_default_admin()
    except Exception as e:
        logger.error(f"Error initializing admin system: {e}")


# Call initialization (will only run once when module is imported)
# In production, you might want to call this explicitly from your startup script
# init_admin_system()
