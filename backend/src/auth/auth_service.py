from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import Session, select
from pydantic import BaseModel

from ..models.user import User
from ..database.database import get_db
from ..config.settings import settings
from .password import verify_password

# Security scheme for API docs
security = HTTPBearer()

class TokenData(BaseModel):
    username: Optional[str] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create access token with expiration
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


async def authenticate_user(db: Session, email: str, password: str):
    """
    Authenticate user by email and password
    """
    # Find user by email
    statement = select(User).where(User.email == email)
    result = db.exec(statement)
    user = result.first()

    if not user or not verify_password(password, user.profile.get("hashed_password", "")):
        return False

    return user


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    Get current user from token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    statement = select(User).where(User.email == token_data.username)
    result = db.exec(statement)
    user = result.first()

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """
    Get current active user
    """
    # In a real application, you might check if the user is active
    # For now, we'll just return the user
    return current_user