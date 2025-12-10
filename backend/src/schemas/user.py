from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime


class UserProfile(BaseModel):
    background: Optional[str] = None  # software/hardware background
    preferences: Optional[Dict] = {}  # user preferences like language, difficulty level
    progress: Optional[Dict] = {}  # learning progress tracking


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    profile: UserProfile
    created_at: datetime
    updated_at: datetime


class UserCreateRequest(BaseModel):
    email: str
    name: str
    password: str
    profile: Optional[UserProfile] = UserProfile()


class UserUpdateRequest(BaseModel):
    email: Optional[str] = None
    name: Optional[str] = None
    profile: Optional[UserProfile] = None