from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, Dict
from datetime import datetime
from .base import Base


class UserBase(SQLModel):
    email: str = Field(..., max_length=255, unique=True)
    name: str = Field(..., max_length=255)
    profile: Dict = Field(default={})


class User(UserBase, Base, table=True):
    __tablename__ = "users"

    # Relationships
    user_progresses: List["UserProgress"] = Relationship(back_populates="user")


class UserCreate(UserBase):
    password: str


class UserUpdate(SQLModel):
    email: Optional[str] = None
    name: Optional[str] = None
    profile: Optional[Dict] = None