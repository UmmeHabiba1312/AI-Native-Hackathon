from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from .base import Base


class UserProgressBase(SQLModel):
    user_id: str = Field(..., foreign_key="users.id")
    chapter_id: str = Field(..., foreign_key="chapters.id")
    module_id: str = Field(..., max_length=255)
    status: str = Field(default="not_started")  # not_started, in_progress, completed
    progress_percentage: float = Field(default=0.0)
    time_spent: int = Field(default=0)  # in seconds
    quiz_scores: List[dict] = Field(default=[])


class UserProgress(UserProgressBase, Base, table=True):
    __tablename__ = "user_progress"

    # Relationships
    user: Optional["User"] = Relationship(back_populates="user_progresses")
    chapter: Optional["Chapter"] = Relationship(back_populates="user_progresses")


class UserProgressCreate(UserProgressBase):
    pass


class UserProgressUpdate(SQLModel):
    user_id: Optional[str] = None
    chapter_id: Optional[str] = None
    module_id: Optional[str] = None
    status: Optional[str] = None
    progress_percentage: Optional[float] = None
    time_spent: Optional[int] = None
    quiz_scores: Optional[List[dict]] = None
    last_accessed: Optional[datetime] = None
    completed_at: Optional[datetime] = None