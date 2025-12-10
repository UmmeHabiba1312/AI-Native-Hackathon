from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class QuizScore(BaseModel):
    quiz_id: str
    score: float


class UserProgressResponse(BaseModel):
    id: str
    user_id: str
    chapter_id: str
    module_id: str
    status: str  # not_started, in_progress, completed
    progress_percentage: float
    time_spent: int  # in seconds
    last_accessed: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    quiz_scores: List[QuizScore] = []


class UserProgressCreateRequest(BaseModel):
    user_id: str
    chapter_id: str
    module_id: str
    status: str = "not_started"
    progress_percentage: float = 0.0
    time_spent: int = 0
    quiz_scores: List[Dict] = []


class UserProgressUpdateRequest(BaseModel):
    status: Optional[str] = None
    progress_percentage: Optional[float] = None
    time_spent: Optional[int] = None
    quiz_scores: Optional[List[Dict]] = None