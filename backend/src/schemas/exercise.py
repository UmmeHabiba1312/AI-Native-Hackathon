from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ExerciseResponse(BaseModel):
    id: str
    chapter_id: str
    type: str  # coding, multiple_choice, essay
    question: str
    solution: str
    difficulty: str  # beginner, intermediate, advanced
    hints: List[str]
    created_at: datetime


class ExerciseCreateRequest(BaseModel):
    chapter_id: str
    type: str  # coding, multiple_choice, essay
    question: str
    solution: str
    difficulty: str = "intermediate"  # default value
    hints: List[str] = []


class ExerciseUpdateRequest(BaseModel):
    type: Optional[str] = None
    question: Optional[str] = None
    solution: Optional[str] = None
    difficulty: Optional[str] = None
    hints: Optional[List[str]] = None