from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CodeExampleResponse(BaseModel):
    id: str
    chapter_id: str
    language: str
    platform: str
    code: str
    description: str
    difficulty: str  # beginner, intermediate, advanced
    validation_status: str  # pending, valid, invalid
    created_at: datetime


class CodeExampleCreateRequest(BaseModel):
    chapter_id: str
    language: str
    platform: str
    code: str
    description: str
    difficulty: str = "intermediate"  # default value


class CodeExampleUpdateRequest(BaseModel):
    language: Optional[str] = None
    platform: Optional[str] = None
    code: Optional[str] = None
    description: Optional[str] = None
    difficulty: Optional[str] = None