from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from .chapter import ChapterResponse


class ModuleResponse(BaseModel):
    id: str
    title: str
    description: str
    chapters: List[str]  # List of chapter IDs
    order: int
    created_at: datetime
    updated_at: datetime


class ModuleListResponse(BaseModel):
    modules: List[ModuleResponse]
    total: int
    limit: int
    offset: int


class ModuleCreateRequest(BaseModel):
    title: str
    description: str
    order: int
    chapters: List[str] = []


class ModuleUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = None
    chapters: Optional[List[str]] = None