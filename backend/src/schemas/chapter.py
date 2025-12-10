from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from .code_example import CodeExampleResponse
from .diagram import DiagramResponse
from .exercise import ExerciseResponse


class ChapterMetadata(BaseModel):
    difficulty: Optional[str] = None  # beginner, intermediate, advanced
    estimated_time: Optional[int] = None  # in minutes
    prerequisites: Optional[List[str]] = None  # list of chapter IDs


class ChapterResponse(BaseModel):
    id: str
    title: str
    module_id: str
    content: str
    learning_objectives: List[str]
    code_examples: List[CodeExampleResponse] = []
    diagrams: List[DiagramResponse] = []
    exercises: List[ExerciseResponse] = []
    metadata: ChapterMetadata
    created_at: datetime
    updated_at: datetime
    version: str


class ChapterListResponse(BaseModel):
    chapters: List[ChapterResponse]
    total: int
    limit: int
    offset: int


class ChapterCreateRequest(BaseModel):
    title: str
    module_id: str
    content: str
    learning_objectives: List[str] = []
    code_examples: List[dict] = []  # Will be processed separately
    diagrams: List[dict] = []  # Will be processed separately
    exercises: List[dict] = []  # Will be processed separately
    metadata: ChapterMetadata = ChapterMetadata()


class ChapterUpdateRequest(BaseModel):
    title: Optional[str] = None
    module_id: Optional[str] = None
    content: Optional[str] = None
    learning_objectives: Optional[List[str]] = None
    metadata: Optional[ChapterMetadata] = None