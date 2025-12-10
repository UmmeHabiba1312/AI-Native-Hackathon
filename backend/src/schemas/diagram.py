from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DiagramResponse(BaseModel):
    id: str
    chapter_id: str
    title: str
    description: str
    file_path: str
    type: str  # flowchart, illustration, urdf
    alt_text: str
    created_at: datetime


class DiagramCreateRequest(BaseModel):
    chapter_id: str
    title: str
    description: str
    file_path: str
    type: str  # flowchart, illustration, urdf
    alt_text: str = ""


class DiagramUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    file_path: Optional[str] = None
    type: Optional[str] = None
    alt_text: Optional[str] = None