from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from .base import Base


class DiagramBase(SQLModel):
    chapter_id: str = Field(..., foreign_key="chapters.id")
    title: str = Field(..., max_length=255)
    description: str
    file_path: str = Field(..., max_length=500)
    type: str = Field(..., max_length=50)  # flowchart, illustration, urdf
    alt_text: str = Field(default="")


class Diagram(DiagramBase, Base, table=True):
    __tablename__ = "diagrams"

    # Relationships
    chapter: Optional["Chapter"] = Relationship(back_populates="diagrams")


class DiagramCreate(DiagramBase):
    pass


class DiagramUpdate(SQLModel):
    chapter_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    file_path: Optional[str] = None
    type: Optional[str] = None
    alt_text: Optional[str] = None