from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from .base import Base


class ChapterBase(SQLModel):
    title: str = Field(..., max_length=255)
    module_id: str = Field(..., max_length=255)
    content: str
    learning_objectives: List[str] = Field(default=[])
    metadata: dict = Field(default={})
    version: str = Field(default="1.0.0")


class Chapter(ChapterBase, Base, table=True):
    __tablename__ = "chapters"

    # Relationships
    module: Optional["Module"] = Relationship(back_populates="chapters")
    code_examples: List["CodeExample"] = Relationship(back_populates="chapter")
    diagrams: List["Diagram"] = Relationship(back_populates="chapter")
    exercises: List["Exercise"] = Relationship(back_populates="chapter")
    user_progresses: List["UserProgress"] = Relationship(back_populates="chapter")


class ChapterCreate(ChapterBase):
    pass


class ChapterUpdate(SQLModel):
    title: Optional[str] = None
    module_id: Optional[str] = None
    content: Optional[str] = None
    learning_objectives: Optional[List[str]] = None
    metadata: Optional[dict] = None
    version: Optional[str] = None