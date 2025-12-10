from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from .base import Base


class CodeExampleBase(SQLModel):
    chapter_id: str = Field(..., foreign_key="chapters.id")
    language: str = Field(..., max_length=50)  # python, cpp, etc.
    platform: str = Field(..., max_length=50)  # ros2, isaac, unity
    code: str
    description: str
    difficulty: str = Field(default="intermediate")  # beginner, intermediate, advanced
    validation_status: str = Field(default="pending")  # pending, valid, invalid


class CodeExample(CodeExampleBase, Base, table=True):
    __tablename__ = "code_examples"

    # Relationships
    chapter: Optional["Chapter"] = Relationship(back_populates="code_examples")


class CodeExampleCreate(CodeExampleBase):
    pass


class CodeExampleUpdate(SQLModel):
    chapter_id: Optional[str] = None
    language: Optional[str] = None
    platform: Optional[str] = None
    code: Optional[str] = None
    description: Optional[str] = None
    difficulty: Optional[str] = None
    validation_status: Optional[str] = None