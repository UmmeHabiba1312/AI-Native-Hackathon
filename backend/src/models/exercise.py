from sqlmodel import SQLModel, Field, Relationship
from typing import Optional
from .base import Base


class ExerciseBase(SQLModel):
    chapter_id: str = Field(..., foreign_key="chapters.id")
    type: str = Field(..., max_length=50)  # coding, multiple_choice, essay
    question: str
    solution: str
    difficulty: str = Field(default="intermediate")  # beginner, intermediate, advanced
    hints: list = Field(default=[])


class Exercise(ExerciseBase, Base, table=True):
    __tablename__ = "exercises"

    # Relationships
    chapter: Optional["Chapter"] = Relationship(back_populates="exercises")


class ExerciseCreate(ExerciseBase):
    pass


class ExerciseUpdate(SQLModel):
    chapter_id: Optional[str] = None
    type: Optional[str] = None
    question: Optional[str] = None
    solution: Optional[str] = None
    difficulty: Optional[str] = None
    hints: Optional[list] = None