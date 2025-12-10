from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from .base import Base


class ModuleBase(SQLModel):
    title: str = Field(..., max_length=255)
    description: str
    order: int
    chapters: List[str] = Field(default=[])


class Module(ModuleBase, Base, table=True):
    __tablename__ = "modules"

    # Relationships
    chapters: List["Chapter"] = Relationship(back_populates="module")


class ModuleCreate(ModuleBase):
    pass


class ModuleUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = None
    chapters: Optional[List[str]] = None