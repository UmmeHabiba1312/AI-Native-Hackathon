from sqlmodel import SQLModel
from sqlalchemy import Column, DateTime, func
from datetime import datetime
import uuid


class Base(SQLModel):
    """Base class for all database models"""
    id: str
    created_at: datetime = Column(DateTime(timezone=True), default=func.now())
    updated_at: datetime = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    def __init__(self, **kwargs):
        # Generate a UUID if id is not provided
        if 'id' not in kwargs or kwargs['id'] is None:
            kwargs['id'] = str(uuid.uuid4())
        super().__init__(**kwargs)