from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from .base import Base


class ChatSessionBase(SQLModel):
    user_id: Optional[str] = Field(default=None, foreign_key="users.id")
    session_data: dict = Field(default={})


class ChatSession(ChatSessionBase, Base, table=True):
    __tablename__ = "chat_sessions"

    # Relationships
    messages: List["ChatMessage"] = Relationship(back_populates="session")


class ChatSessionCreate(ChatSessionBase):
    pass


class ChatSessionUpdate(SQLModel):
    user_id: Optional[str] = None
    session_data: Optional[dict] = None


# Additional model for chat messages
class ChatMessageBase(SQLModel):
    session_id: str = Field(..., foreign_key="chat_sessions.id")
    role: str = Field(..., max_length=20)  # user, assistant
    content: str
    sources: List[str] = Field(default=[])


class ChatMessage(ChatMessageBase, Base, table=True):
    __tablename__ = "chat_messages"

    # Relationships
    session: Optional["ChatSession"] = Relationship(back_populates="messages")


class ChatMessageCreate(ChatMessageBase):
    pass


class ChatMessageUpdate(SQLModel):
    session_id: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    sources: Optional[List[str]] = None