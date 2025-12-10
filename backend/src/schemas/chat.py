from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
    user_id: str
    context: Optional[str] = None  # chapter ID or other context


class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    context_used: List[str] = []