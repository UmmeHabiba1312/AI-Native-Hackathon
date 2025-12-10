"""
API routes for the AI-Native Textbook platform
"""
from .chapters import router as chapters_router
from .chat import router as chat_router
from .user_progress import router as user_progress_router
from .translation import router as translation_router

__all__ = ["chapters_router", "chat_router", "user_progress_router", "translation_router"]