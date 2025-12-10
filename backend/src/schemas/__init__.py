"""
API response models for the AI-Native Textbook platform
"""
from .chapter import ChapterResponse, ChapterListResponse, ChapterCreateRequest, ChapterUpdateRequest
from .module import ModuleResponse, ModuleListResponse
from .user import UserResponse, UserCreateRequest, UserUpdateRequest
from .user_progress import UserProgressResponse, UserProgressCreateRequest, UserProgressUpdateRequest
from .chat import ChatRequest, ChatResponse
from .translation import TranslationRequest, TranslationResponse
from .common import ApiResponse, ErrorResponse

__all__ = [
    "ChapterResponse", "ChapterListResponse", "ChapterCreateRequest", "ChapterUpdateRequest",
    "ModuleResponse", "ModuleListResponse",
    "UserResponse", "UserCreateRequest", "UserUpdateRequest",
    "UserProgressResponse", "UserProgressCreateRequest", "UserProgressUpdateRequest",
    "ChatRequest", "ChatResponse",
    "TranslationRequest", "TranslationResponse",
    "ApiResponse", "ErrorResponse"
]