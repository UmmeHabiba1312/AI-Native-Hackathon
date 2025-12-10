from pydantic import BaseModel
from typing import Optional


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: Optional[str] = None