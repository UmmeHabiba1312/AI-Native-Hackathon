from typing import Optional
import logging
from fastapi import HTTPException, status
from ..schemas.common import ErrorResponse


class TextbookException(Exception):
    """
    Base exception class for the textbook application
    """
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", details: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)

    def to_error_response(self) -> ErrorResponse:
        """
        Convert exception to error response model
        """
        return ErrorResponse(
            error=self.message,
            code=self.error_code,
            details=self.details
        )


def handle_exception(exc: Exception, logger: logging.Logger) -> TextbookException:
    """
    Handle an exception and return a standardized TextbookException
    """
    logger.error(f"Exception occurred: {str(exc)}", exc_info=True)

    if isinstance(exc, TextbookException):
        return exc
    elif isinstance(exc, HTTPException):
        return TextbookException(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            details=str(exc)
        )
    else:
        return TextbookException(
            message="An unexpected error occurred",
            error_code="INTERNAL_ERROR",
            details=str(exc)
        )


# Specific exception classes
class ValidationException(TextbookException):
    """Exception for validation errors"""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class DatabaseException(TextbookException):
    """Exception for database errors"""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, "DATABASE_ERROR", details)


class AuthenticationException(TextbookException):
    """Exception for authentication errors"""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationException(TextbookException):
    """Exception for authorization errors"""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class NotFoundException(TextbookException):
    """Exception for resource not found errors"""
    def __init__(self, message: str = "Resource not found", details: Optional[str] = None):
        super().__init__(message, "NOT_FOUND", details)


class RAGException(TextbookException):
    """Exception for RAG-related errors"""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, "RAG_ERROR", details)


class ContentGenerationException(TextbookException):
    """Exception for content generation errors"""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, "CONTENT_GENERATION_ERROR", details)