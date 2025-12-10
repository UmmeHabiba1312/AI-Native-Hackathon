from fastapi import HTTPException, status
from typing import Any, Dict
from ..utils.exceptions import TextbookException


def handle_api_error(exc: Exception, logger=None):
    """
    Standard error handler for API endpoints
    """
    if logger:
        logger.error(f"API Error: {str(exc)}", exc_info=True)

    if isinstance(exc, TextbookException):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.to_error_response().dict()
        )
    elif isinstance(exc, HTTPException):
        raise exc
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


def create_response(data: Any = None, message: str = "Success", success: bool = True):
    """
    Create a standard API response
    """
    return {
        "success": success,
        "message": message,
        "data": data
    }