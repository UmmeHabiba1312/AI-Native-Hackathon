import logging
import sys
from datetime import datetime
from typing import Any
from ..config.settings import settings


def setup_logging():
    """
    Set up logging configuration for the application
    """
    # Create custom formatter
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # Add timestamp, level, and custom fields
            log_format = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
            formatter = logging.Formatter(
                log_format,
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            return formatter.format(record)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.log_level.upper())
    console_handler.setFormatter(CustomFormatter())

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler to root logger
    root_logger.addHandler(console_handler)

    # Set specific loggers to WARNING level to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    """
    logger = logging.getLogger(name)
    return logger