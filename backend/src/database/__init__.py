"""
Database connection and session management for the AI-Native Textbook platform
"""
from .database import engine, SessionLocal, get_db

__all__ = ["engine", "SessionLocal", "get_db"]