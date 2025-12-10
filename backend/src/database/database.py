from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import QueuePool
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
from sqlmodel import SQLModel

from ..config.settings import settings

# Create database engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create session maker for async sessions
AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Synchronous engine for sync operations if needed
sync_engine = create_engine(
    settings.database_url.replace('+asyncpg', ''),
    echo=settings.debug,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Session maker for sync operations
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
    expire_on_commit=False
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        yield session


def get_db():
    """Get sync database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def create_tables():
    """Create all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def drop_tables():
    """Drop all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)