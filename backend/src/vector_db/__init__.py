"""
Vector database connection and management for RAG functionality
"""
from .qdrant_client import get_qdrant_client, initialize_qdrant_collections

__all__ = ["get_qdrant_client", "initialize_qdrant_collections"]