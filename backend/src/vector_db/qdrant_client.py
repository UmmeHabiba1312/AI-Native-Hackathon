from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional, Any
from uuid import uuid4
import logging
from ..config.settings import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Collection names
TEXTBOOK_CONTENT_COLLECTION = "textbook_content"
CHAPTER_CONTENT_COLLECTION = "chapter_content"
EXERCISE_CONTENT_COLLECTION = "exercise_content"

# Initialize Qdrant client
def get_qdrant_client() -> QdrantClient:
    """
    Get Qdrant client instance
    """
    if settings.qdrant_api_key:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=10
        )
    else:
        client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=10
        )

    return client


async def initialize_qdrant_collections():
    """
    Initialize required collections in Qdrant
    """
    client = get_qdrant_client()

    # Define vector configuration
    vector_config = models.VectorParams(
        size=1536,  # Default OpenAI embedding size
        distance=models.Distance.COSINE
    )

    # Create collections if they don't exist
    collections = [TEXTBOOK_CONTENT_COLLECTION, CHAPTER_CONTENT_COLLECTION, EXERCISE_CONTENT_COLLECTION]

    for collection_name in collections:
        try:
            # Check if collection exists
            client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
        except:
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config
            )
            logger.info(f"Created collection '{collection_name}'")

    logger.info("Qdrant collections initialized successfully")


def upsert_textbook_content(
    collection_name: str,
    content_id: str,
    content: str,
    chapter_id: str,
    module_id: str,
    content_type: str,
    embedding: List[float],
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Upsert textbook content into Qdrant collection
    """
    client = get_qdrant_client()

    if metadata is None:
        metadata = {}

    # Prepare payload
    payload = {
        "content": content,
        "chapter_id": chapter_id,
        "module_id": module_id,
        "content_type": content_type,
        "metadata": metadata
    }
    payload.update(metadata)

    try:
        # Upsert the record
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=content_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        logger.info(f"Upserted content to collection '{collection_name}' with ID: {content_id}")
        return True
    except Exception as e:
        logger.error(f"Error upserting content to Qdrant: {str(e)}")
        return False


def search_similar_content(
    collection_name: str,
    query_embedding: List[float],
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar content in Qdrant collection
    """
    client = get_qdrant_client()

    # Prepare filters if provided
    qdrant_filters = None
    if filters:
        filter_conditions = []
        for key, value in filters.items():
            filter_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )

        if filter_conditions:
            qdrant_filters = models.Filter(
                must=filter_conditions
            )

    try:
        # Search for similar vectors
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=qdrant_filters
        )

        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "content": result.payload.get("content", ""),
                "chapter_id": result.payload.get("chapter_id", ""),
                "module_id": result.payload.get("module_id", ""),
                "content_type": result.payload.get("content_type", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            })

        logger.info(f"Found {len(results)} similar content items in collection '{collection_name}'")
        return results
    except Exception as e:
        logger.error(f"Error searching content in Qdrant: {str(e)}")
        return []


def delete_content_by_id(collection_name: str, content_id: str) -> bool:
    """
    Delete content by ID from Qdrant collection
    """
    client = get_qdrant_client()

    try:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[content_id]
            )
        )
        logger.info(f"Deleted content with ID: {content_id} from collection '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"Error deleting content from Qdrant: {str(e)}")
        return False


def delete_content_by_filter(collection_name: str, filters: Dict[str, Any]) -> bool:
    """
    Delete content by filter from Qdrant collection
    """
    client = get_qdrant_client()

    # Prepare filter conditions
    filter_conditions = []
    for key, value in filters.items():
        filter_conditions.append(
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            )
        )

    qdrant_filter = models.Filter(
        must=filter_conditions
    )

    try:
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=qdrant_filter
            )
        )
        logger.info(f"Deleted content from collection '{collection_name}' using filters")
        return True
    except Exception as e:
        logger.error(f"Error deleting content from Qdrant using filters: {str(e)}")
        return False