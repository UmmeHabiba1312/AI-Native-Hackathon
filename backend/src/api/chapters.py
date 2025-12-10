from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlmodel import Session
from typing import List, Optional

from ..database.database import get_db
from ..models.chapter import Chapter
from ..schemas.chapter import ChapterResponse, ChapterListResponse, ChapterCreateRequest, ChapterUpdateRequest
from ..schemas.common import ApiResponse
from ..utils.logger import get_logger
from ..utils.exceptions import NotFoundException

router = APIRouter()
logger = get_logger(__name__)


@router.get("/", response_model=ChapterListResponse)
async def list_chapters(
    module_id: Optional[str] = Query(None, description="Filter chapters by module ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of chapters to return"),
    offset: int = Query(0, ge=0, description="Number of chapters to skip"),
    db: Session = Depends(get_db)
):
    """
    List all textbook chapters with optional filtering
    """
    try:
        # Build query with optional filters
        query = db.query(Chapter)

        if module_id:
            query = query.filter(Chapter.module_id == module_id)

        # Get total count
        total = query.count()

        # Apply pagination
        chapters = query.offset(offset).limit(limit).all()

        logger.info(f"Retrieved {len(chapters)} chapters (total: {total})")

        # Convert SQLModel objects to Pydantic responses
        chapter_responses = []
        for chapter in chapters:
            chapter_resp = ChapterResponse(
                id=chapter.id,
                title=chapter.title,
                module_id=chapter.module_id,
                content=chapter.content,
                learning_objectives=chapter.learning_objectives,
                code_examples=[],  # Will be populated separately if needed
                diagrams=[],       # Will be populated separately if needed
                exercises=[],      # Will be populated separately if needed
                metadata=chapter.metadata,
                created_at=chapter.created_at,
                updated_at=chapter.updated_at,
                version=chapter.version
            )
            chapter_responses.append(chapter_resp)

        return ChapterListResponse(
            chapters=chapter_responses,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing chapters: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chapters")


@router.get("/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(
    chapter_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the content of a specific chapter
    """
    try:
        chapter = db.query(Chapter).filter(Chapter.id == chapter_id).first()

        if not chapter:
            logger.warning(f"Chapter with ID {chapter_id} not found")
            raise NotFoundException(f"Chapter with ID {chapter_id} not found")

        logger.info(f"Retrieved chapter: {chapter.title}")

        return ChapterResponse(
            id=chapter.id,
            title=chapter.title,
            module_id=chapter.module_id,
            content=chapter.content,
            learning_objectives=chapter.learning_objectives,
            code_examples=[],  # Will be populated separately if needed
            diagrams=[],       # Will be populated separately if needed
            exercises=[],      # Will be populated separately if needed
            metadata=chapter.metadata,
            created_at=chapter.created_at,
            updated_at=chapter.updated_at,
            version=chapter.version
        )
    except NotFoundException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chapter {chapter_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chapter")