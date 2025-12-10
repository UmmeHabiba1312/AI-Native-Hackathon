from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List

from ..database.database import get_db
from ..models.user_progress import UserProgress
from ..schemas.user_progress import UserProgressResponse, UserProgressCreateRequest, UserProgressUpdateRequest
from ..utils.logger import get_logger
from ..utils.exceptions import NotFoundException

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{user_id}", response_model=List[UserProgressResponse])
async def get_user_progress(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the learning progress for a specific user
    """
    try:
        # Find all progress records for the user
        progress_records = db.query(UserProgress).filter(UserProgress.user_id == user_id).all()

        logger.info(f"Retrieved {len(progress_records)} progress records for user: {user_id}")

        # Convert to response models
        progress_responses = []
        for record in progress_records:
            progress_resp = UserProgressResponse(
                id=record.id,
                user_id=record.user_id,
                chapter_id=record.chapter_id,
                module_id=record.module_id,
                status=record.status,
                progress_percentage=record.progress_percentage,
                time_spent=record.time_spent,
                last_accessed=record.last_accessed,
                completed_at=record.completed_at,
                quiz_scores=record.quiz_scores
            )
            progress_responses.append(progress_resp)

        return progress_responses
    except Exception as e:
        logger.error(f"Error retrieving progress for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving user progress")


@router.put("/{user_id}", response_model=UserProgressResponse)
async def update_user_progress(
    user_id: str,
    request: UserProgressUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update the learning progress for a specific user
    """
    try:
        # Find existing progress record
        progress = db.query(UserProgress).filter(
            UserProgress.user_id == user_id,
            UserProgress.chapter_id == request.chapter_id
        ).first()

        if not progress:
            logger.warning(f"Progress record not found for user {user_id}, chapter {request.chapter_id}")
            raise NotFoundException(f"Progress record not found for user {user_id}, chapter {request.chapter_id}")

        # Update the progress record with provided values
        if request.status is not None:
            progress.status = request.status
        if request.progress_percentage is not None:
            progress.progress_percentage = request.progress_percentage
        if request.time_spent is not None:
            progress.time_spent = request.time_spent
        if request.quiz_scores is not None:
            progress.quiz_scores = request.quiz_scores

        # Commit changes to database
        db.commit()
        db.refresh(progress)

        logger.info(f"Updated progress for user {user_id}, chapter {request.chapter_id}")

        # Return updated record
        return UserProgressResponse(
            id=progress.id,
            user_id=progress.user_id,
            chapter_id=progress.chapter_id,
            module_id=progress.module_id,
            status=progress.status,
            progress_percentage=progress.progress_percentage,
            time_spent=progress.time_spent,
            last_accessed=progress.last_accessed,
            completed_at=progress.completed_at,
            quiz_scores=progress.quiz_scores
        )
    except NotFoundException:
        raise
    except Exception as e:
        logger.error(f"Error updating progress for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating user progress")