"""
Matches Router
"""

from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from app.api.database import get_db
from app.api.models import Match, User, ProcessingStatus
from app.api.schemas import MatchCreate, MatchResponse, MatchListResponse, MatchStatusResponse
from app.api.dependencies import analyst_or_admin, get_current_active_user, get_client_ip
from app.api.logging_config import get_logger

router = APIRouter(prefix="/matches", tags=["Matches"])
logger = get_logger("matches")


@router.get("", response_model=MatchListResponse)
async def get_matches(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    season: Optional[str] = None,
    competition: Optional[str] = None,
    team_id: Optional[int] = None,
    status: Optional[ProcessingStatus] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of matches with filtering and pagination."""
    query = db.query(Match)
    
    # Apply filters
    if season:
        query = query.filter(Match.season == season)
    if competition:
        query = query.filter(Match.competition == competition)
    if team_id:
        query = query.filter((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
    if status:
        query = query.filter(Match.processing_status == status)
    if start_date:
        query = query.filter(Match.match_date >= start_date.date())
    if end_date:
        query = query.filter(Match.match_date <= end_date.date())
    
    # Order by date descending
    query = query.order_by(Match.match_date.desc())
    
    total = query.count()
    matches = query.offset(skip).limit(limit).all()
    
    logger.info(f"Retrieved {len(matches)} matches for user {current_user.email}")
    
    return {
        "total": total,
        "limit": limit,
        "offset": skip,
        "matches": [MatchResponse.from_orm(m) for m in matches]
    }


@router.post("/upload", response_model=MatchResponse, status_code=status.HTTP_201_CREATED)
async def upload_match(
    video_file: UploadFile = File(...),
    metadata: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(analyst_or_admin)
):
    """Upload match video for processing."""
    import json
    from pathlib import Path
    
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    # Create match record
    match = Match(
        home_team_id=metadata_dict["home_team_id"],
        away_team_id=metadata_dict["away_team_id"],
        competition=metadata_dict["competition"],
        season=metadata_dict["season"],
        match_date=metadata_dict["match_date"],
        venue=metadata_dict.get("venue"),
        processing_status=ProcessingStatus.PENDING
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    
    # Save video file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    video_path = upload_dir / f"match_{match.match_id}_{video_file.filename}"
    
    with open(video_path, "wb") as f:
        content = await video_file.read()
        f.write(content)
    
    match.video_path = str(video_path)
    db.commit()
    
    logger.info(f"Match {match.match_id} uploaded by {current_user.email}")
    
    # TODO: Queue processing task with Celery
    # process_match_video.delay(match.match_id, str(video_path))
    
    return match


@router.get("/{match_id}/status", response_model=MatchStatusResponse)
async def get_match_status(
    match_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get match processing status."""
    match = db.query(Match).filter(Match.match_id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Calculate progress (simplified - real implementation would track actual progress)
    progress = 0.0
    current_stage = "queued"
    estimated_time = 0
    
    if match.processing_status == ProcessingStatus.PROCESSING:
        progress = 50.0  # Simplified
        current_stage = "processing"
        estimated_time = 300
    elif match.processing_status == ProcessingStatus.COMPLETED:
        progress = 100.0
        current_stage = "completed"
        estimated_time = 0
    elif match.processing_status == ProcessingStatus.FAILED:
        progress = 0.0
        current_stage = "failed"
        estimated_time = 0
    
    return {
        "match_id": match.match_id,
        "status": match.processing_status.value,
        "progress_pct": progress,
        "current_stage": current_stage,
        "estimated_completion_seconds": estimated_time
    }


@router.get("/{match_id}/analytics")
async def get_match_analytics(
    match_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get match analytics."""
    match = db.query(Match).filter(Match.match_id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if match.processing_status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Match processing not completed")
    
    # Load analytics from output directory
    from pathlib import Path
    import json
    
    output_dir = Path(match.output_dir)
    analytics = {}
    
    analytics_file = output_dir / "analytics.json"
    if analytics_file.exists():
        with open(analytics_file) as f:
            analytics = json.load(f)
    
    return {
        "match_id": match.match_id,
        "analytics": analytics
    }


@router.delete("/{match_id}")
async def delete_match(
    match_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_only)
):
    """Delete match (admin only)."""
    match = db.query(Match).filter(Match.match_id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Delete associated files
    if match.video_path:
        video_path = Path(match.video_path)
        if video_path.exists():
            video_path.unlink()
    
    db.delete(match)
    db.commit()
    
    logger.info(f"Match {match_id} deleted by {current_user.email}")
    
    return {"message": "Match deleted successfully"}