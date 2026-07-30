"""
Match Service - Business Logic for Match Operations
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from app.api.models import Match, Team, User
from app.api.schemas import MatchCreate, MatchUpdate


class MatchService:
    """Service for match-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_match(self, match_data: MatchCreate, created_by: User) -> Match:
        """Create a new match."""
        match = Match(
            home_team_id=match_data.home_team_id,
            away_team_id=match_data.away_team_id,
            competition=match_data.competition,
            season=match_data.season,
            match_date=match_data.match_date,
            venue=match_data.venue,
            processing_status="pending"
        )
        self.db.add(match)
        self.db.commit()
        self.db.refresh(match)
        return match
    
    def get_match(self, match_id: int) -> Optional[Match]:
        """Get match by ID."""
        return self.db.query(Match).filter(Match.match_id == match_id).first()
    
    def get_matches(self, skip: int = 0, limit: int = 20, **filters) -> tuple[List[Match], int]:
        """Get matches with filtering."""
        query = self.db.query(Match)
        
        # Apply filters
        for key, value in filters.items():
            if value is not None:
                if key == "season":
                    query = query.filter(Match.season == value)
                elif key == "competition":
                    query = query.filter(Match.competition == value)
                elif key == "team_id":
                    query = query.filter(
                        (Match.home_team_id == value) | (Match.away_team_id == value)
                    )
                elif key == "status":
                    query = query.filter(Match.processing_status == value)
        
        total = query.count()
        matches = query.order_by(Match.match_date.desc()).offset(skip).limit(limit).all()
        
        return matches, total
    
    def update_match(self, match_id: int, match_update: MatchUpdate) -> Optional[Match]:
        """Update match."""
        match = self.get_match(match_id)
        if not match:
            return None
        
        update_data = match_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(match, key, value)
        
        self.db.commit()
        self.db.refresh(match)
        return match
    
    def delete_match(self, match_id: int) -> bool:
        """Delete match."""
        match = self.get_match(match_id)
        if not match:
            return False
        
        self.db.delete(match)
        self.db.commit()
        return True
    
    def get_match_teams(self, match_id: int) -> tuple[Team, Team]:
        """Get home and away teams for a match."""
        match = self.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")
        
        return match.home_team, match.away_team


# Import here to avoid circular dependency
from fastapi import HTTPException