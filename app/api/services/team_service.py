"""
Team Service - Business Logic for Team Operations
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from app.api.models import Team, User, Match
from app.api.schemas import TeamCreate, TeamUpdate


class TeamService:
    """Service for team-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_team(self, team_data: TeamCreate, created_by: User) -> Team:
        """Create a new team."""
        team = Team(
            team_name=team_data.team_name,
            short_name=team_data.short_name,
            country=team_data.country,
            competition=team_data.competition,
            founded_year=team_data.founded_year,
            stadium=team_data.stadium,
            manager=team_data.manager
        )
        self.db.add(team)
        self.db.commit()
        self.db.refresh(team)
        return team
    
    def get_team(self, team_id: int) -> Optional[Team]:
        """Get team by ID."""
        return self.db.query(Team).filter(Team.team_id == team_id).first()
    
    def get_teams(self, skip: int = 0, limit: int = 20, **filters) -> tuple[List[Team], int]:
        """Get teams with filtering."""
        query = self.db.query(Team)
        
        # Apply filters
        for key, value in filters.items():
            if value is not None:
                if key == "name":
                    query = query.filter(Team.team_name.ilike(f"%{value}%"))
                elif key == "competition":
                    query = query.filter(Team.competition == value)
        
        total = query.count()
        teams = query.order_by(Team.team_name).offset(skip).limit(limit).all()
        
        return teams, total
    
    def update_team(self, team_id: int, team_update: TeamUpdate) -> Optional[Team]:
        """Update team."""
        team = self.get_team(team_id)
        if not team:
            return None
        
        update_data = team_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(team, key, value)
        
        self.db.commit()
        self.db.refresh(team)
        return team
    
    def delete_team(self, team_id: int) -> bool:
        """Delete team."""
        team = self.get_team(team_id)
        if not team:
            return False
        
        self.db.delete(team)
        self.db.commit()
        return True
    
    def get_team_matches(self, team_id: int, skip: int = 0, limit: int = 20) -> tuple[List[Match], int]:
        """Get matches for a team."""
        query = self.db.query(Match).filter(
            (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
        )
        
        total = query.count()
        matches = query.order_by(Match.match_date.desc()).offset(skip).limit(limit).all()
        
        return matches, total