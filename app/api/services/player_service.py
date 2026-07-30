"""
Player Service - Business Logic for Player Operations
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from app.api.models import Player, Team, User
from app.api.schemas import PlayerCreate, PlayerUpdate


class PlayerService:
    """Service for player-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_player(self, player_data: PlayerCreate, created_by: User) -> Player:
        """Create a new player."""
        player = Player(
            full_name=player_data.full_name,
            short_name=player_data.short_name,
            position=player_data.position,
            date_of_birth=player_data.date_of_birth,
            nationality=player_data.nationality,
            height_cm=player_data.height_cm,
            preferred_foot=player_data.preferred_foot,
            shirt_number=player_data.shirt_number,
            team_id=player_data.team_id
        )
        self.db.add(player)
        self.db.commit()
        self.db.refresh(player)
        return player
    
    def get_player(self, player_id: int) -> Optional[Player]:
        """Get player by ID."""
        return self.db.query(Player).filter(Player.player_id == player_id).first()
    
    def get_players(self, skip: int = 0, limit: int = 20, **filters) -> tuple[List[Player], int]:
        """Get players with filtering."""
        query = self.db.query(Player)
        
        # Apply filters
        for key, value in filters.items():
            if value is not None:
                if key == "name":
                    query = query.filter(Player.full_name.ilike(f"%{value}%"))
                elif key == "team_id":
                    query = query.filter(Player.team_id == value)
                elif key == "position":
                    query = query.filter(Player.position == value)
        
        total = query.count()
        players = query.order_by(Player.full_name).offset(skip).limit(limit).all()
        
        return players, total
    
    def update_player(self, player_id: int, player_update: PlayerUpdate) -> Optional[Player]:
        """Update player."""
        player = self.get_player(player_id)
        if not player:
            return None
        
        update_data = player_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(player, key, value)
        
        self.db.commit()
        self.db.refresh(player)
        return player
    
    def delete_player(self, player_id: int) -> bool:
        """Delete player."""
        player = self.get_player(player_id)
        if not player:
            return False
        
        self.db.delete(player)
        self.db.commit()
        return True
    
    def get_player_team(self, player_id: int) -> Optional[Team]:
        """Get player's team."""
        player = self.get_player(player_id)
        if not player:
            return None
        return player.team