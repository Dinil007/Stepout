# Football Analytics Platform - Production Architecture

**Version:** 1.0.0  
**Date:** 2025-10-26  
**Status:** ARCHITECTURE COMPLETE

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Database Schema](#database-schema)
3. [API Endpoints](#api-endpoints)
4. [Dashboard Structure](#dashboard-structure)
5. [User Workflows](#user-workflows)
6. [Permissions & RBAC](#permissions--rbac)
7. [Deployment Architecture](#deployment-architecture)
8. [Module Specifications](#module-specifications)

---

## SYSTEM OVERVIEW

### Architecture Pattern

The platform follows a **modular microservices architecture** with:
- **Backend API** (FastAPI) - Business logic, authentication, data access
- **Frontend Dashboard** (Streamlit) - User interfaces for different roles
- **Database Layer** (JSON files for MVP, PostgreSQL for production)
- **Analytics Engine** (existing Python modules)
- **File Storage** (Local filesystem, S3 for production)

### High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                       │
│  ┌──────────────┬──────────────┬──────────────────┐   │
│  │ Match Center │   Player     │    Team Dashboard │   │
│  │   Dashboard  │   Profile    │                  │   │
│  ├──────────────┼──────────────┼──────────────────┤   │
│  │Match Reports │ Scout Search │  Coach Dashboard │   │
│  ├──────────────┴──────────────┴──────────────────┤   │
│  │              Admin Portal                        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │ HTTP/REST API
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    BACKEND API                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  FastAPI Application                             │   │
│  │  - Authentication (OAuth2 + JWT)                 │   │
│  │  - Authorization (RBAC)                          │   │
│  │  - CRUD Operations                               │   │
│  │  - Analytics Integration                         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 ANALYTICS ENGINE                        │
│  ┌────────────┬────────────┬────────────┬───────────┐  │
│  │  Tracking  │  Detection │  Season    │ Evaluation │  │
│  │  Evaluator │  Detector  │  Analyzer  │ Framework  │  │
│  ├────────────┼────────────┼────────────┴───────────┤  │
│  │  Player    │  Team      │  Formation            │  │
│  │  Metrics   │  Analytics │  Detection            │  │
│  └────────────┴────────────┴───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 DATA LAYER                              │
│  ┌────────────────┬────────────────────────────────┐   │
│  │  File System   │  PostgreSQL (Production)        │   │
│  │  (JSON/Pandas) │                                 │   │
│  │  - matches/    │  - users                        │   │
│  │  - players/    │  - matches                      │   │
│  │  - teams/      │  - analytics_events             │   │
│  │  - outputs/    │  - season_stats                 │   │
│  └────────────────┴────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## DATABASE SCHEMA

### Core Tables (PostgreSQL)

#### 1. Users

```sql
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'analyst', 'coach', 'scout')),
    team_id INTEGER REFERENCES teams(team_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_team ON users(team_id);
```

#### 2. Teams

```sql
CREATE TABLE teams (
    team_id SERIAL PRIMARY KEY,
    team_name VARCHAR(255) UNIQUE NOT NULL,
    short_name VARCHAR(50),
    country VARCHAR(100),
    competition VARCHAR(100),
    founded_year INTEGER,
    stadium VARCHAR(255),
    manager VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_teams_name ON teams(team_name);
CREATE INDEX idx_teams_competition ON teams(competition);
```

#### 3. Players

```sql
CREATE TABLE players (
    player_id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    full_name VARCHAR(255) NOT NULL,
    short_name VARCHAR(100),
    position VARCHAR(50) CHECK (position IN ('GK', 'DEF', 'MID', 'FWD')),
    date_of_birth DATE,
    nationality VARCHAR(100),
    height_cm INTEGER,
    preferred_foot VARCHAR(10),
    shirt_number INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_players_team ON players(team_id);
CREATE INDEX idx_players_position ON players(position);
CREATE INDEX idx_players_name ON players(full_name);
```

#### 4. Matches

```sql
CREATE TABLE matches (
    match_id SERIAL PRIMARY KEY,
    home_team_id INTEGER REFERENCES teams(team_id),
    away_team_id INTEGER REFERENCES teams(team_id),
    competition VARCHAR(100) NOT NULL,
    season VARCHAR(20) NOT NULL,
    match_date DATE NOT NULL,
    venue VARCHAR(255),
    home_score INTEGER DEFAULT 0,
    away_score INTEGER DEFAULT 0,
    video_path VARCHAR(500),
    duration_seconds INTEGER,
    processing_status VARCHAR(50) DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    analytics_version VARCHAR(50),
    processing_time_seconds INTEGER,
    output_dir VARCHAR(500),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_matches_season ON matches(season);
CREATE INDEX idx_matches_competition ON matches(competition);
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_teams ON matches(home_team_id, away_team_id);
CREATE INDEX idx_matches_status ON matches(processing_status);
```

#### 5. Match Events

```sql
CREATE TABLE match_events (
    event_id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    frame_number INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    team_id INTEGER REFERENCES teams(team_id),
    player_id INTEGER REFERENCES players(player_id),
    x REAL,
    y REAL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_match ON match_events(match_id);
CREATE INDEX idx_events_type ON match_events(event_type);
CREATE INDEX idx_events_player ON match_events(player_id);
```

#### 6. Player Match Statistics

```sql
CREATE TABLE player_match_stats (
    stat_id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    player_id INTEGER REFERENCES players(player_id),
    team_id INTEGER REFERENCES teams(team_id),
    minutes_played REAL DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    passes_attempted INTEGER DEFAULT 0,
    pass_accuracy_pct REAL DEFAULT 0,
    defensive_actions INTEGER DEFAULT 0,
    distance_m REAL DEFAULT 0,
    max_speed_kmh REAL DEFAULT 0,
    avg_speed_kmh REAL DEFAULT 0,
    sprint_count INTEGER DEFAULT 0,
    xg REAL DEFAULT 0,
    xa REAL DEFAULT 0,
    xt REAL DEFAULT 0,
    rating REAL DEFAULT 0,
    heatmap_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, player_id)
);

CREATE INDEX idx_player_stats_match ON player_match_stats(match_id);
CREATE INDEX idx_player_stats_player ON player_match_stats(player_id);
CREATE INDEX idx_player_stats_rating ON player_match_stats(rating DESC);
```

#### 7. Team Match Statistics

```sql
CREATE TABLE team_match_stats (
    stat_id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    team_id INTEGER REFERENCES teams(team_id),
    possession_pct REAL DEFAULT 0,
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    passes_attempted INTEGER DEFAULT 0,
    pass_accuracy_pct REAL DEFAULT 0,
    corners INTEGER DEFAULT 0,
    fouls INTEGER DEFAULT 0,
    ppda REAL DEFAULT 0,
    xg REAL DEFAULT 0,
    xa REAL DEFAULT 0,
    xt REAL DEFAULT 0,
    formation_detected VARCHAR(50),
    formation_confidence REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, team_id)
);

CREATE INDEX idx_team_stats_match ON team_match_stats(match_id);
CREATE INDEX idx_team_stats_team ON team_match_stats(team_id);
```

#### 8. Season Statistics

```sql
CREATE TABLE season_stats (
    stat_id SERIAL PRIMARY KEY,
    season VARCHAR(20) NOT NULL,
    competition VARCHAR(100) NOT NULL,
    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('player', 'team')),
    entity_id INTEGER NOT NULL,
    matches_played INTEGER DEFAULT 0,
    minutes_played REAL DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    shots INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    passes_attempted INTEGER DEFAULT 0,
    defensive_actions INTEGER DEFAULT 0,
    distance_m REAL DEFAULT 0,
    max_speed_kmh REAL DEFAULT 0,
    xg REAL DEFAULT 0,
    xa REAL DEFAULT 0,
    xt REAL DEFAULT 0,
    average_rating REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, competition, entity_type, entity_id)
);

CREATE INDEX idx_season_stats_season ON season_stats(season, competition);
CREATE INDEX idx_season_stats_entity ON season_stats(entity_type, entity_id);
```

#### 9. Formation History

```sql
CREATE TABLE formation_history (
    formation_id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    team_id INTEGER REFERENCES teams(team_id),
    frame_number INTEGER NOT NULL,
    formation VARCHAR(50) NOT NULL,
    confidence REAL DEFAULT 0,
    team_width_m REAL DEFAULT 0,
    team_length_m REAL DEFAULT 0,
    compactness_m REAL DEFAULT 0,
    defensive_line_m REAL DEFAULT 0,
    midfield_line_m REAL DEFAULT 0,
    forward_line_m REAL DEFAULT 0,
    is_formation_change BOOLEAN DEFAULT FALSE,
    change_from VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_formation_match ON formation_history(match_id);
CREATE INDEX idx_formation_team ON formation_history(team_id);
```

#### 10. Tactical Events

```sql
CREATE TABLE tactical_events (
    event_id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(match_id),
    event_type VARCHAR(50) NOT NULL,
    frame_number INTEGER NOT NULL,
    team_id INTEGER REFERENCES teams(team_id),
    player_id INTEGER REFERENCES players(player_id),
    x REAL,
    y REAL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tactical_match ON tactical_events(match_id);
CREATE INDEX idx_tactical_type ON tactical_events(event_type);
```

---

## API ENDPOINTS

### Base URL

```
Production: https://api.footballanalytics.com/v1
Development: http://localhost:8000/v1
```

### Authentication

All endpoints require `Authorization: Bearer <token>` header.

#### POST /auth/login
```json
Request:
{
  "email": "user@example.com",
  "password": "password"
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "user_id": 1,
    "email": "user@example.com",
    "full_name": "John Doe",
    "role": "analyst"
  }
}
```

#### POST /auth/refresh
```json
Request:
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

### Match Management Endpoints

#### GET /matches
Search and filter matches.

**Query Parameters:**
- `season` (string) - Filter by season
- `competition` (string) - Filter by competition
- `team_id` (int) - Filter by team
- `status` (string) - Filter by status (pending/processing/completed/failed)
- `start_date` (date) - Filter matches from date
- `end_date` (date) - Filter matches to date
- `limit` (int, default=20) - Results per page
- `offset` (int, default=0) - Pagination offset

**Response:**
```json
{
  "total": 150,
  "limit": 20,
  "offset": 0,
  "matches": [
    {
      "match_id": 1,
      "home_team": "Manchester City",
      "away_team": "Arsenal",
      "competition": "EPL",
      "season": "2024-25",
      "match_date": "2024-08-15",
      "home_score": 3,
      "away_score": 1,
      "processing_status": "completed",
      "venue": "Etihad Stadium"
    }
  ]
}
```

#### POST /matches/upload
Upload match video for processing.

**Request:** Multipart form data
- `video_file` (file) - Video file
- `metadata` (JSON) - Match metadata

**Response:**
```json
{
  "match_id": 2,
  "status": "queued",
  "estimated_time_seconds": 600,
  "message": "Match queued for processing"
}
```

#### GET /matches/{match_id}/status
Get processing status.

**Response:**
```json
{
  "match_id": 2,
  "status": "processing",
  "progress_pct": 45.2,
  "current_stage": "tracking",
  "estimated_completion_seconds": 120
}
```

#### GET /matches/{match_id}/analytics
Get complete match analytics.

**Response:**
```json
{
  "match_id": 2,
  "analytics": {
    "summary": {...},
    "player_stats": [...],
    "team_stats": {...},
    "events": {...},
    "formations": [...],
    "tactical": {...}
  }
}
```

#### DELETE /matches/{match_id}
Delete match and associated data.

**Response:**
```json
{
  "message": "Match deleted successfully"
}
```

---

### Player Endpoints

#### GET /players
Search players.

**Query Parameters:**
- `name` (string) - Search by name
- `team_id` (int) - Filter by team
- `position` (string) - Filter by position (GK/DEF/MID/FWD)
- `season` (string) - Filter by season
- `limit` (int, default=20)
- `offset` (int, default=0)

**Response:**
```json
{
  "total": 550,
  "limit": 20,
  "offset": 0,
  "players": [
    {
      "player_id": 1,
      "full_name": "Kevin De Bruyne",
      "team_name": "Manchester City",
      "position": "MID",
      "matches_played": 35,
      "average_rating": 7.25,
      "total_xg": 12.5,
      "total_xa": 8.3
    }
  ]
}
```

#### GET /players/{player_id}
Get player profile.

**Response:**
```json
{
  "player_id": 1,
  "profile": {
    "full_name": "Kevin De Bruyne",
    "team_name": "Manchester City",
    "position": "MID",
    "date_of_birth": "1991-06-28",
    "nationality": "Belgium",
    "height_cm": 181,
    "preferred_foot": "Right"
  },
  "season_stats": {
    "matches_played": 35,
    "minutes_played": 3100.5,
    "average_rating": 7.25,
    "total_xg": 12.5,
    "total_xa": 8.3,
    "total_xt": 45.2,
    "total_distance_m": 425000.0,
    "max_speed_kmh": 33.2
  },
  "match_history": [...],
  "development_trends": {...},
  "strengths": ["Passing", "Vision", "Crossing"],
  "weaknesses": ["Defensive contribution", "Aerial duels"]
}
```

#### GET /players/{player_id}/matches
Get player match history.

**Query Parameters:**
- `season` (string)
- `limit` (int)
- `offset` (int)

**Response:**
```json
{
  "player_id": 1,
  "matches": [
    {
      "match_id": 1,
      "date": "2024-08-15",
      "opponent": "Arsenal",
      "venue": "Home",
      "rating": 7.5,
      "goals": 1,
      "assists": 2,
      "xg": 0.85,
      "xa": 1.2
    }
  ]
}
```

#### GET /players/compare
Compare multiple players.

**Query Parameters:**
- `player_ids` (string, comma-separated) - "1,2,3"

**Response:**
```json
{
  "players": [
    {
      "player_id": 1,
      "name": "Kevin De Bruyne",
      "rating": 7.25,
      "xg": 12.5,
      "xa": 8.3,
      "pass_accuracy": 89.5
    }
  ],
  "comparison_metrics": {
    "rating": {"leader": 1, "value": 7.25},
    "xg": {"leader": 1, "value": 12.5},
    "xa": {"leader": 1, "value": 8.3}
  }
}
```

---

### Team Endpoints

#### GET /teams
List teams.

**Query Parameters:**
- `name` (string) - Search by name
- `competition` (string) - Filter by competition
- `season` (string) - Filter by season

**Response:**
```json
{
  "teams": [
    {
      "team_id": 1,
      "team_name": "Manchester City",
      "competition": "EPL",
      "season": "2024-25",
      "matches_played": 35,
      "wins": 28,
      "draws": 5,
      "losses": 2,
      "points": 89,
      "goal_difference": 45
    }
  ]
}
```

#### GET /teams/{team_id}
Get team profile.

**Response:**
```json
{
  "team_id": 1,
  "team_name": "Manchester City",
  "season_stats": {
    "matches_played": 35,
    "wins": 28,
    "draws": 5,
    "losses": 2,
    "goals_scored": 78,
    "goals_conceded": 32,
    "total_xg": 68.5,
    "total_xa": 52.3
  },
  "formation_history": [...],
  "tactical_trends": {...},
  "pressing_metrics": {...},
  "possession_stats": {...},
  "strengths": ["Build-up play", "Wide attacking"],
  "weaknesses": ["Counter-pressing", "Set pieces"]
}
```

#### GET /teams/{team_id}/matches
Get team match history.

**Response:**
```json
{
  "team_id": 1,
  "matches": [
    {
      "match_id": 1,
      "date": "2024-08-15",
      "opponent": "Arsenal",
      "venue": "Home",
      "result": "W",
      "score": "3-1",
      "formation": "4-3-3",
      "possession_pct": 62.5
    }
  ]
}
```

---

### Report Endpoints

#### GET /matches/{match_id}/report
Generate match report.

**Query Parameters:**
- `format` (string) - pdf, docx, html (default: json)

**Response:**
```json
{
  "match_id": 1,
  "report": {
    "scoreline": "Manchester City 3-1 Arsenal",
    "match_statistics": {...},
    "tactical_summary": "...",
    "key_events": [...],
    "best_performers": [...],
    "worst_performers": [...],
    "player_ratings": [...],
    "team_ratings": {...},
    "xg_timeline": [...],
    "pass_network": {...},
    "formation_timeline": [...]
  }
}
```

#### GET /matches/{match_id}/report/pdf
Download PDF report.

**Response:** Binary PDF file

#### GET /matches/{match_id}/report/docx
Download DOCX report.

**Response:** Binary DOCX file

#### GET /matches/{match_id}/report/html
Download HTML report.

**Response:** HTML file

---

### Season Endpoints

#### GET /seasons/{season}/summary
Get season summary.

**Response:**
```json
{
  "season": "2024-25",
  "competition": "EPL",
  "total_matches": 150,
  "total_goals": 425,
  "avg_goals_per_match": 2.83,
  "top_scorers": [...],
  "top_assists": [...],
  "top_rated_players": [...]
}
```

#### GET /seasons/{season}/players
Get player season statistics.

**Query Parameters:**
- `team_id` (int) - Filter by team
- `position` (string) - Filter by position
- `limit` (int)
- `offset` (int)

**Response:**
```json
{
  "players": [
    {
      "player_id": 1,
      "full_name": "Kevin De Bruyne",
      "team_name": "Manchester City",
      "position": "MID",
      "matches_played": 35,
      "goals": 12,
      "assists": 15,
      "xg": 12.5,
      "xa": 8.3,
      "average_rating": 7.25
    }
  ]
}
```

#### GET /seasons/{season}/teams
Get team season statistics.

**Response:**
```json
{
  "teams": [
    {
      "team_id": 1,
      "team_name": "Manchester City",
      "matches_played": 35,
      "wins": 28,
      "draws": 5,
      "losses": 2,
      "points": 89,
      "goal_difference": 45,
      "total_xg": 68.5
    }
  ]
}
```

---

## DASHBOARD STRUCTURE

### Navigation Structure

```
┌──────────────────────────────────────────────────┐
│  Football Analytics Platform                     │
├────────┬────────┬────────┬────────┬─────────────┤
│ Match  │Player  │ Team   │Scout  │  Coach      │ Admin │
│ Center │Profile │Profile │Dash   │  Dashboard  │ Portal│
└────────┴────────┴────────┴────────┴─────────────┘
```

### 1. Match Center Dashboard

**URL:** `/dashboard/matches`

**Features:**
- Upload video button (drag & drop)
- Processing queue with progress bars
- Recent matches table
- Search and filter (season, competition, team, date)
- Match cards with status indicators

**Widgets:**
- Statistics cards (total matches, completed, failed, processing)
- Processing queue (active matches with progress)
- Recent matches list
- Quick filters

**Interactions:**
- Click match card → View match details
- Upload video → Modal upload dialog
- Refresh button → Update queue status

---

### 2. Player Profile Page

**URL:** `/dashboard/players/{player_id}`

**Tabs:**
1. **Overview**
   - Player info card (photo, name, position, team)
   - Season statistics grid
   - Key metrics (rating, distance, speed, possession impact)
   - Strengths/Weaknesses tags

2. **Match History**
   - Table of all matches
   - Filter by season/competition
   - Click row → View match report

3. **Heatmaps**
   - Season heatmap (all matches)
   - Per-match heatmap selector
   - Team comparison overlay

4. **Development Trends**
   - Rating progression chart
   - Speed progression chart
   - Tactical improvement chart
   - Shooting improvement chart

5. **Similar Players**
   - List of players with similar profiles
   - Comparison metrics
   - Similarity score

**Charts:**
- Line charts for progression
- Radar chart for skill profile
- Heatmap visualization

---

### 3. Team Dashboard

**URL:** `/dashboard/teams/{team_id}`

**Tabs:**
1. **Overview**
   - Team info card
   - Season summary (points, GD, W-D-L)
   - League position
   - Recent form (last 5 matches)

2. **Formation History**
   - Formation timeline (Gantt chart)
   - Formation usage pie chart
   - Formation change log

3. **Tactical Trends**
   - Possession trend line chart
   - Pressing trend (PPDA) line chart
   - Home vs Away comparison

4. **Season Statistics**
   - Goals scored/conceded
   - Possession, passing, and pressing totals
   - Passing networks
   - Shot maps

5. **Team Intelligence**
   - Preferred formation
   - Tactical evolution
   - Strongest attacking pattern
   - Weakest defensive pattern

**Charts:**
- Time series for trends
- Bar charts for comparisons
- Network graphs for passing

---

### 4. Match Report Page

**URL:** `/dashboard/matches/{match_id}/report`

**Sections:**
1. **Header**
   - Scoreline (large)
   - Competition, date, venue
   - Match status

2. **Match Statistics**
   - Possession bar chart
   - Shot comparison
   - Pass accuracy comparison
   - PPDA comparison

3. **Tactical Summary**
   - Formation comparison
   - Tactical evolution
   - Key tactical moments

4. **Key Events Timeline**
   - Interactive timeline
   - Goals, shots, passes
   - Click to jump to video frame

5. **Best/Worst Performers**
   - Top 3 players (rating)
   - Bottom 3 players (rating)

6. **Player Ratings**
   - Table with all players
   - Sorting by rating
   - Filter by team

7. **Team Ratings**
   - Overall team rating
   - Component scores (attack, defense, midfield)

8. **Shot Timeline**
   - Cumulative shot chart
   - Shot map overlay

9. **Pass Network**
   - Team passing network
   - Key passers highlighted

10. **Formation Timeline**
    - Formation changes throughout match
    - With confidence scores

**Export Buttons:**
- PDF Report
- DOCX Report
- HTML Report

---

### 5. Scout Dashboard

**URL:** `/dashboard/scout`

**Features:**
1. **Player Search**
   - Search by name/team
   - Filters: age, position, team, competition
   - Results grid

2. **Player Comparison**
   - Select 2-5 players
   - Side-by-side comparison
   - Radar charts
   - Stat comparison table

3. **Player Watchlist**
   - Saved players
   - Notes
   - Alerts

4. **Trend Analysis**
   - Player performance trends
   - Team trends

5. **Export**
   - PDF scouting report
   - Excel data export

---

### 6. Coach Dashboard

**URL:** `/dashboard/coach`

**Features:**
1. **Match Review**
   - Recent matches list
   - Match report viewer
   - Video playback with analytics overlay

2. **Match Comparison**
   - Select 2 matches
   - Side-by-side statistics
   - Tactical differences

3. **Formation Review**
   - Formation timeline
   - Formation effectiveness
   - Formation change history

4. **Tactical Analysis**
   - Pressing metrics (PPDA)
   - Possession trends
   - Attacking patterns
   - Defensive vulnerabilities

5. **Player Workload**
   - Distance covered per player
   - Speed profiles
   - Fatigue indicators
   - Injury risk

---

### 7. Admin Portal

**URL:** `/dashboard/admin`

**Features:**
1. **User Management**
   - User list (table)
   - Add/edit/delete users
   - Role assignment
   - Team assignment

2. **Role Management**
   - Roles list (Admin, Analyst, Coach, Scout)
   - Permissions matrix
   - Role editor

3. **System Settings**
   - Processing queue settings
   - Storage management
   - API keys
   - Analytics version

4. **Audit Log**
   - User actions
   - System events
   - Processing logs

5. **Database Management**
   - Backup/restore
   - Migration status
   - Data quality checks

---

## USER WORKFLOWS

### 1. Match Processing Workflow

```
Analyst uploads video
    ↓
Backend creates MatchRecord (status: pending)
    ↓
Match queued for processing
    ↓
Processing pipeline starts
    ↓
├── Detection
├── Tracking
├── Homography
├── Ball tracking
├── Event detection
├── Formation detection
├── Tactical analysis
└── Intelligence engine
    ↓
All analytics completed
    ↓
MatchRecord status → completed
    ↓
SeasonAggregationEngine updates
    ↓
Notifications sent to coaches/scouts
```

### 2. Scout Workflow

```
Scout logs in
    ↓
Search for players by position/age/team
    ↓
View player profile
    ↓
├── Check season stats
├── Review match history
├── Analyze heatmaps
├── View development trends
└── Compare with other players
    ↓
Add to watchlist
    ↓
Export scouting report
```

### 3. Coach Workflow

```
Coach logs in
    ↓
View team dashboard
    ↓
├── Check recent match results
├── Review formation changes
├── Analyze tactical trends
├── Check player workload
└── Review opponent analysis
    ↓
Generate match report
    ↓
Export PDF for team meeting
```

### 4. Analyst Workflow

```
Analyst logs in
    ↓
Upload match video
    ↓
Monitor processing queue
    ↓
Validate match analytics
    ↓
├── Check tracking quality
├── Verify event detection
├── Review formation detection
└── Validate season statistics
    ↓
Run evaluation framework
    ↓
Generate season reports
```

### 5. Admin Workflow

```
Admin logs in
    ↓
View system health
    ↓
├── Check processing queue
├── Monitor storage usage
├── Review user activity
└── Check audit logs
    ↓
Manage users
    ↓
├── Create new accounts
├── Assign roles
└── Configure permissions
```

---

## PERMISSIONS & RBAC

### Role Definitions

#### 1. Admin
**Full system access**
- User management (create, edit, delete)
- Role management
- System settings
- Database management
- View all data
- Export all reports

#### 2. Analyst
**Data processing and validation**
- Upload matches
- View processing queue
- Validate analytics
- Run evaluation framework
- View all matches
- Export match reports
- No user management

#### 3. Coach
**Team-specific access**
- View own team matches
- View own team players
- Generate match reports
- View tactical analysis
- View player workload
- Export team reports
- No access to opponent data (unless shared)

#### 4. Scout
**Player search and comparison**
- Search players
- View player profiles
- Compare players
- Add to watchlist
- Export scouting reports
- No access to team tactical data

### Permission Matrix

| Feature | Admin | Analyst | Coach | Scout |
|---------|-------|---------|-------|-------|
| Upload matches | ✓ | ✓ | ✗ | ✗ |
| View all matches | ✓ | ✓ | ✗ | ✗ |
| View team matches | ✓ | ✓ | ✓ (own) | ✗ |
| View player profiles | ✓ | ✓ | ✓ (own) | ✓ |
| Compare players | ✓ | ✓ | ✓ (own) | ✓ |
| Generate reports | ✓ | ✓ | ✓ (own) | ✓ (limited) |
| Export PDF/DOCX | ✓ | ✓ | ✓ (own) | ✓ (limited) |
| View tactical trends | ✓ | ✓ | ✓ (own) | ✗ |
| View formation history | ✓ | ✓ | ✓ (own) | ✗ |
| Manage users | ✓ | ✗ | ✗ | ✗ |
| System settings | ✓ | ✗ | ✗ | ✗ |

### Implementation

```python
# FastAPI dependency
def require_role(required_roles: List[str]):
    async def role_checker(
        current_user: User = Depends(get_current_user)
    ):
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Requires one of: {required_roles}"
            )
        return current_user
    return role_checker

# Usage
@app.get("/matches", dependencies=[Depends(require_role(["admin", "analyst"]))])
async def get_matches():
    ...
```

---

## DEPLOYMENT ARCHITECTURE

### Development Environment

```
┌─────────────────────────────────────────┐
│  Developer Machine                       │
│  ┌────────────┬──────────────┐          │
│  │  Frontend  │   Backend    │          │
│  │ (Streamlit │   (FastAPI)  │          │
│  │  :8501)    │   (:8000)    │          │
│  └────────────┴──────────────┘          │
│  ┌─────────────────────────────┐        │
│  │  File System (JSON)         │        │
│  │  outputs/                   │        │
│  └─────────────────────────────┘        │
└─────────────────────────────────────────┘
```

### Production Environment

```
┌─────────────────────────────────────────────────┐
│                    Cloud (AWS/GCP/Azure)        │
│                                                  │
│  ┌─────────────┐    ┌──────────────┐           │
│  │   CloudFlux │    │    Load      │           │
│  │     CDN     │    │   Balancer   │           │
│  └─────────────┘    └──────┬───────┘           │
│                             │                   │
│           ┌─────────────────┼─────────────────┐ │
│           │                 │                 │ │
│  ┌────────▼────────┐  ┌──────▼──────────┐     │ │
│  │   Frontend      │  │   Backend       │     │ │
│  │   (Streamlit    │  │   (FastAPI      │     │ │
│  │    Cloud)       │  │   Container)    │     │ │
│  └─────────────────┘  └────────┬───────┘     │ │
│                                 │              │ │
│  ┌──────────────────────────────┼───────┐     │ │
│  │  Worker Queue (Celery)        │       │     │ │
│  │  - Video processing          │       │     │ │
│  │  - Analytics computation     │       │     │ │
│  └──────────────────────────────┼───────┘     │ │
│                                 │              │ │
│           ┌─────────────────────┼─────────┐   │ │
│           │                     │         │   │ │
│  ┌────────▼────────┐  ┌─────────▼───────┐│   │ │
│  │  PostgreSQL     │  │   S3 / File     ││   │ │
│  │  Database       │  │   Storage       ││   │ │
│  │  (RDS)          │  │                 ││   │ │
│  └─────────────────┘  └─────────────────┘│   │ │
│                                          │   │ │
│  ┌───────────────────────────────────────┘   │ │
│  │  Redis Cache (Session + Queue)            │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- Streamlit (Dashboard)
- Plotly (Charts)
- Pandas (Data processing)
- Requests (API client)

**Backend:**
- FastAPI (API framework)
- SQLAlchemy (ORM)
- Pydantic (Validation)
- OAuth2 + JWT (Authentication)
- Celery (Task queue)
- Redis (Cache + Queue)

**Database:**
- PostgreSQL (Primary)
- Redis (Cache)
- S3 (File storage)

**Analytics:**
- OpenCV (Video processing)
- YOLOv8 (Detection)
- ByteTrack (Tracking)
- NumPy, Pandas (Analytics)

**Deployment:**
- Docker (Containers)
- Docker Compose (Local)
- Kubernetes (Production)
- Nginx (Reverse proxy)
- Prometheus + Grafana (Monitoring)

### Scaling Considerations

1. **Video Processing**
   - Celery workers for async processing
   - GPU workers for YOLOv8 inference
   - S3 for video storage

2. **API Performance**
   - Redis caching for frequent queries
   - Database indexing
   - Connection pooling

3. **Frontend**
   - Streamlit Cloud for hosting
   - CDN for static assets
   - WebSocket for real-time updates

4. **Database**
   - Read replicas for analytics queries
   - Partitioning by season for large tables
   - Archive old matches to cold storage

---

## MODULE SPECIFICATIONS

### Module 1: Match Center

**File:** `app/dashboard/pages/01_Match_Center.py`

**Features:**
- Upload video (drag & drop, 50GB max)
- Processing queue with real-time progress
- Match metadata editor
- Search and filter
- Batch operations (delete, reprocess)

**API Integration:**
- POST /matches/upload
- GET /matches
- GET /matches/{id}/status
- DELETE /matches/{id}

---

### Module 2: Player Profile

**File:** `app/dashboard/pages/02_Player_Profile.py`

**Features:**
- Player info display
- Season statistics
- Match history table
- Heatmap visualization (Plotly)
- Development trends (line charts)
- Similar players (k-nearest neighbors)

**API Integration:**
- GET /players/{id}
- GET /players/{id}/matches
- GET /players/compare

---

### Module 3: Team Dashboard

**File:** `app/dashboard/pages/03_Team_Dashboard.py`

**Features:**
- Team overview
- Formation history timeline
- Tactical trends (possession, PPDA, pressing)
- Season statistics
- Team intelligence summary

**API Integration:**
- GET /teams/{id}
- GET /teams/{id}/matches

---

### Module 4: Match Report

**File:** `app/dashboard/pages/04_Match_Report.py`

**Features:**
- Professional report layout
- Export to PDF/DOCX/HTML
- Interactive charts
- Key events timeline
- Formation timeline

**API Integration:**
- GET /matches/{id}/report
- GET /matches/{id}/report/pdf
- GET /matches/{id}/report/docx
- GET /matches/{id}/report/html

**Libraries:**
- ReportLab (PDF)
- python-docx (DOCX)
- Jinja2 (HTML)

---

### Module 5: Scout Dashboard

**File:** `app/dashboard/pages/05_Scout_Dashboard.py`

**Features:**
- Player search with filters
- Player comparison (2-5 players)
- Watchlist management
- Scouting report export

**API Integration:**
- GET /players
- GET /players/compare

---

### Module 6: Coach Dashboard

**File:** `app/dashboard/pages/12_Coach_Dashboard.py`

**Features:**
- Team match review
- Match comparison
- Formation review
- Tactical analysis
- Player workload

**API Integration:**
- GET /teams/{id}/matches
- GET /matches/{id}/analytics

---

### Module 7: Admin Portal

**File:** `app/dashboard/pages/13_Admin_Portal.py`

**Features:**
- User management
- Role management
- System settings
- Audit logs
- Database management

**API Integration:**
- POST /auth/users
- GET /admin/users
- POST /admin/roles
- GET /admin/audit-log

---

## SECURITY

### Authentication
- OAuth2 with JWT tokens
- Password hashing (bcrypt)
- Token refresh mechanism
- Session management

### Authorization
- Role-based access control (RBAC)
- Resource-level permissions
- Team-level isolation (coaches see only their team)

### Data Protection
- HTTPS only
- CORS configuration
- Rate limiting
- Input validation
- SQL injection prevention (ORM)

### Audit Logging
```sql
CREATE TABLE audit_logs (
    log_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id INTEGER,
    metadata JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## MONITORING

### Metrics
- API response times
- Database query times
- Video processing duration
- Queue lengths
- Error rates

### Logging
- Application logs (structured JSON)
- Access logs
- Error logs
- Audit logs

### Alerts
- Processing failures
- High error rates
- Storage capacity
- Database connection pool exhaustion

---

## BACKUP & DISASTER RECOVERY

### Database Backups
- Daily automated backups
- 30-day retention
- Point-in-time recovery

### File Backups
- Daily S3 sync
- Versioning enabled
- Cross-region replication

### Disaster Recovery
- RTO: 4 hours
- RPO: 24 hours
- Multi-region deployment (optional)

---

## CONFIDENCE LEVEL

**HIGH** - The platform architecture is complete, scalable, and follows industry best practices. All modules are specified, API endpoints defined, database schema normalized, and deployment strategy outlined.

**Next Steps:**
1. Implement FastAPI backend with authentication
2. Build Streamlit dashboard pages
3. Set up PostgreSQL database
4. Deploy to cloud infrastructure
5. Conduct user acceptance testing

**Estimated Implementation Time:** 8-12 weeks for full platform