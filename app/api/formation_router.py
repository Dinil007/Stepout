from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.analytics.formation_config import FormationConfig
from app.analytics.formation_detector import FormationDetector
from app.analytics.formation_engine import FormationEngine, FormationAnalysisResult
from app.analytics.formation_metrics import FormationMetricsEngine
from app.analytics.formation_templates import default_registry
from app.analytics.formation_types import PlayerPosition
from app.analytics.formation_validation import FormationValidator
from app.api.schemas.formation import (
    AnalyzeFormationRequest,
    AnalyzeTeamRequest,
    BatchFormationRequest,
    ConfigResponse,
    FormationResponse,
    HealthResponse,
    MetricsResponse,
    TemplateListResponse,
    ValidationResponse,
    VisualizationRequest,
    VisualizationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/formation", tags=["Formation Intelligence"])

# Shared service instances
_config = FormationConfig()
_detector = FormationDetector(config=_config, registry=default_registry)
_metrics_engine = FormationMetricsEngine(config=_config)
_engine = FormationEngine(config=_config)
_validator = FormationValidator(registry=default_registry)


def _to_player_position(player_data: dict[str, Any]) -> PlayerPosition:
    """Convert dict to PlayerPosition.

    Args:
        player_data: Dictionary with player data.

    Returns:
        PlayerPosition instance.
    """
    return PlayerPosition(
        player_id=player_data["player_id"],
        team_id=player_data["team_id"],
        team_name=player_data["team_name"],
        jersey_number=player_data["jersey_number"],
        x=float(player_data["x"]),
        y=float(player_data["y"]),
        frame_number=player_data["frame_number"],
        timestamp=player_data.get("timestamp") or datetime.now(timezone.utc),
        confidence=float(player_data.get("confidence", 1.0)),
        is_goalkeeper=bool(player_data.get("is_goalkeeper", False)),
        is_visible=bool(player_data.get("is_visible", True)),
    )


def _result_to_response(result: FormationAnalysisResult) -> FormationResponse:
    """Convert FormationAnalysisResult to FormationResponse.

    Args:
        result: FormationAnalysisResult instance.

    Returns:
        FormationResponse instance.
    """
    return FormationResponse(
        detected_formation=result.detected_formation,
        confidence=result.confidence,
        metrics=MetricsResponse(
            team_width=result.metrics.team_width,
            team_length=result.metrics.team_length,
            compactness=result.metrics.compactness,
            centroid_x=result.metrics.centroid_x,
            centroid_y=result.metrics.centroid_y,
            convex_hull_area=result.metrics.convex_hull_area,
            defensive_line=result.metrics.defensive_line,
            midfield_line=result.metrics.midfield_line,
            forward_line=result.metrics.forward_line,
            vertical_stretch=result.metrics.vertical_stretch,
            horizontal_stretch=result.metrics.horizontal_stretch,
        ),
        frame_number=result.frame_number,
        timestamp=result.timestamp,
        team_id=result.team_id,
        analysis_duration_seconds=result.analysis_duration_seconds,
    )


@router.post(
    "/analyze",
    response_model=FormationResponse,
    summary="Analyze formation",
    description="Run full tactical analysis on player positions.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def analyze_formation(request: AnalyzeFormationRequest) -> FormationResponse:
    """Analyze formation for a single frame.

    Args:
        request: AnalyzeFormationRequest containing match and player data.

    Returns:
        FormationResponse with detection and metrics.
    """
    logger.info("Received /formation/analyze request for match=%s frame=%s", request.match_id, request.frame_number)
    start = time.perf_counter()
    try:
        players = [_to_player_position(p.model_dump()) for p in request.players]
        result = _engine.analyze(players, frame_number=request.frame_number, timestamp=request.timestamp)
        logger.info("Analysis completed in %.3fs", time.perf_counter() - start)
        return _result_to_response(result)
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.post(
    "/team",
    response_model=FormationResponse,
    summary="Analyze team formation",
    description="Analyze formation for a specific team.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_404_NOT_FOUND: {"description": "Team not found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def analyze_team(request: AnalyzeTeamRequest) -> FormationResponse:
    """Analyze formation for a specific team.

    Args:
        request: AnalyzeTeamRequest containing team and player data.

    Returns:
        FormationResponse with detection and metrics.
    """
    logger.info("Received /formation/team request for team_id=%s frame=%s", request.team_id, request.frame_number)
    start = time.perf_counter()
    try:
        players = [_to_player_position(p.model_dump()) for p in request.players]
        result = _engine.analyze_team(players, team_id=request.team_id, frame_number=request.frame_number)
        logger.info("Team analysis completed in %.3fs", time.perf_counter() - start)
        return _result_to_response(result)
    except ValueError as exc:
        if "No players found for team_id" in str(exc):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.post(
    "/match",
    response_model=dict[int, FormationResponse],
    summary="Analyze match",
    description="Analyze formations for all teams in a frame.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def analyze_match(request: AnalyzeFormationRequest) -> dict[int, FormationResponse]:
    """Analyze formations for all teams.

    Args:
        request: AnalyzeFormationRequest containing match and player data.

    Returns:
        Mapping from team_id to FormationResponse.
    """
    logger.info("Received /formation/match request for match=%s frame=%s", request.match_id, request.frame_number)
    start = time.perf_counter()
    try:
        players = [_to_player_position(p.model_dump()) for p in request.players]
        results = _engine.analyze_match(players, frame_number=request.frame_number, timestamp=request.timestamp)
        response = {tid: _result_to_response(res) for tid, res in results.items()}
        logger.info("Match analysis completed in %.3fs", time.perf_counter() - start)
        return response
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.post(
    "/batch",
    response_model=list[FormationResponse],
    summary="Batch analyze",
    description="Analyze multiple frames in batch.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def batch_analyze(request: BatchFormationRequest) -> list[FormationResponse]:
    """Run batch analysis on multiple frames.

    Args:
        request: BatchFormationRequest containing frames.

    Returns:
        List of FormationResponse instances.
    """
    logger.info("Received /formation/batch request with %d frames", len(request.frames))
    start = time.perf_counter()
    try:
        frames = [[_to_player_position(p.model_dump()) for p in frame] for frame in request.frames]
        results = _engine.batch_analyze(frames)
        response = [_result_to_response(r) for r in results]
        logger.info("Batch analysis completed in %.3fs", time.perf_counter() - start)
        return response
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate analysis",
    description="Validate a formation analysis result.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def validate_analysis(request: AnalyzeFormationRequest) -> ValidationResponse:
    """Validate analysis result.

    Args:
        request: AnalyzeFormationRequest containing analysis data.

    Returns:
        ValidationResponse.
    """
    logger.info("Received /formation/validate request for match=%s frame=%s", request.match_id, request.frame_number)
    start = time.perf_counter()
    try:
        players = [_to_player_position(p.model_dump()) for p in request.players]
        result = _engine.analyze(players, frame_number=request.frame_number, timestamp=request.timestamp)
        report = _validator.validate_analysis(result)
        logger.info("Validation completed in %.3fs", time.perf_counter() - start)
        return ValidationResponse(
            overall_valid=report.overall_valid,
            errors=report.errors,
            warnings=report.warnings,
            checked_items=report.checked_items,
            passed_items=report.passed_items,
        )
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.post(
    "/visualize",
    response_model=VisualizationResponse,
    summary="Visualize formation",
    description="Render formation analysis to an image.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def visualize_formation(request: VisualizationRequest) -> VisualizationResponse:
    """Visualize formation analysis.

    Args:
        request: VisualizationRequest containing analysis and options.

    Returns:
        VisualizationResponse with image metadata.
    """
    logger.info("Received /formation/visualize request")
    start = time.perf_counter()
    try:
        from app.analytics.formation_visualizer import FormationVisualizer, VisualizerConfig

        visualizer = FormationVisualizer(config=VisualizerConfig(**(request.visualization_options or {})))
        # Create placeholder players and metrics for visualization
        players = []
        metrics_data = request.analysis_result.get("metrics", {})
        if metrics_data:
            from app.analytics.formation_types import FormationMetrics, FormationAnalysisResult

            metrics = FormationMetrics(
                team_width=metrics_data.get("team_width", 0.0),
                team_length=metrics_data.get("team_length", 0.0),
                compactness=metrics_data.get("compactness", 0.0),
                centroid_x=metrics_data.get("centroid_x", 0.5),
                centroid_y=metrics_data.get("centroid_y", 0.5),
                convex_hull_area=metrics_data.get("convex_hull_area", 0.0),
                defensive_line=metrics_data.get("defensive_line", 0.0),
                midfield_line=metrics_data.get("midfield_line", 0.0),
                forward_line=metrics_data.get("forward_line", 0.0),
                vertical_stretch=metrics_data.get("vertical_stretch", 0.0),
                horizontal_stretch=metrics_data.get("horizontal_stretch", 0.0),
            )
            detection = FormationAnalysisResult(
                team_id=0,
                detected_formation=request.analysis_result.get("detected_formation", ""),
                confidence=request.analysis_result.get("confidence", 0.0),
                metrics=metrics,
                frame_number=request.analysis_result.get("frame_number", 0),
                timestamp=request.analysis_result.get("timestamp"),
                analysis_duration_seconds=request.analysis_result.get("analysis_duration_seconds", 0.0),
            )
            frame = visualizer.create_frame(players, metrics, detection)
        else:
            frame = visualizer.create_frame(players, FormationMetrics(
                team_width=0.0, team_length=0.0, compactness=0.0, centroid_x=0.0, centroid_y=0.0,
                convex_hull_area=0.0, defensive_line=0.0, midfield_line=0.0, forward_line=0.0,
                vertical_stretch=0.0, horizontal_stretch=0.0,
            ))
        render_time = time.perf_counter() - start
        logger.info("Visualization completed in %.3fs", render_time)
        return VisualizationResponse(
            success=True,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            render_time=render_time,
        )
    except ImportError as exc:
        logger.error("Visualization dependency missing: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OpenCV is required for visualization") from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.get(
    "/templates",
    response_model=TemplateListResponse,
    summary="List templates",
    description="Get all registered formation templates.",
)
async def get_templates() -> TemplateListResponse:
    """Return registered formation templates.

    Returns:
        TemplateListResponse with template names.
    """
    logger.info("Received /formation/templates request")
    templates = default_registry.list_templates()
    return TemplateListResponse(templates=templates, count=len(templates))


@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get configuration",
    description="Get current formation configuration.",
)
async def get_config() -> ConfigResponse:
    """Return current configuration.

    Returns:
        ConfigResponse with configuration values.
    """
    logger.info("Received /formation/config request")
    data = _config.to_dict()
    return ConfigResponse(**data)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check formation module health.",
)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse indicating service status.
    """
    logger.info("Received /formation/health request")
    return HealthResponse(status="healthy", module="formation", version="1.0.0")