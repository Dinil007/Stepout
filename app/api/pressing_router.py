from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_detector import PressingDetector
from app.analytics.pressing_engine import PressingAnalysisResult, PressingEngine
from app.analytics.pressing_metrics import PressingMetricsEngine
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressingZone,
    PressureEvent,
)
from app.analytics.pressing_validation import PressingValidator
from app.analytics.pressing_visualizer import PressingVisualizer, VisualizerConfig
from app.api.schemas.pressing import (
    AnalyzeMatchRequest,
    AnalyzePressingRequest,
    AnalyzeTeamRequest,
    BatchAnalyzeRequest,
    ConfigResponse,
    HealthResponse,
    PressingAnalysisResponse,
    PressingDetectionSchema,
    PressingMetricsSchema,
    PressingSequenceSchema,
    PressureEventSchema,
    ValidationResponse,
    VisualizationRequest,
    VisualizationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pressing", tags=["Pressing Intelligence"])

# Shared service instances
_config = PressingConfig()
_detector = PressingDetector(config=_config)
_metrics_engine = PressingMetricsEngine(config=_config)
_engine = PressingEngine(config=_config)
_validator = PressingValidator()
_visualizer = PressingVisualizer()


def _to_player_tuple(player: dict[str, Any]) -> tuple[float, float, float, float]:
    """Convert player dict to (x, y, vx, vy) tuple.

    Args:
        player: Dictionary with player data.

    Returns:
        Tuple of (x, y, vx, vy).
    """
    return (
        float(player["x"]),
        float(player["y"]),
        float(player.get("vx", 0.0)),
        float(player.get("vy", 0.0)),
    )


def _event_to_schema(event: PressureEvent) -> PressureEventSchema:
    """Convert PressureEvent to PressureEventSchema.

    Args:
        event: PressureEvent instance.

    Returns:
        PressureEventSchema instance.
    """
    return PressureEventSchema(
        attacker_id=event.attacker_id,
        defender_id=event.defender_id,
        team_id=event.team_id,
        frame_number=event.frame_number,
        distance=event.distance,
        closing_speed=event.closing_speed,
        pressure_angle=event.pressure_angle,
        successful=event.successful,
    )


def _sequence_to_schema(seq: PressingSequence) -> PressingSequenceSchema:
    """Convert PressingSequence to PressingSequenceSchema.

    Args:
        seq: PressingSequence instance.

    Returns:
        PressingSequenceSchema instance.
    """
    return PressingSequenceSchema(
        sequence_id=seq.sequence_id,
        team_id=seq.team_id,
        start_frame=seq.start_frame,
        end_frame=seq.end_frame,
        duration_seconds=seq.duration_seconds,
        event_count=seq.event_count(),
    )


def _metrics_to_schema(metrics: PressingMetrics) -> PressingMetricsSchema:
    """Convert PressingMetrics to PressingMetricsSchema.

    Args:
        metrics: PressingMetrics instance.

    Returns:
        PressingMetricsSchema instance.
    """
    return PressingMetricsSchema(
        total_pressures=metrics.total_pressures,
        successful_pressures=metrics.successful_pressures,
        pressure_success_rate=metrics.pressure_success_rate,
        average_pressure_time=metrics.average_pressure_time,
        average_closing_speed=metrics.average_closing_speed,
        ppda=metrics.ppda,
        high_press_count=metrics.high_press_count,
        mid_block_count=metrics.mid_block_count,
        low_block_count=metrics.low_block_count,
    )


def _detection_to_schema(detection: PressingDetection) -> PressingDetectionSchema:
    """Convert PressingDetection to PressingDetectionSchema.

    Args:
        detection: PressingDetection instance.

    Returns:
        PressingDetectionSchema instance.
    """
    return PressingDetectionSchema(
        pressing_style=detection.pressing_style.value,
        confidence=detection.confidence,
        frame_number=detection.frame_number,
    )


def _result_to_response(
    result: PressingAnalysisResult, frame_number: int
) -> PressingAnalysisResponse:
    """Convert PressingAnalysisResult to PressingAnalysisResponse.

    Args:
        result: PressingAnalysisResult instance.
        frame_number: Frame number for the response.

    Returns:
        PressingAnalysisResponse instance.
    """
    return PressingAnalysisResponse(
        pressure_events=[_event_to_schema(e) for e in result.pressure_events],
        pressing_sequences=[_sequence_to_schema(s) for s in result.pressing_sequences],
        pressing_detection=(
            _detection_to_schema(result.pressing_detection)
            if result.pressing_detection
            else None
        ),
        pressing_metrics=(
            _metrics_to_schema(result.pressing_metrics)
            if result.pressing_metrics
            else None
        ),
        processing_time_ms=result.processing_time_ms,
        frame_number=frame_number,
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=PressingAnalysisResponse,
    summary="Analyze pressing",
    description="Run pressing analysis on a single frame of attacker/defender data.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def analyze_pressing(request: AnalyzePressingRequest) -> PressingAnalysisResponse:
    """Analyze pressing for a single frame.

    Args:
        request: AnalyzePressingRequest containing attacker/defender data.

    Returns:
        PressingAnalysisResponse with events, sequences, metrics, and detection.
    """
    logger.info(
        "Received /pressing/analyze request for frame=%s",
        request.frame_number,
    )
    start = time.perf_counter()
    try:
        attackers = [_to_player_tuple(p.model_dump()) for p in request.attackers]
        defenders = [_to_player_tuple(p.model_dump()) for p in request.defenders]
        result = _engine.analyze(
            attackers=attackers,
            defenders=defenders,
            frame_number=request.frame_number,
            timestamp=request.timestamp,
        )
        logger.info("Analysis completed in %.3fms", result.processing_time_ms)
        return _result_to_response(result, request.frame_number)
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.post(
    "/team",
    response_model=PressingAnalysisResponse,
    summary="Analyze team pressing",
    description="Analyze pressing for a specific team.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def analyze_team(request: AnalyzeTeamRequest) -> PressingAnalysisResponse:
    """Analyze pressing for a specific team.

    Args:
        request: AnalyzeTeamRequest containing team attacker/defender data.

    Returns:
        PressingAnalysisResponse with events, sequences, metrics, and detection.
    """
    logger.info(
        "Received /pressing/team request for frame=%s",
        request.frame_number,
    )
    start = time.perf_counter()
    try:
        team_attackers = [
            _to_player_tuple(p.model_dump()) for p in request.team_attackers
        ]
        team_defenders = [
            _to_player_tuple(p.model_dump()) for p in request.team_defenders
        ]
        result = _engine.analyze_team(
            team_attackers=team_attackers,
            team_defenders=team_defenders,
            frame_number=request.frame_number,
            timestamp=request.timestamp,
        )
        logger.info("Team analysis completed in %.3fms", result.processing_time_ms)
        return _result_to_response(result, request.frame_number)
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.post(
    "/match",
    response_model=dict[int, PressingAnalysisResponse],
    summary="Analyze match",
    description="Analyze pressing across multiple frames.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def analyze_match(
    request: AnalyzeMatchRequest,
) -> dict[int, PressingAnalysisResponse]:
    """Analyze pressing across multiple frames.

    Args:
        request: AnalyzeMatchRequest containing frame data.

    Returns:
        Mapping from frame number to PressingAnalysisResponse.
    """
    logger.info("Received /pressing/match request with %d frames", len(request.frames))
    start = time.perf_counter()
    try:
        frames = [
            (
                [_to_player_tuple(p.model_dump()) for p in frame.attackers],
                [_to_player_tuple(p.model_dump()) for p in frame.defenders],
            )
            for frame in request.frames
        ]
        results = _engine.analyze_match(
            frames=frames,
            frame_numbers=request.frame_numbers,
            timestamps=request.timestamps,
        )
        response = {
            fn: _result_to_response(res, fn) for fn, res in results.items()
        }
        elapsed = time.perf_counter() - start
        logger.info("Match analysis completed in %.3fs", elapsed)
        return response
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.post(
    "/batch",
    response_model=list[PressingAnalysisResponse],
    summary="Batch analyze",
    description="Analyze multiple frames in batch.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def batch_analyze(request: BatchAnalyzeRequest) -> list[PressingAnalysisResponse]:
    """Run batch pressing analysis on multiple frames.

    Args:
        request: BatchAnalyzeRequest containing frames.

    Returns:
        List of PressingAnalysisResponse instances.
    """
    logger.info(
        "Received /pressing/batch request with %d frames", len(request.frames)
    )
    start = time.perf_counter()
    try:
        frames = [
            (
                [_to_player_tuple(p.model_dump()) for p in frame.attackers],
                [_to_player_tuple(p.model_dump()) for p in frame.defenders],
            )
            for frame in request.frames
        ]
        results = _engine.batch_analyze(
            frames=frames, frame_numbers=request.frame_numbers
        )
        response = [
            _result_to_response(res, res.metadata.get("frame_number", idx))
            for idx, res in enumerate(results)
        ]
        elapsed = time.perf_counter() - start
        logger.info("Batch analysis completed in %.3fs", elapsed)
        return response
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Validate pressing analysis",
    description="Validate a pressing analysis result.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def validate_pressing(
    request: AnalyzePressingRequest,
) -> ValidationResponse:
    """Validate pressing analysis result.

    Args:
        request: AnalyzePressingRequest containing analysis data.

    Returns:
        ValidationResponse with validation report.
    """
    logger.info(
        "Received /pressing/validate request for frame=%s", request.frame_number
    )
    start = time.perf_counter()
    try:
        attackers = [_to_player_tuple(p.model_dump()) for p in request.attackers]
        defenders = [_to_player_tuple(p.model_dump()) for p in request.defenders]
        result = _engine.analyze(
            attackers=attackers,
            defenders=defenders,
            frame_number=request.frame_number,
            timestamp=request.timestamp,
        )
        report = _validator.validate_analysis(result)
        elapsed = time.perf_counter() - start
        logger.info("Validation completed in %.3fs", elapsed)
        return ValidationResponse(
            overall_valid=report.overall_valid,
            errors=report.errors,
            warnings=report.warnings,
            checked_items=report.checked_items,
            passed_items=report.passed_items,
        )
    except ValueError as exc:
        logger.error("Bad request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.post(
    "/visualize",
    response_model=VisualizationResponse,
    summary="Visualize pressing",
    description="Render pressing analysis to an image.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
    },
)
async def visualize_pressing(
    request: VisualizationRequest,
) -> VisualizationResponse:
    """Visualize pressing analysis.

    Args:
        request: VisualizationRequest containing data and options.

    Returns:
        VisualizationResponse with image metadata.
    """
    logger.info("Received /pressing/visualize request")
    start = time.perf_counter()
    try:
        viz_config = VisualizerConfig(
            **(request.visualization_options or {})
        )
        visualizer = PressingVisualizer(config=viz_config)

        # Convert player positions
        player_positions = None
        if request.player_positions:
            player_positions = [
                (int(p[0]), float(p[1]), float(p[2]), int(p[3]))
                for p in request.player_positions
            ]

        # Convert ball position
        ball_position = None
        if request.ball_position and len(request.ball_position) >= 2:
            ball_position = (float(request.ball_position[0]), float(request.ball_position[1]))

        # Convert attacker/defender positions
        attacker_positions = None
        if request.attacker_positions:
            attacker_positions = {
                int(k): (float(v[0]), float(v[1]))
                for k, v in request.attacker_positions.items()
            }
        defender_positions = None
        if request.defender_positions:
            defender_positions = {
                int(k): (float(v[0]), float(v[1]))
                for k, v in request.defender_positions.items()
            }

        # Convert metrics
        metrics = None
        if request.metrics:
            metrics = PressingMetrics(
                total_pressures=request.metrics.get("total_pressures", 0),
                successful_pressures=request.metrics.get("successful_pressures", 0),
                pressure_success_rate=request.metrics.get("pressure_success_rate", 0.0),
                average_pressure_time=request.metrics.get("average_pressure_time", 0.0),
                average_closing_speed=request.metrics.get("average_closing_speed", 0.0),
                ppda=request.metrics.get("ppda", 0.0),
                high_press_count=request.metrics.get("high_press_count", 0),
                mid_block_count=request.metrics.get("mid_block_count", 0),
                low_block_count=request.metrics.get("low_block_count", 0),
            )

        # Convert detection
        detection = None
        if request.detection:
            style_str = request.detection.get("pressing_style", "low_block")
            try:
                pressing_style = PressingZone(style_str)
            except ValueError:
                pressing_style = PressingZone.LOW_BLOCK
            detection = PressingDetection(
                pressing_style=pressing_style,
                confidence=request.detection.get("confidence", 0.0),
                frame_number=request.detection.get("frame_number", 0),
                timestamp=request.detection.get("timestamp"),
            )

        frame = visualizer.render_frame(
            player_positions=player_positions,
            ball_position=ball_position,
            pressure_events=None,
            sequences=None,
            metrics=metrics,
            detection=detection,
            attacker_positions=attacker_positions,
            defender_positions=defender_positions,
        )

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenCV is required for visualization",
        ) from exc
    except Exception as exc:
        logger.exception("Internal error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get configuration",
    description="Get current pressing configuration.",
)
async def get_config() -> ConfigResponse:
    """Return current pressing configuration.

    Returns:
        ConfigResponse with configuration values.
    """
    logger.info("Received /pressing/config request")
    data = _config.to_dict()
    return ConfigResponse(**data)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check pressing module health.",
)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse indicating service status.
    """
    logger.info("Received /pressing/health request")
    return HealthResponse(status="healthy", module="pressing", version="1.0.0")