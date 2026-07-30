"""
Upload API Router Module

Handles video file upload, validation, storage, metadata retrieval, and deletion for the
StepOut Football Analytics Platform.
"""

import logging
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import (
    MAX_UPLOAD_SIZE_BYTES,
    SUPPORTED_VIDEO_FORMATS,
    get_current_timestamp,
    validate_uploaded_file,
    validate_video_file
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# ==========================================
# Upload Directory Configuration
# ==========================================
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Pydantic Response Models
# ==========================================
class UploadMetadata(BaseModel):
    """Metadata response for an uploaded video file."""
    model_config = ConfigDict(from_attributes=True)

    video_id: str = Field(..., description="Unique video identifier (UUID)")
    filename: str = Field(..., description="Original uploaded filename")
    stored_filename: str = Field(..., description="Internal stored filename on disk")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    upload_time: str = Field(..., description="ISO 8601 upload timestamp")
    path: str = Field(..., description="Relative filesystem path to stored video")


class UploadResponse(UploadMetadata):
    """Response returned upon successful video upload."""
    success: bool = True
    message: str = "Video uploaded successfully."


class UploadListResponse(BaseModel):
    """Response model for listing uploaded video files."""
    success: bool = True
    total_count: int = Field(..., description="Total number of uploaded videos stored")
    videos: List[UploadMetadata] = Field(default_factory=list, description="List of uploaded video metadata")
    message: str = "Uploaded videos retrieved successfully."


class DeleteResponse(BaseModel):
    """Response model for video deletion."""
    success: bool = True
    video_id: str
    message: str = "Video deleted successfully."


# ==========================================
# Router Definition
# ==========================================
router = APIRouter(tags=["Upload"])


# ==========================================
# Helper Functions
# ==========================================
def _get_video_path_by_id(video_id: str) -> Optional[Path]:
    """
    Searches the uploads/ directory for a file matching the video_id prefix.

    Args:
        video_id: The UUID string of the video.

    Returns:
        Path object if found, None otherwise.
    """
    if not UPLOAD_DIR.exists():
        return None

    for file_path in UPLOAD_DIR.glob(f"{video_id}_*"):
        if file_path.is_file():
            return file_path
    return None


def _build_metadata_from_path(file_path: Path) -> UploadMetadata:
    """
    Extracts metadata from a stored video file path.

    Args:
        file_path: Path object pointing to the stored file.

    Returns:
        UploadMetadata model instance.
    """
    name_parts = file_path.name.split("_", 1)
    video_id = name_parts[0]
    original_filename = name_parts[1] if len(name_parts) > 1 else file_path.name

    stat = file_path.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    ext = file_path.suffix.lower()
    content_type_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska"
    }
    content_type = content_type_map.get(ext, "video/mp4")

    return UploadMetadata(
        video_id=video_id,
        filename=original_filename,
        stored_filename=file_path.name,
        file_size=stat.st_size,
        content_type=content_type,
        upload_time=mtime_iso,
        path=str(file_path).replace("\\", "/")
    )


# ==========================================
# Endpoints
# ==========================================
@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Match Video",
    description="Uploads a football match video file (.mp4, .mov, .avi, .mkv) up to 500MB."
)
async def upload_video(
    file: UploadFile = Depends(validate_video_file)
) -> UploadResponse:
    """
    Receives an uploaded match video, validates format and size, stores it to disk,
    and returns file metadata.

    Args:
        file: Validated UploadFile instance.

    Returns:
        UploadResponse detailing the stored video metadata.
    """
    video_id = str(uuid.uuid4())[:8]  # Short 8-char UUID prefix for clean naming
    clean_filename = os.path.basename(file.filename or "match_video.mp4")
    stored_filename = f"{video_id}_{clean_filename}"
    destination_path = UPLOAD_DIR / stored_filename

    logger.info("Upload started: filename='%s' (video_id=%s)", clean_filename, video_id)

    try:
        # Save file efficiently in chunks
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = destination_path.stat().st_size

        # Verify max file size post-stream if header was missing
        if file_size > MAX_UPLOAD_SIZE_BYTES:
            destination_path.unlink(missing_ok=True)
            max_mb = MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            logger.warning("Upload rejected: file size %d bytes exceeds %d MB limit", file_size, max_mb)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed limit of {max_mb:.0f} MB."
            )

        upload_time_iso = get_current_timestamp().isoformat()
        content_type = file.content_type or "video/mp4"

        logger.info("Upload completed successfully: '%s' [%d bytes]", stored_filename, file_size)

        return UploadResponse(
            video_id=video_id,
            filename=clean_filename,
            stored_filename=stored_filename,
            file_size=file_size,
            content_type=content_type,
            upload_time=upload_time_iso,
            path=str(destination_path).replace("\\", "/"),
            success=True,
            message="Video uploaded successfully."
        )

    except HTTPException:
        raise
    except Exception as exc:
        destination_path.unlink(missing_ok=True)
        logger.error("Unexpected error saving video file: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while saving the video file."
        )
    finally:
        await file.close()


@router.get(
    "/",
    response_model=UploadListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Uploaded Videos",
    description="Returns metadata for all match video files currently stored in uploads/."
)
async def list_uploaded_videos() -> UploadListResponse:
    """
    Lists all uploaded video files stored in the uploads directory.

    Returns:
        UploadListResponse containing total count and list of UploadMetadata.
    """
    try:
        videos: List[UploadMetadata] = []
        if UPLOAD_DIR.exists():
            for file_path in sorted(UPLOAD_DIR.glob("*"), key=os.path.getmtime, reverse=True):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                    videos.append(_build_metadata_from_path(file_path))

        return UploadListResponse(
            success=True,
            total_count=len(videos),
            videos=videos,
            message="Uploaded videos retrieved successfully."
        )
    except Exception as exc:
        logger.error("Failed to list uploaded videos: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve list of uploaded videos."
        )


@router.get(
    "/{video_id}",
    response_model=UploadMetadata,
    status_code=status.HTTP_200_OK,
    summary="Get Video Metadata",
    description="Retrieves metadata for a single uploaded match video by video_id."
)
async def get_video_metadata(video_id: str) -> UploadMetadata:
    """
    Retrieves metadata for a specific uploaded video file by video_id.

    Args:
        video_id: The video identifier.

    Returns:
        UploadMetadata object.

    Raises:
        HTTPException: HTTP 404 Not Found if video does not exist.
    """
    file_path = _get_video_path_by_id(video_id)
    if not file_path or not file_path.exists():
        logger.warning("Video not found: video_id='%s'", video_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' was not found."
        )

    return _build_metadata_from_path(file_path)


@router.delete(
    "/{video_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Uploaded Video",
    description="Deletes an uploaded match video file from disk by video_id."
)
async def delete_video(video_id: str) -> DeleteResponse:
    """
    Deletes a video file from the uploads directory.

    Args:
        video_id: The video identifier to delete.

    Returns:
        DeleteResponse with confirmation message.

    Raises:
        HTTPException: HTTP 404 Not Found if video does not exist.
    """
    file_path = _get_video_path_by_id(video_id)
    if not file_path or not file_path.exists():
        logger.warning("Delete failed: video_id='%s' not found", video_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID '{video_id}' was not found."
        )

    try:
        file_path.unlink()
        logger.info("Upload deleted: video_id='%s' (file='%s')", video_id, file_path.name)
        return DeleteResponse(
            success=True,
            video_id=video_id,
            message=f"Video '{video_id}' was deleted successfully."
        )
    except Exception as exc:
        logger.error("Failed to delete video_id='%s': %s", video_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete video file for ID '{video_id}'."
        )
