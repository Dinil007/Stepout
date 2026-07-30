"""
Centralized Logging Configuration Module

Configures root logger to write structured execution logs to logs/pipeline.log
and error/warning traces to logs/errors.log.
"""

import logging
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.log"
ERRORS_LOG_PATH = LOGS_DIR / "errors.log"


def setup_central_logging(level: int = logging.INFO) -> None:
    """Configures centralized logging with file handlers for pipeline and error logs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Pipeline Log Handler (INFO & above)
    pipeline_handler = logging.FileHandler(PIPELINE_LOG_PATH, encoding="utf-8")
    pipeline_handler.setLevel(logging.INFO)
    pipeline_handler.setFormatter(formatter)
    root_logger.addHandler(pipeline_handler)

    # 2. Errors Log Handler (WARNING & above)
    errors_handler = logging.FileHandler(ERRORS_LOG_PATH, encoding="utf-8")
    errors_handler.setLevel(logging.WARNING)
    errors_handler.setFormatter(formatter)
    root_logger.addHandler(errors_handler)

    # 3. Console Stream Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info("Centralized logging system initialized.")
