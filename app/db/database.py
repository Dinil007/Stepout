"""
Database Configuration & Session Management

Provides SQLAlchemy 2.x engine, session factory, declarative base, and
FastAPI dependency injection for PostgreSQL database connectivity.
Configuration is loaded exclusively from environment variables / .env file.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from pydantic import PostgresDsn, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Load .env file before settings are evaluated
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


# ==========================================
# Database Settings (Pydantic v2 BaseSettings)
# ==========================================
class DatabaseSettings(BaseSettings):
    """
    Reads all database connection parameters from environment variables or .env file.
    Pydantic v2 validates types and raises clear errors on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str

    # Connection pool configuration
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_RECYCLE: int = 1800       # Recycle connections after 30 minutes
    POOL_PRE_PING: bool = True     # Verify connection liveness before use
    POOL_TIMEOUT: int = 30         # Max seconds to wait for a connection

    @property
    def database_url(self) -> str:
        """
        Constructs the PostgreSQL connection URL from individual components.

        Returns:
            A fully qualified postgresql+psycopg2 database DSN string.
        """
        return (
            f"postgresql+psycopg2://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )


# ==========================================
# Load & Validate Settings at Module Level
# ==========================================
try:
    db_settings = DatabaseSettings()
    logger.info(
        "Database configuration loaded: host=%s port=%d dbname=%s",
        db_settings.DATABASE_HOST,
        db_settings.DATABASE_PORT,
        db_settings.DATABASE_NAME
    )
except ValidationError as exc:
    logger.critical("Invalid database configuration. Check environment variables.\n%s", exc)
    raise


# ==========================================
# SQLAlchemy 2.x Engine
# ==========================================
engine = create_engine(
    db_settings.database_url,
    pool_pre_ping=db_settings.POOL_PRE_PING,
    pool_recycle=db_settings.POOL_RECYCLE,
    pool_size=db_settings.POOL_SIZE,
    max_overflow=db_settings.MAX_OVERFLOW,
    pool_timeout=db_settings.POOL_TIMEOUT,
    future=True,          # Enable SQLAlchemy 2.x style execution
    echo=False,           # Set True for SQL query logging in development
)

# Attach engine-level event listeners for connection diagnostics
@event.listens_for(engine, "connect")
def on_connect(dbapi_connection, connection_record) -> None:
    """Logs every successful new raw DBAPI connection checkout."""
    logger.debug("New database connection established: %s", connection_record)


@event.listens_for(engine, "checkout")
def on_checkout(dbapi_connection, connection_record, connection_proxy) -> None:
    """Logs every connection pool checkout."""
    logger.debug("Database connection checked out from pool.")


# ==========================================
# Session Factory
# ==========================================
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ==========================================
# Declarative Base (SQLAlchemy 2.x style)
# ==========================================
class Base(DeclarativeBase):
    """
    SQLAlchemy 2.x DeclarativeBase for all ORM model classes.
    All models should inherit from this Base.
    """
    pass


# ==========================================
# FastAPI Dependency: get_db()
# ==========================================
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per request.
    The session is automatically closed after the request completes,
    and any uncommitted transaction is rolled back on exception.

    Yields:
        Session: An active SQLAlchemy ORM session.

    Example:
        @router.get("/players")
        def list_players(db: Session = Depends(get_db)):
            return db.query(Player).all()
    """
    db: Session = SessionLocal()
    try:
        logger.debug("Database session opened.")
        yield db
        db.commit()
        logger.debug("Database session committed successfully.")
    except SQLAlchemyError as exc:
        logger.error("Database error during request, rolling back: %s", exc)
        db.rollback()
        raise
    except Exception as exc:
        logger.error("Unexpected error, rolling back database session: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()
        logger.debug("Database session closed.")


# ==========================================
# Context Manager: db_session()
# ==========================================
@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Reusable context manager for database sessions in non-FastAPI contexts
    (e.g., background tasks, CLI scripts, pipeline integration).

    Example:
        with db_session() as db:
            db.add(player_record)

    Yields:
        Session: An active SQLAlchemy ORM session.
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as exc:
        logger.error("Database session error, rolling back: %s", exc)
        db.rollback()
        raise
    except Exception as exc:
        logger.error("Unexpected error in db_session context, rolling back: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


# ==========================================
# Database Initialization
# ==========================================
def init_db() -> None:
    """
    Creates all tables defined in SQLAlchemy ORM models that inherit from Base.

    Should be called once at application startup (e.g., FastAPI lifespan event).
    Safe to call multiple times — only creates tables that do not already exist.

    Raises:
        OperationalError: If the database is unreachable during initialization.
    """
    try:
        logger.info("Initializing database schema...")

        # Verify connectivity before attempting DDL
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            logger.info("Database connectivity verified.")

        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")

    except OperationalError as exc:
        logger.critical(
            "Cannot connect to the database at %s:%d. "
            "Ensure PostgreSQL is running and credentials are correct.\nError: %s",
            db_settings.DATABASE_HOST,
            db_settings.DATABASE_PORT,
            exc
        )
        raise
    except SQLAlchemyError as exc:
        logger.error("Failed to initialize database schema: %s", exc)
        raise


# ==========================================
# Health Check Utility
# ==========================================
def check_db_health() -> bool:
    """
    Executes a lightweight connectivity probe against the database.

    Returns:
        True if the database is reachable, False otherwise.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database health check: OK")
        return True
    except OperationalError as exc:
        logger.error("Database health check FAILED: %s", exc)
        return False
