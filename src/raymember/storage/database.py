"""Database engine and session management for Raymember."""

from contextlib import contextmanager
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""
    pass


class DatabaseManager:
    """Manages SQLite connections and SQLAlchemy sessions."""

    def __init__(self, database_path: str = "raymember.db"):
        self.database_path = database_path
        if database_path == ":memory:":
            self.connection_url = "sqlite:///:memory:"
            self.engine = create_engine(
                self.connection_url,
                connect_args={"check_same_thread": False},
            )
        else:
            abs_path = os.path.abspath(database_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            self.connection_url = f"sqlite:///{abs_path}"
            self.engine = create_engine(self.connection_url)

        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.create_tables()

    def create_tables(self) -> None:
        """Create all tables defined by SQLAlchemy Base."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Dispose database engine resources."""
        self.engine.dispose()
