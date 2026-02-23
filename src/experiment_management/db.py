from __future__ import annotations


from typing import Dict, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .dto import Base

DB_PATH = "experiments.db"

_ENGINE_BY_PATH: Dict[str, Engine] = {}
_SESSION_FACTORY_BY_PATH: Dict[str, sessionmaker] = {}


def _resolve_db_path(path: Optional[str] = None) -> str:
    return path or DB_PATH


def _sqlite_url(path: str) -> str:
    return f"sqlite:///{path}"


def get_engine(path: Optional[str] = None) -> Engine:
    resolved_path = _resolve_db_path(path)
    engine = _ENGINE_BY_PATH.get(resolved_path)
    if engine is None:
        engine = create_engine(_sqlite_url(resolved_path), future=True)
        _ENGINE_BY_PATH[resolved_path] = engine
    return engine


def get_session_factory(path: Optional[str] = None) -> sessionmaker:
    resolved_path = _resolve_db_path(path)
    factory = _SESSION_FACTORY_BY_PATH.get(resolved_path)
    if factory is None:
        factory = sessionmaker(bind=get_engine(resolved_path), expire_on_commit=False)
        _SESSION_FACTORY_BY_PATH[resolved_path] = factory
    return factory


def get_session(path: Optional[str] = None) -> Session:
    return get_session_factory(path)()


def init_schema(path: Optional[str] = None) -> None:
    engine = get_engine(path)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_schema()
