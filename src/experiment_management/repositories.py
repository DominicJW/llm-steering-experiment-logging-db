from typing import Any, Dict, Sequence

from sqlalchemy import inspect, select
from sqlalchemy.orm import selectinload
from .db import get_session


class BaseRepository:
    def __init__(self, model):
        self.Model = model

    def _relationship_options(self) -> tuple:
        rels = inspect(self.Model).relationships
        return tuple(selectinload(getattr(self.Model, rel.key)) for rel in rels)

    def persist(self, model, load_relationships: bool = False):
        with get_session() as session:
            session.add(model)
            session.commit()
            session.refresh(model)
        if load_relationships:
            return self.refresh_all(model)
        return model

    def select_filter(self, criteria: Dict[str, Any], load_relationships: bool = False):
        options = self._relationship_options() if load_relationships else tuple()
        with get_session() as session:
            stmt = select(self.Model).filter_by(**criteria).options(*options)
            return session.scalars(stmt).all()

    def refresh(self, model):
        with get_session() as session:
            attached = session.merge(model)
            session.refresh(attached)
            return attached

    def find_by(self, criteria: Dict[str, Any], load_relationships: bool = False):
        return self.select_filter(criteria, load_relationships=load_relationships)

    def find_matching(self, model, excluded: Sequence[str] | None = None):
        state = inspect(model)
        excluded_keys = set(excluded or [])
        excluded_keys.update(pk.key for pk in state.mapper.primary_key) #doesn't really make sense for composite keys

        criteria: Dict[str, Any] = {}
        for attr in state.mapper.column_attrs:
            key = attr.key
            if key in excluded_keys:
                continue
            criteria[key] = state.attrs[key].value
        return self.find_by(criteria)

    def refresh_all(self, model):
        state = inspect(model)
        criteria: Dict[str, Any] = {}
        for pk in state.mapper.primary_key:
            key = pk.key
            value = state.attrs[key].value
            if value is None:
                raise ValueError("primary key is None, unpersisted object cannot be refreshed")
            criteria[key] = value

        result = self.select_filter(criteria, load_relationships=True)
        if not result:
            raise LookupError("object not found during refresh_all")
        return result[0]

    def update(self, model, load_relationships: bool = False):
        state = inspect(model)
        for pk in state.mapper.primary_key:
            if state.attrs[pk.key].value is None:
                raise ValueError("primary key is None, unpersisted object cannot be updated")

        with get_session() as session:
            attached = session.merge(model)
            session.commit()
            session.refresh(attached)

        if load_relationships:
            return self.refresh_all(attached)
        return attached


class VectorRepository(BaseRepository):
    pass
