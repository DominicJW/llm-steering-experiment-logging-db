from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from sqlalchemy import and_, distinct, func, select
from sqlalchemy.orm import Session

from .db import get_session
from .dto import (
    ExperimentLiveInstanceDTO,
    ExperimentSnapshotDTO,
    ExperimentTemplateDTO,
    GeneratedOutputDTO,
    GroupPromptLinkDTO,
    MetricDTO,
    PromptDTO,
    PromptGroupDTO,
    VectorDTO,
)
from .utils import make_repro_tensor


def _get_persisted_fields(dto_type: type, include_pk: bool = True) -> List[str]:
    mapper = dto_type.__mapper__
    pk_names = {col.key for col in mapper.primary_key}
    fields = [prop.key for prop in mapper.column_attrs]
    if include_pk:
        return fields
    return [name for name in fields if name not in pk_names]


def _get_primary_key_field(dto_type: type) -> str:
    pk_fields = [col.key for col in dto_type.__mapper__.primary_key]
    if not pk_fields:
        raise ValueError(f"No primary key metadata for DTO type: {dto_type}")
    if len(pk_fields) > 1:
        raise ValueError(f"Multiple primary key fields found for DTO type: {dto_type}")
    return pk_fields[0]


def _get_model_column(dto_type: type, field_name: str):
    dto_fields = set(_get_persisted_fields(dto_type, include_pk=True))
    if field_name not in dto_fields:
        raise ValueError(f"Unknown filter/orderby field: {field_name}")
    return getattr(dto_type, field_name)


def convert_dto_into_filter(
    dto,
    exclude: Optional[List[str]] = None,
):
    dto_field_names = _get_persisted_fields(type(dto), include_pk=True)
    exclude_set = set(exclude or [])
    unknown = exclude_set - set(dto_field_names)
    if unknown:
        raise ValueError(f"Unknown exclude fields: {sorted(unknown)}")

    filt = {}
    for field_name in dto_field_names:
        if field_name in exclude_set:
            continue
        filt[field_name] = {"=": getattr(dto, field_name)}
    return filt


def build_predicates_from_filter(dto_type, filt: Optional[dict] = None):
    if not filt:
        return []

    allowed_ops = {"=", "!=", "<>", "<", "<=", ">", ">=", "LIKE", "IS", "IS NOT"}
    predicates = []
    for field_name, constraints in filt.items():
        column = _get_model_column(dto_type, field_name)
        if not isinstance(constraints, dict):
            raise ValueError(
                f"Constraints for '{field_name}' must be a dict of operator to value"
            )
        for op, value in constraints.items():
            op_upper = str(op).upper()
            if op_upper not in allowed_ops:
                raise ValueError(f"Unsupported operator: {op}")

            if op_upper in {"=", "IS"} and value is None:
                predicates.append(column.is_(None))
            elif op_upper in {"!=", "<>", "IS NOT"} and value is None:
                predicates.append(column.is_not(None))
            elif op_upper == "=":
                predicates.append(column == value)
            elif op_upper in {"!=", "<>"}:
                predicates.append(column != value)
            elif op_upper == "<":
                predicates.append(column < value)
            elif op_upper == "<=":
                predicates.append(column <= value)
            elif op_upper == ">":
                predicates.append(column > value)
            elif op_upper == ">=":
                predicates.append(column >= value)
            elif op_upper == "LIKE":
                predicates.append(column.like(value))
            elif op_upper == "IS":
                predicates.append(column.is_(value))
            elif op_upper == "IS NOT":
                predicates.append(column.is_not(value))

    return predicates


def build_orderby_from_filter(dto_type, orderby: Optional[dict] = None):
    if not orderby:
        return []
    if not isinstance(orderby, dict):
        raise ValueError("orderby must be a dict of field to 'asc'/'desc'")

    order_parts = []
    for field_name, direction in orderby.items():
        column = _get_model_column(dto_type, field_name)
        direction_upper = str(direction).upper()
        if direction_upper == "ASC":
            order_parts.append(column.asc())
        elif direction_upper == "DESC":
            order_parts.append(column.desc())
        else:
            raise ValueError(
                f"Invalid order direction for '{field_name}': {direction}"
            )

    return order_parts


class _BaseRepository:
    DTOClass = None

    def __init__(self, conn: Optional[Session] = None):
        self._external_session = conn

    def _open_session(self) -> Tuple[Session, bool]:
        if self._external_session is not None:
            return self._external_session, False
        return get_session(), True

    @staticmethod
    def _close_session(session: Session, should_close: bool) -> None:
        if should_close:
            session.close()

    def get_all(self, result_filter=None, orderby=None):
        session, should_close = self._open_session()
        try:
            stmt = select(self.DTOClass)
            predicates = build_predicates_from_filter(self.DTOClass, result_filter)
            if predicates:
                stmt = stmt.where(and_(*predicates))
            order_clauses = build_orderby_from_filter(self.DTOClass, orderby)
            if order_clauses:
                stmt = stmt.order_by(*order_clauses)
            return list(session.scalars(stmt).all())
        finally:
            self._close_session(session, should_close)

    def _create(self, dto):
        matches = self.existance_check(dto)
        if len(matches) == 1:
            print("Match found")
            return matches[0]
        if len(matches) > 1:
            print("Warning: multiple duplicates exist")
            return matches[0]
        print("New Object")

        session, should_close = self._open_session()
        try:
            session.add(dto)
            session.commit()
            session.refresh(dto)
            return dto
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session, should_close)

    def get(self, id):
        session, should_close = self._open_session()
        try:
            return session.get(self.DTOClass, id)
        finally:
            self._close_session(session, should_close)

    def existance_check(self, dto):
        pk_field = _get_primary_key_field(self.DTOClass)
        filt = convert_dto_into_filter(dto, exclude=[pk_field])
        matches = self.get_all(result_filter=filt)
        return matches

    # should bear in mind most repos dont want rows updating
    def update(self, dto, exclude=None) -> None:
        exclude_set = set(exclude or [])
        session, should_close = self._open_session()
        try:
            pk_field = _get_primary_key_field(self.DTOClass)
            pk = getattr(dto, pk_field)
            if pk is None:
                raise Exception(f"None PK found in {dto}")

            persisted = session.get(self.DTOClass, pk)
            if persisted is None:
                raise Exception(f"{pk} not in {self.DTOClass.__tablename__}")

            set_columns = [
                col
                for col in _get_persisted_fields(self.DTOClass, include_pk=False)
                if col not in exclude_set
            ]
            for col in set_columns:
                setattr(persisted, col, getattr(dto, col))

            session.commit()

            # Keep caller's object synchronized with persisted state
            for col in _get_persisted_fields(self.DTOClass, include_pk=True):
                setattr(dto, col, getattr(persisted, col))
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session, should_close)


class ExperimentTemplateRepository(_BaseRepository):
    DTOClass = ExperimentTemplateDTO

    def create_from_args(self, *args, **kwargs) -> ExperimentTemplateDTO:
        if args:
            field_names = _get_persisted_fields(self.DTOClass, include_pk=True)
            if len(args) > len(field_names):
                raise TypeError(
                    f"Expected at most {len(field_names)} positional args, got {len(args)}"
                )
            for name, value in zip(field_names, args):
                if name in kwargs:
                    raise TypeError(f"Got multiple values for argument '{name}'")
                kwargs[name] = value
        return self._create(ExperimentTemplateDTO(**kwargs))


class VectorRepository(_BaseRepository):
    DTOClass = VectorDTO

    # over-ridden as vector_data must not be compared, this impacts the _create function as its called there
    def existance_check(self, dto):
        if dto.vector_data is None:
            raise ValueError("vector_data must be provided")
        pk_field = _get_primary_key_field(self.DTOClass)
        filt = convert_dto_into_filter(dto, exclude=[pk_field, "vector_data"])
        matches = self.get_all(result_filter=filt)
        if dto.seed is not None and matches:
            dto.vector_id = matches[0].vector_id
            return [dto]
        return matches

    def create_from_seed(
        self,
        seed: int,
        shape: Union[Tuple[int, ...], torch.Size],
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> VectorDTO:
        tensor = make_repro_tensor(shape, seed=seed, device=device, dtype=dtype)
        return self._create(VectorDTO(vector_data=tensor, vector_id=None, seed=seed))

    def create_from_tensor(self, tensor: torch.Tensor) -> VectorDTO:
        return self._create(VectorDTO(vector_data=tensor, vector_id=None))

    def create_from_shape(self, shape):
        seed = random.randint(1, int(1e8))
        return self.create_from_seed(seed=seed, shape=shape)


class ExperimentLiveInstanceRepository(_BaseRepository):
    DTOClass = ExperimentLiveInstanceDTO

    def __init__(
        self,
        conn: Optional[Session] = None,
        vector_repo: Optional[VectorRepository] = None,
    ):
        super().__init__(conn)
        self.vector_repo = vector_repo or VectorRepository(conn)

    def existance_check(self, dto):
        if dto.vector_data is None:
            raise ValueError("vector_data must be provided")
        pk_field = _get_primary_key_field(self.DTOClass)
        filt = convert_dto_into_filter(dto, exclude=[pk_field, "vector_data"])
        matches = self.get_all(result_filter=filt)
        print(f"INFO LiveInstnace Matches Found, but new one being used: {len(matches)}")
        return []

    def create_from_vec_dto(
        self, et_id: int, vec_dto: VectorDTO
    ) -> ExperimentLiveInstanceDTO:
        dto = ExperimentLiveInstanceDTO(
            vector_data=vec_dto.vector_data,
            initial_vector_id=vec_dto.vector_id,
            experiment_template_id=et_id,
        )
        return self._create(dto)

    def create_from_initial_tensor(
        self, et_id: int, tensor: torch.Tensor
    ) -> ExperimentLiveInstanceDTO:
        vec_dto = self.vector_repo.create_from_tensor(tensor)
        return self.create_from_vec_dto(et_id, vec_dto)

    def create_from_seed(
        self,
        et_id: int,
        shape: Union[Tuple[int, ...], torch.Size],
        seed: int,
    ) -> ExperimentLiveInstanceDTO:
        vec_dto = self.vector_repo.create_from_seed(seed=seed, shape=shape)
        return self.create_from_vec_dto(et_id, vec_dto)

    def create_from_template(
        self,
        et_id: int,
        shape: Union[Tuple[int, ...], torch.Size],
    ) -> ExperimentLiveInstanceDTO:
        seed = random.randint(1, int(1e8))
        return self.create_from_seed(et_id, shape, seed)


class ExperimentSnapshotRepository(_BaseRepository):
    DTOClass = ExperimentSnapshotDTO

    def __init__(
        self,
        conn: Optional[Session] = None,
        vector_repo: Optional[VectorRepository] = None,
    ):
        super().__init__(conn)
        self.vector_repo = vector_repo or VectorRepository(conn)

    def create_from_live(
        self, inst_dto: ExperimentLiveInstanceDTO, save_vector: bool = True
    ) -> ExperimentSnapshotDTO:
        if save_vector:
            vector_dto = self.vector_repo.create_from_tensor(inst_dto.vector_data)
            vector_id = vector_dto.vector_id
        else:
            vector_id = None

        snapshot = ExperimentSnapshotDTO(
            snapshot_id=None,
            vector_id=vector_id,
            iteration_count=inst_dto.iteration_count,
            experiment_instance_id=inst_dto.experiment_instance_id,
        )
        return self._create(snapshot)


class GeneratedOutputRepository(_BaseRepository):
    DTOClass = GeneratedOutputDTO


class MetricRepository(_BaseRepository):
    DTOClass = MetricDTO


class PromptRepository(_BaseRepository):
    DTOClass = PromptDTO

    def create_group(self) -> PromptGroupDTO:
        session, should_close = self._open_session()
        try:
            group = PromptGroupDTO()
            session.add(group)
            session.commit()
            session.refresh(group)
            return group
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session, should_close)

    def get_prompts_from_group(self, group_id: int):
        session, should_close = self._open_session()
        try:
            stmt = (
                select(PromptDTO)
                .join(GroupPromptLinkDTO, GroupPromptLinkDTO.prompt_id == PromptDTO.prompt_id)
                .where(GroupPromptLinkDTO.group_id == group_id)
            )
            return list(session.scalars(stmt).all())
        finally:
            self._close_session(session, should_close)

    # shouldn't be used by user
    def add_dto_to_group(self, group_dto: PromptGroupDTO, prompt_dto: PromptDTO) -> None:
        if group_dto.group_id is None or prompt_dto.prompt_id is None:
            raise ValueError("Both group_id and prompt_id must be set")

        session, should_close = self._open_session()
        try:
            link_dto = GroupPromptLinkDTO(
                group_id=group_dto.group_id,
                prompt_id=prompt_dto.prompt_id,
            )
            session.add(link_dto)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session, should_close)

    def create_group_from_dtos(self, prompts: List[PromptDTO]) -> PromptGroupDTO:  # assuming persisted dtos
        prompt_ids = sorted(set(prompt.prompt_id for prompt in prompts if prompt.prompt_id is not None))
        if not prompt_ids:
            raise ValueError("At least one persisted prompt DTO is required")

        session, should_close = self._open_session()
        try:
            # Find a group containing exactly this prompt-id set.
            candidate_group_ids = (
                select(GroupPromptLinkDTO.group_id)
                .where(GroupPromptLinkDTO.prompt_id.in_(prompt_ids))
                .group_by(GroupPromptLinkDTO.group_id)
                .having(func.count(distinct(GroupPromptLinkDTO.prompt_id)) == len(prompt_ids))
            )

            exact_group_stmt = (
                select(GroupPromptLinkDTO.group_id)
                .where(GroupPromptLinkDTO.group_id.in_(candidate_group_ids))
                .group_by(GroupPromptLinkDTO.group_id)
                .having(func.count(GroupPromptLinkDTO.prompt_id) == len(prompt_ids))
                .limit(1)
            )
            group_id = session.scalar(exact_group_stmt)
            if group_id is not None:
                return session.get(PromptGroupDTO, group_id)

            group = PromptGroupDTO()
            session.add(group)
            session.flush()
            for prompt_id in prompt_ids:
                session.add(GroupPromptLinkDTO(group_id=group.group_id, prompt_id=prompt_id))

            session.commit()
            session.refresh(group)
            return group
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session, should_close)

    def create_group_from_strings(self, prompts: List[str]) -> PromptGroupDTO:
        prompt_dtos = [self._create(PromptDTO(text=s)) for s in prompts]
        return self.create_group_from_dtos(prompt_dtos)
