from dataclasses import fields
from typing import Dict, List, Optional, Tuple, Union

import random
import sqlite3
from uuid import uuid4

import torch

from .db import get_connection
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
from .utils import bytes_to_tensor, make_repro_tensor, tensor_to_bytes


_DTO_TABLES: Dict[type, str] = {
    ExperimentTemplateDTO: "ExperimentTemplate",
    VectorDTO: "Vectors",
    ExperimentLiveInstanceDTO: "ExperimentLiveInstance",
    ExperimentSnapshotDTO: "ExperimentSnapshot",
    GeneratedOutputDTO: "GeneratedOutput",
    MetricDTO: "Metric",
    PromptDTO: "Prompt",
    PromptGroupDTO: "PromptGroup",
    GroupPromptLinkDTO: "GroupPrompts",
}

def _table_name_for_dto(dto_type: type) -> str:
    try:
        return _DTO_TABLES[dto_type]
    except KeyError as exc:
        raise ValueError(f"No table mapping for DTO type: {dto_type}") from exc


def _get_persisted_fields(dto_type: type, include_pk: bool = True) -> List[str]:
    result = []
    for f in fields(dto_type):
        if not f.metadata.get("persist", True):
            continue
        if not include_pk and f.metadata.get("primary_key", False):
            continue
        result.append(f.name)
    return result


def _get_primary_key_field(dto_type: type) -> str:
    pk_fields = [
        f.name
        for f in fields(dto_type)
        if f.metadata.get("persist", True) and f.metadata.get("primary_key", False)
    ]
    if not pk_fields:
        raise ValueError(f"No primary key metadata for DTO type: {dto_type}")
    if len(pk_fields) > 1:
        raise ValueError(f"Multiple primary key fields found for DTO type: {dto_type}")
    return pk_fields[0]


def _get_field_def(dto_type: type, field_name: str):
    for field_def in fields(dto_type):
        if field_def.name == field_name:
            return field_def
    raise ValueError(f"Unknown field '{field_name}' for DTO type: {dto_type.__name__}")


def _encode_field_value(dto_type: type, field_name: str, value):
    field_def = _get_field_def(dto_type, field_name)
    encoder = field_def.metadata.get("encode_callback", lambda x: x)
    if value is None:
        return None
    return encoder(value)


def _decode_field_value(dto_type: type, field_name: str, value):
    field_def = _get_field_def(dto_type, field_name)
    decoder = field_def.metadata.get("decode_callback", lambda x: x)
    if value is None:
        return None
    return decoder(value)


def _dto_to_row(
    dto,
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
) -> Dict[str, object]:
    dto_type = type(dto)
    persisted_fields = set(_get_persisted_fields(dto_type, include_pk=True))

    if include_fields is None:
        include_fields = _get_persisted_fields(dto_type, include_pk=True)
    unknown_includes = set(include_fields) - persisted_fields
    if unknown_includes:
        raise ValueError(f"Unknown include fields: {sorted(unknown_includes)}")

    exclude_set = set(exclude_fields or [])
    unknown_excludes = exclude_set - persisted_fields
    if unknown_excludes:
        raise ValueError(f"Unknown exclude fields: {sorted(unknown_excludes)}")

    row = {}
    for name in include_fields:
        if name in exclude_set:
            continue
        row[name] = _encode_field_value(dto_type, name, getattr(dto, name))
    return row


def _row_to_dto(dto_type: type, row):
    if row is None:
        return None

    if hasattr(row, "keys"):
        row_data = {k: row[k] for k in row.keys()}
    else:
        row_data = dict(row)

    kwargs = {}
    for f in fields(dto_type):
        if f.name in row_data:
            kwargs[f.name] = _decode_field_value(dto_type, f.name, row_data[f.name])
    return dto_type(**kwargs)


def _build_insert_sql(table: str, columns: List[str]) -> str:
    if not columns:
        return f"INSERT INTO {table} DEFAULT VALUES"
    placeholders = ", ".join("?" for _ in columns)
    col_sql = ", ".join(columns)
    return f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"


def _build_select_sql(
    table: str,
    columns: List[str],
    where_sql: str = "",
    order_sql: str = "",
) -> str:
    select_cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {select_cols} FROM {table}"
    if where_sql:
        sql += f" WHERE {where_sql}"
    if order_sql:
        sql += order_sql
    return sql


def _build_update_sql(table: str, set_columns: List[str], pk_column: str) -> str:
    set_clause = ", ".join(f"{col} = ?" for col in set_columns)
    return f"UPDATE {table} SET {set_clause} WHERE {pk_column} = ?"


def _build_exact_match_where(columns: List[str]) -> str:
    if not columns:
        return "1=1"
    return " AND ".join(f"{col} IS ?" for col in columns)


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
        return ("1=1", {})

    dto_fields = set(_get_persisted_fields(dto_type, include_pk=True))
    allowed_ops = {"=", "!=", "<>", "<", "<=", ">", ">=", "LIKE", "IS", "IS NOT"}
    predicates = []
    query_params = {}
    for field_name, constraints in filt.items():
        if field_name not in dto_fields:
            raise ValueError(f"Unknown filter field: {field_name}")
        if not isinstance(constraints, dict):
            raise ValueError(
                f"Constraints for '{field_name}' must be a dict of operator to value"
            )
        for op, value in constraints.items():
            op_upper = str(op).upper()
            if op_upper not in allowed_ops:
                raise ValueError(f"Unsupported operator: {op}")
            encoded_value = _encode_field_value(dto_type, field_name, value)
            if encoded_value is None and op_upper in {"=", "IS"}:
                predicates.append(f"{field_name} IS NULL")
            elif encoded_value is None and op_upper in {"!=", "<>", "IS NOT"}:
                predicates.append(f"{field_name} IS NOT NULL")
            else:
                sql_op = "!=" if op_upper == "<>" else op_upper

                param_name = f"filter_{uuid4().hex}"
                query_params[param_name] = encoded_value
                predicates.append(f"{field_name} {sql_op} :{param_name}")

    predicate_sql = " AND ".join(predicates) if predicates else "1=1"
    return (predicate_sql, query_params)


def build_orderby_from_filter(dto_type, orderby: Optional[dict] = None) -> str:
    if not orderby:
        return ""
    if not isinstance(orderby, dict):
        raise ValueError("orderby must be a dict of field to 'asc'/'desc'")

    dto_fields = set(_get_persisted_fields(dto_type, include_pk=True))
    order_parts = []
    for field_name, direction in orderby.items():
        if field_name not in dto_fields:
            raise ValueError(f"Unknown orderby field: {field_name}")
        direction_upper = str(direction).upper()
        if direction_upper not in {"ASC", "DESC"}:
            raise ValueError(
                f"Invalid order direction for '{field_name}': {direction}"
            )
        order_parts.append(f"{field_name} {direction_upper}")

    return " ORDER BY " + ", ".join(order_parts) if order_parts else ""


class _BaseRepository:
    def __init__(self, conn: Optional[sqlite3.Connection] = None):
        self._external_conn = conn

    def _conn(self) -> sqlite3.Connection:
        conn = self._external_conn or get_connection()
        if conn.row_factory is None:
            conn.row_factory = sqlite3.Row
        return conn

    def _close(self, conn: sqlite3.Connection) -> None:
        if not self._external_conn:
            conn.close()

    def get_all(result_filter):
        conn = self._conn()
        cur = conn.cursor()
        table_name = _table_name_for_dto(self.DTOClass)
        
        filter_predicate ,filter_params = build_predicates_from_filter(self.DTOClass,result_filter)
        query_params = {}
        query_params.update(filter_params)

        select_fields = _get_persisted_fields(self.DTOClass, include_pk=True)

        select_sql = _build_select_sql(
            table_name,
            select_fields,
            where_sql=filter_predicate,
            )
        cur.execute(select_sql, query_params)
        conn.commit()
        self._close(conn)
        return dto

    def _create(self, dto):
        matches = self.existance_check(dto)
        if len(matches) == 0:
            print("Match found")
            return matches[0]
        if len(matches) > 1:
            print("Warning: multiple duplicates exist")
            return matches[0]
        print("New Object")
        
        cur = conn.cursor()
        table = _table_name_for_dto(self.DTOClass)
        non_pk_fields = _get_persisted_fields(self.DTOClass, include_pk=False)
        row_values = _dto_to_row(dto, include_fields=non_pk_fields)
        lookup_params = tuple(row_values[field] for field in non_pk_fields)
        insert_sql = _build_insert_sql(table, non_pk_fields)
        cur.execute(insert_sql, lookup_params)
        dto.experiment_template_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

    def get(self,id):
        conn = self._conn()
        cur = conn.cursor()

        table = _table_name_for_dto(self.DTOClass)
        pk_field = _get_primary_key_field(self.DTOClass)
        select_fields = _get_persisted_fields(self.DTOClass, include_pk=True)
        sql = _build_select_sql(table, select_fields, where_sql=f"{pk_field} = ?")

        cur.execute(sql, (experiment_template_id,))
        row = cur.fetchone()
        self._close(conn)
        return _row_to_dto(self.DTOClass, row)

    def existance_check(self):
        conn = self._conn()
        pk_field = _get_primary_key_field(self.DTOClass)
        filt = convert_dto_into_filter(dto,exculde=[pk_field])
        matches = self.get_all(result_filter=filt)
        return matches
    
    #should bear in mind most repos dont want rows updating
    def update(self, dto,exclude=[]) -> None:
        conn = self._conn()
        cur = conn.cursor()
        table = _table_name_for_dto(self.DTOClass)
        pk_field = _get_primary_key_field(self.DTOClass)
        pk = getattr(dto,pk_field)
        if pk is None:
            raise Exception(f"None PK found in {dto}")
        row = self.get(pk)
        if row is None:
            raise Exception(f"{pk} not in {_table_name_for_dto(dto)}")
        set_columns = _get_persisted_fields(self.DTOClass,include_pk=False)
        update_sql = _build_update_sql(table, set_columns, pk_field)
        row_values = _dto_to_row(dto, include_fields=set_columns)
        params = [row_values[col] for col in set_columns]
        params.append(dto.experiment_instance_id)
        cur.execute(update_sql, tuple(params))
        conn.commit()
        self._close(conn)
            
class ExperimentTemplateRepository(_BaseRepository):
    DTOClass = ExperimentTemplateDTO
    def create_from_args(self, *args, **kwargs) -> ExperimentTemplateDTO:
        return self._create(ExperimentTemplateDTO(*args, **kwargs))


class VectorRepository(_BaseRepository):
    DTOClass = VectorDTO

    #over-ridden as vector_data must not be compared, this impacts the _create function as its called there
    def existance_check(self,dto):
        if dto.vetor_data is None:
            raise ValueError("vector_data must be provided")
        pk_field = _get_primary_key_field(self.DTOClass)
        filt = convert_dto_into_filter(dto,exclude = [pk_field,"vector_data"])
        if dto.seed is not None:
            matches = self.get_all(result_filter=filt)
            if matches:
                dto.vector_id = matches[0].vector_id
                return [dto]

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
        conn: Optional[sqlite3.Connection] = None,
        vector_repo: Optional[VectorRepository] = None,
    ):
        super().__init__(conn)
        self.vector_repo = vector_repo or VectorRepository(conn)

    def existance_check(self,dto):
        if dto.vector_data is None:
            raise ValueError("vector_data must be provided") #must it?
        pk_field = _get_primary_key_field(self.DTOClass)
        filt = convert_dto_into_filter(dto,exclude = [pk_field,"vector_data"])
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
        conn: Optional[sqlite3.Connection] = None,
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

#probably just leave this alone
class PromptRepository(_BaseRepository):
    DTOClass = PromptDTO
    def create_group(self) -> PromptGroupDTO:
        conn = self._conn()
        cur = conn.cursor()
        group_table = _table_name_for_dto(PromptGroupDTO)
        insert_sql = _build_insert_sql(group_table, [])
        cur.execute(insert_sql)
        group = PromptGroupDTO(group_id=cur.lastrowid)
        conn.commit()
        self._close(conn)
        return group

    def get_prompts_from_group(self, group_id: int):  
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT p.prompt_id, p.text as text FROM Prompt p JOIN GroupPrompts g ON g.prompt_id = p.prompt_id WHERE g.group_id = ?",
            (group_id,),
        )
        rows = cur.fetchall()
        prompts = [_row_to_dto(PromptDTO, r) for r in rows]
        cur.close()
        self._close(conn)
        return prompts

    # shouldn't be used by user
    def add_dto_to_group(self, group_dto: PromptGroupDTO, prompt_dto: PromptDTO) -> None:
        if group_dto.group_id is None or prompt_dto.prompt_id is None:
            raise ValueError("Both group_id and prompt_id must be set")
        conn = self._conn()
        cur = conn.cursor()

        link_dto = GroupPromptLinkDTO(group_id=group_dto.group_id, prompt_id=prompt_dto.prompt_id)
        link_fields = _get_persisted_fields(GroupPromptLinkDTO, include_pk=False)
        link_values = _dto_to_row(link_dto, include_fields=link_fields)
        insert_sql = _build_insert_sql(_table_name_for_dto(GroupPromptLinkDTO), link_fields)
        params = tuple(link_values[field] for field in link_fields)

        cur.execute(insert_sql, params)
        conn.commit()
        self._close(conn)

    def create_group_from_dtos(self, prompts: List[PromptDTO]) -> PromptGroupDTO:  # assuming persisted dtos
        conn = self._conn()
        prompt_ids = sorted(set(prompt.prompt_id for prompt in prompts))
        placeholders = ",".join("?" for _ in prompt_ids)
        sql = f"""
            SELECT gp.group_id
            FROM GroupPrompts gp
            WHERE gp.prompt_id IN ({placeholders})
            GROUP BY gp.group_id
            HAVING COUNT(DISTINCT gp.prompt_id) = ?
            LIMIT 1
            """
        row = conn.execute(sql, [*prompt_ids, len(prompt_ids)]).fetchone()
        group_id = row["group_id"] if row and hasattr(row, "keys") else (row[0] if row else None)
        if group_id:
            return PromptGroupDTO(group_id)

        group = self.create_group()
        for p in prompts:
            self.add_dto_to_group(group, p)
        return group

    def create_group_from_strings(self, prompts: List[str]) -> PromptGroupDTO:
        prompt_dtos = [self._create(PromptDTO(text=s)) for s in prompts]
        return self.create_group_from_dtos(prompt_dtos)
