from typing import Optional, List, Tuple, Union
import sqlite3
import io
import random
import torch
from dataclasses import fields

from .db import get_connection
from .dto import (
    ExperimentTemplateDTO,
    VectorDTO,
    ExperimentLiveInstanceDTO,
    ExperimentSnapshotDTO,
    GeneratedOutputDTO,
    MetricDTO,
    PromptDTO,
    PromptGroupDTO,
)
from .utils import tensor_to_bytes, bytes_to_tensor, make_repro_tensor

#possibly refactor such that ids are not passed, apart from to the get methods of respective repos
#so dtos are passed instead


def _sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def convert_dto_into_filter(
    dto,
    exclude: Optional[List[str]] = None,
):
    dto_field_names = [f.name for f in fields(type(dto))]
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


def build_predicates_from_filter(dto_type, filt: Optional[dict] = None) -> str:
    if not filt:
        return "1=1"

    dto_fields = {f.name for f in fields(dto_type)}
    allowed_ops = {"=", "!=", "<>", "<", "<=", ">", ">=", "LIKE", "IS", "IS NOT"}
    predicates = []
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

            if value is None and op_upper in {"=", "IS"}:
                predicates.append(f"{field_name} IS NULL")
            elif value is None and op_upper in {"!=", "<>", "IS NOT"}:
                predicates.append(f"{field_name} IS NOT NULL")
            else:
                sql_op = "!=" if op_upper == "<>" else op_upper
                predicates.append(f"{field_name} {sql_op} {_sql_literal(value)}")

    return " AND ".join(predicates) if predicates else "1=1"


def build_orderby_from_filter(dto_type, orderby: Optional[dict] = None) -> str:
    if not orderby:
        return ""
    if not isinstance(orderby, dict):
        raise ValueError("orderby must be a dict of field to 'asc'/'desc'")

    dto_fields = {f.name for f in fields(dto_type)}
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


class ExperimentTemplateRepository(_BaseRepository):
    def _create(self, dto: ExperimentTemplateDTO) -> ExperimentTemplateDTO:
        conn = self._conn()
        cur = conn.cursor()

        # Reuse existing template only when all fields match exactly.
        cur.execute(
            """
            SELECT experiment_template_id
            FROM ExperimentTemplate
            WHERE prompt_group IS ?
              AND loss_name IS ?
              AND loss_additional_parameters IS ?
              AND optimizer_name IS ?
              AND optimizer_additional_parameters IS ?
              AND module_name IS ?
              AND batch_size IS ?
              AND normalization IS ?
            LIMIT 1
            """,
            (
                dto.prompt_group,
                dto.loss_name,
                dto.loss_additional_parameters,
                dto.optimizer_name,
                dto.optimizer_additional_parameters,
                dto.module_name,
                dto.batch_size,
                dto.normalization,
            ),
        )
        row = cur.fetchone()
        if row:
            dto.experiment_template_id = (
                row["experiment_template_id"] if hasattr(row, "keys") else row[0]
            )
            self._close(conn)
            return dto

        cur.execute(
            """
            INSERT INTO ExperimentTemplate
              (prompt_group, loss_name, loss_additional_parameters,
               optimizer_name, optimizer_additional_parameters,
               module_name, batch_size, normalization)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dto.prompt_group,
                dto.loss_name,
                dto.loss_additional_parameters,
                dto.optimizer_name,
                dto.optimizer_additional_parameters,
                dto.module_name,
                dto.batch_size,
                dto.normalization,
            ),
        )
        dto.experiment_template_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

    def get(self, experiment_template_id: int) -> Optional[ExperimentTemplateDTO]:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                prompt_group,
                loss_name,
                loss_additional_parameters,
                optimizer_name,
                optimizer_additional_parameters,
                module_name,
                batch_size,
                normalization,
                experiment_template_id
            FROM ExperimentTemplate
            WHERE experiment_template_id = ?
            """,
            (experiment_template_id,),
        )
        row = cur.fetchone()
        self._close(conn)
        if not row:
            return None
        return ExperimentTemplateDTO(
            prompt_group=row["prompt_group"],
            loss_name=row["loss_name"],
            loss_additional_parameters=row["loss_additional_parameters"],
            optimizer_name=row["optimizer_name"],
            optimizer_additional_parameters=row["optimizer_additional_parameters"],
            module_name=row["module_name"],
            batch_size=row["batch_size"],
            normalization=row["normalization"],
            experiment_template_id=row["experiment_template_id"],
        )

    def create_from_args(self, *args, **kwargs) -> ExperimentTemplateDTO:
        return self._create(ExperimentTemplateDTO(*args, **kwargs))

    def find_matching(
        self,
        et_dto: ExperimentTemplateDTO,
        exclude: Optional[List[str]] = None,
        result_filter: Optional[dict] = None,
        orderby: Optional[dict] = None,
    ) -> List[ExperimentTemplateDTO]:
        template_columns = [f.name for f in fields(ExperimentTemplateDTO)]
        exclude = list(exclude or [])
        if "experiment_template_id" not in exclude:
            exclude.append("experiment_template_id")

        filter_predicates = build_predicates_from_filter(
            ExperimentTemplateDTO, result_filter
        )
        match_filter = convert_dto_into_filter(exclude=exclude, dto=et_dto)
        match_predicates = build_predicates_from_filter(ExperimentTemplateDTO, match_filter)
        all_predicates = match_predicates + " AND " + filter_predicates

        # Exclude the input DTO itself from results.
        if et_dto.experiment_template_id is not None:
            all_predicates += (
                f" AND experiment_template_id != {et_dto.experiment_template_id}"
            )
        else:
            self_filter = convert_dto_into_filter(
                exclude=[],
                dto=et_dto,
            )
            self_predicates = build_predicates_from_filter(
                ExperimentTemplateDTO, self_filter
            )
            all_predicates += f" AND NOT ({self_predicates})"

        order_clause = build_orderby_from_filter(ExperimentTemplateDTO, orderby)

        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT {", ".join(template_columns)}
            FROM ExperimentTemplate
            WHERE {all_predicates}{order_clause}
            """
        )
        rows = cur.fetchall()
        self._close(conn)
        return [
            ExperimentTemplateDTO(
                prompt_group=row["prompt_group"],
                loss_name=row["loss_name"],
                loss_additional_parameters=row["loss_additional_parameters"],
                optimizer_name=row["optimizer_name"],
                optimizer_additional_parameters=row["optimizer_additional_parameters"],
                module_name=row["module_name"],
                batch_size=row["batch_size"],
                normalization=row["normalization"],
                experiment_template_id=row["experiment_template_id"],
            )
            for row in rows
        ]


class VectorRepository(_BaseRepository):
    def _create(self, dto: VectorDTO) -> VectorDTO:
        if dto.vector_data is None:
            raise ValueError("vector_data must be provided")
        conn = self._conn()
        cur = conn.cursor()


        # Reuse existing vector by seed when available.
        if not (dto.seed is None) :
            cur.execute(
                """
                SELECT vector_id 
                FROM Vectors
                WHERE seed IS ?
                LIMIT 1
                """,
                (dto.seed,),
            )
            row = cur.fetchone()
            if row:
                dto.vector_id = row["vector_id"] if hasattr(row, "keys") else row[0]
                self._close(conn)
                return dto

        vector_bytes = sqlite3.Binary(tensor_to_bytes(dto.vector_data))
        cur.execute(
            "INSERT INTO Vectors (vector_data, seed) VALUES (?, ?)",
            (vector_bytes, dto.seed),
        )
        dto.vector_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

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

    def get(self, vector_id: int) -> Optional[VectorDTO]:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT vector_id, vector_data, seed FROM Vectors WHERE vector_id = ?",
            (vector_id,),
        )
        row = cur.fetchone()
        self._close(conn)
        if not row:
            return None
        tensor = bytes_to_tensor(row["vector_data"])
        return VectorDTO(vector_data=tensor, vector_id=row["vector_id"], seed=row["seed"])

    def create_from_shape(self,shape):
        seed = random.randint(1,int(1e8))
        return self.create_from_seed(seed=seed,shape=shape)



class ExperimentLiveInstanceRepository(_BaseRepository):
    def __init__(
        self,
        conn: Optional[sqlite3.Connection] = None,
        vector_repo: Optional[VectorRepository] = None,
    ):
        super().__init__(conn)
        self.vector_repo = vector_repo or VectorRepository(conn)

    def _create(self, dto: ExperimentLiveInstanceDTO) -> ExperimentLiveInstanceDTO:
        if dto.vector_data is None:
            raise ValueError("vector_data must be provided")
        vector_bytes = sqlite3.Binary(tensor_to_bytes(dto.vector_data))
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ExperimentLiveInstance
              (vector_data, initial_vector_id, iteration_count, experiment_template_id)
            VALUES (?, ?, ?, ?)
            """,
            (
                vector_bytes,
                dto.initial_vector_id,
                dto.iteration_count,
                dto.experiment_template_id,
            ),
        )
        dto.experiment_instance_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

    def update(self, dto: ExperimentLiveInstanceDTO) -> None:
        if dto.experiment_instance_id is None:
            raise ValueError("experiment_instance_id required for update")
        vector_bytes = (
            sqlite3.Binary(tensor_to_bytes(dto.vector_data))
            if dto.vector_data is not None
            else None
        )
        conn = self._conn()
        cur = conn.cursor()
        if vector_bytes is not None:
            cur.execute(
                """
                UPDATE ExperimentLiveInstance
                SET iteration_count = ?, vector_data = ?
                WHERE experiment_instance_id = ?
                """,
                (dto.iteration_count, vector_bytes, dto.experiment_instance_id),
            )
        else:
            cur.execute(
                """
                UPDATE ExperimentLiveInstance
                SET iteration_count = ?
                WHERE experiment_instance_id = ?
                """,
                (dto.iteration_count, dto.experiment_instance_id),
            )
        conn.commit()
        self._close(conn)

    def create_from_vec_dto(
        self, experiment_template_dto: ExperimentTemplateDTO, vec_dto: VectorDTO
    ) -> ExperimentLiveInstanceDTO:
        dto = ExperimentLiveInstanceDTO(
            experiment_instance_id=None,
            vector_data=vec_dto.vector_data,
            initial_vector_id=vec_dto.vector_id,
            iteration_count=0,
            experiment_template_id=experiment_template_dto.experiment_template_id,
        )
        return self._create(dto)

    def create_from_initial_tensor(
        self, experiment_template_dto: ExperimentTemplateDTO, tensor: torch.Tensor
    ) -> ExperimentLiveInstanceDTO:
        vec_dto = self.vector_repo.create_from_tensor(tensor)
        return self.create_from_vec_dto(experiment_template_dto, vec_dto)

    def create_from_seed(
        self,
        experiment_template_dto: ExperimentTemplateDTO,
        shape: Union[Tuple[int, ...], torch.Size],
        seed: int,
    ) -> ExperimentLiveInstanceDTO:
        vec_dto = self.vector_repo.create_from_seed(seed=seed, shape=shape) 
        return self.create_from_vec_dto(experiment_template_dto, vec_dto)

    def create_from_template(
        self,
        experiment_template_dto: ExperimentTemplateDTO,
        shape: Union[Tuple[int, ...], torch.Size],
    ) -> ExperimentLiveInstanceDTO:
        seed = random.randint(1, int(1e8))
        return self.create_from_seed(experiment_template_dto, shape, seed)

    def get_all_from_experiment_template(
        self,
        et_dto: ExperimentTemplateDTO,
        result_filter: Optional[dict] = None,
        orderby: Optional[dict] = None,
    ) -> List[ExperimentLiveInstanceDTO]:
        template_fields = {f.name for f in fields(ExperimentTemplateDTO)}
        exclude = [name for name in template_fields if name != "experiment_template_id"]

        match_filter = convert_dto_into_filter(dto=et_dto, exclude=exclude)
        match_predicates = build_predicates_from_filter(
            ExperimentLiveInstanceDTO, match_filter
        )#matches only the id

        filter_predicates = build_predicates_from_filter(
            ExperimentLiveInstanceDTO, result_filter
        )
        all_predicates = match_predicates + " AND " + filter_predicates
        order_clause = build_orderby_from_filter(ExperimentLiveInstanceDTO, orderby)

        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                experiment_instance_id,
                vector_data,
                initial_vector_id,
                iteration_count,
                experiment_template_id
            FROM ExperimentLiveInstance
            WHERE {all_predicates}{order_clause}
            """
        )
        rows = cur.fetchall()
        self._close(conn)
        return [
            ExperimentLiveInstanceDTO(
                experiment_instance_id=row["experiment_instance_id"],
                vector_data=(
                    bytes_to_tensor(row["vector_data"])
                    if row["vector_data"] is not None
                    else None
                ),
                initial_vector_id=row["initial_vector_id"],
                iteration_count=row["iteration_count"],
                experiment_template_id=row["experiment_template_id"],
            )
            for row in rows
        ]


class ExperimentSnapshotRepository(_BaseRepository):
    def __init__(
        self,
        conn: Optional[sqlite3.Connection] = None,
        vector_repo: Optional[VectorRepository] = None,
    ):
        super().__init__(conn)
        self.vector_repo = vector_repo or VectorRepository(conn)

    def _create(self, dto: ExperimentSnapshotDTO) -> ExperimentSnapshotDTO:
        conn = self._conn()
        cur = conn.cursor()

        # Reuse existing snapshot if an identical row already exists.
        cur.execute(
            """
            SELECT snapshot_id
            FROM ExperimentSnapshot
            WHERE vector_id IS ?
              AND iteration_count = ?
              AND experiment_instance_id = ?
            LIMIT 1
            """,
            (dto.vector_id, dto.iteration_count, dto.experiment_instance_id),
        )
        row = cur.fetchone()
        if row:
            dto.snapshot_id = row["snapshot_id"] if hasattr(row, "keys") else row[0]
            self._close(conn)
            return dto

        cur.execute(
            """
            INSERT INTO ExperimentSnapshot
              (vector_id, iteration_count, experiment_instance_id)
            VALUES (?, ?, ?)
            """,
            (dto.vector_id, dto.iteration_count, dto.experiment_instance_id),
        )
        dto.snapshot_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

    def create_from_live(self, inst_dto: ExperimentLiveInstanceDTO,save_vector: bool = True) -> ExperimentSnapshotDTO:
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

    def get_all_from_live(
        self,
        live_dto: ExperimentLiveInstanceDTO,
        result_filter: Optional[dict] = None,
        orderby: Optional[dict] = None,
    ) -> List[ExperimentSnapshotDTO]:
        live_fields = {f.name for f in fields(ExperimentLiveInstanceDTO)}
        exclude = [name for name in live_fields if name != "experiment_instance_id"]

        match_filter = convert_dto_into_filter(dto=live_dto, exclude=exclude)
        match_predicates = build_predicates_from_filter(
            ExperimentSnapshotDTO, match_filter
        )
        filter_predicates = build_predicates_from_filter(
            ExperimentSnapshotDTO, result_filter
        )
        all_predicates = match_predicates + " AND " + filter_predicates
        order_clause = build_orderby_from_filter(ExperimentSnapshotDTO, orderby)

        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                snapshot_id,
                vector_id,
                iteration_count,
                experiment_instance_id
            FROM ExperimentSnapshot
            WHERE {all_predicates}{order_clause}
            """
        )
        rows = cur.fetchall()
        self._close(conn)
        return [
            ExperimentSnapshotDTO(
                snapshot_id=row["snapshot_id"],
                vector_id=row["vector_id"],
                iteration_count=row["iteration_count"],
                experiment_instance_id=row["experiment_instance_id"],
            )
            for row in rows
        ]


class GeneratedOutputRepository(_BaseRepository):
    def _create(self, dto: GeneratedOutputDTO) -> GeneratedOutputDTO:
        conn = self._conn()
        cur = conn.cursor()
        vanilla_value = int(bool(dto.vanilla)) if dto.vanilla is not None else None

        # Reuse existing output only when all fields match exactly.
        cur.execute(
            """
            SELECT output_id
            FROM GeneratedOutput
            WHERE prompt_id IS ?
              AND text IS ?
              AND snapshot_id IS ?
              AND vanilla IS ?
              AND generation_details IS ?
            LIMIT 1
            """,
            (
                dto.prompt_id,
                dto.text,
                dto.snapshot_id,
                vanilla_value,
                dto.generation_details,
            ),
        )
        row = cur.fetchone()
        if row:
            dto.output_id = row["output_id"] if hasattr(row, "keys") else row[0]
            self._close(conn)
            return dto

        cur.execute(
            """
            INSERT INTO GeneratedOutput
              (prompt_id, text, snapshot_id, vanilla, generation_details)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                dto.prompt_id,
                dto.text,
                dto.snapshot_id,
                vanilla_value,
                dto.generation_details,
            ),
        )
        dto.output_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

    def create_from_snapshot(
        self,
        snap_dto: ExperimentSnapshotDTO = None,
        text: str = None,
        prompt_id: int = None,
        vanilla: bool = True,
        generation_details: Optional[str] = None,
    ) -> GeneratedOutputDTO:
        dto = GeneratedOutputDTO(
            output_id=None,
            prompt_id=prompt_id,
            text=text,
            snapshot_id=snap_dto.snapshot_id,
            vanilla=False,
            generation_details=generation_details,
        )
        return self._create(dto)

    def create_vanilla(self,text:str,prompt_id:int,generation_details:Optional[str] = None):
        dto = GeneratedOutputDTO(
            output_id=None,
            prompt_id=prompt_id,
            text=text,
            snapshot_id=None,
            vanilla=True,
            generation_details=generation_details,
        )
        return self._create(dto)

    def get_all_from_snapshot(
        self,
        snap_dto: ExperimentSnapshotDTO,
        result_filter: Optional[dict] = None,
        orderby: Optional[dict] = None,
    ) -> List[GeneratedOutputDTO]:
        snap_fields = {f.name for f in fields(ExperimentSnapshotDTO)}
        exclude = [name for name in snap_fields if name != "snapshot_id"]

        match_filter = convert_dto_into_filter(dto=snap_dto, exclude=exclude)
        match_predicates = build_predicates_from_filter(
            GeneratedOutputDTO, match_filter
        )
        filter_predicates = build_predicates_from_filter(
            GeneratedOutputDTO, result_filter
        )
        all_predicates = match_predicates + " AND " + filter_predicates
        order_clause = build_orderby_from_filter(GeneratedOutputDTO, orderby)

        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                output_id,
                prompt_id,
                text,
                snapshot_id,
                vanilla,
                generation_details
            FROM GeneratedOutput
            WHERE {all_predicates}{order_clause}
            """
        )
        rows = cur.fetchall()
        self._close(conn)
        return [
            GeneratedOutputDTO(
                output_id=row["output_id"],
                prompt_id=row["prompt_id"],
                text=row["text"],
                snapshot_id=row["snapshot_id"],
                vanilla=(bool(row["vanilla"]) if row["vanilla"] is not None else None),
                generation_details=row["generation_details"],
            )
            for row in rows
        ]


class MetricRepository(_BaseRepository):
    def _create(self, dto: MetricDTO) -> MetricDTO:
        conn = self._conn()
        cur = conn.cursor()
        if dto.generated_output_id:
            cur.execute(
                """
                INSERT INTO Metric
                  (value, description, snapshot_id, prompt_id, generated_output_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    dto.value,
                    dto.description,
                    dto.snapshot_id,
                    dto.prompt_id,
                    dto.generated_output_id,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO Metric
                  (value, description, snapshot_id, prompt_id)
                VALUES (?, ?, ?, ?)
                """,
                (dto.value, dto.description, dto.snapshot_id, dto.prompt_id),
            )
        dto.metric_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

    def create_from_snapshot(
        self,
        snap_dto: ExperimentSnapshotDTO,
        value: float,
        description: str,
        prompt_id: Optional[int] = None,
        generated_output_id: Optional[int] = None,
    ) -> MetricDTO:
        dto = MetricDTO(
            value=value,
            description=description,
            metric_id=None,
            snapshot_id=snap_dto.snapshot_id,
            prompt_id=prompt_id,
            generated_output_id=generated_output_id,
        )
        return self._create(dto)


class PromptRepository(_BaseRepository):
    #groups should be created with all prompts at once, not piece meal
    #user should use create_group_from_strings or create_group_from_dtos
    #its acceptable for the user to use _create to create a prompt

    def _create(self, dto: PromptDTO) -> PromptDTO:
        conn = self._conn()
        cur = conn.cursor()

        # Check if prompt text already exists
        cur.execute("SELECT prompt_id FROM Prompt WHERE text = ? LIMIT 1", (dto.text,))
        row = cur.fetchone()
        if row:
            dto.prompt_id = row["prompt_id"] if hasattr(row, "keys") else row[0]
            self._close(conn)
            return dto

        # Insert only if it doesn't exist
        cur.execute("INSERT INTO Prompt (text) VALUES (?)", (dto.text,))
        dto.prompt_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto


    def get(self, prompt_id: int) -> Optional[PromptDTO]:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT prompt_id, text FROM Prompt WHERE prompt_id = ?", (prompt_id,)
        )
        row = cur.fetchone()
        self._close(conn)
        if not row:
            return None
        return PromptDTO(text=row["text"], prompt_id=row["prompt_id"])

    #shouldn't be used by user
    def create_group(self) -> PromptGroupDTO:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO PromptGroup DEFAULT VALUES")
        group = PromptGroupDTO(group_id=cur.lastrowid)
        conn.commit()
        self._close(conn)
        return group
    
    def get_prompts_from_group(self,group_id: int):#should be dto but group dto only has id field
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT p.prompt_id, p.text as text FROM Prompt p JOIN GroupPrompts g ON g.prompt_id = p.prompt_id WHERE g.group_id = ?",(group_id,))
        rows = cur.fetchall()
        prompts = [PromptDTO(prompt_id= r[0],text=r[1]) for r in rows] 
        cur.close()
        self._close(conn)
        return  prompts

    #shouldn't be used by user
    def add_dto_to_group(self, group_dto: PromptGroupDTO, prompt_dto: PromptDTO) -> None:
        if group_dto.group_id is None or prompt_dto.prompt_id is None:
            raise ValueError("Both group_id and prompt_id must be set")
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO GroupPrompts (group_id, prompt_id) VALUES (?, ?)",
            (group_dto.group_id, prompt_dto.prompt_id),
        )
        conn.commit()
        self._close(conn)

    
    def create_group_from_dtos(self, prompts: List[PromptDTO]) -> PromptGroupDTO: #assuming persisted dtos

        #prevents duplication
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
        params = [*prompt_ids, len(prompt_ids)]
        row = conn.execute(sql, [*prompt_ids, len(prompt_ids)]).fetchone()
        group_id = row[0] if row else None   # None => no such group exists
        if group_id:
            return PromptGroupDTO(group_id)
        

        group = self.create_group()
        for p in prompts:
            self.add_dto_to_group(group, p)
        return group

    def create_group_from_strings(self, prompts: List[str]) -> PromptGroupDTO:
        prompt_dtos = [self._create(PromptDTO(text=s)) for s in prompts]
        return self.create_group_from_dtos(prompt_dtos)
