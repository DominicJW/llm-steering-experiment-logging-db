from typing import Optional, List, Tuple, Union
import sqlite3
import io
import random
import torch

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


class VectorRepository(_BaseRepository):
    def _create(self, dto: VectorDTO) -> VectorDTO:
        if dto.vector_data is None:
            raise ValueError("vector_data must be provided")
        vector_bytes = sqlite3.Binary(tensor_to_bytes(dto.vector_data))
        conn = self._conn()
        cur = conn.cursor()
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
        dtype: torch.dtype = torch.float32,
    ) -> VectorDTO:
        tensor = make_repro_tensor(shape, seed=seed, device=device, dtype=dtype)
        return self._create(VectorDTO(vector_data=tensor, vector_id=None, seed=seed))

    def create_from_tensor(self, tensor: torch.Tensor, seed: Optional[int] = None) -> VectorDTO:
        return self._create(VectorDTO(vector_data=tensor, vector_id=None, seed=seed))

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
            vector_dto = self.vector_repo.create_from_tensor(
                inst_dto.vector_data, seed=inst_dto.initial_vector_id
            )
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
    def _create(self, dto: GeneratedOutputDTO) -> GeneratedOutputDTO:
        conn = self._conn()
        cur = conn.cursor()
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
                int(bool(dto.vanilla)) if dto.vanilla is not None else None,
                dto.generation_details,
            ),
        )
        dto.output_id = cur.lastrowid
        conn.commit()
        self._close(conn)
        return dto

    def create_from_snapshot(
        self,
        snap_dto: ExperimentSnapshotDTO,
        text: str,
        prompt_id: Optional[int] = None,
        vanilla: bool = True,
        generation_details: Optional[str] = None,
    ) -> GeneratedOutputDTO:
        dto = GeneratedOutputDTO(
            output_id=None,
            prompt_id=prompt_id,
            text=text,
            snapshot_id=snap_dto.snapshot_id,
            vanilla=vanilla,
            generation_details=generation_details,
        )
        return self._create(dto)


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
    def _create(self, dto: PromptDTO) -> PromptDTO:
        conn = self._conn()
        cur = conn.cursor()
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

    def create_group(self) -> PromptGroupDTO:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO PromptGroup DEFAULT VALUES")
        group = PromptGroupDTO(group_id=cur.lastrowid)
        conn.commit()
        self._close(conn)
        return group
    
    def get_prompts_from_group(self,group_id: int):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT p.text as text FROM Prompt p JOIN GroupPrompts g ON g.prompt_id = p.prompt_id WHERE g.group_id = ?",(group_id,))
        rows = cur.fetchall()          # list of tuples, e.g. [("text1",), ("text2",)]
        prompts = [r[0] for r in rows]   # extract strings: ["text1", "text2"]
        cur.close()
        self._close(conn)
        return  prompts

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

    def create_group_from_dtos(self, prompts: List[PromptDTO]) -> PromptGroupDTO:
        group = self.create_group()
        for p in prompts:
            self.add_dto_to_group(group, p)
        return group

    def create_group_from_strings(self, prompts: List[str]) -> PromptGroupDTO:
        prompt_dtos = [self._create(PromptDTO(text=s)) for s in prompts]
        return self.create_group_from_dtos(prompt_dtos)

