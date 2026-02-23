from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional
import torch
import sqlite3
from .utils import tensor_to_bytes, bytes_to_tensor

# shared helper used in FK metadata declarations
def _get_primary_key_field(dto_type: type) -> str:
    for f in fields(dto_type):
        if f.metadata.get("persist", True) and f.metadata.get("primary_key", False):
            return f.name
    raise ValueError(f"No primary key metadata for DTO type: {dto_type}")

#add foregin key to field
def db_field(
    *,
    sql_type: str,
    persist: bool = True,
    primary_key: bool = False,
    foreign_key: bool = False,
    foreign_dto_type:Optional = None,
    foreign_field:Optional[str] = None,
    autoincrement: bool = False,
    default: Any = MISSING,
    encode_callback: Callable = lambda x: x,
    decode_callback: Callable = lambda x: x
):
    metadata = {
        "sql_type": sql_type,
        "persist": persist,
        "primary_key": primary_key,
        "foreign_key":foreign_key,
        "foreign_dto_type":foreign_dto_type,
        "foreign_field":foreign_field,
        "autoincrement": autoincrement,
        "encode_callback": encode_callback,
        "decode_callback": decode_callback
    }
    if default is MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


@dataclass
class PromptDTO:
    text: str = db_field(sql_type="TEXT")
    prompt_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )

@dataclass
class PromptGroupDTO:
    group_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )


@dataclass
class GroupPromptLinkDTO:
    group_id: int = db_field(
        sql_type="INTEGER",foreign_key=True,
        foreign_dto_type=PromptGroupDTO,
        foreign_field=_get_primary_key_field(PromptGroupDTO)
        )
    prompt_id: int = db_field(
        sql_type="INTEGER",
        foreign_dto_type=PromptDTO,
        foreign_field=_get_primary_key_field(PromptDTO)
        )
    id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )

#should have sane defaults
@dataclass
class ExperimentTemplateDTO:
    prompt_group: Optional[int] = db_field(
        sql_type="INTEGER",
        default=None,
        foreign_key=True,
        foreign_dto_type=PromptGroupDTO,
        foreign_field=_get_primary_key_field(PromptGroupDTO))
    loss_name: Optional[str] = db_field(sql_type="TEXT", default=None)
    loss_additional_parameters: Optional[str] = db_field(sql_type="TEXT", default=None)
    optimizer_name: Optional[str] = db_field(sql_type="TEXT", default=None) 
    optimizer_additional_parameters: Optional[str] = db_field(sql_type="TEXT", default=None) 
    module_name: Optional[str] = db_field(sql_type="TEXT", default=None) 
    batch_size: Optional[int] = db_field(sql_type="INTEGER", default=None) 
    normalization: Optional[float] = db_field(sql_type="REAL", default=None) 
    experiment_template_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )

@dataclass
class VectorDTO:
    vector_data: Optional[torch.Tensor] = db_field(
        sql_type="BLOB",
        encode_callback=lambda value: sqlite3.Binary(tensor_to_bytes(value)),
        decode_callback=lambda value: bytes_to_tensor(value),
        default=None
        ) #shouldn't be optional
    vector_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )
    seed: Optional[int] = db_field(sql_type="INTEGER", default=None)

@dataclass
class ExperimentLiveInstanceDTO:
    experiment_instance_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )
    vector_data: Optional[torch.Tensor] = db_field(
        sql_type="BLOB",
        encode_callback=lambda value: sqlite3.Binary(tensor_to_bytes(value)),
        decode_callback=lambda value: bytes_to_tensor(value),
        default=None,
    )
    initial_vector_id: Optional[int] = db_field(
        sql_type="INTEGER",
        default=None,
        foreign_key=True,
        foreign_dto_type=VectorDTO,
        foreign_field=_get_primary_key_field(VectorDTO)) #shouldn't be optional
    iteration_count: Optional[int] = db_field(sql_type="INTEGER", default=0)
    experiment_template_id: Optional[int] = db_field(
        sql_type="INTEGER",
         default=None,
        foreign_key=True,
        foreign_dto_type=ExperimentTemplateDTO,
        foreign_field=_get_primary_key_field(ExperimentTemplateDTO))#shouldn't be optional

@dataclass
class ExperimentSnapshotDTO:
    snapshot_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )
    vector_id: Optional[int] = db_field(
        sql_type="INTEGER",
        default=None,
        foreign_key=True,
        foreign_dto_type=VectorDTO,
        foreign_field=_get_primary_key_field(VectorDTO)) 
    iteration_count: Optional[int] = db_field(sql_type="INTEGER", default=None) #shouldn't be optional
    experiment_instance_id: Optional[int] = db_field(
        sql_type="INTEGER",
        default=None,
        foreign_key=True,
        foreign_dto_type=ExperimentLiveInstanceDTO,
        foreign_field=_get_primary_key_field(ExperimentLiveInstanceDTO)) #shouldn't be optional

@dataclass
class GeneratedOutputDTO:
    output_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )
    prompt_id: Optional[int] = db_field(
        sql_type="INTEGER",
        foreign_dto_type=PromptDTO,
        foreign_field=_get_primary_key_field(PromptDTO)
        )
    text: Optional[str] = db_field(sql_type="TEXT", default=None)
    snapshot_id: Optional[int] = db_field(
        sql_type="INTEGER",
        default=None,
        foreign_key=True,
        foreign_dto_type=ExperimentSnapshotDTO,
        foreign_field=_get_primary_key_field(ExperimentSnapshotDTO)) 
    vanilla: Optional[bool] = db_field(
        sql_type="INTEGER",
        encode_callback=lambda value: int(bool(value)),
        decode_callback=lambda value: bool(value),
        default=True,
    )
    generation_details: Optional[str] = db_field(sql_type="TEXT", default=None)

@dataclass
class MetricDTO:
    value: float = db_field(sql_type="REAL")
    description: Optional[str] = db_field(sql_type="TEXT")
    metric_id: Optional[int] = db_field(
        sql_type="INTEGER",
        primary_key=True,
        autoincrement=True,
        default=None,
    )
    snapshot_id: Optional[int] = db_field(
        sql_type="INTEGER",
        default=None,
        foreign_key=True,
        foreign_dto_type=ExperimentSnapshotDTO,
        foreign_field=_get_primary_key_field(ExperimentSnapshotDTO)) 
    prompt_id: Optional[int] = db_field(
        sql_type="INTEGER",
        foreign_dto_type=PromptDTO,
        foreign_field=_get_primary_key_field(PromptDTO))
    generated_output_id: Optional[int] = db_field(
        sql_type="INTEGER",
        foreign_dto_type=GeneratedOutputDTO,
        foreign_field=_get_primary_key_field(GeneratedOutputDTO))


_DTO_TABLES: Dict[type, str] = {
    ExperimentTemplateDTO: "ExperimentTemplate",
    VectorDTO: "Vectors",
    ExperimentLiveInstanceDTO: "ExperimentLiveInstance",
    ExperimentSnapshotDTO: "ExperimentSnapshot",
    PromptDTO: "Prompt",
    PromptGroupDTO: "PromptGroup",
    GeneratedOutputDTO: "GeneratedOutput",
    MetricDTO: "Metric",
    GroupPromptLinkDTO: "GroupPrompts",
}

_TABLE_REGISTRATION_ORDER: List[type] = [
    ExperimentTemplateDTO,
    VectorDTO,
    ExperimentLiveInstanceDTO,
    ExperimentSnapshotDTO,
    PromptDTO,
    PromptGroupDTO,
    GeneratedOutputDTO,
    MetricDTO,
    GroupPromptLinkDTO,
]


def _table_name_for_dto(dto_type: type) -> str:
    try:
        return _DTO_TABLES[dto_type]
    except KeyError as exc:
        raise ValueError(f"No table mapping for DTO type: {dto_type}") from exc
