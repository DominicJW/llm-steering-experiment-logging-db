from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import LargeBinary, TypeDecorator, JSON

from .utils import bytes_to_tensor, tensor_to_bytes


class Base(DeclarativeBase):
    pass


class TensorBlob(TypeDecorator):
    """Store torch.Tensor values as bytes in SQLite BLOB columns."""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return tensor_to_bytes(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return bytes_to_tensor(value)


class Prompt(Base):
    __tablename__ = "Prompt"

    prompt_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)


class PromptGroup(Base):
    __tablename__ = "PromptGroup"

    group_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    prompts: Mapped[List[Prompt]] = relationship(
        "Prompt",
        secondary="PromptGroupItem",
        primaryjoin="PromptGroup.group_id == PromptGroupItem.group_id",
        secondaryjoin="Prompt.prompt_id == PromptGroupItem.prompt_id",
        lazy="selectin",
    )



class PromptGroupItem(Base):
    __tablename__ = "PromptGroupItem"

    id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    group_id: Mapped[int] = mapped_column(ForeignKey("PromptGroup.group_id"), nullable=False)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("Prompt.prompt_id"), nullable=False)




class ExperimentTemplate(Base):
    __tablename__ = "ExperimentTemplate"

    experiment_template_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    group_id: Mapped[Optional[int]] = mapped_column(ForeignKey("PromptGroup.group_id"),nullable=False)
    loss_name: Mapped[Optional[str]] = mapped_column(Text,nullable=False)
    loss_additional_parameters: Mapped[Dict[Any,Any]] = mapped_column(JSON,nullable=True)
    optimizer_name: Mapped[Optional[str]] = mapped_column(Text,nullable=False)
    optimizer_additional_parameters: Mapped[Dict[Any,Any]] = mapped_column(JSON,nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(Text,nullable=False)
    module_name: Mapped[Optional[str]] = mapped_column(Text,nullable=False)
    batch_size: Mapped[Optional[int]] = mapped_column(Integer,nullable=False)
    normalization: Mapped[Optional[float]] = mapped_column(Float,nullable=False)
    
    prompt_group: Mapped[PromptGroup] = relationship(
        "PromptGroup",
        foreign_keys=[group_id],
        lazy="selectin",
    )

class Vector(Base):
    __tablename__ = "Vector"

    vector_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_data: Mapped[Optional[torch.Tensor]] = mapped_column(TensorBlob,nullable=False)
    seed: Mapped[Optional[int]] = mapped_column(Integer,nullable=True)
    shape: Mapped[Optional[Tuple[int]]] = mapped_column(JSON, nullable=False)


class ExperimentLiveInstance(Base):
    __tablename__ = "ExperimentLiveInstance"

    experiment_instance_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_data: Mapped[Optional[torch.Tensor]] = mapped_column(TensorBlob,nullable=True)
    initial_vector_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Vector.vector_id"),nullable=False)
    iteration_count: Mapped[Optional[int]] = mapped_column(Integer, default=0,nullable=False)
    experiment_template_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ExperimentTemplate.experiment_template_id"),nullable=False
    )

    experiment_template: Mapped[ExperimentTemplate] = relationship(
        "ExperimentTemplate",
        foreign_keys="ExperimentTemplate.experiment_template_id",
        lazy="selectin",
    )


    snapshots: Mapped[List[ExperimentSnapshot]] = relationship(
        "ExperimentSnapshot",
        foreign_keys="ExperimentSnapshot.experiment_instance_id",
        lazy="selectin",
    )


class VanillaBaseline(Base):
    __tablename__ = "VanillaBaseline"
    vanilla_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    model_name: Mapped[Optional[str]] = mapped_column(Text,nullable=False) #should be huggingface model id ideally




class ExperimentSnapshot(Base):
    __tablename__ = "ExperimentSnapshot"
    __table_args__ = (
        CheckConstraint(
            "(experiment_instance_id IS NOT NULL AND vanilla_baseline_id IS NULL) OR "
            "(experiment_instance_id IS NULL AND vanilla_baseline_id IS NOT NULL AND iteration_count = 0 AND vector_id is NULL)",
            name="ck_snapshot_exactly_one_owner",
        ),
    )
    

    snapshot_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Vector.vector_id"),nullable=True)
    iteration_count: Mapped[Optional[int]] = mapped_column(Integer,nullable=False)
    
    experiment_instance_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ExperimentLiveInstance.experiment_instance_id"),nullable=True
    )

    vanilla_baseline_id:  Mapped[Optional[int]] = mapped_column(
        ForeignKey("VanillaBaseline.vanilla_id"),nullable=True
    )


    generated_outputs: Mapped[List[GeneratedOutput]] = relationship(
        "GeneratedOutput",
        foreign_keys="GeneratedOutput.snapshot_id",
        lazy="selectin",
    )
    metrics: Mapped[List[Metric]] = relationship(
        "Metric",
        foreign_keys="Metric.snapshot_id",
        lazy="selectin",
    )


class GeneratedOutput(Base):
    __tablename__ = "GeneratedOutput"

    output_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    prompt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Prompt.prompt_id"),nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text,nullable=False)
    snapshot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ExperimentSnapshot.snapshot_id"),nullable=False)
    eos: Mapped[Optional[bool]] = mapped_column(Boolean, default=True)
    generation_details: Mapped[Optional[str]] = mapped_column(Text,default="")
    token_ids: Mapped[Optional[str]] = mapped_column(JSON) #new #should use cutom type decorator

    prompt : Mapped[Prompt] = relationship(
        "Prompt",
        foreign_keys=[prompt_id],
        lazy="selectin",
    )

class Metric(Base):
    __tablename__ = "Metric"
    __table_args__ = (
        CheckConstraint(
            "(prompt_id IS NOT NULL OR generated_output_id IS NOT NULL)",
            name="ck_at_least_prompt_or_generated_output",
        ),
    )

    metric_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text,nullable=False)

    snapshot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ExperimentSnapshot.snapshot_id"),nullable=False)
    secondary_snapshot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ExperimentSnapshot.snapshot_id"),nullable=True)
    
    prompt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Prompt.prompt_id"),nullable=True)
    generated_output_id: Mapped[Optional[int]] = mapped_column(ForeignKey("GeneratedOutput.output_id"),nullable=True)

    prompt : Mapped[Optional[Prompt]] = relationship(
        "Prompt",
        foreign_keys=[prompt_id],
        lazy="selectin",
    )
    generated_output : Mapped[Optional[GeneratedOutput]] = relationship(
        "GeneratedOutput",
        foreign_keys=[generated_output_id],
        lazy="selectin",
    )
