from __future__ import annotations

from typing import Dict, List, Optional,Tuple

import torch
from sqlalchemy import Boolean, Float, ForeignKey, Integer, Text
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

    group_links: Mapped[List[PromptGroupItem]] = relationship(
        "PromptGroupItem",
        back_populates="prompt",
        cascade="all, delete-orphan",
    )
    generated_outputs: Mapped[List[GeneratedOutput]] = relationship(
        "GeneratedOutput",
        back_populates="prompt",
    )
    metrics: Mapped[List[Metric]] = relationship("Metric", back_populates="prompt")


class PromptGroup(Base):
    __tablename__ = "PromptGroup"

    group_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    group_links: Mapped[List[PromptGroupItem]] = relationship(
        "PromptGroupItem",
        back_populates="group",
        cascade="all, delete-orphan",
    )
    experiment_templates: Mapped[List[ExperimentTemplate]] = relationship(
        "ExperimentTemplate",
        back_populates="prompt_group_ref",
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

    group: Mapped[PromptGroup] = relationship("PromptGroup", back_populates="group_links")
    prompt: Mapped[Prompt] = relationship("Prompt", back_populates="group_links")


class ExperimentTemplate(Base):
    __tablename__ = "ExperimentTemplate"

    experiment_template_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    prompt_group: Mapped[Optional[int]] = mapped_column(ForeignKey("PromptGroup.group_id"))
    loss_name: Mapped[Optional[str]] = mapped_column(Text)
    loss_additional_parameters: Mapped[Optional[str]] = mapped_column(Text)
    optimizer_name: Mapped[Optional[str]] = mapped_column(Text)
    optimizer_additional_parameters: Mapped[Optional[str]] = mapped_column(Text)
    module_name: Mapped[Optional[str]] = mapped_column(Text)
    batch_size: Mapped[Optional[int]] = mapped_column(Integer)
    normalization: Mapped[Optional[float]] = mapped_column(Float)

    prompt_group_ref: Mapped[Optional[PromptGroup]] = relationship(
        "PromptGroup",
        back_populates="experiment_templates",
    )
    live_instances: Mapped[List[ExperimentLiveInstance]] = relationship(
        "ExperimentLiveInstance",
        back_populates="experiment_template",
    )


class Vector(Base):
    __tablename__ = "Vector"

    vector_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_data: Mapped[Optional[torch.Tensor]] = mapped_column(TensorBlob)
    seed: Mapped[Optional[int]] = mapped_column(Integer)
    shape: Mapped[Optional[Tuple[int]]] = mapped_column(JSON, nullable=True)


    live_instances: Mapped[List[ExperimentLiveInstance]] = relationship(
        "ExperimentLiveInstance",
        back_populates="initial_vector",
    )
    snapshots: Mapped[List[ExperimentSnapshot]] = relationship(
        "ExperimentSnapshot",
        back_populates="vector",
    )


class ExperimentLiveInstance(Base):
    __tablename__ = "ExperimentLiveInstance"

    experiment_instance_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_data: Mapped[Optional[torch.Tensor]] = mapped_column(TensorBlob)
    initial_vector_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Vector.vector_id"))
    iteration_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    experiment_template_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ExperimentTemplate.experiment_template_id")
    )

    initial_vector: Mapped[Optional[Vector]] = relationship(
        "Vector",
        back_populates="live_instances",
    )
    experiment_template: Mapped[Optional[ExperimentTemplate]] = relationship(
        "ExperimentTemplate",
        back_populates="live_instances",
    )
    snapshots: Mapped[List[ExperimentSnapshot]] = relationship(
        "ExperimentSnapshot",
        back_populates="live_instance",
    )


class ExperimentSnapshot(Base):
    __tablename__ = "ExperimentSnapshot"

    snapshot_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Vector.vector_id"))
    iteration_count: Mapped[Optional[int]] = mapped_column(Integer)
    experiment_instance_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ExperimentLiveInstance.experiment_instance_id")
    )

    vector: Mapped[Optional[Vector]] = relationship("Vector", back_populates="snapshots")
    live_instance: Mapped[Optional[ExperimentLiveInstance]] = relationship(
        "ExperimentLiveInstance",
        back_populates="snapshots",
    )
    generated_outputs: Mapped[List[GeneratedOutput]] = relationship(
        "GeneratedOutput",
        back_populates="snapshot",
    )
    metrics: Mapped[List[Metric]] = relationship("Metric", back_populates="snapshot")


class GeneratedOutput(Base):
    __tablename__ = "GeneratedOutput"

    output_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    prompt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Prompt.prompt_id"))
    text: Mapped[Optional[str]] = mapped_column(Text)
    snapshot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ExperimentSnapshot.snapshot_id"))
    vanilla: Mapped[Optional[bool]] = mapped_column(Boolean, default=True)
    eos: Mapped[Optional[bool]] = mapped_column(Boolean, default=True)
    generation_details: Mapped[Optional[str]] = mapped_column(Text)

    prompt: Mapped[Optional[Prompt]] = relationship("Prompt", back_populates="generated_outputs")
    snapshot: Mapped[Optional[ExperimentSnapshot]] = relationship(
        "ExperimentSnapshot",
        back_populates="generated_outputs",
    )
    metrics: Mapped[List[Metric]] = relationship("Metric", back_populates="generated_output")


class Metric(Base):
    __tablename__ = "Metric"

    metric_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    snapshot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ExperimentSnapshot.snapshot_id"))
    prompt_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Prompt.prompt_id"))
    generated_output_id: Mapped[Optional[int]] = mapped_column(ForeignKey("GeneratedOutput.output_id"))

    snapshot: Mapped[Optional[ExperimentSnapshot]] = relationship(
        "ExperimentSnapshot",
        back_populates="metrics",
    )
    prompt: Mapped[Optional[Prompt]] = relationship("Prompt", back_populates="metrics")
    generated_output: Mapped[Optional[GeneratedOutput]] = relationship(
        "GeneratedOutput",
        back_populates="metrics",
    )