from __future__ import annotations

from typing import Dict, List, Optional

import torch
from sqlalchemy import Boolean, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import LargeBinary, TypeDecorator

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


class PromptDTO(Base):
    __tablename__ = "Prompt"

    prompt_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)

    group_links: Mapped[List[GroupPromptLinkDTO]] = relationship(
        "GroupPromptLinkDTO",
        back_populates="prompt",
        cascade="all, delete-orphan",
    )
    generated_outputs: Mapped[List[GeneratedOutputDTO]] = relationship(
        "GeneratedOutputDTO",
        back_populates="prompt",
    )
    metrics: Mapped[List[MetricDTO]] = relationship("MetricDTO", back_populates="prompt")


class PromptGroupDTO(Base):
    __tablename__ = "PromptGroup"

    group_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    group_links: Mapped[List[GroupPromptLinkDTO]] = relationship(
        "GroupPromptLinkDTO",
        back_populates="group",
        cascade="all, delete-orphan",
    )
    experiment_templates: Mapped[List[ExperimentTemplateDTO]] = relationship(
        "ExperimentTemplateDTO",
        back_populates="prompt_group_ref",
    )


class GroupPromptLinkDTO(Base):
    __tablename__ = "GroupPrompts"

    id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    group_id: Mapped[int] = mapped_column(ForeignKey("PromptGroup.group_id"), nullable=False)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("Prompt.prompt_id"), nullable=False)

    group: Mapped[PromptGroupDTO] = relationship("PromptGroupDTO", back_populates="group_links")
    prompt: Mapped[PromptDTO] = relationship("PromptDTO", back_populates="group_links")


class ExperimentTemplateDTO(Base):
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

    prompt_group_ref: Mapped[Optional[PromptGroupDTO]] = relationship(
        "PromptGroupDTO",
        back_populates="experiment_templates",
    )
    live_instances: Mapped[List[ExperimentLiveInstanceDTO]] = relationship(
        "ExperimentLiveInstanceDTO",
        back_populates="experiment_template",
    )


class VectorDTO(Base):
    __tablename__ = "Vectors"

    vector_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_data: Mapped[Optional[torch.Tensor]] = mapped_column(TensorBlob)
    seed: Mapped[Optional[int]] = mapped_column(Integer)

    live_instances: Mapped[List[ExperimentLiveInstanceDTO]] = relationship(
        "ExperimentLiveInstanceDTO",
        back_populates="initial_vector",
    )
    snapshots: Mapped[List[ExperimentSnapshotDTO]] = relationship(
        "ExperimentSnapshotDTO",
        back_populates="vector",
    )


class ExperimentLiveInstanceDTO(Base):
    __tablename__ = "ExperimentLiveInstance"

    experiment_instance_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_data: Mapped[Optional[torch.Tensor]] = mapped_column(TensorBlob)
    initial_vector_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Vectors.vector_id"))
    iteration_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    experiment_template_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ExperimentTemplate.experiment_template_id")
    )

    initial_vector: Mapped[Optional[VectorDTO]] = relationship(
        "VectorDTO",
        back_populates="live_instances",
    )
    experiment_template: Mapped[Optional[ExperimentTemplateDTO]] = relationship(
        "ExperimentTemplateDTO",
        back_populates="live_instances",
    )
    snapshots: Mapped[List[ExperimentSnapshotDTO]] = relationship(
        "ExperimentSnapshotDTO",
        back_populates="live_instance",
    )


class ExperimentSnapshotDTO(Base):
    __tablename__ = "ExperimentSnapshot"

    snapshot_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    vector_id: Mapped[Optional[int]] = mapped_column(ForeignKey("Vectors.vector_id"))
    iteration_count: Mapped[Optional[int]] = mapped_column(Integer)
    experiment_instance_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ExperimentLiveInstance.experiment_instance_id")
    )

    vector: Mapped[Optional[VectorDTO]] = relationship("VectorDTO", back_populates="snapshots")
    live_instance: Mapped[Optional[ExperimentLiveInstanceDTO]] = relationship(
        "ExperimentLiveInstanceDTO",
        back_populates="snapshots",
    )
    generated_outputs: Mapped[List[GeneratedOutputDTO]] = relationship(
        "GeneratedOutputDTO",
        back_populates="snapshot",
    )
    metrics: Mapped[List[MetricDTO]] = relationship("MetricDTO", back_populates="snapshot")


class GeneratedOutputDTO(Base):
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
    generation_details: Mapped[Optional[str]] = mapped_column(Text)

    prompt: Mapped[Optional[PromptDTO]] = relationship("PromptDTO", back_populates="generated_outputs")
    snapshot: Mapped[Optional[ExperimentSnapshotDTO]] = relationship(
        "ExperimentSnapshotDTO",
        back_populates="generated_outputs",
    )
    metrics: Mapped[List[MetricDTO]] = relationship("MetricDTO", back_populates="generated_output")


class MetricDTO(Base):
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

    snapshot: Mapped[Optional[ExperimentSnapshotDTO]] = relationship(
        "ExperimentSnapshotDTO",
        back_populates="metrics",
    )
    prompt: Mapped[Optional[PromptDTO]] = relationship("PromptDTO", back_populates="metrics")
    generated_output: Mapped[Optional[GeneratedOutputDTO]] = relationship(
        "GeneratedOutputDTO",
        back_populates="metrics",
    )