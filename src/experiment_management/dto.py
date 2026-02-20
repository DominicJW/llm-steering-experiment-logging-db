from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch

@dataclass
class PromptDTO:
    text: str
    prompt_id: Optional[int] = None

@dataclass
class PromptGroupDTO:
    group_id: Optional[int] = None

@dataclass
class ExperimentTemplateDTO:
    prompt_group: Optional[int] = None
    loss_name: Optional[str] = None 
    loss_additional_parameters: Optional[str] = None
    optimizer_name : Optional[str] = None 
    optimizer_additional_parameters : Optional[str] = None
    module_name: Optional[str] = None
    batch_size: Optional[int] = None
    normalization: Optional[float] = None
    experiment_template_id: Optional[int] = None

@dataclass
class VectorDTO:
    vector_data: Optional[torch.Tensor] = None
    vector_id: Optional[int] = None
    seed: Optional[int] = None

@dataclass
class ExperimentLiveInstanceDTO:
    experiment_instance_id: Optional[int] = None
    vector_data: Optional[torch.Tensor] = None
    initial_vector_id: Optional[int] = None
    iteration_count: Optional[int] = None
    experiment_template_id: Optional[int] = None

@dataclass
class ExperimentSnapshotDTO:
    snapshot_id: Optional[int] = None
    vector_id: Optional[int] = None
    iteration_count: Optional[int] = None
    experiment_instance_id: Optional[int] = None

@dataclass
class GeneratedOutputDTO:
    output_id: Optional[int] = None
    prompt_id: Optional[int] = None
    text: Optional[str] = None #hold only the generated text
    snapshot_id: Optional[int] = None
    vanilla: Optional[bool] = True
    generation_details: Optional[str] = None

@dataclass
class MetricDTO:
    value: float
    description: Optional[str]
    metric_id: Optional[int] = None
    snapshot_id: Optional[int] = None
    prompt_id: Optional[int] = None
    generated_output_id: Optional[int] = None
