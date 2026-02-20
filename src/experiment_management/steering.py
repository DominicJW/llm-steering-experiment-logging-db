# steering.py
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple, Union,Dict,Callable,Type

import torch

from .dto import ExperimentLiveInstanceDTO, ExperimentTemplateDTO
from .factories import loss_factory, optimizer_factory
from .utils import slice_batch_output, normalize
from .repositories import ExperimentLiveInstanceRepository, ExperimentTemplateRepository

#the optimizer is optimizing a specific tensor
#I am seeing a risk for problems


class LiveInstance:
    """
    Wraps an ExperimentLiveInstanceDTO together with the repository that created it.
    Provides convenience methods for stepping the instance and for constructing new
    live instances from a template, a seed, a tensor or an existing vector id.
    """

    def __init__(
        self,
        live_dto: ExperimentLiveInstanceDTO,
        live_repo: ExperimentLiveInstanceRepository,
        et_repo: ExperimentTemplateRepository,
    ):
        self.live_dto = live_dto
        self.live_repo = live_repo
        self.et_repo = et_repo

        # expose frequently‑used fields

        self.et_dto = self.et_repo.get(live_dto.experiment_template_id)
        if self.et_dto is None:
            raise ValueError(
                f"ExperimentTemplate {live_dto.experiment_template_id} not found"
            )
        self.vector_data = normalize(live_dto.vector_data,et_dto.normalization)
        self.update_dto_vector()

        # build loss & optimizer from the template
        self.loss_fn = loss_factory(
            self.et_dto.loss_name, self.et_dto.loss_additional_parameters
        )
        self.optimizer_constructor = optimizer_factory(
            self.et_dto.optimizer_name, self.et_dto.optimizer_additional_parameters
        )
        
        self.optimizer = self.optimizer_constructor([self.vector_data])

        self.module_name = self.et_dto.module_name
        self.experiment_instance_id = live_dto.experiment_instance_id
        self.last_training_loss = None

    # ----------------------------------------------------------------------
    # Simple state mutation
    # ----------------------------------------------------------------------
    def add_iteration(self) -> None:
        """Increment the iteration counter on the underlying DTO."""
        self.live_dto.iteration_count += 1
        
    def update_dto_vector(self) -> None:
        self.live_dto.vector_data = self.vector_data

    # ----------------------------------------------------------------------
    # Factory helpers – mirror the repository API
    # ----------------------------------------------------------------------
    @classmethod
    def create_from_template(
        cls,
        et_dto: ExperimentTemplateDTO,
        shape: Union[torch.Size, Tuple[int, ...]],
        live_repo: ExperimentLiveInstanceRepository,
        et_repo: ExperimentTemplateRepository,
    ) -> LiveInstance:
        """Create a live instance using the repository’s `create_from_template`."""
        live_dto = live_repo.create_from_template(et_dto, shape)
        return cls(live_dto, live_repo, et_repo)

    @classmethod
    def create_from_seed(
        cls,
        et_dto: ExperimentTemplateDTO,
        shape: Union[torch.Size, Tuple[int, ...]],
        seed: int,
        live_repo: ExperimentLiveInstanceRepository,
        et_repo: ExperimentTemplateRepository,
    ) -> LiveInstance:
        """Create a live instance from a random seed."""
        live_dto = live_repo.create_from_seed(et_dto, shape, seed)
        return cls(live_dto, live_repo, et_repo)

    @classmethod
    def create_from_tensor(
        cls,
        et_dto: ExperimentTemplateDTO,
        tensor: torch.Tensor,
        live_repo: ExperimentLiveInstanceRepository,
        et_repo: ExperimentTemplateRepository,
    ) -> LiveInstance:
        """Create a live instance from an already‑prepared tensor."""
        live_dto = live_repo.create_from_initial_tensor(et_dto, tensor)
        return cls(live_dto, live_repo, et_repo)

    @classmethod
    def create_from_initial_vector_id(
        cls,
        et_dto: ExperimentTemplateDTO,
        vector_id: int,
        live_repo: ExperimentLiveInstanceRepository,
        et_repo: ExperimentTemplateRepository,
    ) -> LiveInstance:
        """
        Create a live instance when you already have a persisted vector id.
        The repository does not expose a direct method for this, so we fetch the
        vector, then build the DTO manually.
        """
        # Retrieve the vector DTO (raises if not found)
        vector_repo = live_repo.vector_repo
        vec_dto = vector_repo.get(vector_id)
        if vec_dto is None:
            raise ValueError(f"Vector {vector_id} does not exist")

        live_dto = live_repo.create_from_vec_dto(experiment_template_dto=et_dto,vec_dto=vec_dto)
        return cls(live_dto, live_repo, et_repo)


class BatchSteer:
    """
    Handles a batch of LiveInstance objects, registers forward‑hooks that
    inject the learned vectors into the model, and aggregates per‑instance loss.
    """

    def __init__(self, live_objs: List[LiveInstance], model: Type[torch.nn.Module]):#actually meant to be transformer
        self.live_objs = live_objs
        self.inst_id_to_slice: dict[int, slice] = {}
        # All instances are expected to share the same batch size
        self.num_prompts_in_batch = self.live_objs[0].et_dto.batch_size
        self.model = model
        self.module_to_hookfn : Dict[str,Callable] = {}

    def _make_hookfns(self):#has to be called each time steer is called
        # Group vectors by the submodule they should be injected into
        module_to_instlist = defaultdict(list)
        module_to_veclist = defaultdict(list)
        for live_obj in self.live_objs:
            module_to_instlist[live_obj.module_name].append(live_obj)
            module_to_veclist[live_obj.module_name].append(
                live_obj.vector_data.unsqueeze(0).to(dtype = self.model.dtype,device=self.model.device) #that is a copy 
            )

        start = self.num_prompts_in_batch
        tensorstart = start
        for module,veclist in module_to_veclist.items():
            tensor = torch.concat(veclist, dim=0)  # (num_vecs, width)
            for i, _ in enumerate(veclist):
                inst_id = module_to_instlist[module][i].experiment_instance_id
                self.inst_id_to_slice[inst_id] = slice(
                    start, start + self.num_prompts_in_batch
                )
                start += self.num_prompts_in_batch
            tensor_slice = slice(tensorstart,start)
            self.module_to_hookfn[module] = self._make_hookfn(tensor,tensor_slice)

    # ----------------------------------------------------------------------
    # Hook creation
    # ----------------------------------------------------------------------
    def _make_hookfn(self, tensor: torch.Tensor,tensor_slice: slice):
        """
        Returns a forward‑hook that appends the learned vector (broadcasted)
        to the module’s output.
        """

        def hook(module, inp, output):
            stacked_vectors = tensor.repeat(self.num_prompts_in_batch, 1)# (num_vecs*num_prompts, width)
            batch_slice = tensor_slice
            stacked_vectors = stacked_vectors.unsqueeze(1)  # (num_vecs*num_prompts, 1, width)
            output[batch_slice] = output[batch_slice] + stacked_vectors #inplace modificatin of output
            return output

        return hook
#incorrect, new hook function must be made each time its steered


    # ----------------------------------------------------------------------
    # Loss aggregation
    # ----------------------------------------------------------------------
    def calc_loss(self, output, input_ids: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((1,),dtype=output.hidden_states[0].dtype,device=output.hidden_states[0].device) #not ideal
        vanilla_slice = slice(0, self.num_prompts_in_batch)

        for live_obj in self.live_objs:
            inst_slice = self.inst_id_to_slice[live_obj.experiment_instance_id]
            inst_loss = live_obj.loss_fn(
                output, vanilla_slice, inst_slice, input_ids
            )
            live_obj.last_training_loss = inst_loss
            loss += inst_loss

        return loss

    # ----------------------------------------------------------------------
    # Context manager that registers and removes hooks
    # ----------------------------------------------------------------------
    @contextmanager
    def steer(self):
        handles = []
        try:
            self._make_hookfns()#need to make new hookfn each time, as tensors are copied into the function
            for module_name, hookfn in self.module_to_hookfn.items():
                submodule = self.model.get_submodule(module_name)
                handles.append(submodule.register_forward_hook(hookfn))
            yield
        finally:
            for h in handles:
                h.remove()
                
    

