# steering.py
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple, Union,Dict,Callable,Type

import torch
import torch.nn.functional as F


from .dto import ExperimentLiveInstanceDTO, ExperimentTemplateDTO
from .factories import loss_factory, optimizer_factory
from .utils import slice_batch_output
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
        #note that 
        with torch.no_grad():
            self.vector_data.data = et_dto.normalization*F.normalize(self.vector_data,dim=0)

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

    def step_optimizer(self):
        with torch.no_grad():
            self.vector_data.grad.sub_(torch.dot(self.vector_data.grad, self.vector_datas) * self.vector_data / (self.et_dto.normalization**2))
        self.optimzer.step()
        with torch.no_grad():
            self.vector_data.copy_(self.et_dto.normalization*F.normalize(self.vector_data,dim=0))
        self.optimizer.zero_grad()




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
        self.inst_id_to_slice = {}
        self.module_to_hookfn = {}
        # Group vectors by the submodule they should be injected into
        module_to_instlist = defaultdict(list)
        module_to_veclist = defaultdict(list)
        for live_obj in self.live_objs:
            module_to_instlist[live_obj.module_name].append(live_obj)
            module_to_veclist[live_obj.module_name].append(
                live_obj.vector_data.unsqueeze(0).to(dtype = self.model.dtype,device=self.model.device) #that is a copy 
            )

        start = 0
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



    # ----------------------------------------------------------------------
    # Loss aggregation
    # ----------------------------------------------------------------------
    def calc_loss(self, output, vanilla_output, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != self.num_prompts_in_batch:
            raise ValueError(
                f"Expected input_ids batch size {self.num_prompts_in_batch}, got {input_ids.shape[0]}"
            )
        loss = torch.zeros((1,),dtype=self.model.dtype,device=self.model.device)

        for live_obj in self.live_objs:
            inst_slice = self.inst_id_to_slice[live_obj.experiment_instance_id]
            steered_output = slice_batch_output(output, inst_slice)
            inst_loss = live_obj.loss_fn(
                steered_output, vanilla_output, input_ids
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
                
    
