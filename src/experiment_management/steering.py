# steering.py
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple, Union,Dict,Callable,Type
import warnings

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
        snap_repo: ExperimentSnapshotRepository
    ):
        self.live_dto = live_dto
        self.live_repo = live_repo
        self.et_repo = et_repo
        self.snap_repo = snap_repo
        self.last_snapshot = None


        # expose frequently‑used fields
        self.experiment_instance_id = self.live_dto.experiment_instance_id
        self.experiment_template_id = self.live_dto.experiment_template_id
        self.et_dto = self.et_repo.get(self.experiment_template_id)
        if self.et_dto is None:
            raise ValueError(
                f"ExperimentTemplate {self.live_dto.experiment_template_id} not found"
            )
        self.module_name = self.et_dto.module_name

        with torch.no_grad():
            self.live_dto.vector_data = self.live_dto.vector_data.detach().clone()
            self.vector_data.copy_(self.et_dto.normalization*F.normalize(self.vector_data,dim=0))
        self.vector_data.requires_grad_(True)

        # build loss & optimizer from the template
        self.loss_fn = loss_factory(
            self.et_dto.loss_name, self.et_dto.loss_additional_parameters
        )
        self.optimizer_constructor = optimizer_factory(
            self.et_dto.optimizer_name, self.et_dto.optimizer_additional_parameters
        )
        self.optimizer = self.optimizer_constructor([self.vector_data])
        self.optimizer.zero_grad()
        self.last_training_loss = None

    def step_optimizer(self):
        if self.vector_data.grad is None:
            warnings.warn(f"None gradient encountered in LiveInstance.step_optimizer {self.live_dto}")
        with torch.no_grad():
            self.vector_data.grad.sub_(torch.dot(self.vector_data.grad, self.vector_data) * self.vector_data / (self.et_dto.normalization**2))
        self.optimizer.step()
        with torch.no_grad():
            self.vector_data.copy_(self.et_dto.normalization*F.normalize(self.vector_data,dim=0))
        self.optimizer.zero_grad()

    @property
    def vector_data(self):
        return self.live_dto.vector_data
    
    @property
    def iteration_count(self):
        return self.live_dto.iteration_count

    @iteration_count.setter
    def iteration_count(self,value):
        self.live_dto.iteration_count = value

    def create_snapshot(self,save_vector=False):
        self.last_snapshot = self.snap_repo.create_from_live(self.live_dto,save_vector=save_vector)
        return self.last_snapshot

    def update_repo(self):
        return self.live_repo.update(self.live_dto)


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
        #could enforce that all live_objs have same prompt_group


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
        tensor_start = start
        for module,veclist in module_to_veclist.items():
            tensor = torch.concat(veclist, dim=0)  # (num_vecs, width)
            for i, _ in enumerate(veclist):
                inst_id = module_to_instlist[module][i].experiment_instance_id
                self.inst_id_to_slice[inst_id] = slice(
                    start, start + self.num_prompts_in_batch
                )
                start += self.num_prompts_in_batch
            tensor_slice = slice(tensor_start,start)
            tensor_start = start
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
                
    
