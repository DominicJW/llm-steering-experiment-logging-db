# steering.py
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Dict, List, Type
import warnings

import torch
import torch.nn.functional as F

from .orm import ExperimentLiveInstance
from .services import (
    ExperimentLiveInstanceService,
    ExperimentSnapshotService,
    VectorService,
)
from .utils import slice_batch_output, loss_factory, optimizer_factory

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
        live: ExperimentLiveInstance,
    ):
        self.live = live #should choose bettr name
        self.last_snapshot = None

        self.live = ExperimentLiveInstanceService.refresh_all(self.live) #loads the relationships
        if self.live.vector_data is None:
            raise ValueError("live.vector_data is None; initialize the live instance vector before steering")

        normalization = self.live.experiment_template.normalization
        if normalization is None or normalization <= 0:
            raise ValueError(f"normalization must be > 0, got {normalization}")

        self.module_name = self.live.experiment_template.module_name
        with torch.no_grad():
            self.live.vector_data = self.live.vector_data.detach().clone()
            self.vector_data.copy_(normalization * F.normalize(self.vector_data,dim=0))
        self.vector_data.requires_grad_(True)

        # build loss & optimizer from the template
        loss_kwargs = self.live.experiment_template.loss_additional_parameters or {}
        self.loss_fn = loss_factory(
            self.live.experiment_template.loss_name,
            **loss_kwargs,
        )
        optimizer_kwargs = self.live.experiment_template.optimizer_additional_parameters or {}
        self.optimizer_constructor = optimizer_factory(
            self.live.experiment_template.optimizer_name,
            **optimizer_kwargs,
        )
        self.optimizer = self.optimizer_constructor([self.vector_data])
        self.optimizer.zero_grad()
        self.last_training_loss = None

    def step_optimizer(self):
        if self.vector_data.grad is None:
            warnings.warn(f"None gradient encountered in LiveInstance.step_optimizer {self.live}")
            return
        normalization = self.live.experiment_template.normalization
        with torch.no_grad():
            self.vector_data.grad.sub_(torch.dot(self.vector_data.grad, self.vector_data) * self.vector_data / (normalization**2))
        self.optimizer.step()
        with torch.no_grad():
            self.vector_data.copy_(normalization * F.normalize(self.vector_data,dim=0))
        self.optimizer.zero_grad()

    @property
    def vector_data(self):
        return self.live.vector_data
    
    @property
    def iteration_count(self):
        return self.live.iteration_count

    @iteration_count.setter
    def iteration_count(self,value):
        self.live.iteration_count = value

    def create_snapshot(self,save_vector=False):
        vector_id = None
        if save_vector:
            vector = VectorService.create_persisted(
                tensor=self.vector_data.detach().clone()
            )
            vector_id = vector.vector_id
        self.last_snapshot = ExperimentSnapshotService.create_persisted(
                vector_id=vector_id,
                iteration_count=self.iteration_count,
                experiment_instance_id=self.live.experiment_instance_id,
        )
        return self.last_snapshot

    def update(self):
        self.live = ExperimentLiveInstanceService.update(self.live)


class BatchSteer:
    """
    Handles a batch of LiveInstance objects, registers forward‑hooks that
    inject the learned vectors into the model, and aggregates per‑instance loss.
    """

    def __init__(self, live_objs: List[LiveInstance], model):#actually meant to be transformer
        if  len(live_objs) > 0:
            ValueError("live_objs cannot be empty")
        self.live_objs = live_objs
        self.inst_id_to_slice: dict[int, slice] = {}
        # All instances are expected to share the same batch size
        self.num_prompts_in_batch = self.live_objs[0].live.experiment_template.batch_size
        self.model = model
        self.module_to_hookfn : Dict[str,Callable] = {}
        #could enforce that all live_objs have same prompt_group


    def _make_hookfns(self):#has to be called each time steer is called
        self.inst_id_to_slice = {}
        self.module_to_hookfn = {}
        # Group vectors by the submodule they should be injected into
        module_to_instlist = defaultdict(list)
        for live_obj in self.live_objs:
            module_to_instlist[live_obj.module_name].append(live_obj)

        start = 0
        tensor_start = start
        for module,live_objs in module_to_instlist.items():
            vector_list = []
            for live in live_objs:
                vector_list.append(live.vector_data)
                inst_id = live.live.experiment_instance_id
                self.inst_id_to_slice[inst_id] = slice(
                    start, start + self.num_prompts_in_batch
                )
                start += self.num_prompts_in_batch
            tensor = torch.concat([vec.unsqueeze(0) for vec in vector_list],dim=0) # (num_vecs, width)
            tensor_slice = slice(tensor_start,start)
            tensor_start = start
            stacked_vectors = tensor.repeat(self.num_prompts_in_batch, 1)# (num_vecs*num_prompts, width)
            stacked_vectors = stacked_vectors.unsqueeze(1)
            self.module_to_hookfn[module] = self._make_hookfn(stacked_vectors,tensor_slice)

    # ----------------------------------------------------------------------
    # Hook creation
    # ----------------------------------------------------------------------
    def _make_hookfn(self, stacked_vectors: torch.Tensor,tensor_slice: slice):
        """
        Returns a forward‑hook that appends the learned vector (broadcasted)
        to the module’s output.
        """
        def hook(module, inp, output):
            output[tensor_slice,:,:] = output[tensor_slice,:,:] + stacked_vectors.to(device = self.model.device,dtype = self.model.dtype) #inplace modificatin of output
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