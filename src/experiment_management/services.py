import random
from typing import Optional, Tuple

import torch

from .orm import (
    Base,
    ExperimentLiveInstance,
    ExperimentSnapshot,
    ExperimentTemplate,
    GeneratedOutput,
    Metric,
    Prompt,
    PromptGroup,
    PromptGroupItem,
    VanillaBaseline,
    Vector,
)
from .repositories import BaseRepository, VectorRepository
from .utils import make_repro_tensor


class BaseService:
    Model = Base
    Repository = BaseRepository

    @classmethod
    def _repo(cls):
        return cls.Repository(cls.Model)

    @classmethod
    def persist(cls, model):
        matching = cls.find_matching(model, excluded=[])
        if matching:
            return matching[0]
        return cls._repo().persist(model, load_relationships=True)

    @classmethod
    def create_non_persisted(cls, **kwargs):
        return cls.Model(**kwargs)

    @classmethod
    def create_persisted(cls, **kwargs):
        model = cls.create_non_persisted(**kwargs)
        return cls.persist(model)

    @classmethod
    def refresh_all(cls, model):
        return cls._repo().refresh_all(model)

    @classmethod
    def find_matching(cls, model, excluded=None):
        return cls._repo().find_matching(model, excluded)

    @classmethod
    def find_by(cls, criteria, load_relationships: bool = False):
        return cls._repo().find_by(criteria, load_relationships=load_relationships)
    



class ExperimentTemplateService(BaseService):
    Model = ExperimentTemplate


class ExperimentLiveInstanceService(BaseService):
    Model = ExperimentLiveInstance

    @classmethod
    def update(cls, model):
        return cls._repo().update(model, load_relationships=True)

    @classmethod
    def create_persisted_from_snap(cls,snapshot=None,**kwargs):
        if snapshot is None:
            raise ValueError("snapshot argument not set, use create_persisted method instead")
        if snapshot.vector_id is None:
            raise ValueError("Snapshot vector_id cannot be none when initalising from snapshot")
        if snapshot.iteration_count == 0:
            raise ValueError("Snapshot iteration count must be greater than 0 when initalising from snapshot")
        if "initial_vector_id" in kwargs.keys():
            if kwargs["initial_vector_id"] != snapshot.vector_id: #forgive user for passing both as long as they are the same
                raise ValueError("Cannot initialise from both initial vector and snapshot")
            del kwargs["initial_vector_id"]
        live_instance = cls.create_persisted(initial_vector_id=snapshot.vector_id,**kwargs)
        return cls.refresh_all(live_instance)
        


class VanillaBaselineService(BaseService):
    Model = VanillaBaseline


class ExperimentSnapshotService(BaseService):
    Model = ExperimentSnapshot


class GeneratedOutputService(BaseService):
    Model = GeneratedOutput


class MetricService(BaseService):
    Model = Metric


class VectorService(BaseService):
    Model = Vector
    Repository = VectorRepository

    @classmethod
    def find_matching(cls, model, excluded=None):
        # Only de-dupe reproducible vectors where both seed and shape are set.
        # If seed is None (e.g. ad-hoc tensor snapshots), always persist a new vector.
        if model.seed is None or model.shape is None:
            return []
        excluded_keys = list(excluded or [])
        excluded_keys.append("vector_data")
        return super().find_matching(model, excluded=excluded_keys)

    @classmethod
    def create_non_persisted(
        cls,
        seed: Optional[int] = None,
        shape: Optional[Tuple[int, ...] | torch.Size] = None,
        tensor: Optional[torch.Tensor] = None,
    ):
        if tensor is not None:
            shape = tuple(tensor.shape)
            seed = None
        else:
            if seed is None:
                seed = random.randint(1, int(1e8))
            if shape is None:
                raise ValueError("shape or tensor is required")
            shape = tuple(shape)
            tensor = make_repro_tensor(shape=shape, seed=seed)
        return super().create_non_persisted(seed=seed, shape=shape, vector_data=tensor)


class PromptService(BaseService):
    Model = Prompt


class PromptGroupItemService(BaseService):
    Model = PromptGroupItem


class PromptGroupService(BaseService):
    Model = PromptGroup

    @classmethod
    def create_from_strings(cls, strings=None):
        prompt_texts = list(strings or [])
        prompts = [PromptService.create_persisted(text=text) for text in prompt_texts]
        group = cls.create_non_persisted()
        groups = PromptGroupService.find_matching(group,excluded=["group_id"]) #gets all groups
        group_ids_set = set(g.group_id for g in groups)
        for prompt in prompts:
            prompt_group_item = PromptGroupItemService.create_non_persisted(
                prompt_id=prompt.prompt_id,
            )
            existing_prompt_group_items = PromptGroupItemService.find_matching(prompt_group_item,excluded=["group_id"])
            group_ids_set = group_ids_set - set(item.group_id for item in existing_prompt_group_items)


        if len(group_ids_set) == 0: #means there is no group which has all prompts in group
            group = cls.persist(group)
            for prompt in prompts:
                PromptGroupItemService.create_persisted(group_id = group.group_id,prompt_id=prompt.prompt_id)
            return cls.refresh_all(group)
        
        else: #there is a group which has all prompts in group (and possibly more)
            for group_id in group_ids_set:
                temp = PromptGroupItemService.create_non_persisted(group_id=group_id)
                prompt_group_items = PromptGroupItemService.find_matching(temp,excluded=["prompt_id"]) #gets all prompt_ids associated with group
                extra_prompts = set(item.prompt_id for item in prompt_group_items) - set(prompt.prompt_id for prompt in prompts)
                if len(extra_prompts) == 0: #there is a matching group
                    group = cls.create_non_persisted(group_id=group_id)
                    return cls.refresh_all(cls._repo().find_matching(group)[0])
            
            #if function not returned yet, that means all the groups have extra prompts in them
            group = cls.persist(group)
            for prompt in prompts:
                PromptGroupItemService.create_persisted(group_id = group.group_id,prompt_id=prompt.prompt_id)
            return cls.refresh_all(group)
    
    @classmethod
    def persist(cls,model):
        return cls._repo().persist(model)

