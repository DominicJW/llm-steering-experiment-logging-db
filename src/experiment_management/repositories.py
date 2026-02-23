from .orm import (
    Vector,
    PromptGroup,
    PromptGroupItem,
    Prompt,
    ExperimentTemplate,
    ExperimentLiveInstance,
)
from .utils import make_repro_tensor
from typing import Optional
import torch
from sqlalchemy import select, and_, case, func, distinct
from sqlalchemy.orm import selectinload
from .db import get_session


def _hydrate_for_detached_use(session, item):
    if isinstance(item, ExperimentLiveInstance):
        return session.scalar(
            select(ExperimentLiveInstance)
            .where(ExperimentLiveInstance.experiment_instance_id == item.experiment_instance_id)
            .options(selectinload(ExperimentLiveInstance.experiment_template))
        )
    return item

def create_vector(shape:Optional=None,seed:Optional=None,tensor:Optional=None) -> Vector:
    assert type(tensor) is torch.Tensor or not (shape  is None)
    assert type(shape) is tuple or type(shape) is torch.Size
    assert type(seed) is int

    if seed is None:
        seed = random.randint(1,int(1e8))

    if not (tensor is None):
        seed = None
        shape = tuple(tensor.shape)
    else:
        shape = tuple(shape)
        tensor = make_repro_tensor(seed=seed,shape=shape)
    
    clauses = [
        Vector.seed == seed,
        Vector.shape == shape,
    ]
    
    with get_session() as session:
        stmt = select(Vector).where(and_(*clauses))
        existing = session.scalars(stmt).first()
        if existing is not None:
            print("vector of matching seed and shape exists, returning that vector")
            return existing
        
        new_vec = Vector(seed=seed, shape=shape, vector_data=tensor)
        session.add(new_vec)
        session.commit()
        session.refresh(new_vec)
        return new_vec

def create_prompt_group_from_prompts(prompts):
    prompt_ids = [prompt.prompt_id for prompt in prompts]
    if any(prompt_id is None for prompt_id in prompt_ids):
        raise ValueError("All prompts must be persisted before creating a prompt group")

    unique_prompt_ids = sorted(set(prompt_ids))

    with get_session() as session:
        if len(unique_prompt_ids) == 0:
            # Reuse an existing empty group, if any.
            stmt = (
                select(PromptGroup)
                .outerjoin(PromptGroupItem, PromptGroup.group_id == PromptGroupItem.group_id)
                .group_by(PromptGroup.group_id)
                .having(func.count(PromptGroupItem.id) == 0)
            )
        else:
            # Exact-set match:
            # 1) group size equals input size
            # 2) all group prompt_ids are in the input set
            stmt = (
                select(PromptGroup)
                .join(PromptGroupItem, PromptGroup.group_id == PromptGroupItem.group_id)
                .group_by(PromptGroup.group_id)
                .having(func.count(PromptGroupItem.prompt_id) == len(unique_prompt_ids))
                .having(func.count(distinct(PromptGroupItem.prompt_id)) == len(unique_prompt_ids))
                .having(
                    func.sum(
                        case(
                            (PromptGroupItem.prompt_id.in_(unique_prompt_ids), 1),
                            else_=0,
                        )
                    )
                    == len(unique_prompt_ids)
                )
            )
        existing = session.scalars(stmt).first()
        if existing is not None:
            print("Exact same group exists, returning that")
            hydrated_existing = session.scalar(
                select(PromptGroup)
                .where(PromptGroup.group_id == existing.group_id)
                .options(
                    selectinload(PromptGroup.group_links).selectinload(PromptGroupItem.prompt)
                )
            )
            return hydrated_existing

        new_group = PromptGroup()
        session.add(new_group)
        session.commit()
        session.refresh(new_group)
        for prompt_id in unique_prompt_ids:
            session.add(PromptGroupItem(group_id=new_group.group_id, prompt_id=prompt_id))
        session.commit()
        hydrated_new_group = session.scalar(
            select(PromptGroup)
            .where(PromptGroup.group_id == new_group.group_id)
            .options(selectinload(PromptGroup.group_links).selectinload(PromptGroupItem.prompt))
        )
        return hydrated_new_group


        
def create_prompt_group_from_strings(strings):
    prompts = [persist(Prompt(text=text)) for text in strings]
    return create_prompt_group_from_prompts(prompts)
    

        
def persist(item,check_dupe=True):
    _CLASS = type(item)
    non_pk_columns = [
        column for column in _CLASS.__table__.columns if not column.primary_key
    ]
    clauses = [
        getattr(_CLASS, column.name) == getattr(item, column.name)
        for column in non_pk_columns
    ]

    with get_session() as session:
        stmt = select(_CLASS).where(and_(*clauses))
        existing = session.scalars(stmt).first()
        if existing is not None and check_dupe is True:
            print(f"{_CLASS} Exists, returning that")
            return _hydrate_for_detached_use(session, existing)
        elif existing is not None and check_dupe =="warn":
            print(f"{_CLASS} Exists, Not returning that")

        session.add(item)
        session.commit()
        session.refresh(item)
        return _hydrate_for_detached_use(session, item)

def persist_experiment_template(et):
    non_pk_columns = [
        column for column in ExperimentTemplate.__table__.columns if not column.primary_key
    ]
    clauses = [
        getattr(ExperimentTemplate, column.name) == getattr(et, column.name)
        for column in non_pk_columns
    ]

    with get_session() as session:
        stmt = select(ExperimentTemplate).where(and_(*clauses))
        existing = session.scalars(stmt).first()
        if existing is not None:
            print("Experiment Exists, returning that")
            return existing
        session.add(et)
        session.commit()
        session.refresh(et)
        return et
