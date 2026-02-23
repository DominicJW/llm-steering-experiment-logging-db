from .orm import Vector, PromptGroup, PromptGroupItem,Prompt
from .utils import make_repro_tensor
from typing import Optional
import torch
from sqlalchemy import select,and_, insert

def create_vector(self,shape:Optional,seed:Optional,tensor:Optional) -> Vector:
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
    
    with Session(engine) as session:
        stmt = select(Vector).where(and_(*clauses))
        existing = session.scalars(stmt).first()
        if existing is not None:
            print("vector of matching seed and shape exists, returning that vector")
            return existing
        
        new_vec = Vector(seed=seed, shape=shape, tensor=tensor)
        session.add(new_vec)
        session.commit()
        session.refresh(new_vec)
        return new_vec

def create_prompt_group_from_prompts(prompts):
    with Session(engine):
        stmt = select(PromptGroupItem
        ).join(PromptGroup, PromptGroup.group_id == PromptGroupItem.group_id
        ).join(Prompt.prompt_id, Prompt.prompt_id == PromptGroupItem.prompt_id
        ).where()
        existing = session.scalars(stmt).first()
        if existing is not None:
            print("Exact same group exists, returning that")
            return existing
        new_group = PromptGroup()
        session.add(new_group)
        session.commit()
        session.refresh(new_group)
        for prompt in prompts:
            session.add(PromptGroupItem(group_id=new_group.group_id,prompt_id=prompt.prompt_id))
            session.commit()
        session.refresh(new_group)
        return new_group


def persist_prompt(prompt):
    with Session(engine):
        stmt = select(Prompt).where(text=prompt.text)
        existing = session.scalars(stmt).first()
        if existing is not None:
            print("Exact same prompt exists, returning that")
            return existing
        session.add(prompt)
        session.commit()
        session.refresh(prompt)
        return prompt        
        


        
        
