from typing import Callable,Dict,Type
import json
from torch import optim

loss_registry : Dict[str,Callable] = {}

def register_loss_fn(name:str, function:Callable):
    loss_registry[name] = function

def loss_factory(name:str,kwargs:str):
    kwargs = json.loads(kwargs)
    loss_fn = loss_registry[name]
    return lambda output,vanilla_slice,inst_slice,input_ids : loss_fn(output,vanilla_slice,inst_slice,input_ids,**kwargs)
    
    
optimizer_registry : Dict[str,Type[optim.Optimizer]] = {"AdamW":optim.AdamW}

def optimizer_factory(name:str,kwargs:str):
    kwargs = json.loads(kwargs)
    optimizer_class = optimizer_registry[name]
    return lambda parameter_list : optimizer_class(parameter_list,**kwargs)
