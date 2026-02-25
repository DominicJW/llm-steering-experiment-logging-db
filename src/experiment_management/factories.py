from .utils import (
    loss_factory,
    loss_registry,
    optimizer_factory,
    optimizer_registry,
    register_loss_fn,
)


def register_optimizer(name, optimizer_class):
    optimizer_registry[name] = optimizer_class
