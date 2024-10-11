from collections.abc import Callable
import equinox as eqx 
from jaxtyping import PyTree


def swap_model_trainables(model: PyTree[..., "T"], trained: PyTree[..., "T"], where_train: Callable):
    return eqx.tree_at(
        where_train,
        model,
        where_train(trained),
    )
    

def subdict(dct, keys):
    return {k: dct[k] for k in keys}