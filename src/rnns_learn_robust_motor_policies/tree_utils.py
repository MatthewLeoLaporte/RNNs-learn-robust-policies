from collections.abc import Callable, Mapping
from typing import Any, TypeVar, Sequence

import equinox as eqx 
import jax.tree as jt
from jaxtyping import Array, PyTree
import plotly.graph_objects as go

from feedbax import is_module, tree_take
from feedbax._tree import anyf, is_type
from feedbax.intervene import AbstractIntervenor


T = TypeVar("T")


def swap_model_trainables(model: PyTree[..., "T"], trained: PyTree[..., "T"], where_train: Callable):
    return eqx.tree_at(
        where_train,
        model,
        where_train(trained),
    )
    

def subdict(dct: dict[T, Any], keys: Sequence[T]):
    """Returns the dict containing only the keys `keys`."""
    return type(dct)({k: dct[k] for k in keys})


def dictmerge(*dicts: dict) -> dict:
    """Merges all """
    return {k: v for d in dicts for k, v in d.items()}


def tree_subset_dict_level(tree: PyTree[dict[T, Any]], keys: Sequence[T], dict_type=dict):
    """Maps `subdict` over dicts of a given type"""
    return jt.map(
        lambda d: subdict(d, keys),
        tree,
        is_leaf=is_type(dict_type),
    )


def pp(tree):
    """Pretty-prints PyTrees, truncating objects commonly treated as leaves during data analysis."""
    eqx.tree_pprint(tree, truncate_leaf=anyf(is_module, is_type(go.Figure)))


def take_replicate(i, tree: PyTree[Array, 'T']) -> PyTree[Array, 'T']:
    """"""
    # TODO: Wrap non-batched array leaves in a `Module`? 
    # e.g. `WithoutBatches[0]` means the wrapped array is missing axis 0 relative to the "full" state;
    # for ensembled models, this is the ensemble (or model replicate) axis. So in this function, we should
    # be able to check for `WithoutBatches[0]`, given that the model is ensembled.
    # Need to partition since there are non-vmapped *arrays* in the intervenors...
    intervenors, other = eqx.partition(
        tree, 
        jt.map(
            lambda x: isinstance(x, AbstractIntervenor), 
            tree, 
            is_leaf=is_type(AbstractIntervenor),
        ),
    )
    return eqx.combine(intervenors, tree_take(other, i))


def deep_update(d1, d2):
    """Updates a dict with another, recursively.
    
    ```
    deep_update(dict(a=dict(b=2, c=3)), dict(a=dict(b=4)))
    # Returns dict(a=dict(b=4, c=3)), not dict(a=dict(b=4)).
    ```
    """
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1