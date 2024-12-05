from collections.abc import Callable, Mapping
from typing import Any, TypeVar, Sequence

import equinox as eqx 
import jax.tree as jt
from jaxtyping import PyTree
import plotly.graph_objects as go

from feedbax import is_module, tree_take
from feedbax._tree import eitherf, is_type
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


def subset_dict_tree_level(tree: PyTree[dict[T, Any]], keys: Sequence[T], dict_type=dict):
    """Maps `subdict` over dicts of a given type"""
    return jt.map(
        lambda d: subdict(d, keys),
        tree,
        is_leaf=is_type(dict_type),
    )


def pp(tree):
    """Pretty-prints PyTrees, truncating objects commonly treated as leaves during data analysis."""
    eqx.tree_pprint(tree, truncate_leaf=eitherf(is_module, is_type(go.Figure)))


def take_single_replicate(models, i):
    # Need to partition since there are non-vmapped arrays in the intervenors...
    intervenors, other = eqx.partition(
        models, 
        jt.map(lambda x: isinstance(x, AbstractIntervenor), models, is_leaf=is_type(AbstractIntervenor)),
    )
    return eqx.combine(intervenors, tree_take(other, i))
