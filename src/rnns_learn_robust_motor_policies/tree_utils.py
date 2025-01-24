from collections.abc import Callable
from typing import Any, TypeVar, Sequence

import equinox as eqx
import jax as jax 
import jax.tree as jt
from jaxtyping import Array, PyTree
import plotly.graph_objects as go

from feedbax import is_module, tree_take, tree_key_tuples
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


# TODO: This exists because I was thinking of generalizing the way that
# the model PyTree is constructed in training notebook 2. If that doesn't get done, 
# then it would make sense to just do a dict comprehension explicitly when 
# constructing the `task_model_pairs` dict, instead of making an 
# opaque call to this function.
def map_kwargs_to_dict(
    func: Callable[..., Any],  #! kwargs only
    keyword: str,
    values: Sequence[Any],
):
    """Given a function that takes optional kwargs, evaluate the function over 
    a sequence of values of a single kwarg
    """
    return dict(zip(
        values, 
        map(
            lambda value: func(**{keyword: value}), 
            values,
        )
    ))


def falsef(x):
    return False

    
def tree_level_types(tree: PyTree, is_leaf=falsef) -> list[type]:
    """Given a PyTree, return a PyTree of the types of each node along the path to the first leaf."""
    treedef = jt.structure(tree)
    
    subtreedef = treedef
    types = []
    
    while any(subtreedef.children()):
        node_data = subtreedef.node_data()
        if node_data is not None:
            if is_leaf(node_data[0]):
                break
            types.append(node_data[0])
        subtreedef = subtreedef.children()[0]
    
    return types


def tree_map_with_keys(func, tree: PyTree, *rest, is_leaf=None, **kwargs):
    """Maps `func` over a PyTree, returning a PyTree of the results and the paths to the leaves.
    
    The first argument of `func` must be the path
    """
    return jt.map(
        func,
        tree,
        tree_key_tuples(tree, is_leaf=is_leaf),
        *rest,
        is_leaf=is_leaf,
        **kwargs,
    )


def tree_subset_dict_level(tree: PyTree[dict[T, Any]], keys: Sequence[T], dict_type=dict):
    """Maps `subdict` over dicts of a given type"""
    return jt.map(
        lambda d: subdict(d, keys),
        tree,
        is_leaf=is_type(dict_type),
    )
    

def flatten_with_paths(tree, is_leaf=None):
    return jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)


def index_multi(obj, *idxs):
    """Index zero or more times into a Python object."""
    if not idxs:
        return obj
    return index_multi(obj[idxs[0]], *idxs[1:])


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