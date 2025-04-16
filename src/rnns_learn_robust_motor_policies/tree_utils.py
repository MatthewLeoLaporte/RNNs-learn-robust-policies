from collections.abc import Callable, Mapping
import logging
from types import SimpleNamespace
from typing import Any, TypeVar, Sequence

import equinox as eqx
import jax as jax 
import jax.tree as jt
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree
import plotly.graph_objects as go

from feedbax.intervene import AbstractIntervenor
from jax_cookbook import anyf, is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.config import STRINGS
from rnns_learn_robust_motor_policies.types import LDict, TreeNamespace



T = TypeVar("T")
M = TypeVar("M", bound=Mapping)


logger = logging.getLogger(__name__)


def swap_model_trainables(model: PyTree[..., "T"], trained: PyTree[..., "T"], where_train: Callable):
    return eqx.tree_at(
        where_train,
        model,
        where_train(trained),
    )


def _get_mapping_constructor(d: Mapping):
    if isinstance(d, LDict):
        return LDict.of(d.label)   
    else:
        return type(d)


def subdict(d: Mapping[T, Any], keys: Sequence[T]):
    """Returns the mapping containing only the keys `keys`."""
    return _get_mapping_constructor(d)({k: d[k] for k in keys})


def dictmerge(*dicts: Mapping) -> Mapping:
    if len(set(type(d) for d in dicts)) == 1:
        constructor = _get_mapping_constructor(dicts[0])
    else: 
        constructor = dict
    return constructor({k: v for d in dicts for k, v in d.items()})


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

    
def tree_level_labels(tree: LDict, is_leaf=falsef, sep=None) -> list[str]:
    """
    Given a PyTree of LDict nodes, return a list of labels, one for each level of the tree.
    
    This function assumes a homogeneous tree structure where all nodes at the same level
    have the same label. It traverses the tree from root to first leaf, collecting LDict
    labels along the way.
    """
    # Get the path to the first leaf
    paths, _ = jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)
    if not paths:
        return []
    first_path, _ = paths[0]
    
    # Collect the labels from all LDict nodes in the path
    labels = []
    current_node = tree
    for path_element in first_path:
        # If this is an LDict, collect its label
        if isinstance(current_node, LDict):
            labels.append(current_node.label)   
        else:
            labels.append(current_node.__class__.__name__)
            logger.warning(
                f"Non-LDict node encountered when labeling tree levels: {type(current_node)}; " 
                "assuming it is a leaf, and stopping."
            )
            break
        
        # Get the node at this level
        if isinstance(current_node, dict) or hasattr(current_node, '__getitem__'):
            current_node = current_node[path_element.key if hasattr(path_element, 'key') else path_element]
        
        if is_leaf(current_node):
            break
        
    if sep is not None:
        labels = [label.replace(STRINGS.hps_level_label_sep, sep) for label in labels]
        
    return labels


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
        jtree.key_tuples(tree, is_leaf=is_leaf),
        *rest,
        is_leaf=is_leaf,
        **kwargs,
    )
    
    
K = TypeVar('K')
V = TypeVar('V')

LT = TypeVar('LT', bound=str) 


def tree_subset_ldict_level(tree: PyTree[LDict[K, V]], keys: Sequence[K], label: str):
    """Maps `subdict` over LabeledDict nodes with a specific label in a PyTree.
    """
    ldicts, other = eqx.partition(tree, LDict.is_of(label), is_leaf=LDict.is_of(label))
    ldicts = [subdict(ld, keys) for ld in ldicts if ld is not None]
    return eqx.combine(ldicts, other)
    

def flatten_with_paths(tree, is_leaf=None):
    return jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)


def index_multi(obj, *idxs):
    """Index zero or more times into a Python object."""
    if not idxs:
        return obj
    return index_multi(obj[idxs[0]], *idxs[1:])


_is_leaf = anyf(is_module, is_type(go.Figure, TreeNamespace))


def pp(tree, truncate_leaf=_is_leaf):
    """Pretty-prints PyTrees, truncating objects commonly treated as leaves during data analysis."""
    eqx.tree_pprint(tree, truncate_leaf=truncate_leaf)


def pp2(tree, truncate_leaf=_is_leaf):
    """Substitute for `pp` given that `truncate_leaf` of `eqx.tree_pprint` appears to be broken atm."""
    tree = jax.tree_map(
        lambda x: type(x).__name__ if truncate_leaf(x) else x,
        tree,
        is_leaf=truncate_leaf,
    )
    eqx.tree_pprint(tree)


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
    return eqx.combine(intervenors, jtree.take(other, i))


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


def at_path(path):
    def at_func(obj):
        """Navigate to `path` in `obj` and return the value there."""
        # TODO: Generalize this to use the usual key types from `jax.tree_utils`
        # We can then create a separate function to translate "simple" representations
        # like `('step', 'feedback_channels', 0, 'noise_func', 'std')` into paths that use 
        # e.g. `DictKey`
        for key in path:
            if isinstance(obj, (eqx.Module, TreeNamespace)):
                # Assume the key can be cast to the attribute name (string)
                obj = getattr(obj, str(key))
            elif isinstance(obj, (dict, list, tuple)):
                # Assume the key types match with the tree level types so this doesn't err 
                obj = obj[key]

        return obj
    return at_func


