from collections.abc import Callable, KeysView, Mapping
import logging
from types import SimpleNamespace
from typing import Any, Optional, TypeVar, Sequence
import hashlib
import json
import re
import types

import equinox as eqx
import jax as jax 
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree
import plotly.graph_objects as go

from feedbax.intervene import AbstractIntervenor
from jax_cookbook import anyf, is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.config import STRINGS
from rnns_learn_robust_motor_policies.types import _Wrapped, LDict, LDictConstructor, TreeNamespace



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

    
def tree_level_labels(
    tree: LDict, 
    sep: Optional[str] = None,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> list[str]:
    """
    Given a PyTree of LDict nodes, return a list of labels, one for each level of the tree.
    
    This function assumes a homogeneous tree structure where all nodes at the same level
    have the same label. It traverses the tree from root to first leaf, collecting LDict
    labels along the way.
    """
    # Get the path to the first leaf
    leaves_with_typed_paths = jtree.leaves_with_annotated_path(
        tree, is_leaf=is_leaf, annotation_func=_annotation_func,
    )
    if not leaves_with_typed_paths:
        return []
    first_path, _ = leaves_with_typed_paths[0]
    
    # Collect the labels from all LDict nodes in the path
    labels = []
    for node_type, _ in first_path:
        if isinstance(node_type, LDictConstructor):
            labels.append(node_type.label)   
        else:
            labels.append(node_type.__name__)
        
    if sep is not None:
        labels = [label.replace(STRINGS.hps_level_label_sep, sep) for label in labels]
        
    return labels


def _annotation_func(node):
    #! For use with `jtree.leaves_with_annotated_path`
    if is_type(LDict)(node):
        return LDict.of(node.label)
    else:
        return type(node)


def print_ldict_tree_summary(tree):
    """Prints a summary of the labels and keys of a pure `LDict` tree."""
    while True:
        try:
            print(tree.label, '\t', list(tree.keys()))
            tree = next(iter(tree.values()))
        except AttributeError:
            break


def ldict_level_to_top(label: str, tree: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None) -> list[type]:
    """Given an `LDict` label and a PyTree containing such nodes, move them to the top of the tree."""
    #? `is_leaf` doesn't seem to work as an argument to an inner `jt.structure` in transpose,
    #? e.g. when we want to keep `Responses` on the inside; 
    #? instead, wrap these inner nodes as non-PyTree objects to force them to be treated as leaves.

    level_labels = tree_level_labels(tree, is_leaf=is_leaf)

    if label not in level_labels:
        logger.warning(f"{label} not found in tree; doing nothing.")
        return tree
    
    if label == level_labels[0]:
        logger.debug(f"Level {label} is already at the top of the tree; doing nothing.")
        return tree
    
    if is_leaf is not None:
        wrapped_tree = jt.map(_Wrapped, tree, is_leaf=is_leaf)
    else:
        if label == level_labels[-1]:
            wrapped_tree = tree  # No need to wrap
        else:
            # Treat nodes inside the given level as leaves, so they are not carried outward
            subtree_outer_label = level_labels[level_labels.index(label) + 1]
            wrapped_tree = jt.map(_Wrapped, tree, is_leaf=LDict.is_of(subtree_outer_label))

    result = jt.transpose(
        jt.structure(wrapped_tree, is_leaf=LDict.is_of(label)),
        None,
        wrapped_tree,
    )

    return jt.map(lambda x: x.unwrap() if isinstance(x, _Wrapped) else x, result)


def ldict_level_to_bottom(label: str, tree: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None) -> list[type]:
    """Given an `LDict` label and a PyTree containing such nodes, move them to the bottom of the tree.
    
    Similarly to other functions that operate on PyTrees of `LDict`, assumes that the same-labelled `LDict`
    does not appear at multiple levels of the tree.
    """
    #! Use an inefficient implementation that moves each other level to the top.
    #! TODO: Find a more efficient implementation. 
    # 1. Could do a `jt.map` starting at the level of `label`, and only move *lower* levels to the top of the subtrees.
    # 2. Could find a way to use `jt.transpose`, maybe.

    level_labels = tree_level_labels(tree, is_leaf=is_leaf)

    if label not in level_labels:
        logger.warning(f"{label} not found in tree; doing nothing.")
        return tree

    if label == level_labels[-1]:
        logger.debug(f"Level {label} is already at the bottom of the tree; doing nothing.")
        return tree

    level_labels.remove(label)

    wrapped_tree = jt.map(_Wrapped, tree, is_leaf=is_leaf)

    current_tree = wrapped_tree
    for level_label in reversed(level_labels):
        current_tree = ldict_level_to_top(level_label, current_tree, is_leaf=is_leaf)

    return jt.map(lambda x: x.unwrap() if isinstance(x, _Wrapped) else x, current_tree)


def move_ldict_level_above(inner_label: str, outer_label: str, tree: PyTree, is_leaf: Optional[Callable[[Any], bool]] = None) -> list[type]:
    """Move an `LDict` level just above another, in a PyTree."""
    level_labels = tree_level_labels(tree, is_leaf=is_leaf)
    inner_level_idx = level_labels.index(inner_label)
    outer_level_idx = level_labels.index(outer_label)

    if inner_level_idx < outer_level_idx:
        logger.warning(f"{inner_label} is outside of {outer_label} in the tree; doing nothing.")
        return tree

    return jt.map(
        lambda x: ldict_level_to_top(inner_label, x, is_leaf=is_leaf),
        tree,
        is_leaf=LDict.is_of(outer_label),
    )


def ldict_level_keys(tree: PyTree, label: str) -> KeysView:
    """Returns the keys of the first `LDict` node with the given label."""
    return first(tree, is_leaf=LDict.is_of(label)).keys()


# def align_levels_and_map(
#     func, tree, *rest, is_leaf=None, 
# ):
#     """
#     Map over the nodes within the lowest `LDict` level of `tree`.
    
#     Before mapping, rearrange the levels of the tree(s) in `*rest` 
#     so that they are a prefix for `tree`, for the purpose of this mapping.
    
#     The argument `is_leaf` can be used to terminate the mapping at a higher 
#     `LDict` level of `tree`.
    
#     NOTE: This assumes there are no non-LDict nodes in `tree` above the level 
#     of the nodes to be mapped; otherwise `tree_level_labels` will return a 
#     truncated list.  
#     """
#     level_labels = tree_level_labels(tree, is_leaf=is_leaf)
#     other_trees = []
#     for tree in rest:
#         reordered = tree 
#         # TODO: Move the levels of `tree` to the front of `reordered` 
#     # TODO: Map over `tree`


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


def check_nan_in_pytree(tree: PyTree) -> tuple[PyTree, PyTree]:
    """
    Checks for NaN values in the array leaves of a PyTree.

    Args:
        tree: A PyTree (e.g., nested dict, list, tuple) potentially
              containing JAX arrays as leaves.

    Returns:
        A tuple containing two PyTrees with the same structure as the input:
        1. has_nans_tree: A PyTree with boolean leaves. Each leaf is True if
           the corresponding leaf array in the input tree contains any NaN
           values, and False otherwise.
        2. where_nans_tree: A PyTree where each leaf contains the output of
           jnp.where(jnp.isnan(input_leaf)). This is typically a tuple of
           index arrays indicating the locations of NaNs in the corresponding
           input leaf array.
    """

    # Define a function that processes a single leaf (assumed to be an array)
    def process_leaf(leaf):
        # Check if the leaf is likely an array type JAX can handle
        if not hasattr(leaf, 'dtype') or not hasattr(leaf, 'shape'):
             # Handle non-array leaves if necessary. 
             pass 

        is_nan_mask = jnp.isnan(leaf)
        has_nans = jnp.any(is_nan_mask)
        # jnp.where called with only the condition returns a tuple of
        # arrays representing the indices where the condition is True.
        nan_indices = _Wrapped(jnp.where(is_nan_mask))
        return has_nans, nan_indices # Return both results as a tuple

    processed_tree = jt.map(process_leaf, tree)

    return jt.map(
        lambda x: x.unwrap() if isinstance(x, _Wrapped) else x, 
        jtree.unzip(processed_tree),
    )


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


def first(tree, is_leaf: Callable = _is_leaf):
    """Return the first leaf of a tree."""
    return jt.leaves(tree, is_leaf=is_leaf)[0]


def first_shape(tree):
    """Return the shape of the first leaf of a tree of arrays."""
    arrays = eqx.filter(tree, eqx.is_array)
    return first(arrays, is_leaf=None).shape


@jtree.filter_wrap(eqx.is_array)
def shapes(tree):
    """Returns a tree of the shapes of the leaves of `tree`."""
    return jax.tree_map(lambda x: x.shape, tree)


def _hash_pytree(tree) -> str:
    """Return a deterministic MD5 digest of a PyTree **content**.

    Strategy
    --------
    • Stream stable byte-representations of every leaf into a single ``hashlib.md5``
      object – no use of Python's salted ``hash``.
    • Arrays → ``arr.tobytes(order='C')``.
    • Primitive JSON scalars → their UTF-8 string.
    • Callables → qualified name ``<module>.<qualname>``.
    • Everything else → ``repr()`` with memory addresses (``0x…``) masked out
      so it stays identical across interpreter sessions.
    """



    md5 = hashlib.md5()

    # Pre-compiled regex to erase memory addresses like 0x7ffde82aaf80
    _ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

    def _bytes_from_leaf(leaf):
        """Convert *leaf* into a stable sequence of bytes."""
        # Arrays (JAX DeviceArray, numpy.ndarray, etc.)
        if eqx.is_array(leaf):
            return leaf.tobytes(order='C')

        # JSON primitives – cheap & deterministic
        if isinstance(leaf, (int, float, bool, str)) or leaf is None:
            return json.dumps(leaf, sort_keys=True).encode()

        # Bytes object – already bytes
        if isinstance(leaf, (bytes, bytearray)):
            return bytes(leaf)

        # Callables / functions
        if isinstance(leaf, types.FunctionType):
            qname = f"{leaf.__module__}.{leaf.__qualname__}"
            return qname.encode()

        # Equinox Module or arbitrary object – fall back to repr w/o memory addr
        rep = _ADDR_RE.sub("0x", repr(leaf))
        return rep.encode()

    try:
        leaves = jt.leaves(tree)
    except Exception as e:
        raise TypeError(f"Failed to flatten PyTree for hashing: {e}") from e

    for leaf in leaves:
        md5.update(_bytes_from_leaf(leaf))

    return md5.hexdigest()
