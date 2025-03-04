from collections.abc import Callable
from copy import deepcopy
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

from rnns_learn_robust_motor_policies.constants import LEVEL_LABEL_SEP
from rnns_learn_robust_motor_policies.types import LDict



T = TypeVar("T")
NT = TypeVar("NT", bound=SimpleNamespace)
DT = TypeVar("DT", bound=dict)


logger = logging.getLogger(__name__)


def swap_model_trainables(model: PyTree[..., "T"], trained: PyTree[..., "T"], where_train: Callable):
    return eqx.tree_at(
        where_train,
        model,
        where_train(trained),
    )


def get_dict_constructor(d: dict):
    if isinstance(d, LDict):
        return LDict.of(d.label)   
    else:
        return type(d)


def subdict(d: dict[T, Any], keys: Sequence[T]):
    """Returns the dict containing only the keys `keys`."""
    return get_dict_constructor(d)({k: d[k] for k in keys})


def dictmerge(*dicts: dict) -> dict:
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
            raise NotImplementedError("")
        
        # Get the node at this level
        if isinstance(current_node, dict) or hasattr(current_node, '__getitem__'):
            current_node = current_node[path_element.key if hasattr(path_element, 'key') else path_element]
        
        if is_leaf(current_node):
            break
        
    if sep is not None:
        labels = [label.replace(LEVEL_LABEL_SEP, sep) for label in labels]
        
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


def is_dict_with_int_keys(d: dict) -> bool:
    return isinstance(d, dict) and len(d) > 0 and all(isinstance(k, int) for k in d.keys())


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


def dict_to_namespace(
    d: dict,
    to_type: type[NT] = SimpleNamespace,
    exclude: Callable = lambda x: False,
) -> NT:
    """Convert a nested dictionary to a nested SimpleNamespace.

    This is the inverse operation of namespace_to_dict.
    """
    return convert_kwargy_node_type(d, to_type=to_type, from_type=dict, exclude=exclude)


@jtu.register_pytree_with_keys_class
class TreeNamespace(SimpleNamespace):
    """A simple namespace that's a PyTree.

    This is useful when we want to attribute-like access to the data in
    a nested dict. For example, `hyperparameters['train']['n_batches']` 
    becomes `TreeNamespace(**hyperparameters).train.n_batches`.
    """
    def tree_flatten_with_keys(self):
        children_with_keys = [(jtu.GetAttrKey(k), v) for k, v in self.__dict__.items()]
        aux_data = self.__dict__.keys()
        return children_with_keys, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))
    
    def __repr__(self):
        return self._repr_with_indent(0)
        
    def _repr_with_indent(self, level):
        INDENT = "  "  # or "\t" for tabs
        current_indent = INDENT * level
        attr_strs = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, TreeNamespace):
                attr_repr = attr._repr_with_indent(level + 1)
            else:
                attr_repr = repr(attr)
            attr_strs.append(f"{name}={attr_repr},")
            
        return (f"{self.__class__.__name__}(\n" + 
                '\n'.join(current_indent + INDENT + s for s in attr_strs) +
                f"\n{current_indent})")

    def update_none_leaves(self, other):
        # I would just use `jt.map` or `eqx.combine` to do this, however I don't want to assume
        # that `other` will have identical PyTree structure to `self` -- only that it contains at 
        # least the keys whose values are `None` in `self`.
        #? Could work on flattened trees.
        def _update_none_leaves(target: TreeNamespace, source: TreeNamespace) -> TreeNamespace:
            result = deepcopy(target)
            source = deepcopy(source)

            for attr_name in vars(result):
                if attr_name == 'load': 
                    continue
                
                result_value = getattr(result, attr_name)
                source_value = getattr(source, attr_name, None)

                if result_value is None:
                    if source_value is None:
                        raise ValueError(f"Cannot replace `None` value of key {attr_name}; no matching key available in source")
                    setattr(result, attr_name, source_value)

                elif isinstance(result_value, TreeNamespace):
                    if source_value is None:
                        continue
                    if not isinstance(source_value, TreeNamespace):
                        raise ValueError(f"Source must contain all the parent keys (but not necessarily all the leaves) of the target")
                    setattr(result, attr_name, _update_none_leaves(result_value, source_value))

            return result
        return _update_none_leaves(self, other)

    def __or__(self, other: 'TreeNamespace | dict') -> 'TreeNamespace':
        """Merge two TreeNamespaces, with values from other taking precedence.

        Handles nested TreeNamespaces recursively.
        """
        result = deepcopy(self)

        if isinstance(other, dict):
            other = dict_to_namespace(other, to_type=TreeNamespace, exclude=is_dict_with_int_keys)

        for attr_name, other_value in vars(other).items():
            self_value = getattr(result, attr_name, None)

            if isinstance(other_value, TreeNamespace) and isinstance(self_value, TreeNamespace):
                # Recursively merge nested TreeNamespaces
                setattr(result, attr_name, self_value | other_value)
            else:
                # Simply update, when at least one side isn't a TreeNamespace
                setattr(result, attr_name, other_value)

        return result


def _convert_value(value: Any, to_type: type, from_type: type, exclude: Callable) -> Any:
    recurse_func = lambda x: _convert_value(x, to_type, from_type, exclude)
    map_recurse_func = lambda tree: jt.map(recurse_func, tree, is_leaf=is_type(from_type))

    if exclude(value):
        subtrees, treedef = eqx.tree_flatten_one_level(value)
        subtrees = [map_recurse_func(subtree) for subtree in subtrees]
        return jt.unflatten(treedef, subtrees)

    elif isinstance(value, from_type):
        if isinstance(value, SimpleNamespace):
            value = vars(value)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a dict or namespace, got {type(value)}")

        return to_type(**{
            str(k): recurse_func(v)
            for k, v in value.items()
        })

    elif isinstance(value, (str, ArrayLike, type(None))):
        return value

    # Map over any remaining PyTrees, except 
    elif isinstance(value, PyTree):
        # `object` is an atomic PyTree, so without this check we'll get infinite recursion
        if value is not object:
            return map_recurse_func(value)

    return value


def convert_kwargy_node_type(x, to_type: type, from_type: type, exclude: Callable = lambda x: False):
    """Convert a nested dictionary to a nested SimpleNamespace.

    !!! dev 
        This should convert all the dicts to namespaces, even if the dicts are not contiguous all 
        the way down (e.g. a dict in a list in a list in a dict)
    """
    return _convert_value(x, to_type, from_type, exclude)


def namespace_to_dict(
    ns: SimpleNamespace,
    to_type: type[DT] = dict,
    exclude: Callable = lambda x: False,
) -> DT:
    """Convert a nested SimpleNamespace to a nested dictionary.

    This is the inverse operation of dict_to_namespace.
    """
    return convert_kwargy_node_type(ns, to_type=to_type, from_type=SimpleNamespace, exclude=exclude)