from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, Generic, NamedTuple, Optional, TypeVar, overload
import equinox as eqx
import jax
import jax.tree as jt
import jax.tree_util as jtu
from jax_cookbook import is_type
from jaxtyping import ArrayLike, PyTree
import yaml


TaskModelPair = namedtuple("TaskModelPair", ["task", "model"])


# This can't be placed in `config.STRINGS` since that would cause a circular import
TNS_REPR_INDENT_STR = "  "


K = TypeVar('K')
V = TypeVar('V')
NT = TypeVar("NT", bound=SimpleNamespace)
DT = TypeVar("DT", bound=dict)


def convert_kwargy_node_type(x, to_type: type, from_type: type, exclude: Callable = lambda x: False):
    """Convert a nested dictionary to a nested SimpleNamespace.

    !!! dev 
        This should convert all the dicts to namespaces, even if the dicts are not contiguous all 
        the way down (e.g. a dict in a list in a list in a dict)
    """
    return _convert_value(x, to_type, from_type, exclude)


def dict_to_namespace(
    d: dict,
    to_type: type[NT] = SimpleNamespace,
    exclude: Callable = lambda x: False,
) -> NT:
    """Convert a nested dictionary to a nested SimpleNamespace.

    This is the inverse operation of namespace_to_dict.
    """
    return convert_kwargy_node_type(d, to_type=to_type, from_type=dict, exclude=exclude)


def is_dict_with_int_keys(d: dict) -> bool:
    return isinstance(d, dict) and len(d) > 0 and all(isinstance(k, int) for k in d.keys())


@jtu.register_pytree_with_keys_class
class TreeNamespace(SimpleNamespace):
    """A simple namespace that's a PyTree.

    This is useful when we want to attribute-like access to the data in
    a nested dict. For example, `hyperparameters['train']['n_batches']` 
    becomes `TreeNamespace(**hyperparameters).train.n_batches`.
    
    NOTE:
        If it weren't for `update_none_leaves`, `__or__`, and perhaps `__repr__`, 
        we could simply register `SimpleNamespace` as a PyTree. Consider whether 
        these methods can be replaced by e.g. functions.
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
        cls_name = self.__class__.__name__
        if not any(self.__dict__):
            return f"{cls_name}()"
        
        attr_strs = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, TreeNamespace):
                attr_repr = attr._repr_with_indent(level + 1)
            else:
                attr_repr = repr(attr)
            attr_strs.append(f"{name}={attr_repr},")

        current_indent = TNS_REPR_INDENT_STR * level
        inner_str = '\n'.join(current_indent + TNS_REPR_INDENT_STR + s for s in attr_strs)
        
        return f"{cls_name}(\n" + inner_str + f"\n{current_indent})"

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
                setattr(result, attr_name, other_value)

        return result
    

def unflatten_dict_keys(flat_dict: dict, sep: str = '__') -> dict:
    """Unflatten a dictionary by splitting keys on the separator.
    
    Supports multiple levels of nesting.
    """
    result = {}
    
    for key, value in flat_dict.items():
        current = result
        
        if sep in key:
            parts = key.split(sep)
            
            for part in parts[:-1]:
                current = current.setdefault(part, {})
                
            current[parts[-1]] = value
        else:
            result[key] = value
            
    return result


@jax.tree_util.register_pytree_with_keys_class
class LDict(Mapping[K, V], Generic[K, V]):
    """Immutable dictionary with a string label for distinguishing dictionary types.
    
    Our PyTrees will contain levels corresponding to training conditions (standard deviation
    of disturbance amplitude during training), evaluation conditions (disturbance
    amplitudes during analysis), and so on. Associating a label with a mapping will allow us 
    to identify and map over specific levels of these PyTrees, as well as to keep track of the 
    names of hyperparameters stored in the PyTree, e.g. so we can automatically determine 
    which columns to store those hyperparameters in, in the DB.
    """
    
    def __init__(self, label: str, data: Mapping[K, V]):
        self._label = label
        self._data = dict(data)  
    
    @property
    def label(self) -> str:
        return self._label
    
    def __getitem__(self, key: K) -> V:
        return self._data[key]
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self) -> str:
        #! TODO: Proper line breaks when nested
        return f"LDict({repr(self._label)}, {self._data})"
    
    def tree_flatten_with_keys(self):
        children_with_keys = [(jtu.DictKey(k), v) for k, v in self.items()]
        return children_with_keys, (self._label, self.keys())
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        label, keys = aux_data
        return cls(label, dict(zip(keys, children)))
    
    def items(self):
        return self._data.items()
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def get(self, key, default=None):
        return self._data.get(key, default)

    @staticmethod
    def of(label: str):
        """Returns a constructor function for the given label."""
        return _LDictConstructor(label)
    
    @staticmethod
    def is_of(label: str) -> Callable[[Any], bool]:
        """Return a predicate checking if a node is a LDict with a specific label."""
        return lambda node: isinstance(node, LDict) and node.label == label
    
    @staticmethod
    @overload
    def fromkeys(label: str, keys: Iterable[K]) -> 'LDict[K, None]': ...
    
    @staticmethod
    @overload
    def fromkeys(label: str, keys: Iterable[K], value: V) -> 'LDict[K, V]': ...
    
    @staticmethod
    def fromkeys(label: str, keys: Iterable[Any], value: Any = None) -> 'LDict[Any, Any]':
        """Create a new LDict with the given label and keys, each with value set to value."""
        return LDict(label, dict.fromkeys(keys, value))


class _LDictConstructor(Generic[K, V]):
    """Constructor for an `LDict` with a particular label."""
    def __init__(self, label: str):
        self.label = label
    
    def __call__(self, data: Mapping[K, V]):
        return LDict(self.label, data)
        
    def fromkeys(self, keys: Iterable[K], value: Optional[V] = None):
        return LDict.fromkeys(self.label, keys, value)
    
    def from_ns(self, namespace: SimpleNamespace):
        """Convert the top level of `namespace` to an `LDict`."""
        return LDict(self.label, namespace.__dict__)


# YAML serialisation/deserialisation for LDict objects
def _ldict_representer(dumper, data):
    # Store both the label and the dictionary data
    # Format: !LDict:label {key1: value1, key2: value2, ...}
    return dumper.represent_mapping(f"!LDict:{data.label}", data._data)

yaml.add_representer(LDict, _ldict_representer)

def _ldict_multi_constructor(loader, tag_suffix, node):
    # Extract the label from the tag suffix (after the colon)
    label = tag_suffix
    mapping = loader.construct_mapping(node)
    return LDict(label, mapping)

yaml.SafeLoader.add_multi_constructor('!LDict:', _ldict_multi_constructor)
    

class ImpulseAmpTuple(tuple, Generic[K, V]):
    def __repr__(self):
        return f"ImpulseAmpTuple({tuple.__repr__(self)})"
    
    
for cls in (ImpulseAmpTuple,): 
    jtu.register_pytree_node(
        cls, 
        lambda x: (x, None), 
        lambda _, children: cls(children)  # type: ignore
    ) 


def pprint_ldict_structure(
        tree: LDict, 
        indent: int = 0, 
        indent_str: str = "  ", 
        homogeneous: bool = True,
):
    """Pretty print the structure of a nested LDict PyTree.
    
    Args:
        tree: An LDict or nested structure of LDicts
        indent: Current indentation level (used recursively)
        indent_str: String used for each level of indentation
        homogeneous: If True, assumes all nodes at each level have the same label and keys,
                    so only prints the first occurrence at each level
    """
    if not isinstance(tree, LDict):
        return
    
    # Print current level's label and keys
    current_indent = indent_str * indent
    print(f"{current_indent}LDict('{tree.label}') with keys: {list(tree.keys())}")
    
    # Process LDict values, breaking after first one if homogeneous
    for value in tree.values():
        if isinstance(value, LDict):
            pprint_ldict_structure(value, indent + 2, indent_str, homogeneous)
            if homogeneous:
                break


# TODO: Rename to Effector, or something; also this probably shouldn't be in this module.
class ResponseVar(str, Enum):
    """Variables available in response state."""
    POSITION = 'pos'
    VELOCITY = 'vel'
    FORCE = 'force'


class Responses(NamedTuple):
    pos: Any
    vel: Any
    force: Any


RESPONSE_VAR_LABELS = Responses('Position', 'Velocity', 'Control force')
            

RESPONSE_VAR_LABELS_SHORT = Responses(
    pos='p',
    vel='v',
    force='F',
)


class Direction(str, Enum):
    """Available directions for vector components."""
    PARALLEL = 'parallel'
    ORTHOGONAL = 'orthogonal'


DIRECTION_IDXS = {
    Direction.PARALLEL: 0,
    Direction.ORTHOGONAL: 1,
}


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

    elif isinstance(value, (str, type(None))) or isinstance(value, ArrayLike):
        return value

    # Map over any remaining PyTrees, except 
    elif isinstance(value, PyTree):
        # `object` is an atomic PyTree, so without this check we'll get infinite recursion
        if value is not object:
            return map_recurse_func(value)

    return value


def namespace_to_dict(
    ns: SimpleNamespace,
    to_type: type[DT] = dict,
    exclude: Callable = lambda x: False,
) -> DT:
    """Convert a nested SimpleNamespace to a nested dictionary.

    This is the inverse operation of dict_to_namespace.
    """
    return convert_kwargy_node_type(ns, to_type=to_type, from_type=SimpleNamespace, exclude=exclude)