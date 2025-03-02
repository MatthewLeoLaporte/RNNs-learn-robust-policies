from collections import namedtuple
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Any, ClassVar, Dict, Generic, Literal, NamedTuple, Type, TypeAlias, TypeVar, TypedDict, cast, Mapping, overload, Union, Protocol, runtime_checkable

import jax
import jax.tree_util as jtu
import yaml


TaskModelPair = namedtuple("TaskModelPair", ["task", "model"])


"""
Our PyTrees will contain levels corresponding to training conditions (standard deviation
of disturbance amplitude during training), evaluation conditions (disturbance
amplitudes during analysis), and so on.

Here, we define some trivial subclasses of `dict` and `tuple` that can be identified by
name, when manipulating such levels in the trees.

For example, `TrainStdDict` behaves like `dict` in almost every way, except it is technically 
a different type. This means in particular that if `a = dict()` and `b = TrainStdDict()`, 
then `isinstance(b, dict) == isinstance(b, TrainStdDict) == True` but 
`isinstance(a, TrainStdDict) == False`. Also, these dict subclasses maintain the order of their
entries through `jax.tree.map`, which is not the case for builtin `dict`.
"""
K = TypeVar('K', bound=str)
V = TypeVar('V')

LT = TypeVar('LT', bound=str)  # For label


@jax.tree_util.register_pytree_node_class
class LDictBase(Mapping[K, V], Generic[K, V, LT]):
    """Immutable dictionary with a string label for distinguishing dictionary types."""
    
    def __init__(self, label: LT, data: Mapping[K, V]):
        self._label = label
        self._data = dict(data)  # Make a copy for immutability
    
    @property
    def label(self) -> LT:
        return self._label
    
    def __getitem__(self, key: K) -> V:
        return self._data[key]
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self) -> str:
        return f"LabeledDict({repr(self._label)}, {self._data})"
    
    # PyTree implementation
    def tree_flatten(self) -> tuple[tuple[Dict[K, V]], LT]:
        """Flatten this LabeledDict for JAX PyTree traversal."""
        # Return a tuple of (children, auxiliary_data)
        return (self._data,), self._label
    
    @classmethod
    def tree_unflatten(cls, label: LT, children: tuple[Dict[K, V]]):
        """Recreate a LabeledDict from flattened data."""
        return cls(label, children[0])
    
    # Common dict methods
    def items(self):
        return self._data.items()
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    

LDictType: TypeAlias = LDictBase[K, Any, LT]
    

class LDictConstructor(Generic[LT]):
    """Constructor for a specific labeled dictionary type."""
    
    def __init__(self, label: LT):
        self.label = label
    
    def __call__(self, data: Mapping[K, V]) -> LDictBase[K, V, LT]:
        return LDictBase(self.label, data)
    
    @property
    def is_leaf(self) -> Callable[[Any], bool]:
        """Return a predicate for JAX pytree traversal."""
        label = self.label
        return lambda node: isinstance(node, LDictBase) and node.label == label


class LDictFactory:
    """Factory for creating labeled dictionary constructors."""
    
    def __call__(self, label: LT) -> LDictConstructor[LT]:
        """Returns a constructor for dictionaries with the specified label."""
        return LDictConstructor(label)
    
    @staticmethod
    def is_any_ldict(node: Any) -> bool:
        """Check if a node is any type of LabeledDict."""
        return isinstance(node, LDictBase)


# The factory instance
LDict = LDictFactory()


class CustomDict(Dict[K, V], Generic[K, V]):
    def __repr__(self):
        return f"{self.__class__.__name__}({dict.__repr__(self)})"

    # In principle we could check if self and other are instances of two 
    # different custom dict classes, and return a plain dict in that case.
    # However I foresee no reason at this point we should want to mix
    # different custom types, so I'll keep this simple.
    def __or__(self, other):
        return type(self)({**self, **other})
    
    def __ror__(self, other):
        return type(self)({**other, **self})
    

class TrainStdDict(CustomDict[K, V], Generic[K, V]):
    ...


class PertAmpDict(CustomDict[K, V], Generic[K, V]):
    ...


class PertVarDict(CustomDict[K, V], Generic[K, V]):
    ...


class ContextInputDict(CustomDict[K, V], Generic[K, V]):
    ...


class TrainingMethodDict(CustomDict[K, V], Generic[K, V]):
    ...


class FPDict(CustomDict[K, V], Generic[K, V]):
    ...


class TrainWhereDict(CustomDict[K, V], Generic[K, V]):
    ...


class LabelDict(CustomDict[str, V], Generic[V]):
    ...
    
    
class MeasureDict(CustomDict[str, V], Generic[V]):
    ...
    

class CoordDict(CustomDict[str, V], Generic[V]):
    ...


class ColorDict(CustomDict[str, V], Generic[V]):
    ...
 
    
_custom_dict_classes = (
    TrainStdDict, 
    PertAmpDict, 
    PertVarDict, 
    ContextInputDict, 
    TrainingMethodDict,
    CoordDict,
    LabelDict,
    FPDict,
    TrainWhereDict,
    MeasureDict,
    ColorDict,
)


def _dict_flatten_with_keys(obj):
    children = [(jtu.DictKey(k), v) for k, v in obj.items()]
    return (children, obj.keys())


def _get_dict_unflatten(cls):
    def dict_unflatten(keys, children):
        return cls(zip(keys, children))
    
    return dict_unflatten


def _yaml_dicttype_representer(cls: type[dict], dumper, data):
    return dumper.represent_mapping(f"!{cls.__name__}", data)


def _yaml_dicttype_constructor(cls: type[dict], loader, node):
    return cls(loader.construct_mapping(node))


for cls in _custom_dict_classes:
    jtu.register_pytree_with_keys(
        cls, 
        _dict_flatten_with_keys, 
        _get_dict_unflatten(cls),
    )

    # Add YAML representers and constructors to enable writing/reading
    # of special dict types to/from YAML.
    yaml.add_representer(cls, partial(_yaml_dicttype_representer, cls))    
    yaml.SafeLoader.add_constructor(
        f"!{cls.__name__}",
        partial(_yaml_dicttype_constructor, cls), 
    )
    

class ImpulseAmpTuple(tuple, Generic[K, V]):
    def __repr__(self):
        return f"ImpulseAmpTuple({tuple.__repr__(self)})"
    
    
for cls in (ImpulseAmpTuple,): 
    jtu.register_pytree_node(
        cls, 
        lambda x: (x, None), 
        lambda _, children: cls(children)  # type: ignore
    ) 


#! TODO: Compute this algorithmically using `camel_to_snake`
#! (Will also need to rename `TrainStdDict` to `PertStdDict` or something)
TYPE_LABELS = {
    TrainStdDict: 'disturbance_std',
    PertAmpDict: 'disturbance_amplitude',  
    PertVarDict: 'pert_var', 
    ContextInputDict: 'context_input', 
    TrainingMethodDict: 'training_method',
    LabelDict: 'label',
    MeasureDict: 'measure',
    CoordDict: 'coord',
    ColorDict: 'color',
    # FPDict: 'fp',
    # TrainWhereDict: 'train_where',    
}


# TODO: Rename to Effector, or something
class ResponseVar(str, Enum):
    """Variables available in response state."""
    POSITION = 'pos'
    VELOCITY = 'vel'
    FORCE = 'force'


class Direction(str, Enum):
    """Available directions for vector components."""
    PARALLEL = 'parallel'
    ORTHOGONAL = 'orthogonal'


DIRECTION_IDXS = {
    Direction.PARALLEL: 0,
    Direction.ORTHOGONAL: 1,
}


class Responses(NamedTuple):
    pos: Any
    vel: Any
    force: Any


RESPONSE_VAR_LABELS = Responses('Position', 'Velocity', 'Control force')
            
  