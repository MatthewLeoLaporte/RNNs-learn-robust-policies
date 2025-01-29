from collections import namedtuple
from functools import partial
from typing import Dict, Generic, TypeVar

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
K = TypeVar('K')
V = TypeVar('V')


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

    
_custom_dict_classes = (
    TrainStdDict, 
    PertAmpDict, 
    PertVarDict, 
    ContextInputDict, 
    TrainingMethodDict,
    FPDict,
    TrainWhereDict,
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
  




            
  