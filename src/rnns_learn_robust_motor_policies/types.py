from collections import namedtuple
from typing import Dict, Generic, TypeVar

import jax.tree_util as jtu


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


class TrainStdDict(Dict[K, V], Generic[K, V]):
    def __repr__(self):
        return f"TrainStdDict({dict.__repr__(self)})"
    
    
class PertAmpDict(Dict[K, V], Generic[K, V]):
    def __repr__(self):
        return f"PertAmpDict({dict.__repr__(self)})"


class PertVarDict(Dict[K, V], Generic[K, V]):
    def __repr__(self):
        return f"PertVarDict({dict.__repr__(self)})"
    
    
class ContextInputDict(Dict[K, V], Generic[K, V]):
    def __repr__(self):
        return f"ContextInputDict({dict.__repr__(self)})"
    

class TrainingMethodDict(Dict[K, V], Generic[K, V]):
    def __repr__(self):
        return f"TrainingMethodDict({dict.__repr__(self)})"    


def _dict_flatten_with_keys(obj):
    children = [(jtu.DictKey(k), v) for k, v in obj.items()]
    return (children, obj.keys())


def _get_dict_unflatten(cls):
    def dict_unflatten(keys, children):
        return cls(zip(keys, children))
    
    return dict_unflatten


for cls in (TrainStdDict, PertAmpDict, PertVarDict, ContextInputDict, TrainingMethodDict):
    jtu.register_pytree_with_keys(
        cls, 
        _dict_flatten_with_keys, 
        _get_dict_unflatten(cls),
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