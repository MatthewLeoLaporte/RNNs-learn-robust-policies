from collections import namedtuple

from feedbax import (
    make_named_dict_subclass,
    make_named_tuple_subclass,
)



TaskModelPair = namedtuple("TaskModelPair", ["task", "model"])

"""
Our PyTrees will contain levels corresponding to training conditions (standard deviation
of disturbance amplitude during training), evaluation conditions (disturbance
amplitudes during analysis), and so one.

Here, define some trivial subclasses of `dict` and `tuple` that can be identified by
name, when manipulating such levels in the trees.

For example, `TrainStdDict` behaves like `dict` in almost every way, except it is technically 
a different type. This means in particular that if `a = dict()` and `b = TrainStdDict()`, 
then `isinstance(b, dict) == isinstance(b, TrainStdDict) == True` but 
`isinstance(a, TrainStdDict) == False`. Also, these dict subclasses maintain the order of their
entries through `jax.tree.map`, which is not the case for builtin `dict`.
"""
TrainStdDict = make_named_dict_subclass('TrainStdDict')
PertAmpDict = make_named_dict_subclass('PertAmpDict')
PertVarDict = make_named_dict_subclass('PertVarDict')
ImpulseAmpTuple = make_named_tuple_subclass('ImpulseAmpTuple')
ContextInputDict = make_named_dict_subclass('ContextInputDict')