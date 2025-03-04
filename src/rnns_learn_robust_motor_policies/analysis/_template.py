"""Skeleton for analysis modules imported in __init__.py.

For example, `analysis.part1.plant_perts` is such a module.
"""

from collections.abc import Callable, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Optional

from equinox import Module

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace
from rnns_learn_robust_motor_policies.types import LDict


"""Specify any additional colorscales needed for this analysis. 
These will be included in the `colors` kwarg passed to `AbstractAnalysis` methods
"""
COLOR_FUNCS: dict[str, Callable[[TreeNamespace], Sequence]] = dict(
    some_variable=lambda hps: hps.some_variable,  #! e.g.
)


def setup_eval_tasks_and_models(task_base: Module, models_base: LDict[float, Module], hps: TreeNamespace):
    """Specify how to set up the PyTrees of evaluation tasks and models, given a base task and 
    a spread of models.
    
    Also, make any necessary modifications to `hps` as they will be available during analysis. 
    """
    # Trivial example
    tasks = task_base
    models = models_base 
    
    # Provides any additional data needed for the analysis
    extras = SimpleNamespace()  
    
    return tasks, models, hps, extras

    
"""Depending on the structure of `setup_eval_tasks_and_models`, e.g. the use of `vmap`, it may be 
necessary to define a more complex function here.

For example, check out `analysis.part2.unit_perts`.
"""
eval_func: Callable = vmap_eval_ensemble


# Define any subclasses of `AbstractAnalysis` that are specific to this task
class SomeAnalysis(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    ...
 
   
"""Determines which analyses are performed by `run_analysis.py`, for this module."""
ALL_ANALYSES = [
    SomeAnalysis(),
]