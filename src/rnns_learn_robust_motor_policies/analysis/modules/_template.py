"""Skeleton for analysis modules imported in __init__.py.

For example, `analysis.part1.plant_perts` is such a module.
"""

from collections.abc import Callable, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Optional

from equinox import Module

from rnns_learn_robust_motor_policies.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.analysis import FigParams
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.types import LDict


"""Should match the filename of the respective YAML config file in the `config` subpackage."""
#! TODO: Could have the user pass `"part1.plant_perts"`, for example, instead of `1-1`.
#! Then the subpackage structure of `config` could match that of `analysis`.
#! However, then how could the user pass a custom YAML file instead of the ones in `config`?
#! Perhaps it is best to keep the IDs in the YAML/modules, but rearrange `config` as described above,
#! and allow the user to pass either an ID or a path to a custom YAML file. And the IDs could be 
#! "part1.plant_perts" etc. instead of "1-1".
ID: str = ""


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
    _pre_ops: tuple[tuple[str, Callable]] = ()
    fig_params: FigParams = FigParams()
    
    ...
 
   
"""Determines which analyses are performed by `run_analysis.py`, for this module."""
ALL_ANALYSES = [
    SomeAnalysis(),
]