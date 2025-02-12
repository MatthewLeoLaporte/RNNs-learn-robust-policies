
from types import MappingProxyType
from typing import ClassVar, Optional
import equinox as eqx

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble


def setup_eval_tasks_and_models(task_base, models_base, hps):
    # 1. Task is steady-state 
    # 2. Disturbance type is always the same (e.g. regardless of training on curl or constant)
    # 3. `models_base` is a `TrainStdDict`
    # 4. 
    return # all_tasks, all_models, hps


# def eval_func(models, task, hps, key_eval):
#     """Vmap over impulse amplitude."""
#     return eqx.filter_vmap(
#         lambda amplitude: vmap_eval_ensemble(
#             models, 
#             task_with_imp_amplitude(task, amplitude), 
#             hps,
#             key_eval,
#         )
#     )(hps.disturbance.amplitudes)
    

eval_func = vmap_eval_ensemble


class Foo(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType({})
    variant: ClassVar[Optional[str]] = "full"
    conditions: tuple[str, ...] = ()


ALL_ANALYSES = []