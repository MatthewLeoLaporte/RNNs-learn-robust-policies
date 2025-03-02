"""What happens if we change nothing but the network's context input, at steady state?
"""


import equinox as eqx

from feedbax.task import TrialSpecDependency

from rnns_learn_robust_motor_policies.analysis.center_out import CenterOutByReplicate
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
from rnns_learn_robust_motor_policies.analysis.state_utils import get_step_task_input
from rnns_learn_robust_motor_policies.types import LDict


COLOR_FUNCS = dict()


def setup_eval_tasks_and_models(task_base, models_base, hps):
    """Modify the task so that context inputs vary over trials.
    
    Note that this is a bit different to how we perturb state variables; normally we'd use an intervenor 
    but since the context input is supplied by the task, we can just change the way that's defined.
    """
    task = eqx.tree_at(
        lambda task: task.input_dependencies,
        task_base,
        # TODO: Use not just a fixed perturbation of the context, but randomly-sampled context endpoints
        dict(context=TrialSpecDependency(get_step_task_input(
            hps.disturbance.context.c_min, 
            hps.disturbance.context.c_max,
            hps.disturbance.context.step,  
            hps.model.n_steps - 1, 
            task_base.n_validation_trials,
        ))),
    )
    
    return task, models_base, hps


eval_func = vmap_eval_ensemble
    

ALL_ANALYSES = [
    CenterOutByReplicate(variant='full'),
]
