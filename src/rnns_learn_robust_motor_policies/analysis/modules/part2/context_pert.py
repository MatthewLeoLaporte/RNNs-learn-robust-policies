"""What happens if we change nothing but the network's context input, at steady state?
"""

from types import MappingProxyType
from typing import ClassVar, Optional
import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

from feedbax.intervene import schedule_intervenor
from feedbax.task import TrialSpecDependency
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.activity import NetworkActivity_ProjectPCA, NetworkActivity_SampleUnits
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_PERT_FUNCS, get_pert_amp_vmap_eval_func
from rnns_learn_robust_motor_policies.analysis.disturbance import task_with_pert_amp
from rnns_learn_robust_motor_policies.analysis.effector import Effector_ByEval, Effector_ByReplicate
from rnns_learn_robust_motor_policies.analysis.state_utils import BestReplicateStates, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.analysis.state_utils import get_step_task_input
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.types import LDict


ID = "2-7"


COLOR_FUNCS = dict(
    pert__amp=lambda hps: [final - hps.pert.context.init for final in hps.pert.context.final],
)


def setup_eval_tasks_and_models(task_base, models_base, hps):
    """Modify the task so that context inputs vary over trials.
    
    Note that this is a bit different to how we perturb state variables; normally we'd use an intervenor 
    but since the context input is supplied by the task, we can just change the way that's defined.
    """
    plant_disturbance = PLANT_PERT_FUNCS[hps.pert.plant.type]
    
    # Add placeholder for plant perturbations
    task_base, models_base = schedule_intervenor(
        task_base, models_base, 
        lambda model: model.step.mechanics,
        plant_disturbance(0),
        default_active=False,
        label=PLANT_INTERVENOR_LABEL,
    )
    
    # 
    tasks, models, hps = jtree.unzip(LDict.of("pert__amp")({
        context_final: (
            eqx.tree_at(
                lambda task: task.input_dependencies,
                task_base,
                # TODO: Use not just a fixed perturbation of the context, but randomly-sampled context endpoints
                dict(context=TrialSpecDependency(get_step_task_input(
                    hps.pert.context.init, 
                    context_final,
                    hps.pert.context.step,  
                    hps.model.n_steps - 1, 
                    task_base.n_validation_trials,
                ))),
            ),
            models_base, 
            hps | dict(pert=dict(amp=context_final - hps.pert.context.init)),
        )
        for context_final in hps.pert.context.final
    }))
    
    return tasks, models, hps, None


eval_func = get_pert_amp_vmap_eval_func(lambda hps: hps.pert.plant.amp, PLANT_INTERVENOR_LABEL)


VARIANT = "steady"
    

ALL_ANALYSES = [
    # 0. Show that context perturbation does not cause a significant change in force output at steady-state.
    #! (might want to go to zero noise, to show how true this actually is)
    Effector_ByEval(
        variant=VARIANT, 
        legend_title="Pert. field amp.",
        mean_exclude_axes=(-3,),  # Average over all extra batch axes *except* reach direction/condition
    ),
    
    # 1. Sample center-out plots for perturbation during reaches; 
    # these aren't very useful once we have the aligned plots.
    Effector_ByEval(
        variant="reach", 
        legend_title="Pert. field amp.",
        mean_exclude_axes=(-3,),
        legend_labels=lambda hps, hps_common: hps_common.pert.plant.amp,   
    ),
    
    # 2. Activity of sample units, to show they change when context input does
    NetworkActivity_SampleUnits(variant=VARIANT),
    
    # 3. Plot aligned vars for +/- plant pert, +/- context pert on same plot
    
    
    # 4. Perform PCA wrt baseline `reach` variant, and project `steady` variant into that space
    # (To show that context input causally varies the network activity in a null direction)
    # NetworkActivity_ProjectPCA(
    #     variant=VARIANT, 
    #     variant_pca="reach_pca",
    # ),
]
