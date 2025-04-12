"""What happens if we change the network's context input, at steady state or during a reach?
"""

import equinox as eqx

from feedbax.intervene import schedule_intervenor
from feedbax.task import TrialSpecDependency
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.activity import NetworkActivity_SampleUnits
from rnns_learn_robust_motor_policies.analysis.aligned import AlignedEffectorTrajectories
from rnns_learn_robust_motor_policies.colors import ColorscaleSpec
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_PERT_FUNCS, get_pert_amp_vmap_eval_func
from rnns_learn_robust_motor_policies.analysis.profiles import VelocityProfiles
from rnns_learn_robust_motor_policies.analysis.state_utils import get_step_task_input
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.types import LDict


ID = "2-7"


COLOR_FUNCS = dict(
    # pert__amp=lambda hps: [final - hps.pert.context.init for final in hps.pert.context.final],
    # context_input=lambda hps: [final - hps.pert.context.init for final in hps.pert.context.final],
    pert__context__amp=ColorscaleSpec(
        sequence_func=lambda hps: [final - hps.pert.context.init for final in hps.pert.context.final],
        colorscale="thermal",
    ),
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
    
    #! Neither `pert__amp` nor `context__input` are entirely valid as labels here, I think
    tasks, models, hps = jtree.unzip(LDict.of("pert__context__amp")({
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


def fig_params_fn_context_pert(fig_params, i, item):
    PLANT_PERT_LABELS = {0: "no curl", 1: "curl"}
    PLANT_PERT_STYLES = dict(line_dash={0: "dot", 1: "solid"})

    return dict(
        # legend_labels=[
        #     f"{label} ({PLANT_PERT_LABELS[i]})"
        #     for label in fig_params.legend_labels
        # ],
        scatter_kws=dict(
            line_dash=PLANT_PERT_STYLES['line_dash'][i],
        ),
    )


ALL_ANALYSES = [
    # # 0. Show that context perturbation does not cause a significant change in force output at steady-state.
    # #! (might want to go to zero noise, to show how true this actually is)
    # Effector_ByEval(
    #     variant="steady", 
    #     mean_exclude_axes=(-3,),  # Average over all extra batch axes *except* reach direction/condition
    # ).with_fig_params(legend_title="Pert. field amp."),
    
    # # 1. Sample center-out plots for perturbation during reaches; 
    # # these aren't very useful once we have the aligned plots.
    # Effector_ByEval(
    #     variant="reach", 
    #     mean_exclude_axes=(-3,),
    #     legend_labels=lambda hps, hps_common: hps_common.pert.plant.amp,   
    # ).with_fig_params(legend_title="Pert. field amp."),
    
    # # 2. Activity of sample units, to show they change when context input does
    NetworkActivity_SampleUnits(variant="steady"),
    
    # 3. Plot aligned vars for +/- plant pert, +/- context pert on same plot
    # (It only makes sense to do this for reaches (not ss), at least for curl fields.)
    AlignedEffectorTrajectories(variant="reach")
        .after_stacking(level="pert__context__amp")
        # Axis 3 and not 2, because of the prior stacking
        .combine_figs_by_axis(axis=3, fig_params_fn=fig_params_fn_context_pert)
        .with_fig_params(legend_title="Final context<br>input"),

    VelocityProfiles(variant="reach")
        .after_level_to_top('train__pert__std')
        .combine_figs_by_axis(axis=2, fig_params_fn=fig_params_fn_context_pert),

    # 4. Perform PCA wrt baseline `reach` variant, and project `steady` variant into that space
    # (To show that context input causally varies the network activity in a null direction)
    # NetworkActivity_ProjectPCA(
    #     variant="steady", 
    #     variant_pca="reach_pca",
    # ),
]
