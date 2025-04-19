"""What happens if we change the network's context input, at steady state or during a reach?
"""

import equinox as eqx

from feedbax.intervene import schedule_intervenor
from feedbax.task import TrialSpecDependency
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.activity import NetworkActivity_SampleUnits
from rnns_learn_robust_motor_policies.analysis.aligned import AlignedEffectorTrajectories
from rnns_learn_robust_motor_policies.analysis.effector import EffectorTrajectories
from rnns_learn_robust_motor_policies.colors import ColorscaleSpec
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_PERT_FUNCS, get_pert_amp_vmap_eval_func
from rnns_learn_robust_motor_policies.analysis.profiles import Profiles
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate_states, get_step_task_input
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


PLANT_PERT_LABELS = {0: "no curl", 1: "curl"}
PLANT_PERT_STYLES = dict(line_dash={0: "dot", 1: "solid"})


ALL_ANALYSES = [
    # -- Steady-state --
    # 0. Show that context perturbation does not cause a significant change in force output at steady-state.
    EffectorTrajectories(
        variant="steady",
        pos_endpoints=False,
        straight_guides=False,
        colorscale_axis=1, 
        colorscale_key="reach_condition",
    )
        .after_transform(get_best_replicate_states)  # By default has `axis=1` for replicates
        .with_fig_params(
            mean_exclude_axes=(-3,),  # Average over all extra batch axes *except* reach direction/condition
            legend_title="Context<br>pert. amp.",
        ),

    #! TODO: Not displaying; debug pytree structure
    #! Also only one of the two legendgroup titles is displayed, even though the respective values/labels appear to be properly passed
    # Profiles(variant="steady")
    #     .after_level_to_top('train__pert__std')
    #     .combine_figs_by_axis(
    #         axis=2,     
    #         fig_params_fn=lambda fig_params, i, item: dict(
    #             scatter_kws=dict(
    #                 line_dash=PLANT_PERT_STYLES['line_dash'][i],
    #                 legendgroup=PLANT_PERT_LABELS[i],
    #                 legendgrouptitle_text=PLANT_PERT_LABELS[i].capitalize(),
    #             ),
    #         ),
    #     ),

    # 1. Activity of sample units, to show they change when context input does
    NetworkActivity_SampleUnits(variant="steady")
        .after_transform(get_best_replicate_states)
        .after_level_to_top('train__pert__std')
        .with_fig_params(
            legend_title="Context pert. amp.",  #! No effect
        ),

    # -- Reaching --
    # 2. Plot aligned vars for reaching +/- plant pert, +/- context pert on same plot
    # (It only makes sense to do this for reaches (not ss), at least for curl fields.)
    # Hide individual trials for this plot, since they make it hard to distinguish the means;
    # the variability should be clear from other plots. 
    AlignedEffectorTrajectories(variant="reach")
        .after_stacking(level="pert__context__amp")
        .combine_figs_by_axis(
            axis=3,  # Not 2, because of the prior stacking
            fig_params_fn=lambda fig_params, i, item: dict(
                mean_scatter_kws=dict(
                    line_dash=PLANT_PERT_STYLES['line_dash'][i],
                    legendgroup=PLANT_PERT_LABELS[i],
                    legendgrouptitle_text=PLANT_PERT_LABELS[i].capitalize(),
                ),
            ),
        )
        .with_fig_params(
            legend_title="Final context<br>input",
            scatter_kws=dict(line_width=0),  # Hide individual trials
            layout_kws=dict(
                legend_title_font_weight="bold",
                #! TODO: Nested dict update so we don't need to pass these redundantly
                width=900, 
                height=300,
                legend_tracegroupgap=1, 
                margin_t=50,
                margin_b=20,
            ),
        ),

    #! Only one of the two legendgroup titles is displayed, even though the respective values/labels appear to be properly passed.
    #! I'm not sure why this is different from `AlignedEffectorTrajectories`, where the legend is displayed correctly
    Profiles(variant="reach")
        .after_level_to_top('train__pert__std')
        .combine_figs_by_axis(
            axis=2,     
            fig_params_fn=lambda fig_params, i, item: dict(
                scatter_kws=dict(
                    line_dash=PLANT_PERT_STYLES['line_dash'][i],
                    legendgroup=PLANT_PERT_LABELS[i],
                    legendgrouptitle_text=PLANT_PERT_LABELS[i].capitalize(),
                ),
            ),
        ),

    # 4. Perform PCA wrt baseline `reach` variant, and project `steady` variant into that space
    # (To show that context input causally varies the network activity in a null direction)
    # NetworkActivity_ProjectPCA(
    #     variant="steady", 
    #     variant_pca="reach_pca",
    # ),
]
