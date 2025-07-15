from functools import partial 

import jax.tree as jt

from feedbax.intervene import add_intervenors, schedule_intervenor
from jax_cookbook import is_module
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import AlignedEffectorTrajectories
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from rnns_learn_robust_motor_policies.analysis.measures import Measures
from rnns_learn_robust_motor_policies.analysis.profiles import Profiles
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate, get_constant_task_input_fn, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.colors import ColorscaleSpec
from rnns_learn_robust_motor_policies.plot import set_axes_bounds_equal
from rnns_learn_robust_motor_policies.tree_utils import move_ldict_level_above
from rnns_learn_robust_motor_policies.types import (
    LDict,
)


COLOR_FUNCS = dict(
    sisu=ColorscaleSpec(
        sequence_func=lambda hps: hps.sisu,
        colorscale="thermal",
    ),
)


eval_func = vmap_eval_ensemble


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.type}")
    
    pert_amps = hps.pert.amp
    
    # Tasks with varying plant perturbation amplitude 
    tasks_by_amp, _ = jtree.unzip(jt.map( # over disturbance amplitudes
        lambda pert_amp: schedule_intervenor(  # (implicitly) over train stds
            task_base, jt.leaves(models_base, is_leaf=is_module)[0],
            lambda model: model.step.mechanics,
            disturbance(pert_amp),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        ),
        LDict.of("pert__amp")(
            dict(zip(pert_amps, pert_amps)),
        )
    ))
    
    # Add plant perturbation module (placeholder with amp 0.0) to all loaded models
    models_by_std = jt.map(
        lambda models: add_intervenors(
            models,
            lambda model: model.step.mechanics,
            # The first key is the model stage where to insert the disturbance field;
            # `None` means prior to the first stage.
            # The field parameters will come from the task, so use an amplitude 0.0 placeholder.
            {None: {PLANT_INTERVENOR_LABEL: disturbance(0.0)}},
        ),
        models_base,
        is_leaf=is_module,
    )
    
    # Also vary tasks by SISU
    tasks = LDict.of('sisu')({
        sisu: jt.map(
            lambda task: task.add_input(
                name="sisu",
                input_fn=get_constant_task_input_fn(
                    sisu, 
                    hps.model.n_steps - 1, 
                    task.n_validation_trials,
                ),
            ),
            tasks_by_amp,
            is_leaf=is_module,
        )
        for sisu in hps.sisu
    })
    
    # The outer levels of `models` have to match those of `tasks`
    models, hps = jtree.unzip(jt.map(
        lambda _: (models_by_std, hps), tasks, is_leaf=is_module
    ))
    
    return tasks, models, hps, None


MEASURE_KEYS = (
    "max_parallel_vel_forward",
    # "max_orthogonal_vel_signed",
    # "max_orthogonal_vel_left",
    # "max_orthogonal_vel_right",  # -2
    "largest_orthogonal_distance",
    # "max_orthogonal_distance_left",
    # "sum_orthogonal_distance",
    "sum_orthogonal_distance_abs",
    "end_position_error",
    # "end_velocity_error",  # -1
    "max_parallel_force_forward",
    # "sum_parallel_force",  # -2
    # "max_orthogonal_force_right",  # -1
    "sum_orthogonal_force_abs",
    "max_net_force",
    "sum_net_force",
)

        
ANALYSES = {
    # "effector_trajectories_by_condition": (
    #     # By condition, all evals for the best replicate only
    #     EffectorTrajectories(
    #         colorscale_axis=1, 
    #         colorscale_key="reach_condition",
    #     )
    #     .after_transform(get_best_replicate)  # By default has `axis=1` for replicates
    # ),
    "aligned_trajectories_by_sisu": (
        AlignedEffectorTrajectories()
        .after_stacking('sisu')
        .map_figs_at_level("train__pert__std")
        .then_transform_figs(
            partial(
                set_axes_bounds_equal, 
                padding_factor=0.1,
                trace_selector=lambda trace: trace.showlegend is True,
            ),
        )
    ),
    "aligned_trajectories_by_train_std": AlignedEffectorTrajectories().after_stacking("train__pert__std").map_figs_at_level('sisu'),
    "profiles": (
        Profiles()
        .after_transform(get_best_replicate)
        .after_transform(
            lambda tree, **kws: move_ldict_level_above("var", "train__pert__std", tree),
            dependency_name="vars",
        )
    ),
    "measures_by_pert_amp": Measures(measure_keys=MEASURE_KEYS).map_figs_at_level("pert__amp"),
    "measures_by_train_std": Measures(measure_keys=MEASURE_KEYS).map_figs_at_level("train__pert__std"),
    "measures_train_std_by_pert_amp": Measures(measure_keys=MEASURE_KEYS).after_level_to_top("train__pert__std").map_figs_at_level("pert__amp"),
}