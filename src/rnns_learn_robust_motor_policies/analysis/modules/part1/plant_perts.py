from functools import partial

import jax.numpy as jnp
import jax.tree as jt

from feedbax.intervene import add_intervenors, schedule_intervenor
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import AlignedEffectorTrajectories
from rnns_learn_robust_motor_policies.analysis.effector import EffectorTrajectories
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_PERT_FUNCS
from rnns_learn_robust_motor_policies.analysis.measures import Measures, output_corr
from rnns_learn_robust_motor_policies.analysis.profiles import Profiles
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.misc import lohi
from rnns_learn_robust_motor_policies.plot import get_violins, set_axes_bounds_equal, set_axis_bounds_equal
from rnns_learn_robust_motor_policies.types import LDict


ID = "1-1"


COLOR_FUNCS = dict()


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown perturbation type: {hps.pert.type}")

    # Insert the disturbance field component into each model
    models = jt.map(
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

    # Assume a sequence of amplitudes is provided, as in the default config
    pert_amps = hps.pert.amp
    # Construct tasks with different amplitudes of disturbance field
    all_tasks, all_models = jtree.unzip(jt.map(
        lambda pert_amp: schedule_intervenor(
            task_base, models,
            lambda model: model.step.mechanics,
            disturbance(pert_amp),
            label=PLANT_INTERVENOR_LABEL,  
            default_active=False,
        ),
        LDict.of("pert__amp")(
            dict(zip(pert_amps, pert_amps))
        ),
    ))
    
    all_hps = jt.map(lambda _: hps, all_tasks, is_leaf=is_module)
    
    return all_tasks, all_models, all_hps, None


# We aren't vmapping over any other variables, so this is trivial.
eval_func = vmap_eval_ensemble


"""Labels of measures to include in the analysis."""
MEASURE_KEYS = (
    "max_parallel_vel_forward",
    "max_orthogonal_vel_left",
    "max_orthogonal_vel_right",
    "max_orthogonal_distance_left",
    "sum_orthogonal_distance",
    "end_position_error",
    # "end_velocity_error",
    "max_parallel_force_forward",
    "sum_parallel_force",
    "max_orthogonal_force_right",  
    "sum_orthogonal_force_abs",
    "max_net_force",
    "sum_net_force",
)

measures_base = Measures(measure_keys=MEASURE_KEYS)

i_eval = 0  # For single-eval plots   
        

# PyTree levels: 
# State batch shape: (eval, replicate, condition)
ALL_ANALYSES = [
    # By condition, all evals for the best replicate only
    (
        EffectorTrajectories(
            colorscale_axis=1, 
            colorscale_key="reach_condition",
        )
        .after_transform(get_best_replicate)
        .then_transform_figs(
            partial(set_axis_bounds_equal, 'y', padding_factor=0.1),
        )
        # .with_fig_params()
    ),  

    # By replicate, single eval
    (
        EffectorTrajectories(
            colorscale_axis=0, 
            colorscale_key="replicate",
        )
        .after_indexing(0, i_eval, axis_label='eval')
        .with_fig_params(
            scatter_kws=dict(line_width=1),
        )
    ),

    # Single eval for a single replicate
    (
        EffectorTrajectories(
            colorscale_axis=0, 
            colorscale_key="reach_condition",
        )
        .after_transform(get_best_replicate) 
        .after_indexing(0, i_eval, axis_label='eval')
        .with_fig_params(
            curves_mode='markers+lines',
            ms=3,
            scatter_kws=dict(line_width=0.75),
            mean_scatter_kws=dict(line_width=0),
        )
    ),

    AlignedEffectorTrajectories().after_stacking(level='pert__amp'),
    AlignedEffectorTrajectories().after_stacking(level='train__pert__std'),
    Profiles().after_transform(get_best_replicate)
    measures_base,
    measures_base.after_transform(lohi, level='train__pert__std'),
    measures_base.after_transform(lohi, level=['train__pert__std', 'pert__amp']),
]