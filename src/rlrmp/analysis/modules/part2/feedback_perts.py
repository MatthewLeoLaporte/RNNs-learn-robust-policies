from collections.abc import Callable
from functools import partial
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Literal, Optional
import jax.numpy as jnp
import jax.tree as jt 

import equinox as eqx
from jax_cookbook import is_type, is_module, is_none
import jax_cookbook.tree as jtree


from feedbax.intervene import schedule_intervenor
import feedbax.plotly as fbp

# from rlrmp.analysis import measures
from rlrmp.analysis import AbstractAnalysis, AnalysisInputData
from rlrmp.analysis.aligned import AlignedEffectorTrajectories, AlignedVars
from rlrmp.analysis.analysis import _DummyAnalysis, DefaultFigParamNamespace, FigParamNamespace
from rlrmp.analysis.disturbance import FB_INTERVENOR_LABEL, get_pert_amp_vmap_eval_func, task_with_pert_amp
from rlrmp.analysis.effector import EffectorTrajectories
from rlrmp.analysis.measures import Measures
from rlrmp.analysis.profiles import Profiles
from rlrmp.misc import lohi
from rlrmp.plot import PLANT_VAR_LABELS, WHERE_PLOT_PLANT_VARS, set_axis_bounds_equal
from rlrmp.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rlrmp.misc import get_constant_input_fn
from rlrmp.types import LDict, unflatten_dict_keys
from rlrmp.perturbations import feedback_impulse
from rlrmp.colors import ColorscaleSpec


COLOR_FUNCS = dict()


#! TODO: Move; these are redundant with 1-2
PERT_VAR_NAMES = ('fb_pos', 'fb_vel')
COORD_NAMES = ('x', 'y')
I_IMPULSE_AMP_PLOT = -1  # The largest amplitude perturbation
COMPONENTS_LABELS = (r'\parallel', r'\bot')
COMPONENTS_NAMES = ('parallel', 'orthogonal')


def setup_eval_tasks_and_models(task_base, models_base, hps):
    impulse_end_step = hps.pert.start_step + hps.pert.duration
    impulse_time_idxs = slice(hps.pert.start_step, impulse_end_step)

    all_impulse_amplitudes = jt.map(
        lambda max_amp: jnp.linspace(0, max_amp, hps.pert.n_amps + 1)[1:],
        LDict.of("pert__var").from_ns(hps.pert.amp_max),
    )

    all_tasks, all_models, all_hps = jtree.unzip(jt.map(
        lambda feedback_var_idx, impulse_amplitudes: (
            *schedule_intervenor(
                task_base, models_base,
                lambda model: model.step.feedback_channels[0],  # type: ignore
                feedback_impulse(  
                    hps.model.n_steps,
                    1.0,  # Will be varied later
                    hps.pert.duration,
                    feedback_var_idx,   
                    hps.pert.start_step,
                ),
                default_active=False,
                stage_name="update_queue",
                label=FB_INTERVENOR_LABEL,
            ),
            hps | unflatten_dict_keys(dict(pert__amp=impulse_amplitudes)),
        ),
        LDict.of("pert__var")(dict(fb_pos=0, fb_vel=1)),
        all_impulse_amplitudes,
        is_leaf=is_type(tuple),
    ))

    # # Get the perturbation directions, for later:
    # #? I think these values are equivalent to `line_vec` in the functions in `state_utils`
    # impulse_directions = jt.map(
    #     lambda task: task.validation_trials
    #         .intervene[FB_INTERVENOR_LABEL]
    #         .arrays[:, hps.pert.start_step],
    #     all_tasks,
    #     is_leaf=is_module,
    # )

    # Generate tasks with different SISU
    # TODO: Ideally we'd just `tree_at` or `vmap` a single instance, instead of constructing a whole PyTree of them
    all_tasks, all_models, all_hps = jtree.unzip(jt.map(
        lambda task, models, hps: LDict.of('sisu')({
            sisu: (
                task.add_input(
                    name="sisu",
                    input_fn=get_constant_input_fn(
                        sisu, hps.model.n_steps, task.n_validation_trials,
                    ),
                ),
                models,  
                hps | dict(sisu=sisu),
            )
            for sisu in hps.sisu
        }),
        all_tasks,
        all_models,
        all_hps,
        is_leaf=is_module,
    ))
    
    extras = SimpleNamespace(
        # impulse_directions=impulse_directions,
        impulse_time_idxs=impulse_time_idxs,
    )
    
    return all_tasks, all_models, all_hps, extras


eval_func = get_pert_amp_vmap_eval_func(lambda hps: hps.pert.amp, FB_INTERVENOR_LABEL)


MEASURE_KEYS = [
    "max_net_force",
    "max_parallel_force_reverse",
    "sum_net_force",
    "max_parallel_vel_forward",
    "max_parallel_vel_reverse",
    "max_orthogonal_vel_left",
    "max_orthogonal_vel_right",
    "max_deviation",
    "sum_deviation",
]


# TODO: We wouldn't need to hardcode this if we could pass a callable to `after_indexing`
ORIGIN_GRID_IDX = 12

#! Add a couple more measures over specific intervals of the trials
# shortly_after_impulse = slice(impulse_end_step, impulse_end_step + impulse_duration)
# after_impulse = slice(impulse_end_step, None)
# custom_measures, custom_measure_labels = jtree.unzip({
#     "max_parallel_force_forward_shortly_after_impulse": (
#         measures.set_timesteps(
#             measures.max_parallel_force, shortly_after_impulse,
#         ),
#         f"Max forward force within {impulse_duration} steps of pert. end",
#     ),
#     "max_net_force_during_impulse": (
#         measures.set_timesteps(
#             measures.max_net_force, impulse_time_idxs,
#         ),
#         "Max net force during pert.",
#     ),
#     "max_net_force_after_impulse": (
#         measures.set_timesteps(
#             measures.max_net_force, after_impulse,
#         ),
#         "Max net force after pert.",
#     ),
# })
# all_measures = subdict(MEASURES, measure_keys) | custom_measures
# measure_labels = MEASURE_LABELS | custom_measure_labels


def get_impulse_origins_directions(task, models, hps):
    # Steady-state positions
    origins = task.validation_trials.inits["mechanics.effector"].pos

    # Impulse directions
    directions = (
        task
        .validation_trials
        .intervene[FB_INTERVENOR_LABEL]
        .arrays[:, hps.pert.start_step]
    )

    return origins, directions


DEPENDENCIES = {
    "aligned_vars_impulse": AlignedVars(
        origins_directions_func=get_impulse_origins_directions,
    ),
}


def get_impulse_vrect_kws(hps):
    return dict(
        x0=hps.pert.start_step,
        x1=hps.pert.start_step + hps.pert.duration,
        fillcolor="grey",
        opacity=0.2,
        line_width=0,
        name='Perturbation',
    )


def measures_fig_params_fn(fig_params, i, item):
    if i == 0: 
        return fig_params | dict(
            trace_kws=dict(
                    opacity=0.3, line_color='grey',
                ),
            layout_kws=dict(
                showlegend=False,
                xaxis_visible=False, 
                # yaxis_visible=False,
            ),
        )
    return fig_params


# State PyTree structure: ['pert__var', 'sisu', 'train__pert__std']
# Array batch shape: (evals, replicates, impulse amplitudes, reach conditions)
ANALYSES = {
    # "effector_trajectories": (
    #     EffectorTrajectories(
    #         variant="full",
    #         colorscale_axis=1,  # impulse amplitude  # TODO: change to 0 if indexing eval
    #         colorscale_key='pert__amp',
    #     )
    #     .after_transform(get_best_replicate) 
    #     # .after_indexing(2, ORIGIN_GRID_IDX, axis_label='grid')
    #     # .after_indexing(0, i_eval, axis_label='eval')
    #     .with_fig_params(
    #         mean_exclude_axes=(-3,),  # TODO: uncomment if not indexing grid
    #         # curves_mode='markers+lines',
    #         # ms=3,
    #         # scatter_kws=dict(line_width=0.75),
    #         # mean_scatter_kws=dict(line_width=0),
    #     )
    # ),

    "aligned_trajectories": (
        AlignedEffectorTrajectories(
            variant="full",
            colorscale_axis=1,
            colorscale_key='pert__amp',
            custom_inputs={
                "aligned_vars": "aligned_vars_impulse",
            },
        )
        .after_transform(get_best_replicate)
    ),

    "aligned_trajectories_by_train_std": (
        #! This is broken; nothing appears. 
        AlignedEffectorTrajectories(
            variant="full",
            custom_inputs={
                "aligned_vars": "aligned_vars_impulse",
            },
        )
        .after_transform(get_best_replicate)
        .after_stacking(level='train__pert__std')
    ),

    "profiles": (
        Profiles(
            variant="full",
            custom_inputs={
                "vars": "aligned_vars_impulse",
            },
            vrect_kws_func=get_impulse_vrect_kws,  
        )
        .after_transform(get_best_replicate) 
        .after_indexing(1, -2, axis_label='pert__amp') 
        .map_figs_at_level('train__pert__std')
        .with_fig_params(
            # legend_title="SISU",
            layout_kws=dict(
                width=500,
                height=300,
            ),
        )
    ),

    "measures": (
        Measures(
            measure_keys=MEASURE_KEYS,
            custom_inputs={
                "vars": "aligned_vars_impulse",
            },
        )
        .after_transform(get_best_replicate)
        .after_unstacking(1, "pert__amp")
        .after_transform(lohi, level="train__pert__std")
        # Save seperate figures for zero-std, as pared-down all-grey
        .map_figs_at_level(
            'train__pert__std',
            fig_params_fn=measures_fig_params_fn,
        )
        .with_fig_params(
            legend_title="SISU",
            xaxis_title="Feedback impulse amplitude",
            violinmode="group",
        )
        .then_transform_figs(
            partial(set_axis_bounds_equal, 'y'),
            level='train__pert__std'
        )
    ),
}