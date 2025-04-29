from collections.abc import Callable
from functools import partial 
from types import MappingProxyType
from typing import ClassVar, Optional, Dict, Any, Literal as L, Sequence

import equinox as eqx
from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree
import numpy as np
import plotly.graph_objects as go

from feedbax.intervene import add_intervenors, schedule_intervenor
import feedbax.plotly as fbp
from feedbax.task import TrialSpecDependency
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import AlignedEffectorTrajectories, AlignedVars
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from rnns_learn_robust_motor_policies.analysis.effector import EffectorTrajectories
from rnns_learn_robust_motor_policies.analysis.measures import ALL_MEASURE_KEYS, MEASURE_LABELS
from rnns_learn_robust_motor_policies.analysis.measures import Measures
from rnns_learn_robust_motor_policies.analysis.profiles import Profiles
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate, get_constant_task_input, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.colors import ColorscaleSpec
from rnns_learn_robust_motor_policies.config.config import PLOTLY_CONFIG
from rnns_learn_robust_motor_policies.constants import POS_ENDPOINTS_ALIGNED
from rnns_learn_robust_motor_policies.plot import add_endpoint_traces, get_violins, set_axes_bounds_equal
from rnns_learn_robust_motor_policies.types import (
    RESPONSE_VAR_LABELS,
    LDict,
    Responses,
    TreeNamespace,
)


ID = "2-1"


COLOR_FUNCS = dict(
    context_input=ColorscaleSpec(
        sequence_func=lambda hps: hps.context_input,
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
    
    tasks = LDict.of("context_input")({
        context_input: jt.map(
            lambda task: eqx.tree_at( 
                lambda task: task.input_dependencies,
                task, 
                {
                    'context': TrialSpecDependency(get_constant_task_input(
                            context_input, 
                            hps.model.n_steps - 1, 
                            task.n_validation_trials,
                    ))
                },
            ),
            tasks_by_amp,
            is_leaf=is_module,
        )
        for context_input in hps.context_input
    })
    
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
    
    # The outer levels of `all_models` have to match those of `all_tasks`
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

        
ALL_ANALYSES = [
    # By condition, all evals for the best replicate only
#     (
#        EffectorTrajectories(
#             colorscale_axis=1, 
#             colorscale_key="reach_condition",
#         )
#         .after_transform(get_best_replicate)  # By default has `axis=1` for replicates
#     ),
    (
        AlignedEffectorTrajectories()
        .after_stacking("context_input")
        .map_at_level("train__pert__std")
        .then_transform_figs(
            partial(
                set_axes_bounds_equal, 
                padding_factor=0.1,
                trace_selector=lambda trace: trace.showlegend is True,
            ),
        )
    ),
    AlignedEffectorTrajectories().after_stacking("train__pert__std").map_at_level("context_input"),
    Profiles(),  #! TODO
    Measures(measure_keys=MEASURE_KEYS).map_at_level("pert__amp"),
    Measures(measure_keys=MEASURE_KEYS).map_at_level("train__pert__std"),
    Measures(measure_keys=MEASURE_KEYS).after_level_to_top("train__pert__std").map_at_level("pert__amp"),
]