from collections.abc import Callable
from functools import partial 
from types import MappingProxyType
from typing import ClassVar, Optional, Dict, Any, Literal as L, Sequence

import equinox as eqx
from equinox import Module
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree
import numpy as np
import plotly.graph_objects as go

from feedbax.intervene import NetworkConstantInput, TimeSeriesParam, add_intervenors, schedule_intervenor
import feedbax.plotly as fbp
from jax_cookbook import is_module, is_type, is_none
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import AlignedEffectorTrajectories, AlignedVars, get_trivial_reach_origins_directions
from rnns_learn_robust_motor_policies.analysis.analysis import _DummyAnalysis, AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS, get_pert_amp_vmap_eval_func
from rnns_learn_robust_motor_policies.analysis.effector import EffectorTrajectories
from rnns_learn_robust_motor_policies.analysis.measures import ALL_MEASURE_KEYS, MEASURE_LABELS
from rnns_learn_robust_motor_policies.analysis.measures import Measures
from rnns_learn_robust_motor_policies.analysis.network import UnitPreferences
from rnns_learn_robust_motor_policies.analysis.profiles import Profiles
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate, get_constant_task_input_fn, get_segment_trials_func, get_symmetric_accel_decel_epochs, vmap_eval_ensemble
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


COLOR_FUNCS = dict(
    context_input=ColorscaleSpec(
        sequence_func=lambda hps: hps.context_input,
        colorscale="thermal",
    ),
    stim_amp=ColorscaleSpec(
        sequence_func=lambda hps: hps.pert.unit.amp,
        colorscale="viridis",
    )
)


UNIT_STIM_INTERVENOR_LABEL = "UnitStim"


def unit_stim(unit_idx, *, hps):
    # idxs = slice(
    #     hps.pert.unit.start_step, 
    #     hps.pert.unit.start_step + hps.pert.unit.duration,
    # )
    # trial_mask = jnp.zeros((hps.train.model.n_steps - 1,), bool).at[idxs].set(True)
    
    unit_spec = jnp.full(hps.train.model.hidden_size, jnp.nan)
    unit_spec = unit_spec.at[unit_idx].set(1.0)
    
    return NetworkConstantInput.with_params(
        # out_where=lambda state: state.hidden,
        unit_spec=unit_spec,
        active=True,
        # active=TimeSeriesParam(trial_mask),
    )
    
    
def schedule_unit_stim(unit_idx, *, tasks, models, hps):
    tasks, models = jtree.unzip(jt.map(
        lambda task, model, hps: schedule_intervenor(
            task, model, 
            lambda model: model.step.net,
            # unit_stim(unit_idx, hps=hps),
            NetworkConstantInput(),
            default_active=False,
            stage_name=None,  # None -> before RNN forward pass; 'hidden' -> after 
            label=UNIT_STIM_INTERVENOR_LABEL,
        ),
        tasks, models, hps,
        is_leaf=is_module,
    ))
    return tasks, models


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.plant.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.plant.type}")
    
    pert_amps = hps.pert.plant.amp
    
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
    
    # Also vary tasks by context input
    tasks = LDict.of("context_input")({
        context_input: jt.map(
            lambda task: task.add_input(
                name="context",
                input_fn=get_constant_task_input_fn(
                    context_input, 
                    hps.model.n_steps - 1, 
                    task.n_validation_trials,
                ),
            ),
            tasks_by_amp,
            is_leaf=is_module,
        )
        for context_input in hps.context_input
    })
    
    # The outer levels of `models` have to match those of `tasks`
    models, hps = jtree.unzip(jt.map(
        lambda _: (models_by_std, hps), tasks, is_leaf=is_module
    ))
    
    # Schedule unit stim for a placeholder unit (0)
    tasks, models = schedule_unit_stim(
        0, tasks=tasks, models=models, hps=hps,
    )
    
    return tasks, models, hps, None


def task_with_unit_stim(task, unit_idx, stim_amp, hidden_size, intervenor_label):
    return eqx.tree_at(
        lambda task: (
            task.intervention_specs.validation[intervenor_label].intervenor.params.scale,
            task.intervention_specs.validation[intervenor_label].intervenor.params.unit_spec,
        ),
        task,
        (
            stim_amp,
            jnp.full(hidden_size, jnp.nan).at[unit_idx].set(1.0),
        ),
        is_leaf=is_none,
    )


def eval_func(key_eval, hps, models, task):
    states = eqx.filter_vmap(
        lambda stim_amp: eqx.filter_vmap(
            lambda unit_idx: vmap_eval_ensemble(
                key_eval, 
                hps,
                models,
                task_with_unit_stim(
                    task, 
                    unit_idx, 
                    stim_amp, 
                    hps.train.model.hidden_size, 
                    UNIT_STIM_INTERVENOR_LABEL,
                ),
            ),
        )(jnp.arange(hps.train.model.hidden_size))
    )(jnp.array(hps.pert.unit.amp))
    
    return states


MEASURE_KEYS = (
)


PLANT_PERT_LABELS = {0: "no curl", 1: "curl"}
PLANT_PERT_STYLES = dict(line_dash={0: "dot", 1: "solid"})

CONTEXT_LABELS = {0: -2, 1: 0, 2: 2}
CONTEXT_STYLES = dict(line_dash={0: "dot", 1: "dash", 2: "solid"})

UNIT_STIM_IDX = 1

# PyTree structure: [context_input, pert__amp, train__pert__std]
# Array batch shape: [stim_amp, unit_idx, eval, replicate, condition]
ALL_ANALYSES = [
    (
        # TODO: We need profiles *without* AlignedVars (but *with* subset of states in "vars" level)
        # TODO: for steady-state analysis. How can we modify the dependencies of `Profiles` conditionally? 
        Profiles(
            variant="full",
            dependency_params={
                AlignedVars: dict(
                    # Bypass alignment; keep aligned with x-y axes
                    origins_directions_func=get_trivial_reach_origins_directions,
                )
            }
        )
        .after_transform(partial(get_best_replicate, axis=3))
        .after_indexing(1, jnp.array([0,1,2,3,4]), axis_label="unit_stim_idx")  #! Examine only a few stim units
        .after_unstacking(1, 'unit_stim_idx', above_level='pert__amp')#, dependency_name="aligned_vars")
        .after_indexing(0, 1, axis_label="stim_amp")  #! Only examine trials with unit stim
        .combine_figs_by_level(
            level='context_input',
            fig_params_fn=lambda fig_params, i, item: dict(
                scatter_kws=dict(
                    line_dash=CONTEXT_STYLES['line_dash'][i],
                    legendgroup=CONTEXT_LABELS[i],
                    legendgrouptitle_text=CONTEXT_LABELS[i],
                ),
            ),
        )
        .with_fig_params(
            # legend_title="Context",
            layout_kws=dict(
                width=500,
                height=300,
            ),
        )
    ),
    (
        # Result shape: [stim_amp, unit_stim_idx, unit_idx, feature]
        UnitPreferences(
            variant="full",
            feature_fn=lambda task, states: states.efferent.output,
            label="unit_prefs",
        )
        .after_transform(partial(get_best_replicate, axis=3))
        .after_transform(
            get_segment_trials_func(get_symmetric_accel_decel_epochs),
            dependency_name="states",
        )
        .vmap_over_states(axes=[0, 1])
    ),
]