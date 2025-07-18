"""
Analysis of fixed point (FP) structure of the network during reaching.

This analysis finds fixed points during simple reaching tasks, including:
1. Goal-goal fixed points (steady state)
2. Init-goal fixed points (initial state)
3. Fixed points along the trajectory during reaching
"""

from collections.abc import Callable, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Optional

import equinox as eqx
from equinox import Module
import jax.numpy as jnp
import jax.tree as jt

from jax_cookbook import is_module
import jax_cookbook.tree as jtree
import numpy as np

from rlrmp.analysis import AbstractAnalysis
from rlrmp.analysis.analysis import _DummyAnalysis, DefaultFigParamNamespace, FigParamNamespace
from rlrmp.analysis.pca import StatesPCA
from rlrmp.misc import get_constant_input_fn
from rlrmp.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rlrmp.tree_utils import take_replicate
from rlrmp.types import TreeNamespace
from rlrmp.types import LDict
from rlrmp.analysis.fps_tmp2 import (
    ReachFPs,
    ReachFPsInPCSpace,
    ReachTrajectoriesInPCSpace,
    ReachDirectionTrajectories,
)
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from feedbax.intervene import add_intervenors, schedule_intervenor


"""Specify any additional colorscales needed for this analysis. 
These will be included in the `colors` kwarg passed to `AbstractAnalysis` methods
"""
COLOR_FUNCS: dict[str, Callable[[TreeNamespace], Sequence]] = dict(
)


def setup_eval_tasks_and_models(task_base: Module, models_base: LDict[float, Module], hps: TreeNamespace):
    """Set up tasks with plant perturbations and varying SISU inputs."""
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.type}")
    
    pert_amps = hps.pert.amp  # Using amp as perturbation amplitude for this analysis
    
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
                input_fn=get_constant_input_fn(
                    sisu, 
                    hps.model.n_steps, 
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


# Unlike `feedback_perts`, we don't need to vmap over impulse amplitude 
eval_func: Callable = vmap_eval_ensemble
 
# goals_pos = task.validation_trials.targets["mechanics.effector.pos"].value[:, -1]


N_PCA = 30
START_STEP = 50  # Start PCA from midway through trial
END_STEP = 100


DEPENDENCIES = {
    "states_pca": (
        StatesPCA(n_components=N_PCA, where_states=lambda states: states.net.hidden)
        .after_transform(get_best_replicate)
        .after_indexing(-2, np.arange(START_STEP, END_STEP), axis_label="timestep")
    ),
}


# State PyTree structure: ['sisu', 'train__pert__std']
# Array batch shape: (evals, replicates, reach conditions)
ANALYSES = {
    "reach_fps_in_pc_space": (
        ReachFPsInPCSpace(custom_inputs=dict(pca_results="states_pca"))
    ),
    "reach_trajectories_in_pc_space": (
        ReachTrajectoriesInPCSpace(custom_inputs=dict(pca_results="states_pca"))
    ),
    "reach_direction_trajectories": (
        ReachDirectionTrajectories(custom_inputs=dict(pca_results="states_pca"))
    ),
}


