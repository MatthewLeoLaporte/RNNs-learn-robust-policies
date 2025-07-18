"""
Do PCA on hidden states during reaching. See how tangling varies over them. 

I am not sure how well this will work, given that each trajectory won't visit the same place 
in state space many times. However, it may be the case that there are still significant differences 
across SISU. 
"""

from collections.abc import Callable
from functools import partial 
from types import MappingProxyType
from typing import ClassVar, Optional, Dict, Any, Literal as L, Sequence

import equinox as eqx
from equinox import Module
import jax
import jax.numpy as jnp
import jax.tree as jt
from feedbax.misc import batch_reshape  # for flattening/unflattening
from jaxtyping import Array, Float, PyTree
import numpy as np
import plotly.graph_objects as go

from feedbax.intervene import add_intervenors, schedule_intervenor
import feedbax.plotly as fbp
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rlrmp.analysis.aligned import AlignedEffectorTrajectories, AlignedVars
from rlrmp.analysis.analysis import AbstractAnalysis, AnalysisDependenciesType, AnalysisInputData, CallWithDeps, Data, DefaultFigParamNamespace, FigParamNamespace, Required
from rlrmp.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from rlrmp.analysis.effector import EffectorTrajectories
from rlrmp.analysis.measures import ALL_MEASURE_KEYS, MEASURE_LABELS
from rlrmp.analysis.measures import Measures
from rlrmp.analysis.pca import StatesPCA
from rlrmp.analysis.profiles import Profiles
from rlrmp.analysis.state_utils import get_best_replicate, get_constant_task_input_fn, vmap_eval_ensemble
from rlrmp.analysis.tangling import Tangling
from rlrmp.colors import ColorscaleSpec
from rlrmp.config.config import PLOTLY_CONFIG
from rlrmp.constants import POS_ENDPOINTS_ALIGNED
from rlrmp.plot import add_endpoint_traces, get_violins, set_axes_bounds_equal
from rlrmp.tree_utils import move_ldict_level_above
from rlrmp.types import (
    RESPONSE_VAR_LABELS,
    LDict,
    Responses,
    TreeNamespace,
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


N_PCA = 10
TANGLING_AGG_T_SLICE = slice(1, None)


DEPENDENCIES = {
    "hidden_states_pca": (
        StatesPCA(
            n_components=N_PCA, 
            where_states=lambda states: states.net.hidden,
        )
        .after_transform(get_best_replicate)
        # .after_indexing(-2, np.arange(START_STEP, END_STEP), axis_label="timestep")
    ),
}


def rms(x: Array, axis: int = -1) -> Array:
    """Returns the root mean square of `x` along `axis`."""
    return jnp.sqrt(jnp.mean(x ** 2, axis=axis))


class PCPlot(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()  
    variant: Optional[str] = "small"
    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        title="",
        x_label="",
    )


# State batch shape: (eval, replicate, condition)
ANALYSES = {
    # This is an example of how to use a CallWithDeps transform to get the PCs 
    # without having to make `StatesPCA` a dependency of `Tangling`
    "tangling": (
        Tangling(
            variant="small",
            custom_inputs=dict(
                state=Data.states(where=lambda states: states.net.hidden),
            )
        )
        .after_transform(get_best_replicate)
        .after_transform(
            # Pull in the PCA results and use them to transform the hidden states
            CallWithDeps("hidden_states_pca")(
                lambda pca_results, states: pca_results.batch_transform(states),
            ),
            dependency_name="state",
        ) 
        #! TODO: Do the RMS in a separate analysis -> violin plots
        # .then_transform_result(
        #     lambda result: jt.map(lambda x: rms(x[..., TANGLING_AGG_T_SLICE]), result),
        # )              
    )

}