"""Temporary module for fixed-point-based analyses.

This is a refactoring of the logic in `notebooks/markdown/part2__fps_steady.md`
into the declarative analysis framework.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import ClassVar, Optional, TypeVar

import equinox as eqx
from equinox import Module, field
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.bodies import SimpleFeedbackState
from jax_cookbook import is_module, vmap_multi, is_type, is_none
from sqlalchemy import func

from rlrmp.analysis.analysis import (
    AbstractAnalysis,
    AnalysisDefaultInputsType,
    AnalysisInputData,
    DefaultFigParamNamespace,
    FigParamNamespace,
)
from rlrmp.analysis.fp_finder import (
    FPFilteredResults,
)
from rlrmp.analysis.fps import FixedPoints
from rlrmp.analysis.pca import StatesPCA
from rlrmp.analysis.state_utils import exclude_bad_replicates
from rlrmp.misc import take_non_nan
from rlrmp.plot import plot_fp_pcs
from rlrmp.tree_utils import first, ldict_level_to_bottom
from rlrmp.types import LDict, TreeNamespace


T = TypeVar('T')


def origin_only(states, axis=-2, *, hps_common):
    #! TODO: Do not assume "full" variant
    origin_idx = hps_common.task.full.eval_grid_n ** 2 // 2
    idx = jnp.array([origin_idx])
    return jt.map(lambda x: jnp.take_along_axis(x, idx, axis=axis), states)


#! TODO: Finish gutting this
class SteadyStateJacobians(AbstractAnalysis):
    """Compute Jacobians and their eigendecomposition at steady-state FPs."""

    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType(dict(
        fps_results=FixedPoints,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    origin_only: bool = False
    key: PRNGKeyArray = field(default_factory=lambda: jr.PRNGKey(0))

    def compute(
        self,
        data: AnalysisInputData,
        *,
        fps_results: TreeNamespace,
        hps_common: TreeNamespace,
        **kwargs,
    ):        
        
        task_leaf = jt.leaves(data.tasks, is_leaf=is_module)[0]
        goals_pos = task_leaf.validation_trials.targets["mechanics.effector.pos"].value[:, -1]
        
        fps_grid = jnp.moveaxis(fps_results.fps, 0, 1)

        rnn_funcs = jt.map(lambda m: m.step.net.hidden, data.models, is_leaf=is_module)


#! Should be totally scrappable
class FPsInPCSpace(AbstractAnalysis):
    """Plot fixed points in PC space."""

    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType(dict(
        fps_results=FixedPoints,
        pca_results=StatesPCA,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        fps_results: TreeNamespace,
        pca_results: TreeNamespace,
        colors: PyTree,
        replicate_info: PyTree,
        hps_common: TreeNamespace,
        **kwargs,
    ):
        
        fps_grid = jnp.moveaxis(fps_results.fps, 0, 1)

        fps_grid_pre_pc = jt.map(
            lambda fps: take_non_nan(fps, axis=1),
            exclude_bad_replicates(fps_grid, replicate_info=replicate_info),
        )
        
        fps_grid_pc = pca_results.batch_transform(fps_grid_pre_pc)
        
        all_readout_weights = exclude_bad_replicates(
            jt.map(
                lambda model: model.step.net.readout.weight,
                # Weights do not depend on SISU, take first
                jt.map(first, data.models, is_leaf=LDict.is_of('sisu')),
                is_leaf=is_module,
            ),
            replicate_info=replicate_info,
            axis=0,
        )

        all_readout_weights_pc = pca_results.batch_transform(all_readout_weights)
        
        def plot_fp_pcs_by_sisu(fp_pcs_by_sisu, readout_weights_pc):
            fig = go.Figure(
                layout=dict(
                    width=800, height=800,
                    scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                    legend=dict(title='SISU', itemsizing='constant', y=0.85),
                )
            )

            for sisu, fps_pc in fp_pcs_by_sisu.items():
                fig = plot_fp_pcs(
                    fps_pc, fig=fig, label=sisu,
                    colors=colors['sisu'].dark[sisu],
                )

            if readout_weights_pc is not None:
                fig.update_layout(
                    legend2=dict(title='Readout components', itemsizing='constant', y=0.45),
                )
                mean_base_fp_pc = jnp.mean(fp_pcs_by_sisu[min(hps_common.sisu)], axis=1)
                traces = []
                k = 0.25
                for j in range(readout_weights_pc.shape[-2]):
                    start, end = mean_base_fp_pc, mean_base_fp_pc + k * readout_weights_pc[..., j, :]
                    x = np.column_stack((start[..., 0], end[..., 0], np.full_like(start[..., 0], None))).ravel()
                    y = np.column_stack((start[..., 1], end[..., 1], np.full_like(start[..., 1], None))).ravel()
                    z = np.column_stack((start[..., 2], end[..., 2], np.full_like(start[..., 2], None))).ravel()
                    traces.append(go.Scatter3d(
                        x=x, y=y, z=z, mode='lines', line=dict(width=10),
                        showlegend=True, name=j, legend="legend2",
                    ))
                fig.add_traces(traces)
            return fig

        return jt.map(
            plot_fp_pcs_by_sisu,
            fps_grid_pc,
            all_readout_weights_pc,
            is_leaf=LDict.is_of('sisu'),
        ) 