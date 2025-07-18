"""Temporary module for fixed-point-based analyses.

This is a refactoring of the logic in `notebooks/markdown/part2__fps_steady.md`
into the declarative analysis framework.
"""

from collections.abc import Callable, Sequence
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import equinox as eqx
from equinox import Module, field
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax.bodies import SimpleFeedbackState
from jax_cookbook import is_module, is_type, vmap_multi

from rlrmp.analysis.analysis import (
    AbstractAnalysis,
    AnalysisDependenciesType,
    AnalysisInputData,
    Data,
    DefaultFigParamNamespace,
    FigParamNamespace,
    Required,
)
from rlrmp.analysis.fp_finder import (
    FPFilteredResults,
    FixedPointFinder,
    fp_adam_optimizer,
    take_top_fps,
)
from rlrmp.analysis.pca import StatesPCA
from rlrmp.analysis.state_utils import exclude_bad_replicates
from rlrmp.misc import create_arr_df, take_non_nan
from rlrmp.plot import plot_eigvals_df, plot_fp_pcs
from rlrmp.tree_utils import first, ldict_level_to_bottom
from rlrmp.types import LDict, TreeNamespace


#! TODO: Either remove SISU-specific logic, or rename
def process_fps(all_fps: PyTree[FPFilteredResults]):
    """Only keep FPs/replicates that meet criteria."""
    
    n_fps_meeting_criteria = jt.map(
        lambda fps: fps.counts['meets_all_criteria'],
        all_fps,
        is_leaf=is_type(FPFilteredResults),
    )

    satisfactory_replicates = jt.map(
        lambda n_matching_fps_by_sisu: jnp.all(
            jnp.stack(jt.leaves(n_matching_fps_by_sisu), axis=0),
            axis=0,
        ),
        n_fps_meeting_criteria,
        is_leaf=LDict.is_of('sisu'),
    )

    all_top_fps = take_top_fps(all_fps, n_keep=6)

    # Average over the top fixed points, to get a single one for each included replicate and
    # control input.
    fps_final = jt.map(
        lambda top_fps_by_sisu: jt.map(
            lambda fps: jnp.nanmean(fps, axis=-2),
            top_fps_by_sisu,
            is_leaf=is_type(FPFilteredResults),
        ),
        all_top_fps,
        is_leaf=LDict.is_of('sisu'),
    )

    return TreeNamespace(
        fps=fps_final,
        all_top_fps=all_top_fps,
        n_fps_meeting_criteria=n_fps_meeting_criteria,
        satisfactory_replicates=satisfactory_replicates,
    )


def get_ss_rnn_input(pos: Float[Array, "2"], sisu: float, input_size: int):
    input_star = jnp.zeros((input_size,))
    # Set target and feedback inputs to the same position
    input_star = input_star.at[1:3].set(pos)
    input_star = input_star.at[5:7].set(pos)
    return input_star.at[0].set(sisu)


def get_ss_rnn_func(pos: Float[Array, "2"], sisu: float, rnn_cell: Module, key: PRNGKeyArray):
    input_star = get_ss_rnn_input(pos, sisu, rnn_cell.input_size)
    def rnn_func(h):
        return rnn_cell(input_star, h, key=key)
    return rnn_func


class NNSteadyStateFPs(AbstractAnalysis):
    """Find steady-state fixed points of the RNN."""

    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    cache_result: bool = True
    
    ss_func: Callable = get_ss_rnn_func
    fp_tol: float = 1e-5
    unique_tol: float = 0.025
    outlier_tol: float = 1.0
    stride_candidates: int = 16
    key: PRNGKeyArray = field(default_factory=lambda: jr.PRNGKey(0))

    def compute(
        self,
        data: AnalysisInputData,
        hps_common: TreeNamespace,
        **kwargs,
    ):
        fp_optimizer = fp_adam_optimizer()
        fpfinder = FixedPointFinder(fp_optimizer)
        fpf_func = partial(
            fpfinder.find_and_filter,
            outlier_tol=self.outlier_tol,
            unique_tol=self.unique_tol,
            key=self.key,
        )

        models, states = [
            ldict_level_to_bottom('sisu', tree, is_leaf=is_module)
            for tree in (data.models, data.states)
        ]

        rnn_funcs = jt.map(
            lambda model: model.step.net.hidden,
            models,
            is_leaf=is_module,
        )

        task_leaf = jt.leaves(data.tasks, is_leaf=is_module)[0]
        positions = task_leaf.validation_trials.targets["mechanics.effector.pos"].value[:, -1]

        def get_ss_rnn_fps(
            pos: Float[Array, "2"], 
            rnn_cell: Module, 
            candidate_states: Float[Array, "n_candidates n_replicates hidden_size"], 
            sisu: float, 
            fpf_func, 
            fp_tol, 
            key: PRNGKeyArray,
        ):
            fps = fpf_func(
                get_ss_rnn_func(pos, sisu, rnn_cell, key),
                candidate_states,
                fp_tol,
            )
            return fps

        get_fps_partial = partial(
            get_ss_rnn_fps,
            fpf_func=fpf_func,
            fp_tol=self.fp_tol,
            key=self.key,
        )

        #! Does the replicate logic hold when only the best replicate is passed to this analysis?
        if isinstance(positions, Array) and len(positions.shape) == 2:
            get_fps_func = vmap_multi(
                get_fps_partial,
                in_axes_sequence=(
                    (None, 0, 0, None),  # Over replicates
                    (0, None, None, None),  # Over grid positions
                ),
            )
        else:
            get_fps_func = eqx.filter_vmap(
                get_fps_partial,
                in_axes=(None, 0, 0, None),  # Over replicates
            )

        candidates = jt.map(
            lambda s: jnp.reshape(
                s.net.hidden,
                (hps_common.train.model.n_replicates, -1, hps_common.train.model.hidden_size),
            )[:, ::self.stride_candidates],
            states,
            is_leaf=is_module,
        )

        all_fps = jt.map(
            lambda func, candidates_by_sisu: LDict.of('sisu')({
                sisu: get_fps_func(
                    positions,
                    first(func, is_leaf=is_module),
                    candidates_by_sisu[sisu],
                    sisu,
                )
                for sisu in hps_common.sisu
            }),
            rnn_funcs, candidates,
            is_leaf=LDict.is_of('sisu'),
        )

        return process_fps(all_fps)


def origin_only(states, axis=-2, *, hps_common):
    #! TODO: Do not assume "full" variant
    origin_idx = hps_common.task.full.eval_grid_n ** 2 // 2
    idx = jnp.array([origin_idx])
    return jt.map(lambda x: jnp.take_along_axis(x, idx, axis=axis), states)


class Eigendecomposition(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict(
        matrices=Required,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    @partial(jax.jit, device=jax.devices('cpu')[0])
    def _eig_cpu(self, *a, **kw):
        return tuple(jax.lax.linalg.eig(*a, **kw))
    
    def compute(
        self,
        data: AnalysisInputData,
        matrices,
        **kwargs,
    ):
        eigvals, eigvecs_l, eigvecs_r = jtree.unzip(jt.map(self._eig_cpu, matrices))
        return TreeNamespace(
            eigvals=eigvals,
            eigvecs_l=eigvecs_l,
            eigvecs_r=eigvecs_r,
        )
    
    def make_figs(
        self, 
        data: AnalysisInputData, 
        result: PyTree, 
        hps_common: TreeNamespace,
        colors: PyTree,
        **kwargs
    ) -> PyTree[go.Figure]:
        
        eigvals = result.eigvals
        
        #! TODO: Do not hardcode column names here... 
        #! If too difficult, just separate this method off into a different, ad hoc analysis
        col_names = ['sisu', 'pos', 'replicate', 'eigenvalue']
        eigval_dfs = jt.map(
            lambda arr: create_arr_df(arr, col_names=col_names).astype({'sisu': 'str', 'replicate': 'str'}),
            eigvals
        )
        
        plot_func_partial = partial(
            plot_eigvals_df,
            marginals='box',
            color='sisu',
            trace_kws=dict(marker_size=2.5),
            scatter_kws=dict(opacity=1),
            layout_kws=dict(
                legend_title='SISU',
                legend_itemsizing='constant',
                xaxis_title='Re',
                yaxis_title='Im',
            ),
        )

        figs = jt.map(
            lambda df: plot_func_partial(df, color_discrete_sequence=list(colors['sisu'].dark.values())),
            eigval_dfs
        )

        #! TODO: Remove SISU logic 
        if self.sisu_values_to_plot is not None:
             sisu_values_to_plot = self.sisu_values_to_plot
        else:
             sisu_values_to_plot = hps_common.sisu

        def _update_trace_name(trace):
            non_data_trace_names = ['zerolines', 'boundary_circle', 'boundary_line']
            if trace.name is not None and trace.name not in non_data_trace_names:
                return trace.update(name=sisu_values_to_plot[int(trace.name)])
            else:
                return trace

        jt.map(
            lambda fig: fig.for_each_trace(_update_trace_name),
            figs,
            is_leaf=is_type(go.Figure),
        )

        return figs


def get_ss_rnn_input(pos: Float[Array, "2"], sisu: float, input_size: int):
    input_star = jnp.zeros((input_size,))
    # Set target and feedback inputs to the same position
    input_star = input_star.at[1:3].set(pos)
    input_star = input_star.at[5:7].set(pos)
    return input_star.at[0].set(sisu)


def get_ss_rnn_func(pos: Float[Array, "2"], sisu: float, rnn_cell: Module, key: PRNGKeyArray):
    input_star = get_ss_rnn_input(pos, sisu, rnn_cell.input_size)
    def rnn_func(h):
        return rnn_cell(input_star, h, key=key)
    return rnn_func


class Jacobians(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    #! TODO: Generalize to accept inputs/states as `default_inputs`,
    #! since maybe the user wants to construct them differently
    func_where: Callable[[PyTree[Module]], Callable[[Array, Array], Array]] = None  # In `data.models`
    inputs_where: Callable[[PyTree[Array]], Array] = None  # In `data.states`
    states_where: Callable[[PyTree[Array]], Array] = None  # In `data.states`
    
    def compute(
        self,
        data: AnalysisInputData,
        **kwargs,
    ):
        def get_jac_func(position, sisu, func):
            return jax.jacobian(get_ss_rnn_func(position, sisu, func, self.key))

        def get_jacobian(position, sisu, fp, func):
            return get_jac_func(position, sisu, func)(fp)
        
        if None in (self.func_where, self.inputs_where, self.states_where):
            raise ValueError("Must specify `func_where`, `inputs_where`, and `states_where`")

        get_jac = eqx.filter_vmap(get_jacobian, in_axes=(None, None, 0, 0)) # Over replicates

        if isinstance(goals_pos_for_jac, Array) and len(goals_pos_for_jac.shape) == 2:
            get_jac = eqx.filter_vmap(get_jac, in_axes=(0, None, 1, None)) # Over positions

        def _get_jac_by_sisu(func, fps_by_sisu):
            return LDict.of('sisu')({
                sisu: get_jac(
                    goals_pos_for_jac, sisu, fps, first(func, is_leaf=is_module)
                )
                for sisu, fps in fps_by_sisu.items()
            })

        jacobians = jt.map(
            _get_jac_by_sisu,
            rnn_funcs, fps_grid,
            is_leaf=LDict.is_of('sisu')
        )

        #! This probably should not be here, but handled by prep/post ops
        jacobians_stacked = jt.map(
            lambda d: jtree.stack(list(d.values())),
            jacobians,
            is_leaf=LDict.is_of('sisu'),
        )
        

class Hessians(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    func_where: Callable[[PyTree[Module]], Callable[[Array, Array], Array]] = None  # In `data.models`
    inputs_where: Callable[[PyTree[Array]], Array] = None  # In `data.states`
    states_where: Callable[[PyTree[Array]], Array] = None  # In `data.states`
    
    def compute(
        self,
        data: AnalysisInputData,
        **kwargs,
    ):
        ...


#! TODO: Finish gutting this
class SteadyStateJacobians(AbstractAnalysis):
    """Compute Jacobians and their eigendecomposition at steady-state FPs."""

    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict(
        fps_results=NNSteadyStateFPs,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    origin_only: bool = False
    key: PRNGKeyArray = field(default_factory=lambda: jr.PRNGKey(0))

    def compute(
        self,
        data: AnalysisInputData,
        fps_results: TreeNamespace,
        hps_common: TreeNamespace,
        **kwargs,
    ):        
        
        task_leaf = jt.leaves(data.tasks, is_leaf=is_module)[0]
        goals_pos = task_leaf.validation_trials.targets["mechanics.effector.pos"].value[:, -1]
        
        fps_grid = jnp.moveaxis(fps_results.fps, 0, 1)

        rnn_funcs = jt.map(lambda m: m.step.net.hidden, data.models, is_leaf=is_module)



class FPsInPCSpace(AbstractAnalysis):
    """Plot fixed points in PC space."""

    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict(
        fps_results=NNSteadyStateFPs,
        pca_results=StatesPCA,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    def make_figs(
        self,
        data: AnalysisInputData,
        result: PyTree,
        fps_results: TreeNamespace,
        pca_results: TreeNamespace,
        # models: PyTree,
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