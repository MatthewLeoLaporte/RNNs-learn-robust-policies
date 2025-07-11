"""Temporary module for fixed-point-based analyses.

This is a refactoring of the logic in `notebooks/markdown/part2__fps_steady.md`
into the declarative analysis framework.
"""

from collections.abc import Callable, Sequence
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import plotly.graph_objects as go
from jaxtyping import Array, PyTree
from feedbax.bodies import SimpleFeedbackState

from jax_cookbook import is_module, is_type
from rnns_learn_robust_motor_policies.analysis.analysis import (
    AbstractAnalysis,
    AnalysisDependenciesType,
    AnalysisInputData,
    Data,
    DefaultFigParamNamespace,
    FigParamNamespace,
)
from rnns_learn_robust_motor_policies.analysis.fp_finder import (
    FPFilteredResults,
    FixedPointFinder,
    fp_adam_optimizer,
    take_top_fps,
)
from rnns_learn_robust_motor_policies.analysis.pca import StatesPCA
from rnns_learn_robust_motor_policies.misc import create_arr_df
from rnns_learn_robust_motor_policies.plot import plot_eigvals_df, plot_fp_pcs
from rnns_learn_robust_motor_policies.tree_utils import first, ldict_level_to_bottom
from rnns_learn_robust_motor_policies.types import LDict, TreeNamespace


# ########################################################################## #
# Helper functions from the notebook
# ########################################################################## #


def get_ss_network_input_with_context(pos, context, rnn_cell):
    input_star = jnp.zeros((rnn_cell.input_size,))
    # Set target and feedback inputs to the same position
    input_star = input_star.at[1:3].set(pos)
    input_star = input_star.at[5:7].set(pos)
    return input_star.at[0].set(context)


def get_ss_rnn_func_at_context(pos, context, rnn_cell, key):
    input_star = get_ss_network_input_with_context(pos, context, rnn_cell)
    def rnn_func(h):
        return rnn_cell(input_star, h, key=key)
    return rnn_func


def get_ss_rnn_fps(pos, rnn_cell, candidate_states, context, fpf_func, fp_tol, key):
    fps = fpf_func(
        get_ss_rnn_func_at_context(pos, context, rnn_cell, key),
        candidate_states,
        fp_tol,
    )
    return fps


def multi_vmap(func, in_axes_sequence, vmap_func=eqx.filter_vmap):
    """Given a sequence of `in_axes`, construct a nested vmap of `func`."""
    func_v = func
    for ax in in_axes_sequence:
        func_v = vmap_func(func_v, in_axes=ax)
    return func_v


def process_fps(all_fps):
    """Only keep FPs/replicates that meet criteria."""
    n_fps_meeting_criteria = jt.map(
        lambda fps: fps.counts['meets_all_criteria'],
        all_fps,
        is_leaf=is_type(FPFilteredResults),
    )

    satisfactory_replicates = jt.map(
        lambda n_matching_fps_by_context: jnp.all(
            jnp.stack(jt.leaves(n_matching_fps_by_context), axis=0),
            axis=0,
        ),
        n_fps_meeting_criteria,
        is_leaf=LDict.is_of('context_input'),
    )

    all_top_fps = take_top_fps(all_fps, n_keep=6)

    # Average over the top fixed points, to get a single one for each included replicate and
    # control input.
    fps_final = jt.map(
        lambda top_fps_by_context: jt.map(
            lambda fps: jnp.nanmean(fps, axis=-2),
            top_fps_by_context,
            is_leaf=is_type(FPFilteredResults),
        ),
        all_top_fps,
        is_leaf=LDict.is_of('context_input'),
    )

    return TreeNamespace(
        fps=fps_final,
        all_top_fps=all_top_fps,
        n_fps_meeting_criteria=n_fps_meeting_criteria,
        satisfactory_replicates=satisfactory_replicates,
    )


# ########################################################################## #
# Analysis classes
# ########################################################################## #


class SteadyStateFPs(AbstractAnalysis):
    """Find steady-state fixed points of the RNN."""

    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    cache_result: bool = True
    fp_tol: float = 1e-5
    unique_tol: float = 0.025
    outlier_tol: float = 1.0
    stride_candidates: int = 16

    def compute(
        self,
        data: AnalysisInputData,
        hps_common: TreeNamespace,
        **kwargs,
    ):
        key = jr.PRNGKey(0) #! This should be passed in.

        fp_optimizer = fp_adam_optimizer()
        fpfinder = FixedPointFinder(fp_optimizer)
        fpf_func = partial(
            fpfinder.find_and_filter,
            outlier_tol=self.outlier_tol,
            unique_tol=self.unique_tol,
            key=key,
        )

        models, states = [
            ldict_level_to_bottom("context_input", tree, is_leaf=is_module)
            for tree in (data.models, data.states)
        ]

        rnn_funcs = jt.map(
            lambda model: model.step.net.hidden,
            models,
            is_leaf=is_module,
        )

        task_leaf = jt.leaves(data.tasks, is_leaf=is_module)[0]
        positions = task_leaf.validation_trials.targets["mechanics.effector.pos"].value[:, -1]

        get_fps_partial = partial(
            get_ss_rnn_fps,
            fpf_func=fpf_func,
            fp_tol=self.fp_tol,
            key=key,
        )

        if isinstance(positions, Array) and len(positions.shape) == 2:
            get_fps_func = multi_vmap(
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
            lambda func, candidates_by_context: LDict.of('context_input')({
                context_input: get_fps_func(
                    positions,
                    first(func, is_leaf=is_module),
                    candidates_by_context[context_input],
                    context_input,
                )
                for context_input in hps_common.context_input
            }),
            rnn_funcs, candidates,
            is_leaf=LDict.is_of('context_input'),
        )

        return process_fps(all_fps)


class SteadyStateJacobians(AbstractAnalysis):
    """Compute Jacobians and their eigendecomposition at steady-state FPs."""

    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict(
        fps_results=SteadyStateFPs,
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    best_replicate_only: bool = True
    origin_only: bool = False

    def compute(
        self,
        data: AnalysisInputData,
        fps_results: TreeNamespace,
        replicate_info: PyTree,
        hps_common: TreeNamespace,
        **kwargs,
    ):
        key = jr.PRNGKey(0) #! This should be passed in.
        
        task_leaf = jt.leaves(data.tasks, is_leaf=is_module)[0]
        goals_pos = task_leaf.validation_trials.targets["mechanics.effector.pos"].value[:, -1]
        
        fps_grid = jnp.moveaxis(fps_results.fps, 0, 1)

        rnn_funcs = jt.map(lambda m: m.step.net.hidden, data.models, is_leaf=is_module)

        if self.best_replicate_only:
            from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate
            fps_grid_for_jac = get_best_replicate(fps_grid, replicate_info=replicate_info, axis=0, keep_axis=True)
            rnn_funcs_for_jac = get_best_replicate(rnn_funcs, replicate_info=replicate_info, axis=0, keep_axis=True)
        else:
            fps_grid_for_jac = fps_grid
            rnn_funcs_for_jac = rnn_funcs

        if self.origin_only:
            origin_idx = hps_common.task.full.eval_grid_n ** 2 // 2
            idx = jnp.array([origin_idx])
            fps_grid_for_jac = jt.map(lambda x: x[:, idx], fps_grid_for_jac)
            goals_pos_for_jac = goals_pos[idx]
        else:
            goals_pos_for_jac = goals_pos
            
        def get_jac_func(position, context, func):
            return jax.jacobian(get_ss_rnn_func_at_context(position, context, func, key))

        def get_jacobian(position, context, fp, func):
            return get_jac_func(position, context, func)(fp)

        get_jac = eqx.filter_vmap(get_jacobian, in_axes=(None, None, 0, 0)) # Over replicates

        if isinstance(goals_pos_for_jac, Array) and len(goals_pos_for_jac.shape) == 2:
            get_jac = eqx.filter_vmap(get_jac, in_axes=(0, None, 1, None)) # Over positions

        def _get_jac_by_context(func, fps_by_context):
            return LDict.of('context_input')({
                context_input: get_jac(
                    goals_pos_for_jac, context_input, fps, first(func, is_leaf=is_module)
                )
                for context_input, fps in fps_by_context.items()
            })

        jacobians = jt.map(
            _get_jac_by_context,
            rnn_funcs_for_jac, fps_grid_for_jac,
            is_leaf=LDict.is_of('context_input')
        )

        jacobians_stacked = jt.map(
            lambda d: jtree.stack(list(d.values())),
            jacobians,
            is_leaf=LDict.is_of("context_input"),
        )
        
        eig_cpu = jax.jit(
            lambda *a, **kw: tuple(jax.lax.linalg.eig(*a, **kw)),
            device=jax.devices('cpu')[0],
        )

        eigvals, _, _ = jtree.unzip(jt.map(eig_cpu, jacobians_stacked))

        return TreeNamespace(
            jacobians=jacobians_stacked,
            eigvals=eigvals,
        )


class JacobianEigenspectra(AbstractAnalysis):
    """Plot eigendecomposition of Jacobians."""

    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict(
        jac_results=SteadyStateJacobians
    ))
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    contexts_to_plot: Optional[Sequence[float]] = None

    def make_figs(
        self, 
        data: AnalysisInputData, 
        result: PyTree, 
        jac_results: TreeNamespace,
        hps_common: TreeNamespace,
        colors: PyTree,
        **kwargs
    ) -> PyTree[go.Figure]:
        
        eigvals = jac_results.eigvals
        
        col_names = ['context', 'pos', 'replicate', 'eigenvalue']
        eigval_dfs = jt.map(
            lambda arr: create_arr_df(arr, col_names=col_names).astype({'context': 'str', 'replicate': 'str'}),
            eigvals
        )
        
        plot_func_partial = partial(
            plot_eigvals_df,
            marginals='box',
            color='context',
            trace_kws=dict(marker_size=2.5),
            scatter_kws=dict(opacity=1),
            layout_kws=dict(
                legend_title='Context input',
                legend_itemsizing='constant',
                xaxis_title='Re',
                yaxis_title='Im',
            ),
        )

        figs = jt.map(
            lambda df: plot_func_partial(df, color_discrete_sequence=list(colors["context_input"].dark.values())),
            eigval_dfs
        )

        if self.contexts_to_plot is not None:
             contexts_plot = self.contexts_to_plot
        else:
             contexts_plot = hps_common.context_input

        def _update_trace_name(trace):
            non_data_trace_names = ['zerolines', 'boundary_circle', 'boundary_line']
            if trace.name is not None and trace.name not in non_data_trace_names:
                return trace.update(name=contexts_plot[int(trace.name)])
            else:
                return trace

        jt.map(
            lambda fig: fig.for_each_trace(_update_trace_name),
            figs,
            is_leaf=is_type(go.Figure),
        )

        return figs


class FPsInPCSpace(AbstractAnalysis):
    """Plot fixed points in PC space."""

    default_inputs: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict(
        fps_results=SteadyStateFPs,
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
        from rnns_learn_robust_motor_policies.analysis.state_utils import exclude_bad_replicates
        from rnns_learn_robust_motor_policies.misc import take_non_nan
        
        fps_grid = jnp.moveaxis(fps_results.fps, 0, 1)

        fps_grid_pre_pc = jt.map(
            lambda fps: take_non_nan(fps, axis=1),
            exclude_bad_replicates(fps_grid, replicate_info=replicate_info),
        )
        
        fps_grid_pc = pca_results.batch_transform(fps_grid_pre_pc)
        
        all_readout_weights = exclude_bad_replicates(
            jt.map(
                lambda model: model.step.net.readout.weight,
                # Weights do not depend on context input, take first
                jt.map(first, data.models, is_leaf=LDict.is_of('context_input')),
                is_leaf=is_module,
            ),
            replicate_info=replicate_info,
            axis=0,
        )

        all_readout_weights_pc = pca_results.batch_transform(all_readout_weights)
        
        def plot_fp_pcs_by_context(fp_pcs_by_context, readout_weights_pc):
            fig = go.Figure(
                layout=dict(
                    width=800, height=800,
                    scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                    legend=dict(title='Context input', itemsizing='constant', y=0.85),
                )
            )

            for context, fps_pc in fp_pcs_by_context.items():
                fig = plot_fp_pcs(
                    fps_pc, fig=fig, label=context,
                    colors=colors["context_input"].dark[context],
                )

            if readout_weights_pc is not None:
                fig.update_layout(
                    legend2=dict(title='Readout components', itemsizing='constant', y=0.45),
                )
                mean_base_fp_pc = jnp.mean(fp_pcs_by_context[min(hps_common.context_input)], axis=1)
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
            plot_fp_pcs_by_context,
            fps_grid_pc,
            all_readout_weights_pc,
            is_leaf=LDict.is_of("context_input"),
        ) 