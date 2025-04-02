from collections.abc import Callable, Sequence
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import jax.tree as jt
from equinox import Module
from jaxtyping import PyTree
import plotly.graph_objects as go

from feedbax.task import AbstractTask
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.state_utils import BestReplicateStates, get_pos_endpoints
from rnns_learn_robust_motor_policies.colors import COLORSCALES
from rnns_learn_robust_motor_policies.config import PLOTLY_CONFIG
from rnns_learn_robust_motor_policies.constants import REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.plot import add_endpoint_traces, plot_2d_effector_trajectories
from rnns_learn_robust_motor_policies.types import TreeNamespace


MEAN_LIGHTEN_FACTOR = PLOTLY_CONFIG.mean_lighten_factor


class Effector_ByEval(AbstractAnalysis):
    conditions: tuple[str, ...] = ('any_system_noise',)  # TODO: Skip this analysis, if only one eval
    variant: Optional[str] = "small"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        best_replicate_states=BestReplicateStates,
    ))
    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        legend_title="Reach direction",
        mean_exclude_axes=(),
    )
    legend_labels: Optional[Sequence | Callable] = None
    colorscale_axis: int = 1
    colorscale_key: str = "reach_condition"
    pos_endpoints: bool = True

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        best_replicate_states,
        hps_common,
        **kwargs,
    ):
        plot_states = best_replicate_states[self.variant]
        
        if self.fig_params.legend_labels is None:
            if isinstance(self.legend_labels, Callable):
                legend_labels = self.legend_labels(data.hps, hps_common)
            else:
                legend_labels = self.legend_labels
        else:
            legend_labels = self.fig_params.legend_labels

        if self.pos_endpoints:
            pos_endpoints = jt.map(
                #? Will this always be `validation_trials`? For this project, perhaps.
                lambda task: get_pos_endpoints(task.validation_trials),
                data.tasks[self.variant],
                is_leaf=is_type(AbstractTask),
            )
        else: 
            pos_endpoints = None

        figs = jt.map(
            partial(
                plot_2d_effector_trajectories,
                legend_title=self.fig_params.legend_title,
                legend_labels=legend_labels,
                colorscale_key=self.colorscale_key,
                curves_mode='lines',
                colorscale_axis=self.colorscale_axis,
                mean_trajectory_line_width=2.5,
                mean_exclude_axes=self.fig_params.mean_exclude_axes,
                darken_mean=MEAN_LIGHTEN_FACTOR,
                scatter_kws=dict(line_width=0.5),
            ),
            plot_states,
            is_leaf=is_module,
        )

        if self.pos_endpoints:
            figs = jt.map(
                lambda fig, pos_endpoints: add_endpoint_traces(
                    fig, 
                    pos_endpoints, 
                    xaxis='x1', 
                    yaxis='y1', 
                    colorscale=COLORSCALES[self.colorscale_key],
                ),
                figs,
                pos_endpoints,
                is_leaf=is_type(go.Figure),
            )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, replicate_info, train_pert_std, **kwargs):
        return dict(
            i_replicate=replicate_info[train_pert_std]['best_replicates'][REPLICATE_CRITERION],
        )


class Effector_SingleEval(AbstractAnalysis):
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "small"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        best_replicate_states=BestReplicateStates,
    ))
    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        legend_title="Reach direction",
    )
    colorscale_key: str = "reach_condition"
    colorscale_axis: int = 0
    i_trial: int = 0

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        best_replicate_states,
        **kwargs,
    ):
        plot_states = best_replicate_states[self.variant]
        # Assume replicate has already been indexed out, and trial is the first axis
        plot_states_i = jtree.take(plot_states, self.i_trial, 0)

        figs = jt.map(
            partial(
                plot_2d_effector_trajectories,
                legend_title=self.fig_params.legend_title,
                colorscale_key=self.colorscale_key,
                mode='markers+lines',
                colorscale_axis=self.colorscale_axis,
                ms=3,
                scatter_kws=dict(line_width=0.75),
            ),
            plot_states_i,
            is_leaf=is_module,
        )

        # add_endpoint_traces(
        #     fig, pos_endpoints[self.variant], xaxis='x1', yaxis='y1', colorscale='phase'
        # )

        return figs


class Effector_ByReplicate(AbstractAnalysis):
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "small"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        legend_title="Reach direction",
    )
    colorscale_key: str = "reach_condition"
    colorscale_axis: int = 1
    i_trial: int = 0

    def make_figs(
        self,
        data: AnalysisInputData,
        **kwargs,
    ):
        plot_states = jtree.take(data.states[self.variant], self.i_trial, 0)

        figs = jt.map(
            partial(
                plot_2d_effector_trajectories,
                legend_title=self.fig_params.legend_title,
                colorscale_key=self.colorscale_key,
                curves_mode='lines',
                colorscale_axis=self.colorscale_axis,
                mean_trajectory_line_width=2.5,
                darken_mean=MEAN_LIGHTEN_FACTOR,
                scatter_kws=dict(line_width=0.75),
            ),
            plot_states,
            is_leaf=is_module,
        )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, train_pert_std, **kwargs):
        return dict(
            # n=n_replicates_included[train_pert_std],
        )


