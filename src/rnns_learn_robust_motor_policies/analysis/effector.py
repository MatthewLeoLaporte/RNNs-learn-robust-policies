from collections.abc import Callable, Sequence
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import jax.tree as jt
from equinox import Module, field
from jaxtyping import PyTree
import plotly.graph_objects as go

from feedbax.task import AbstractTask
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.state_utils import get_pos_endpoints
from rnns_learn_robust_motor_policies.colors import COLORSCALES
from rnns_learn_robust_motor_policies.config import PLOTLY_CONFIG
from rnns_learn_robust_motor_policies.constants import REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.plot import add_endpoint_traces, plot_2d_effector_trajectories
from rnns_learn_robust_motor_policies.plot_utils import get_label_str
from rnns_learn_robust_motor_policies.types import TreeNamespace


MEAN_LIGHTEN_FACTOR = PLOTLY_CONFIG.mean_lighten_factor


class EffectorTrajectories(AbstractAnalysis):
    conditions: tuple[str, ...] = () # ('any_system_noise',)  # TODO: Skip this analysis, if only one eval
    variant: Optional[str] = "small"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        # legend_title="Reach direction",
        mean_exclude_axes=(),
        curves_mode='lines',
        mean_trajectory_line_width=2.5,
        legend_labels=None,
        darken_mean=MEAN_LIGHTEN_FACTOR,
        scatter_kws=dict(line_width=0.75, opacity=0.4),
    )
    colorscale_key: Optional[str] = None 
    colorscale_axis: Optional[int] = None
    pos_endpoints: bool = True
    straight_guides: bool = True

    def make_figs(
        self,
        data: AnalysisInputData,
        **kwargs,
    ):
        #! TODO: Add a general way to include callables in `fig_params`;
        #! however this probably requires passing `fig_params` to `AbstractAnalysis.make_figs`...
        if self.fig_params.legend_title is None:
            fig_params = self.fig_params | dict(legend_title=get_label_str(self.colorscale_key))
        else: 
            fig_params = self.fig_params

        figs = jt.map(
            partial(
                plot_2d_effector_trajectories,
                colorscale_key=self.colorscale_key,
                colorscale_axis=self.colorscale_axis,
                **fig_params,
            ),
            data.states[self.variant],
            is_leaf=is_module,
        )

        if self.pos_endpoints:
            #! See comment in `aligned.AlignedTrajectories`
            task_0 = jt.leaves(data.tasks[self.variant], is_leaf=is_type(AbstractTask))[0]
            pos_endpoints = get_pos_endpoints(task_0.validation_trials)

            if self.colorscale_key == 'reach_condition':
                colorscale = COLORSCALES['reach_condition']
            else:
                colorscale = None

            init_marker_kws = dict(color="rgb(25, 25, 25)")

            figs = jt.map(
                lambda fig: add_endpoint_traces(
                    fig, 
                    pos_endpoints, 
                    xaxis='x1', 
                    yaxis='y1', 
                    colorscale=colorscale,
                    init_marker_kws=init_marker_kws,
                    straight_guides=self.straight_guides,
                ),
                figs,
                is_leaf=is_type(go.Figure),
            )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, replicate_info, train_pert_std, **kwargs):
        return dict(
            i_replicate=replicate_info[train_pert_std]['best_replicates'][REPLICATE_CRITERION],
        )