from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional, Literal as L

from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree

import feedbax.plotly as fbp
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData
from rnns_learn_robust_motor_policies.analysis.state_utils import get_aligned_vars, get_pos_endpoints
from rnns_learn_robust_motor_policies.colors import COLORSCALES, MEAN_LIGHTEN_FACTOR
from rnns_learn_robust_motor_policies.hyperparams import flat_key_to_where_func
from rnns_learn_robust_motor_policies.plot_utils import get_label_str
from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.types import (
    RESPONSE_VAR_LABELS, 
    Responses, 
    LDict,
)


WHERE_VARS_TO_ALIGN = lambda states, pos_endpoints: Responses(
    # Positions with respect to the origin
    states.mechanics.effector.pos - pos_endpoints[0][..., None, :],
    states.mechanics.effector.vel,
    states.efferent.output,
)


class AlignedVars(AbstractAnalysis):
    """Align spatial variable (e.g. position and velocity) coordinates with the reach direction."""
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: Optional[str] = None
    conditions: tuple[str, ...] = ()

    def compute(
        self,
        data: AnalysisInputData,
        *,
        trial_specs,
        **kwargs,
    ):
        return jt.map(
            lambda specs, states_by_std: jt.map(
                lambda states: get_aligned_vars(
                    states, WHERE_VARS_TO_ALIGN, get_pos_endpoints(specs),
                ),
                states_by_std,
                is_leaf=is_module,
            ),
            trial_specs,
            data.states,
            is_leaf=is_module,
        )


plot_condition_trajectories = partial(
    fbp.trajectories_2D,
    var_labels=RESPONSE_VAR_LABELS,
    axes_labels=('x', 'y'),
    # mode='std',
    mean_trajectory_line_width=3,
    # n_curves_max=n_curves_max,
    darken_mean=MEAN_LIGHTEN_FACTOR,
    layout_kws=dict(
        width=900,
        height=400,
        legend_tracegroupgap=1,
        margin_t=75,
    ),
    scatter_kws=dict(
        line_width=1,
        opacity=0.6,
    ),
)
        
        
class AlignedTrajectories(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: Optional[str] = "small"
    conditions: tuple[str, ...] = ()
    stack_by: Optional[str] = None
    colorscale_axis: int = 0
    colorscale_key: Optional[str] = None
    legend_title: Optional[str] = None
    n_curves_max: int = 20

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        aligned_vars,
        hps_common,
        **kwargs,
    ):
        if self.stack_by is not None:
            vars_plot = jt.map(
                lambda d: jtree.stack(list(d.values())),
                aligned_vars[self.variant],
                is_leaf=LDict.is_of(self.stack_by),
            )
            colorscale_key = self.stack_by 
        else:
            vars_plot = aligned_vars[self.variant]
            if self.colorscale_key is None:
                raise ValueError("both colorscale_key and stack_by are None")
            colorscale_key = self.colorscale_key
        
        if self.legend_title is None:
            legend_title = get_label_str(colorscale_key)
        else:
            legend_title = self.legend_title 
            
        try:
            legend_labels = flat_key_to_where_func(colorscale_key)(hps_common)
        except:
            legend_labels = None

        figs = jt.map(
            partial(
                plot_condition_trajectories,
                colorscale=COLORSCALES[colorscale_key],
                colorscale_axis=self.colorscale_axis,
                legend_title=legend_title,
                legend_labels=legend_labels,
                curves_mode='lines',
            ),
            vars_plot,
            is_leaf=is_type(Responses),
        )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, hps_common, **kwargs):
        return dict(
            # n=min(self.n_curves_max, hps_common.eval_n * n_replicates_included[train_pert_std] * self.n_conditions)
        )