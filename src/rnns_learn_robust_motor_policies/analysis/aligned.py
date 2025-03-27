from collections.abc import Callable
from copy import deepcopy
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional, Literal as L

from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree

import feedbax.plotly as fbp
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.state_utils import get_aligned_vars, get_pos_endpoints
from rnns_learn_robust_motor_policies.config import PLOTLY_CONFIG
from rnns_learn_robust_motor_policies.hyperparams import flat_key_to_where_func
from rnns_learn_robust_motor_policies.plot_utils import get_label_str
from rnns_learn_robust_motor_policies.types import (
    RESPONSE_VAR_LABELS, 
    Responses, 
    LDict,
    TreeNamespace,
)


WHERE_VARS_TO_ALIGN = lambda states, pos_endpoints: Responses(
    # Positions with respect to the origin
    states.mechanics.effector.pos - pos_endpoints[0][..., None, :],
    states.mechanics.effector.vel,
    states.efferent.output,
)


class AlignedVars(AbstractAnalysis):
    """Align spatial variable (e.g. position and velocity) coordinates with the reach direction."""
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = None
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

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
        
        
class AlignedTrajectories(AbstractAnalysis):
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "small"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        var_labels=RESPONSE_VAR_LABELS,
        axes_labels=('x', 'y'),
        # mode='std',
        mean_trajectory_line_width=3,
        # n_curves_max=n_curves_max,
        darken_mean=PLOTLY_CONFIG.mean_lighten_factor,
        n_curves_max=20,
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
    colorscale_key: Optional[str] = None 
    colorscale_axis: Optional[int] = None

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        aligned_vars,
        hps_common,
        colorscales,
        **kwargs,
    ):
        fig_params = deepcopy(self.fig_params)

        if fig_params.legend_title is None and self.colorscale_key is not None:
            fig_params.legend_title = get_label_str(self.colorscale_key)
            
        try:
            fig_params.legend_labels = flat_key_to_where_func(self.colorscale_key)(hps_common)
        except:
            pass

        figs = jt.map(
            partial(
                fbp.trajectories_2D,
                colorscale=colorscales[self.colorscale_key],
                colorscale_axis=self.colorscale_axis,
                curves_mode='lines',
                **fig_params,
            ),
            aligned_vars[self.variant],
            is_leaf=is_type(Responses),
        )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, hps_common, **kwargs):
        return dict(
            # n=min(self.n_curves_max, hps_common.eval_n * n_replicates_included[train_pert_std] * self.n_conditions)
        )