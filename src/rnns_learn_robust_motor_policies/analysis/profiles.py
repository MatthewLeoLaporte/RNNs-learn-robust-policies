
from collections.abc import Callable
from types import MappingProxyType
from typing import ClassVar, Optional

from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree
import numpy as np
import plotly.graph_objects as go

import feedbax.plotly as fbp
from jax_cookbook import is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import AlignedVars
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.plot_utils import get_label_str
from rnns_learn_robust_motor_policies.tree_utils import move_ldict_level_above, tree_level_labels
from rnns_learn_robust_motor_policies.types import Responses
from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.types import LDict


class Profiles(AbstractAnalysis):
    """Generates aligned profile figures.
    """
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    fig_params: FigParamNamespace = DefaultFigParamNamespace(
        mode='std', # or 'curves'
        n_std_plot=1,
        layout_kws=dict(
            width=600,
            height=400,
            legend_tracegroupgap=1,
        ),
    )
    vrect_kws: Optional[Callable[[TreeNamespace], dict]] = None
        

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        aligned_vars,
        colors,
        **kwargs,
    ):
        level_labels = tree_level_labels(aligned_vars[self.variant])

        result = move_ldict_level_above('var', level_labels[-2], aligned_vars[self.variant])

        def _get_fig(fig_data, i, label, colors):                      
            level_colors = list(colors[fig_data.label].dark.values())
            return fbp.profiles(
                jtree.take(fig_data, i, -1),
                varname=label.capitalize(),
                legend_title=get_label_str(fig_data.label),
                hline=dict(y=0, line_color="grey"),
                colors=level_colors,
                # stride_curves=500,
                # curves_kws=dict(opacity=0.7),
                **self.fig_params,
            )

        figs = jt.map(
            lambda results_by_var: LDict.of('var')({
                var_label: LDict.of("direction")({
                    direction_label: _get_fig(
                        var_data, coord_idx, f"{direction_label} {var_label}", colors
                    )
                    for coord_idx, direction_label in enumerate(("parallel", "orthogonal"))
                })
                for var_label, var_data in results_by_var.items()
            }),
            result,
            is_leaf=LDict.is_of('var'),
        )

        if self.vrect_kws is not None:
            # Allows the vrect to parametrize by the outer tree levels only
            jt.map(
                lambda figs_by_var, hps: jt.map(
                    lambda fig: fig.add_vrect(
                        **self.vrect_kws(jt.leaves(hps, is_leaf=is_type(TreeNamespace))[0])
                    ),
                    figs_by_var,
                    is_leaf=is_type(go.Figure),
                ),
                figs,
                data.hps[self.variant],
                is_leaf=LDict.is_of('var'),
            )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, aligned_vars, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(aligned_vars[self.variant])[0].shape[:-2]))
        )