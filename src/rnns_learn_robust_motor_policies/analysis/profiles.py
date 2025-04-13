
from collections.abc import Callable
from types import MappingProxyType
from typing import ClassVar, Optional

import jax.tree as jt
import numpy as np
from equinox import Module
from jaxtyping import PyTree

import feedbax.plotly as fbp
from jax_cookbook import is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import AlignedVars
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.plot_utils import get_label_str
from rnns_learn_robust_motor_policies.types import Responses
from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.types import LDict


class VelocityProfiles(AbstractAnalysis):
    """Generates forward and lateral velocity profile figures.
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

    def compute(
        self,
        data: AnalysisInputData,
        *,
        aligned_vars,
        **kwargs,
    ):
        result = jt.map(
            lambda responses: responses.vel,
            aligned_vars[self.variant],
            is_leaf=is_type(Responses),
        )
        return result
        

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        result,
        colors,
        **kwargs,
    ):
        def _get_fig(fig_data, i, label, colors):                      
            return fbp.profiles(
                jtree.take(fig_data, i, -1),
                varname=f"{label} velocity".capitalize(),
                legend_title=get_label_str(fig_data.label),
                hline=dict(y=0, line_color="grey"),
                colors=list(colors[fig_data.label].dark.values()),
                # stride_curves=500,
                # curves_kws=dict(opacity=0.7),
                **self.fig_params,
            )
        
        figs = LDict.of(result.label)({
            value: LDict.of("direction")({
                label: _get_fig(result[value], coord_idx, label, colors)
                for coord_idx, label in enumerate(("forward", "lateral"))
            })
            for value in result.keys()
        })
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape[:-2]))
        )