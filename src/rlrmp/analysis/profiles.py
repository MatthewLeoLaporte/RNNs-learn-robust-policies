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

from rlrmp.analysis.aligned import AlignedVars
from rlrmp.analysis.analysis import AbstractAnalysis, AnalysisDefaultInputsType, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rlrmp.plot_utils import get_label_str
from rlrmp.tree_utils import move_ldict_level_above, tree_level_labels
from rlrmp.types import Responses
from rlrmp.types import TreeNamespace
from rlrmp.types import LDict


class Profiles(AbstractAnalysis):
    """Generates figures for 
    
    Assumes that all the aligned vars have the same number of coordinates (i.e. 
    length of final array axis), and that these coordinates can be labeled similarly
    by `coord_labels`. For example, this is the case when we align position, velocity, 
    acceleration, and force in 2D. 
    """
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType(dict(
        vars=AlignedVars,
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
    var_level_label: str = "var"
    vrect_kws_func: Optional[Callable[[TreeNamespace], dict]] = None
    var_labels: Optional[dict[str, str]] = None  # e.g. for mapping "pos" to "position"
    coord_labels: Optional[tuple[str, str]] = ("parallel", "orthogonal")  # None for vars with single, unlabelled coordinates (e.g. deviations) 
    
    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        vars,
        colors,
        hps_common,
        **kwargs,
    ):
        def _get_fig(fig_data, i, coord_label, var_key, colors):      
            if self.var_labels is not None:
                var_label = self.var_labels[var_key]
            else:
                var_label = var_key
                
            if coord_label:
                label = f"{coord_label} {var_label}"
            else:
                label = var_label
                
            if isinstance(fig_data, LDict):            
                colors = list(colors[fig_data.label].dark.values())
                legend_title = get_label_str(fig_data.label)
            else:
                colors = None 
                legend_title = None

            return fbp.profiles(
                jtree.take(fig_data, i, -1),
                varname=label.capitalize(),
                legend_title=legend_title,
                hline=dict(y=0, line_color="grey"),
                colors=colors,
                # stride_curves=500,
                # curves_kws=dict(opacity=0.7),
                **self.fig_params,
            )
            
        def _get_figs_by_coord(var_key, var_data):
            if self.coord_labels is None:
                return _get_fig(var_data, 0, "", var_key, colors)
            else:
                return LDict.of("coord")({
                    coord_label: _get_fig(var_data, coord_idx, coord_label, var_key, colors)
                    for coord_idx, coord_label in enumerate(self.coord_labels)
                })
            
        figs = jt.map(
            lambda results_by_var: LDict.of(self.var_level_label)({
                var_key: _get_figs_by_coord(var_key, var_data)
                for var_key, var_data in results_by_var.items()
            }),
            vars[self.variant],
            is_leaf=LDict.is_of(self.var_level_label),
        )

        if self.vrect_kws_func is not None:
            vrect_kws = self.vrect_kws_func(hps_common)
            jt.map(
                lambda fig: fig.add_vrect(**vrect_kws),
                figs,
                is_leaf=is_type(go.Figure),
            )

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, vars, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(vars[self.variant])[0].shape[:-2]))
        )