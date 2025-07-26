from rlrmp.analysis.analysis import AbstractAnalysis, AnalysisDefaultInputsType, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace, RequiredInput
from rlrmp.misc import create_arr_df
from rlrmp.plot import plot_eigvals_df
from rlrmp.types import TreeNamespace


import jax
import jax.tree as jt
import jax_cookbook.tree as jtree
import plotly.graph_objects as go
from jax_cookbook import is_type
from jaxtyping import PyTree


from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional


class Eigendecomposition(AbstractAnalysis):
    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType(dict(
        matrices=RequiredInput,
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
        *,
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
        *,
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