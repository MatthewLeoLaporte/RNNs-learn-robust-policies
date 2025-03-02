from rnns_learn_robust_motor_policies.analysis.aligned import AlignedVars
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.types import Responses
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace
from rnns_learn_robust_motor_policies.types import LDict


import feedbax.plotly as fbp
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
from equinox import Module
from jax_cookbook import is_type
from jaxtyping import PyTree


from types import MappingProxyType
from typing import ClassVar, Optional


class VelocityProfiles(AbstractAnalysis):
    """Generates forward and lateral velocity profile figures."""
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()

    def compute(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        aligned_vars,
        **kwargs,
    ):
        return jt.map(
            lambda responses: responses.vel,
            aligned_vars[self.variant],
            is_leaf=is_type(Responses),
        )

    def make_figs(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        result,
        colors,
        **kwargs,
    ):
        figs = LDict.of("disturbance__amplitude")({
            # TODO: Once the mapping between custom dict types and their column names is automatic
            # (e.g. `PertVarDict` will simply map to 'pert_var'), we can construct a `DirectionDict`
            # ad hoc maybe
            disturbance_amplitude: LDict.of("label")({
                label: fbp.profiles(
                    jtree.take(result, i, -1)[disturbance_amplitude],
                    varname=f"{label} velocity",
                    legend_title="Train<br>field std.",
                    mode='std', # or 'curves'
                    n_std_plot=1,
                    hline=dict(y=0, line_color="grey"),
                    colors=list(colors['full']['disturbance_std']['dark'].values()),
                    # stride_curves=500,
                    # curves_kws=dict(opacity=0.7),
                    layout_kws=dict(
                        width=600,
                        height=400,
                        legend_tracegroupgap=1,
                    ),
                )
                for i, label in enumerate(("Forward", "Lateral"))
            })
            for disturbance_amplitude in hps[self.variant].disturbance.amplitude
        })
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape[:-2]))
        )