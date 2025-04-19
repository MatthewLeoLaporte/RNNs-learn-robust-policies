


from types import MappingProxyType
from typing import ClassVar, Optional

import jax.tree as jt
from jaxtyping import PyTree

from jax_cookbook import is_module
import numpy as np

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.measures import output_corr
from rnns_learn_robust_motor_policies.plot import get_violins
from rnns_learn_robust_motor_policies.types import LDict, TreeNamespace


class OutputWeightCorrelation(AbstractAnalysis):
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    def compute(
        self, 
        data: AnalysisInputData,
        **kwargs,
    ):
        activities = jt.map(
            lambda states: states.net.hidden,
            data.states[self.variant],
            is_leaf=is_module,
        )

        output_weights = jt.map(
            lambda models: models.step.net.readout.weight,
            data.models,
            is_leaf=is_module,
        )
        
        #! TODO: Generalize
        output_corrs = jt.map(
            lambda activities: LDict.of("train__pert__std")({
                train_std: output_corr(
                    activities[train_std], 
                    output_weights[train_std],
                )
                for train_std in activities
            }),
            activities,
            is_leaf=LDict.is_of("train__pert__std"),
        )
        
        return output_corrs
        
    def make_figs(
        self, 
        data: AnalysisInputData,
        *, 
        result, 
        colors, 
        **kwargs,
    ):
        #! TODO: Generalize
        assert result is not None
        fig = get_violins(
            result, 
            yaxis_title="Output correlation", 
            xaxis_title="Train field std.",
            colors=colors['pert__amp'].dark,
        )
        return fig

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape)),
            measure="output_correlation",
        )    