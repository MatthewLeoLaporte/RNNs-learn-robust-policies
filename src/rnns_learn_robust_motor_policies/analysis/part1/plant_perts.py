from collections.abc import Callable
from functools import partial 
from types import MappingProxyType
from typing import ClassVar, Optional, Dict, Any

import equinox as eqx
from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree
import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm

from feedbax.intervene import CurlField, FixedField, add_intervenors, schedule_intervenor
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import Aligned_IdxTrial
from rnns_learn_robust_motor_policies.analysis.aligned import Aligned_IdxPertAmp
from rnns_learn_robust_motor_policies.analysis.aligned import Aligned_IdxTrainStd
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.center_out import CenterOutByEval
from rnns_learn_robust_motor_policies.analysis.center_out import CenterOutSingleEval
from rnns_learn_robust_motor_policies.analysis.center_out import CenterOutByReplicate
from rnns_learn_robust_motor_policies.analysis.measures import output_corr
from rnns_learn_robust_motor_policies.analysis.measures import Measures_LoHiSummary
from rnns_learn_robust_motor_policies.analysis.profiles import VelocityProfiles
from rnns_learn_robust_motor_policies.analysis.state_utils import orthogonal_field, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.constants import INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace
from rnns_learn_robust_motor_policies.misc import camel_to_snake
from rnns_learn_robust_motor_policies.plot import get_violins
from rnns_learn_robust_motor_policies.tree_utils import tree_subset_dict_level
from rnns_learn_robust_motor_policies.types import PertAmpDict, TrainStdDict


"""Labels of measures to include in the analysis."""
MEASURE_KEYS = (
    "max_parallel_vel_forward",
    "max_orthogonal_vel_left",
    "max_orthogonal_vel_right",
    "max_orthogonal_distance_left",
    "sum_orthogonal_distance",
    "end_position_error",
    "end_velocity_error",
    "max_parallel_force_forward",
    "sum_parallel_force",
    "max_orthogonal_force_right",  
    "sum_orthogonal_force_abs",
    "max_net_force",
    "sum_net_force",
)


DISTURBANCE_FUNCS = {
    'curl': lambda amplitude: CurlField.with_params(
        amplitude=amplitude,
    ),
    'constant': lambda amplitude: FixedField.with_params(
        scale=amplitude,
        field=orthogonal_field,
    ),
}

# Could specify extra colorscales needed for this analysis
COLOR_FUNCS = dict()


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = DISTURBANCE_FUNCS[hps.disturbance.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.disturbance.type}")

    # Insert the disturbance field component into each model
    models = jt.map(
        lambda models: add_intervenors(
            models,
            lambda model: model.step.mechanics,
            # The first key is the model stage where to insert the disturbance field;
            # `None` means prior to the first stage.
            # The field parameters will come from the task, so use an amplitude 0.0 placeholder.
            {None: {INTERVENOR_LABEL: disturbance(0.0)}},
        ),
        models_base,
        is_leaf=is_module,
    )

    # Assume a sequence of amplitudes is provided, as in the default config
    disturbance_amplitudes = hps.disturbance.amplitude
    # Construct tasks with different amplitudes of disturbance field
    all_tasks, all_models = jtree.unzip(jt.map(
        lambda disturbance_amplitude: schedule_intervenor(
            task_base, models,
            lambda model: model.step.mechanics,
            disturbance(disturbance_amplitude),
            label=INTERVENOR_LABEL,  
            default_active=False,
        ),
        PertAmpDict(zip(disturbance_amplitudes, disturbance_amplitudes)),
    ))
    
    return all_tasks, all_models, hps


# We aren't vmapping over any other variables, so this is trivial.
eval_func = vmap_eval_ensemble


class OutputWeightCorrelation(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    def compute(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        **kwargs,
    ):
        activities = jt.map(
            lambda states: states.net.hidden,
            states[self.variant],
            is_leaf=is_module,
        )

        output_weights = jt.map(
            lambda models: models.step.net.readout.weight,
            models,
            is_leaf=is_module,
        )

        output_corrs = jt.map(
            lambda activities: TrainStdDict({
                train_std: output_corr(
                    activities[train_std], 
                    output_weights[train_std],
                )
                for train_std in activities
            }),
            activities,
            is_leaf=is_type(TrainStdDict),
        )
        
        return output_corrs
        
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
        assert result is not None
        fig = get_violins(
            result, 
            yaxis_title="Output correlation", 
            xaxis_title="Train field std.",
            colors=colors[self.variant]['disturbance_amplitude']['dark'],
        )
        return fig

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape)),
            measure="output_correlation",
        )    


"""All the analyses to perform in this part."""
ALL_ANALYSES = [
    CenterOutByEval(),
    CenterOutSingleEval(i_trial=0),
    CenterOutByReplicate(i_trial=0),
    Aligned_IdxTrial(),
    Aligned_IdxPertAmp(),
    Aligned_IdxTrainStd(),
    VelocityProfiles(),
    # Measures_ByTrainStd(measure_keys=MEASURE_KEYS),
    # Measures_CompareReplicatesLoHi(measure_keys=MEASURE_KEYS),
    Measures_LoHiSummary(measure_keys=MEASURE_KEYS),
    # OutputWeightCorrelation(),
]