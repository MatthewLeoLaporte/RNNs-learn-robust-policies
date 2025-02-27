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

from feedbax.intervene import add_intervenors, schedule_intervenor
from feedbax.task import TrialSpecDependency
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import Aligned_IdxTrial, AlignedVars, plot_condition_trajectories
from rnns_learn_robust_motor_policies.analysis.aligned import Aligned_IdxPertAmp
from rnns_learn_robust_motor_policies.analysis.aligned import Aligned_IdxTrainStd
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.center_out import CenterOutByEval
from rnns_learn_robust_motor_policies.analysis.center_out import CenterOutSingleEval
from rnns_learn_robust_motor_policies.analysis.center_out import CenterOutByReplicate
from rnns_learn_robust_motor_policies.analysis.disturbance import DISTURBANCE_FUNCS
from rnns_learn_robust_motor_policies.analysis.measures import MEASURE_LABELS, output_corr
from rnns_learn_robust_motor_policies.analysis.measures import Measures
from rnns_learn_robust_motor_policies.analysis.profiles import VelocityProfiles
from rnns_learn_robust_motor_policies.analysis.state_utils import get_constant_task_input, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.colors import COLORSCALES
from rnns_learn_robust_motor_policies.constants import INTERVENOR_LABEL, POS_ENDPOINTS_ALIGNED
from rnns_learn_robust_motor_policies.misc import camel_to_snake
from rnns_learn_robust_motor_policies.plot import add_endpoint_traces, get_violins
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace, tree_subset_dict_level
from rnns_learn_robust_motor_policies.types import (
    ContextInputDict,
    MeasureDict,
    PertAmpDict,
    Responses,
    TrainingMethodDict,
    TrainStdDict, 
)


"""Labels of measures to include in the analysis."""
MEASURE_KEYS = (
    "max_parallel_vel_forward",
    "max_orthogonal_vel_signed",
    "max_orthogonal_vel_left",
    # "max_orthogonal_vel_right",  # -2
    "largest_orthogonal_distance",
    "max_orthogonal_distance_left",
    "sum_orthogonal_distance",
    "sum_orthogonal_distance_abs",
    "end_position_error",
    # "end_velocity_error",  # -1
    "max_parallel_force_forward",
    # "sum_parallel_force",  # -2
    # "max_orthogonal_force_right",  # -1
    "sum_orthogonal_force_abs",
    "max_net_force",
    "sum_net_force",
)

COLOR_FUNCS = dict(
    context_inputs=lambda hps: hps.context_input,
)


#! This might need to be changed; note that in the original 1-1 `evaluate_all_states`,
#! we `tree_map` first over tasks and then over models, but in 2-1 we do the opposite.
eval_func = vmap_eval_ensemble


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = DISTURBANCE_FUNCS[hps.disturbance.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.disturbance.type}")
    
    disturbance_amplitudes = hps.disturbance.amplitude
    
    tasks_by_amp, _ = jtree.unzip(jt.map( # over disturbance amplitudes
        lambda disturbance_amplitude: schedule_intervenor(  # (implicitly) over train stds
            task_base, jt.leaves(models_base, is_leaf=is_module)[0],
            lambda model: model.step.mechanics,
            disturbance(disturbance_amplitude),
            label=INTERVENOR_LABEL,
            default_active=False,
        ),
        PertAmpDict(zip(disturbance_amplitudes, disturbance_amplitudes)),
    ))
    
    all_tasks = ContextInputDict({
        context_input: jt.map(
            lambda task: eqx.tree_at( 
                lambda task: task.input_dependencies,
                task, 
                {
                    'context': TrialSpecDependency(
                        get_constant_task_input(
                            context_input, 
                            hps.model.n_steps - 1, 
                            task.n_validation_trials,
                        )
                    )
                },
            ),
            tasks_by_amp,
            is_leaf=is_module,
        )
        for context_input in hps.context_inputs
    })
    
    all_models = jt.map(
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
    
    # all_models = move_level_to_outside(all_models, TrainingMethodDict)
    
    return all_tasks, all_models, hps


class Aligned_IdxContextInput(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: Optional[str] = "small"
    conditions: tuple[str, ...] = ()
    # n_conditions: int  # all_tasks['small'][disturbance_amplitude].n_validation_trials
    n_curves_max: int = 20
    
    def make_figs(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        aligned_vars,
        **kwargs,
    ):
        plot_vars_stacked = jt.map(
            lambda d: jtree.stack(d.values()),
            aligned_vars[self.variant],
            is_leaf=is_type(ContextInputDict),
        )
        
        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                colorscale=COLORSCALES['context_inputs'],
                colorscale_axis=0,
                # stride=stride,
                legend_title="Context input",
                legend_labels=hps[self.variant].context_input,
                curves_mode='lines',
                var_endpoint_ms=0,
                scatter_kws=dict(line_width=0.5, opacity=0.3),
                # ref_endpoints=(pos_endpoints, None),
            ),
            plot_vars_stacked,
            is_leaf=is_type(Responses),
        )
        
        assert self.variant is not None, "How is it that this `variant` field is None?"

        for fig in jt.leaves(figs, is_leaf=is_type(go.Figure)):
            add_endpoint_traces(fig, POS_ENDPOINTS_ALIGNED[self.variant], xaxis='x1', yaxis='y1')
        
        return figs
        
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, disturbance_std, **kwargs):
        return dict(
            # n=min(self.n_curves_max, n_replicates_included[disturbance_std] * self.n_conditions)
        )


#! I think this is replaceable with `aligned.AlignedIdxTrainStd` if we move the plot var stacking to a separate analysis
class Aligned_IdxTrainStd_PerContext(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: Optional[str] = "small"
    conditions: tuple[str, ...] = ()
    # n_conditions: int  # all_tasks['small'][disturbance_amplitude].n_validation_trials
    n_curves_max: int = 20

    def make_figs(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        aligned_vars,
        **kwargs,
    ):
        plot_vars_stacked = jt.map(
            lambda d: jtree.stack(d.values()),
            aligned_vars[self.variant],
            is_leaf=is_type(ContextInputDict),
        )
        
        plot_vars = jt.map(
            lambda d: {
                context_input: jtree.stack(
                    jt.map(lambda arr: arr[hps.context_input.index(context_input)], d).values()
                )
                for context_input in hps.context_input
            },
            plot_vars_stacked,
            is_leaf=is_type(TrainStdDict),
        )
        
        figs = jt.map(
            partial(
                plot_condition_trajectories,
                colorscale=COLORSCALES['disturbance_std'],
                colorscale_axis=0,
                legend_title="Train<br>field std.",
                legend_labels=hps[self.variant].load.disturbance.std,
                curves_mode='lines',
                var_endpoint_ms=0,
                scatter_kws=dict(line_width=0.5, opacity=0.3),
                # ref_endpoints=(pos_endpoints['full'], None),
            ),
            plot_vars,
            is_leaf=is_type(Responses),
        )
        
        assert self.variant is not None, "How is it that this `variant` field is None?"
        
        for fig in jt.leaves(figs, is_leaf=is_type(go.Figure)):
            add_endpoint_traces(fig, POS_ENDPOINTS_ALIGNED[self.variant], xaxis='x1', yaxis='y1')

        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], **kwargs):
        return dict(
            # TODO: The number of replicates (`n_replicates_included`) may vary with the disturbance train std!
            # n=min(self.n_curves_max, hps.eval_n * n_replicates_included)  #? n: pytree[int]
        )


class Measures_CompareTrainStdAndContext(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values=Measures,
    ))
    measure_keys: tuple[str, ...]
    variant: Optional[str] = "small"
    conditions: tuple[str, ...] = ()
    # n_conditions: int  # all_tasks['small'][disturbance_amplitude].n_validation_trials
    n_curves_max: int = 20    

    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        return dict(
            measure_values_lohi_disturbance_std=dict(
                measure_keys=self.measure_keys,
            )
        )

    def make_figs(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        measure_values,
        colors,
        **kwargs,
    ):
        # Move the disturbance amplitude level to the outside of each measure.
        #! Is this necessary now?
        measure_values = MeasureDict({
            measure_key: jtree.move_level_to_outside(measure_values, PertAmpDict)
            for measure_key, measure_values in measure_values.items()
        })

        figs = MeasureDict({
            measure_key: PertAmpDict({
                pert_amplitude: get_violins(
                    measure_values,
                    yaxis_title=MEASURE_LABELS[measure_key],
                    xaxis_title="Context input",
                    legend_title="Train std.",
                    colors=colors[self.variant]['disturbance_std']['dark'],
                    arr_axis_labels=["Evaluation", "Replicate", "Condition"],
                    zero_hline=True,
                    layout_kws=dict(
                        width=700,
                        height=500,
                        yaxis_fixedrange=False,
                        yaxis_autorange=True,
                        # yaxis_range=[0, measure_ranges_lohi[key][1]],
                    ),
                )
                for pert_amplitude, measure_values in measure_values[measure_key].items()
            })
            for measure_key in measure_values
        })
        
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, measure_values, **kwargs):
        return dict(
            # TODO: The number of replicates (`n_replicates_included`) may vary with the disturbance train std!
            n=int(np.prod(jt.leaves(measure_values)[0].shape)),
        )
                
        

ALL_ANALYSES = [
    Aligned_IdxContextInput(),
    Aligned_IdxTrainStd_PerContext(),
    Measures_CompareTrainStdAndContext(),
]