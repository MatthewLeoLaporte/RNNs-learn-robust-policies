from collections.abc import Callable
from functools import partial 
from types import MappingProxyType
from typing import ClassVar, Optional, Dict, Any, Literal as L

import equinox as eqx
from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree
import numpy as np
import plotly.graph_objects as go

from feedbax.intervene import add_intervenors, schedule_intervenor
from feedbax.task import TrialSpecDependency
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.aligned import AlignedVars, plot_condition_trajectories
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, FigParams
from rnns_learn_robust_motor_policies.analysis.disturbance import PLANT_INTERVENOR_LABEL, PLANT_PERT_FUNCS
from rnns_learn_robust_motor_policies.analysis.measures import MEASURE_LABELS
from rnns_learn_robust_motor_policies.analysis.measures import Measures
from rnns_learn_robust_motor_policies.analysis.state_utils import get_constant_task_input, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.constants import POS_ENDPOINTS_ALIGNED
from rnns_learn_robust_motor_policies.plot import add_endpoint_traces, get_violins
from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.types import (
    LDict,
    Responses,
)


ID = "2-1"


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
    context_input=lambda hps: hps.context_input,
)


#! This might need to be changed; note that in the original 1-1 `evaluate_all_states`,
#! we `tree_map` first over tasks and then over models, but in 2-1 we do the opposite.
eval_func = vmap_eval_ensemble


def setup_eval_tasks_and_models(task_base, models_base, hps):
    try:
        disturbance = PLANT_PERT_FUNCS[hps.pert.type]
    except KeyError:
        raise ValueError(f"Unknown disturbance type: {hps.pert.type}")
    
    pert_amps = hps.pert.amp
    
    tasks_by_amp, _ = jtree.unzip(jt.map( # over disturbance amplitudes
        lambda pert_amp: schedule_intervenor(  # (implicitly) over train stds
            task_base, jt.leaves(models_base, is_leaf=is_module)[0],
            lambda model: model.step.mechanics,
            disturbance(pert_amp),
            label=PLANT_INTERVENOR_LABEL,
            default_active=False,
        ),
        LDict.of("pert__amp")(
            dict(zip(pert_amps, pert_amps)),
        )
    ))
    
    tasks = LDict.of("context_input")({
        context_input: jt.map(
            lambda task: eqx.tree_at( 
                lambda task: task.input_dependencies,
                task, 
                {
                    'context': TrialSpecDependency(get_constant_task_input(
                            context_input, 
                            hps.model.n_steps - 1, 
                            task.n_validation_trials,
                    ))
                },
            ),
            tasks_by_amp,
            is_leaf=is_module,
        )
        for context_input in hps.context_input
    })
    
    models_by_std = jt.map(
        lambda models: add_intervenors(
            models,
            lambda model: model.step.mechanics,
            # The first key is the model stage where to insert the disturbance field;
            # `None` means prior to the first stage.
            # The field parameters will come from the task, so use an amplitude 0.0 placeholder.
            {None: {PLANT_INTERVENOR_LABEL: disturbance(0.0)}},
        ),
        models_base,
        is_leaf=is_module,
    )
    
    # The outer levels of `all_models` have to match those of `all_tasks`
    models, hps = jtree.unzip(jt.map(
        lambda _: (models_by_std, hps), tasks, is_leaf=is_module
    ))
    
    return tasks, models, hps, None


class Aligned_IdxContextInput(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: Optional[str] = "small"
    conditions: tuple[str, ...] = ()
    _pre_ops: tuple[tuple[str, Callable]] = ()
    fig_params: FigParams = FigParams(
        n_curves_max=20,
    )
    # n_conditions: int  # all_tasks['small'][pert_amp].n_validation_trials
    
    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        aligned_vars,
        colorscales,
        **kwargs,
    ):
        plot_vars_stacked: LDict[float, Any] = jt.map(
            lambda d: jtree.stack(d.values()),
            aligned_vars[self.variant],
            is_leaf=LDict.is_of("context_input"),
        )
        
        # Context inputs do not depend on pert amp
        context_inputs = jt.leaves(hps, is_leaf=is_type(TreeNamespace))[0].context_input
        
        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                colorscale=colorscales['context_input'],
                colorscale_axis=0,
                # stride=stride,
                legend_title="Context input",
                legend_labels=context_inputs,
                curves_mode='lines',
                var_endpoint_ms=0,
                scatter_kws=dict(line_width=0.5, opacity=0.3),
                # ref_endpoints=(pos_endpoints, None),
            ),
            plot_vars_stacked,
            is_leaf=is_type(Responses),
        )
        
        assert self.variant is not None, "How is it that the `variant` field is None?"

        for fig in jt.leaves(figs, is_leaf=is_type(go.Figure)):
            add_endpoint_traces(fig, POS_ENDPOINTS_ALIGNED[self.variant], xaxis='x1', yaxis='y1')
        
        return figs
        
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, train_pert_std, **kwargs):
        return dict(
            # n=min(self.n_curves_max, n_replicates_included[train_pert_std] * self.n_conditions)
        )


#! I think this is replaceable with `aligned.AlignedIdxTrainStd` if we move the plot var stacking to a separate analysis
class Aligned_IdxTrainStd_PerContext(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: Optional[str] = "small"
    conditions: tuple[str, ...] = ()
    _pre_ops: tuple[tuple[str, Callable]] = ()
    fig_params: FigParams = FigParams(
        n_curves_max=20,
    )
    # n_conditions: int  # all_tasks['small'][pert_amp].n_validation_trials

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        aligned_vars,
        colorscales,
        **kwargs,
    ):
        plot_vars_stacked: LDict[float, Any] = jt.map(
            lambda d: jtree.stack(d.values()),
            aligned_vars[self.variant],
            is_leaf=LDict.is_of("context_input"),
        )
        
        # Context inputs and train pert stds do not depend on pert amp
        hps_0 = jt.leaves(data.hps, is_leaf=is_type(TreeNamespace))[0]
        context_inputs = hps_0.context_input
        train_pert_stds = hps_0.train.pert.std 
        
        plot_vars = jt.map(
            lambda d: LDict.of("context_input")({
                context_input: jtree.stack(
                    jt.map(lambda arr: arr[i], d).values()
                )
                for i, context_input in enumerate(context_inputs)
            }),
            plot_vars_stacked,
            is_leaf=LDict.is_of("train__pert__std"),
        )
        
        figs = jt.map(
            partial(
                plot_condition_trajectories,
                colorscale=colorscales['train__pert__std'],
                colorscale_axis=0,
                legend_title="Train<br>field std.",
                legend_labels=train_pert_stds,
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

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, hps_common, **kwargs):
        return dict(
            # TODO: The number of replicates (`n_replicates_included`) may vary with the disturbance train std!
            # n=min(self.n_curves_max, hps_common.eval_n * n_replicates_included)  #? n: pytree[int]
        )


class Measures_CompareTrainStdAndContext(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        all_measure_values=Measures,
    ))
    measure_keys: tuple[str, ...]
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    _pre_ops: tuple[tuple[str, Callable]] = ()
    fig_params: FigParams = FigParams(
        n_curves_max=20,
    )

    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        return dict(
            all_measure_values=dict(
                measure_keys=self.measure_keys,
                variant=self.variant,
            )
        )

    def make_figs(
        self,
        data: AnalysisInputData,
        *,
        all_measure_values,
        colors,
        **kwargs,
    ):
        # Move the disturbance amplitude level to the outside of each measure.
        #! Is this necessary now?
        all_measure_values = LDict.of("measure")({
            measure_key: jtree.move_level_to_outside(
                measure_values, 
                LDict[float, Responses],  #! TODO LDict
            )
            for measure_key, measure_values in all_measure_values.items()
        })
        
        # Since train__pert__std is inner in this case, as it typically is, but we want to show violins
        # with train std. in the legend and context on the x-axis, we will need to swap these levels 
        # before passing to `get_violins`
        swap_vars = lambda tree: jt.transpose(
            jt.structure(tree, is_leaf=LDict.is_of("train__pert__std")), None, tree
        )

        figs = LDict.of("measure")({
            measure_key: LDict.of("pert__amp")({
                pert_amplitude: get_violins(
                    swap_vars(measure_values),
                    yaxis_title=MEASURE_LABELS[measure_key],
                    xaxis_title="Context input",
                    legend_title="Train std.",
                    colors=colors['context_input'].dark,
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
                for pert_amplitude, measure_values in all_measure_values[measure_key].items()
            })
            for measure_key in all_measure_values
        })
        
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, all_measure_values, **kwargs):
        return dict(
            # TODO: The number of replicates (`n_replicates_included`) may vary with the disturbance train std!
            n=int(np.prod(jt.leaves(all_measure_values)[0].shape)),
        )
                
        
ALL_ANALYSES = [
    Aligned_IdxContextInput(),
    Aligned_IdxTrainStd_PerContext(),
    Measures_CompareTrainStdAndContext(measure_keys=MEASURE_KEYS),
]