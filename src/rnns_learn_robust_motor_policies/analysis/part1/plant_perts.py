from collections.abc import Callable
from functools import partial 
from types import MappingProxyType
from typing import ClassVar, Optional

import equinox as eqx
from equinox import Module
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree
import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm

from feedbax.intervene import CurlField, FixedField, add_intervenors, schedule_intervenor
import feedbax.plotly as fbp
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.analysis import WHERE_PLOT
from rnns_learn_robust_motor_policies.analysis.analysis import VAR_LABELS
from rnns_learn_robust_motor_policies.analysis.analysis import AlignedVars
from rnns_learn_robust_motor_policies.analysis.measures import MEASURES, MEASURE_LABELS, RESPONSE_VAR_LABELS, Measure, Responses, compute_all_measures, output_corr
from rnns_learn_robust_motor_policies.analysis.state_utils import orthogonal_field, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.colors import MEAN_LIGHTEN_FACTOR, COLORSCALES
from rnns_learn_robust_motor_policies.constants import INTERVENOR_LABEL, REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace
from rnns_learn_robust_motor_policies.misc import camel_to_snake, lohi
from rnns_learn_robust_motor_policies.plot import get_measure_replicate_comparisons, get_violins
from rnns_learn_robust_motor_policies.tree_utils import subdict, tree_subset_dict_level
from rnns_learn_robust_motor_policies.types import LabelDict, MeasureDict, PertAmpDict, TrainStdDict


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


def plot_trajectories(states, *args, **kwargs): 
    return fbp.trajectories_2D(
        WHERE_PLOT(states),
        var_labels=VAR_LABELS,
        axes_labels=('x', 'y'),
        colorscale=COLORSCALES['reach_condition'],
        legend_title='Reach direction',
        # scatter_kws=dict(line_width=0.5),
        layout_kws=dict(
            width=100 + len(VAR_LABELS) * 300,
            height=400,
            legend_tracegroupgap=1,
        ),
        *args, 
        **kwargs,
    )


#! TODO: Move
class BestReplicateStates(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: ClassVar[Optional[str]] = None
    conditions: tuple[str, ...] = ()
    i_replicate: Optional[int] = None
    
    def compute(
        self, 
        models: PyTree[Module],
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        *, 
        replicate_info, 
        **kwargs,
    ):
        return jt.map(
            lambda states_by_std: TrainStdDict({
                std: jtree.take(
                    states, 
                    replicate_info[std]["best_replicates"][REPLICATE_CRITERION], 
                    axis=1,
                )
                for std, states in states_by_std.items()
            }),
            states,
            is_leaf=is_type(TrainStdDict),
        )


class CenterOutByEval(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict( 
        best_replicate_states=BestReplicateStates,
    ))
    variant: ClassVar[Optional[str]] = "small"
    conditions: tuple[str, ...] = ('any_system_noise',)  # Skip this eval comparison, if only one eval

    def make_figs(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        *, 
        best_replicate_states, 
        **kwargs,
    ):
        plot_states = best_replicate_states['small']
        figs = jt.map(
            partial(
                plot_trajectories, 
                curves_mode='lines', 
                colorscale_axis=1, 
                mean_trajectory_line_width=2.5,
                darken_mean=MEAN_LIGHTEN_FACTOR,
                scatter_kws=dict(line_width=0.5),
            ),
            plot_states,
            is_leaf=is_module,
        )
        return figs
    
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, replicate_info, disturbance_std, **kwargs):
        return dict(
            i_replicate=replicate_info[disturbance_std]['best_replicates'][REPLICATE_CRITERION],
        )
        
    
class CenterOutSingleEval(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        best_replicate_states=BestReplicateStates,
    ))
    variant: ClassVar[Optional[str]] = "small"
    conditions: tuple[str, ...] = ('any_system_noise',)
    i_trial: int = 0

    def make_figs(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        *, 
        best_replicate_states, 
        **kwargs,
    ):
        plot_states = best_replicate_states['small']
        plot_states_i = jtree.take(plot_states, self.i_trial, 0)

        figs = jt.map(
            partial(
                plot_trajectories, 
                mode='markers+lines', 
                ms=3,
                scatter_kws=dict(line_width=0.75),
            ),
            plot_states_i,
            is_leaf=is_module,
        )
        
        # add_endpoint_traces(
        #     fig, pos_endpoints['small'], xaxis='x1', yaxis='y1', colorscale='phase'
        # )
        
        return figs
        

# "center_out_sets/single_eval_all_replicates"
class CenterOutByReplicate(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: ClassVar[Optional[str]] = "small"
    conditions: tuple[str, ...] = ('any_system_noise',)
    i_trial: int = 0

    def make_figs(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        **kwargs,
    ):
        plot_states = jtree.take(states['small'], self.i_trial, 0)

        figs = jt.map(
            partial(
                plot_trajectories, 
                curves_mode='lines', 
                colorscale_axis=1, 
                mean_trajectory_line_width=2.5,
                darken_mean=MEAN_LIGHTEN_FACTOR,
                scatter_kws=dict(line_width=0.75),
            ),
            plot_states,
            is_leaf=is_module,
        )
        
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, disturbance_std, **kwargs):
        return dict(
            # n=n_replicates_included[disturbance_std],
        )


plot_condition_trajectories = partial(
    fbp.trajectories_2D,
    var_labels=RESPONSE_VAR_LABELS,
    axes_labels=('x', 'y'),
    # mode='std',
    mean_trajectory_line_width=3,
    # n_curves_max=n_curves_max,
    darken_mean=MEAN_LIGHTEN_FACTOR,
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


#! TODO: Move
class Aligned_IdxTrial(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: ClassVar[Optional[str]] = "small"
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
        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                legend_title="Trial",
                colorscale=COLORSCALES['trial'],
                colorscale_axis=0, 
                curves_mode='lines', 
            ),
            aligned_vars['small'],
            is_leaf=is_type(Responses),
        )
        return figs
        
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, disturbance_std, **kwargs):
        return dict(
            # n=min(self.n_curves_max, n_replicates_included[disturbance_std] * self.n_conditions)
        )
  
        
class Aligned_IdxPertAmp(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: ClassVar[Optional[str]] = "small"
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
        # plot_vars_stacked = jtree.stack(aligned_vars['small'].values())
        plot_vars_stacked = jt.map(
            lambda d: jtree.stack(list(d.values())),
            aligned_vars['small'],
            is_leaf=is_type(PertAmpDict), 
        )

        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                colorscale=COLORSCALES['disturbance_amplitude'],
                colorscale_axis=0,
                legend_title="Field<br>amplitude",
                legend_labels=hps['small'].disturbance.amplitude,
                curves_mode='lines',
            ),
            plot_vars_stacked,
            is_leaf=is_type(Responses),
        )
        
        return figs
                
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, disturbance_std, **kwargs):
        return dict(
            # n=min(self.n_curves_max, hps.eval_n * n_replicates_included[disturbance_std] * self.n_conditions)
        )
        

class Aligned_IdxTrainStd(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: ClassVar[Optional[str]] = "small"
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
            lambda d: jtree.stack(list(d.values())),
            aligned_vars['small'],
            is_leaf=is_type(TrainStdDict), 
        )
        # plot_vars_stacked = PertAmpDict({
        #     # concatenate along the replicate axis, which has variable length
        #     disturbance_amplitude: jtree.stack(list(vars_.values()))
        #     for disturbance_amplitude, vars_ in aligned_vars['small'].items()
        # })
        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                colorscale=COLORSCALES['disturbance_std'],
                colorscale_axis=0,
                legend_title="Train<br>field std.",
                legend_labels=hps['small'].load.disturbance.std,  
                curves_mode='lines',
                var_endpoint_ms=0,
                scatter_kws=dict(line_width=0.5, opacity=0.3),
                # ref_endpoints=(pos_endpoints['full'], None),
            ),
            plot_vars_stacked,
            is_leaf=is_type(Responses),
        )
        
        return figs
                        
    def _params_to_save(self, hps: PyTree[TreeNamespace], **kwargs):
        return dict(
            # TODO: The number of replicates (`n_replicates_included`) may vary with the disturbance train std!
            # n=min(self.n_curves_max, hps.eval_n * n_replicates_included)  #? n: pytree[int]
        )
        

class VelocityProfiles(AbstractAnalysis):
    """Generates forward and lateral velocity profile figures."""
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: ClassVar[Optional[str]] = "full"
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
            aligned_vars['full'],
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
        figs = PertAmpDict({
            # TODO: Once the mapping between custom dict types and their column names is automatic
            # (e.g. `PertVarDict` will simply map to 'pert_var'), we can construct a `DirectionDict`
            # ad hoc maybe
            disturbance_amplitude: LabelDict({
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
            for disturbance_amplitude in hps['full'].disturbance.amplitude
        })
        return figs
        
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape[:-2]))
        )
        

class Measures(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    variant: ClassVar[Optional[str]] = None
    conditions: tuple[str, ...] = ()
    measure_keys: tuple[str, ...] = MEASURE_KEYS

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
        all_measures: MeasureDict[Measure] = subdict(MEASURES, self.measure_keys)  # type: ignore
        all_measure_values = compute_all_measures(all_measures, aligned_vars['full'])
        return all_measure_values


def get_violins_per_measure(measure_values, **kwargs):
    return {
        key: get_violins(
            values, 
            yaxis_title=MEASURE_LABELS[key], 
            xaxis_title="Train field std.",
            **kwargs,
        )
        for key, values in measure_values.items()
    }
    

class Measures_ByTrainStd(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values=Measures,
    ))
    variant: ClassVar[Optional[str]] = "full"
    conditions: tuple[str, ...] = ()
    
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
        figs = get_violins_per_measure(
            measure_values,
            colors=colors['full']['disturbance_amplitude']['dark'],  
        )
        return figs
    
    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape))
        )


def get_one_measure_plot_per_eval_condition(plot_func, measures, colors, **kwargs):
    return {
        key: PertAmpDict({
            disturbance_amplitude: plot_func(
                measure[disturbance_amplitude], 
                MEASURE_LABELS[key], 
                colors,
                **kwargs,
            )
            for disturbance_amplitude in measure
        })
        for key, measure in measures.items()
    }


subset_by_train_stds = partial(tree_subset_dict_level, dict_type=TrainStdDict)


class MeasuresLoHiPertStd(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values=Measures,
    ))
    variant: ClassVar[Optional[str]] = None
    conditions: tuple[str, ...] = ()  
    
    def compute(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        *, 
        measure_values, 
        **kwargs,
    ):
        # Map over analysis variants (e.g. full task vs. small task)
        return jt.map(
            lambda hps_: subset_by_train_stds(
                measure_values,
                lohi(hps_.load.disturbance.std),  # type: ignore
            ),
            hps,
            is_leaf=is_type(TreeNamespace),
        )


class Measures_CompareReplicatesLoHi(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values_lohi_disturbance_std=MeasuresLoHiPertStd,
    ))
    variant: ClassVar[Optional[str]] = "full"
    conditions: tuple[str, ...] = ()    

        
    def make_figs(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        *, 
        measure_values_lohi_disturbance_std, 
        colors: TreeNamespace,
        replicate_info,
        **kwargs,
    ):
        included_replicates = replicate_info['included_replicates'][REPLICATE_CRITERION]
        replicates_all_lohi_included = jt.reduce(jnp.logical_and, lohi(included_replicates))
        figs = get_one_measure_plot_per_eval_condition(
            get_measure_replicate_comparisons, 
            measure_values_lohi_disturbance_std,
            lohi(colors.dark.disturbance_stds),
            included_replicates=np.where(replicates_all_lohi_included)[0],
        )
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, measure_values_lohi_disturbance_std, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(measure_values_lohi_disturbance_std)[0].shape))
        )       


class Measures_LoHiSummary(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values_lohi_disturbance_std=MeasuresLoHiPertStd,
    ))
    variant: ClassVar[Optional[str]] = "full"
    conditions: tuple[str, ...] = ()  
    
    def compute(
        self, 
        models: PyTree[Module], 
        tasks: PyTree[Module], 
        states: PyTree[Module], 
        hps: PyTree[TreeNamespace], 
        *,
        measure_values_lohi_disturbance_std, 
        **kwargs,
    ):
        
        return MeasureDict(**{
            key: subdict(measure, lohi(hps["full"].disturbance.amplitude))  # type: ignore
            # MeasuresLoHiPertStd returns `measure_values_lohi_disturbance_std` for all eval variants,
            # so we choose the right variant
            for key, measure in measure_values_lohi_disturbance_std["full"].items()
        })
        
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
        figs = MeasureDict(**{
            key: get_violins(
                measure, 
                yaxis_title=MEASURE_LABELS[key], 
                xaxis_title="Train field std.",
                legend_title="TODO",
                colors=colors['full']['disturbance_amplitude']['dark'],
                layout_kws=dict(
                    width=300, height=300, 
                )
            )
            for key, measure in result.items()
        })
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape))
        )          
          

class OutputWeightCorrelation(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: ClassVar[Optional[str]] = "full"
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
            states['full'],
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
            colors=colors['full']['disturbance_amplitude']['dark'],
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
    # Measures_ByTrainStd(),
    # Measures_CompareReplicatesLoHi(),
    Measures_LoHiSummary(),
    # OutputWeightCorrelation(),
]