from collections.abc import Callable
from functools import partial 
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
from rnns_learn_robust_motor_policies.analysis.measures import MEASURES, MEASURE_LABELS, RESPONSE_VAR_LABELS, Measure, Responses, compute_all_measures, output_corr
from rnns_learn_robust_motor_policies.analysis.state_utils import orthogonal_field, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.colors import MEAN_LIGHTEN_FACTOR
from rnns_learn_robust_motor_policies.constants import INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.database import add_evaluation_figure
from rnns_learn_robust_motor_policies.hyperparams import TreeNamespace
from rnns_learn_robust_motor_policies.misc import camel_to_snake, lohi
from rnns_learn_robust_motor_policies.plot import get_measure_replicate_comparisons, get_violins
from rnns_learn_robust_motor_policies.plot_utils import figs_flatten_with_paths
from rnns_learn_robust_motor_policies.setup_utils import convert_tasks_to_small
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


def setup_tasks_and_models(task_base, models_base, hps):
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
            task_base, models[0],
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


WHERE_PLOT = where_plot = lambda states: (
    states.mechanics.effector.pos,
    states.mechanics.effector.vel,
    states.efferent.output,
)
VAR_LABELS = ('Position', 'Velocity', 'Control force')


def plot_trajectories(states, *args, colors, **kwargs): 
    return fbp.trajectories_2D(
        WHERE_PLOT(states),
        var_labels=VAR_LABELS,
        axes_labels=('x', 'y'),
        colorscale=colors['reach_condition'],
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


class AlignedVars(AbstractAnalysis):
    dependencies: ClassVar[dict[str, Callable]] = {}
    conditions: ClassVar[tuple[str, ...]] = ()

    def __call__(self, states: PyTree[Module], hps: TreeNamespace, **kwargs):
        raise NotImplementedError


class BestReplicateStates(AbstractAnalysis):
    i_replicate: Optional[int] = None
    dependencies: ClassVar[dict[str, Callable]] = dict(
        #! best_replicate=get_best_replicate, 
    )
    conditions: ClassVar[tuple[str, ...]] = ()
    
    def compute(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        best_replicate, 
        **kwargs,
    ):
        return jt.map(
            lambda states_by_std: TrainStdDict({
                std: jtree.take(
                    states, 
                    best_replicate[std], 
                    axis=1,
                )
                for std, states in states_by_std.items()
            }),
            states,
            is_leaf=is_type(TrainStdDict),
        )


# "center_out_sets/all_evals_single_replicate"
class CenterOutByEval(AbstractAnalysis):
    dependencies: ClassVar[dict[str, Callable]] = dict( 
        plot_states=BestReplicateStates,
    )
    conditions: ClassVar[tuple[str, ...]] = ('any_system_noise',)  # Skip this eval comparison, if only one eval

    def make_figs(self, states: PyTree[Module], hps: TreeNamespace, *, plot_states, **kwargs):
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
    
    def _params_to_save(self, hps: TreeNamespace, *, best_replicate, disturbance_std, **kwargs):
        return dict(
            i_replicate=best_replicate[disturbance_std],
        )
        
    
# "center_out_sets/single_eval_single_replicate"
class CenterOutSingleEval(AbstractAnalysis):
    i_trial: int = 0
    dependencies: ClassVar[dict[str, Callable]] = dict(
        plot_states=BestReplicateStates,
    )
    conditions: ClassVar[tuple[str, ...]] = ('any_system_noise',)

    def make_figs(self, states: PyTree[Module], hps: TreeNamespace, *, plot_states, **kwargs):
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
    i_trial: int = 0
    dependencies: ClassVar[dict[str, Callable]] = dict()
    conditions: ClassVar[tuple[str, ...]] = ('any_system_noise',)

    def make_figs(self, states: PyTree[Module], hps: TreeNamespace, **kwargs):
        plot_states = jtree.take(states, self.i_trial, 0)

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

    def _params_to_save(self, hps: TreeNamespace, *, disturbance_std, **kwargs):
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


class Aligned_IdxTrial(AbstractAnalysis):
    n_conditions: int  # all_tasks['small'][disturbance_amplitude].n_validation_trials
    n_curves_max: int = 20
    dependencies: ClassVar[dict[str, Callable]] = dict(
        aligned_vars=AlignedVars,
    )
    conditions: ClassVar[tuple[str, ...]] = ()

    def make_figs(self, states: PyTree[Module], hps: TreeNamespace, *, aligned_vars, **kwargs):
        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                legend_title="Trial",
                colorscale=COLORSCALES['trials'],
                colorscale_axis=0, 
                curves_mode='lines', 
            ),
            aligned_vars['small'],
            is_leaf=is_type(Responses),
        )
        
    def _params_to_save(self, hps: TreeNamespace, *, disturbance_std, **kwargs):
        return dict(
            # n=min(self.n_curves_max, n_replicates_included[disturbance_std] * self.n_conditions)
        )
  
        
class Aligned_IdxPertAmp(AbstractAnalysis):
    n_conditions: int  # all_tasks['small'][disturbance_amplitude].n_validation_trials
    n_curves_max: int = 20
    dependencies: ClassVar[dict[str, Callable]] = dict(
        aligned_vars=AlignedVars,
        #! colors=Colors,
    )
    conditions: ClassVar[tuple[str, ...]] = ()

    def make_figs(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        aligned_vars, 
        colors, 
        **kwargs,
    ):
        plot_vars_stacked = jtree.stack(aligned_vars['small'].values())

        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                colorscale=colors['disturbance_amplitudes'],
                colorscale_axis=0,
                legend_title="Field<br>amplitude",
                legend_labels=hps.disturbance.amplitude,
                curves_mode='lines',
            ),
            plot_vars_stacked,
            is_leaf=is_type(Responses),
        )
        
        return figs
                
    def _params_to_save(self, hps: TreeNamespace, *, disturbance_std, **kwargs):
        return dict(
            # n=min(self.n_curves_max, hps.eval_n * n_replicates_included[disturbance_std] * self.n_conditions)
        )
        

class Aligned_IdxTrainStd(AbstractAnalysis):
    n_conditions: int  # all_tasks['small'][disturbance_amplitude].n_validation_trials
    n_curves_max: int = 20
    dependencies: ClassVar[dict[str, Callable]] = dict(
        aligned_vars=AlignedVars,
        #! colors=Colors,
    )
    conditions: ClassVar[tuple[str, ...]] = ()

    def make_figs(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        aligned_vars, 
        colors, 
        **kwargs,
    ):
        plot_vars_stacked = {
            # concatenate along the replicate axis, which has variable length
            disturbance_amplitude: jtree.stack(list(subdict(vars_, hps.disturbance.std).values()))
            for disturbance_amplitude, vars_ in aligned_vars['small'].items()
        }


        figs = jt.map(
            partial(
                plot_condition_trajectories, 
                colorscale=colors['disturbance_stds'],
                colorscale_axis=0,
                legend_title="Train<br>field std.",
                legend_labels=hps.disturbance.std,  #!
                curves_mode='lines',
                var_endpoint_ms=0,
                scatter_kws=dict(line_width=0.5, opacity=0.3),
                # ref_endpoints=(pos_endpoints['full'], None),
            ),
            plot_vars_stacked,
            is_leaf=is_type(Responses),
        )
                        
    def _params_to_save(self, hps: TreeNamespace, *, disturbance_std, **kwargs):
        return dict(
            # TODO: The number of replicates (`n_replicates_included`) may vary with the disturbance train std!
            # n=min(self.n_curves_max, hps.eval_n * n_replicates_included)  #? n: pytree[int]
        )
        

class VelocityProfiles(AbstractAnalysis):
    """Generates forward and lateral velocity profile figures."""
    dependencies: ClassVar[dict[str, Callable]] = dict(
        aligned_vars=AlignedVars,
        #! colors=Colors,
    )
    conditions: ClassVar[tuple[str, ...]] = ()

    def compute(
        self,
        states: PyTree[Module], 
        hps: TreeNamespace, 
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
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        result,
        **kwargs,
    ):
        figs = {
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
                    # colors=colors.dark.disturbance_std, #! TODO
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
            for disturbance_amplitude in hps.disturbance.amplitude
        }
        return figs
        
    def _params_to_save(self, hps: TreeNamespace, *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape[:-2]))
        )
        

class Measures(AbstractAnalysis):
    measure_keys: tuple[str, ...] = MEASURE_KEYS
    dependencies: ClassVar[dict[str, Callable]] = dict(
        aligned_vars=AlignedVars,
    )
    conditions: ClassVar[tuple[str, ...]] = ()

    def compute(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
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
    dependencies: ClassVar[dict[str, Callable]] = dict(
        measure_values=Measures,
    )
    conditions: ClassVar[tuple[str, ...]] = ()
    
    def make_figs(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        measure_values, 
        colors: TreeNamespace,
        **kwargs,
    ):
        figs = get_violins_per_measure(
            measure_values,
            colors=colors.dark.disturbance_amplitude,  #! TODO
        )
        return figs
    
    def _params_to_save(self, hps: TreeNamespace, *, result, **kwargs):
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
    dependencies: ClassVar[dict[str, Callable]] = dict(
        measure_values=Measures,
        #! replicates_included=IncludedReplicates,
    )
    conditions: ClassVar[tuple[str, ...]] = ()  
    
    def compute(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        measure_values, 
        **kwargs,
    ):
        return subset_by_train_stds(
            measure_values,
            lohi(hps.load.disturbance.std),  # type: ignore
        )      


class Measures_CompareReplicatesLoHi(AbstractAnalysis):
    dependencies: ClassVar[dict[str, Callable]] = dict(
        measure_values_lohi_disturbance_std=MeasuresLoHiPertStd,
        #! replicates_included=IncludedReplicates,
    )
    conditions: ClassVar[tuple[str, ...]] = ()    

        
    def make_figs(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        measure_values_lohi_disturbance_std, 
        colors: TreeNamespace,
        included_replicates,
        **kwargs,
    ):
        replicates_all_lohi_included = jt.reduce(jnp.logical_and, lohi(included_replicates))
        figs = get_one_measure_plot_per_eval_condition(
            get_measure_replicate_comparisons, 
            measure_values_lohi_disturbance_std,
            lohi(colors.dark.disturbance_stds),
            included_replicates=np.where(replicates_all_lohi_included)[0],
        )
        return figs

    def _params_to_save(self, hps: TreeNamespace, *, measure_values_lohi_disturbance_std, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(measure_values_lohi_disturbance_std)[0].shape))
        )       


class Measures_LoHiSummary(AbstractAnalysis):
    dependencies: ClassVar[dict[str, Callable]] = dict(
        measure_values_lohi_disturbance_std=MeasuresLoHiPertStd,
        #! replicates_included=IncludedReplicates,
    )
    conditions: ClassVar[tuple[str, ...]] = ()  
    
    def compute(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        measure_values_lohi_disturbance_std, 
        **kwargs,
    ):
        return MeasureDict(**{
            key: subdict(measure, lohi(hps.disturbance.amplitude))  # type: ignore
            for key, measure in measure_values_lohi_disturbance_std.items()
        })
        
    def make_figs(
        self, 
        states: PyTree[Module], 
        hps: TreeNamespace, 
        *, 
        result, 
        colors: TreeNamespace,
        included_replicates,
        **kwargs,
    ):
        figs = {
            key: get_violins(
                measure, 
                yaxis_title=MEASURE_LABELS[key], 
                xaxis_title="Train field std.",
                legend_title="smee",
                colors=colors.dark.disturbance_amplitude,  #! TODO
                layout_kws=dict(
                    width=300, height=300, 
                )
            )
            for key, measure in result.items()
        }

    def _params_to_save(self, hps: TreeNamespace, *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape))
        )          
          

class OutputWeightCorrelation(AbstractAnalysis):
    dependencies: ClassVar[dict[str, Callable]] = dict(
        colors=Colors,
    )
    conditions: ClassVar[tuple[str, ...]] = ()
    
    def compute(self, states: PyTree[Module], hps: TreeNamespace, **kwargs):
        activities = jt.map(
            lambda states: states.net.hidden,
            states['full'],
            is_leaf=is_module,
        )

        output_weights = jt.map(
            lambda models: models.step.net.readout.weight,
            #! TODO: Pass `models` to `AbstractAnalysis`
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
        
    def make_figs(self, states: PyTree[Module], hps: TreeNamespace, *, result, colors, **kwargs):
        assert result is not None
        fig = get_violins(
            result, 
            yaxis_title="Output correlation", 
            xaxis_title="Train field std.",
            colors=colors.dark.disturbance_amplitude,
        )

    def _params_to_save(self, hps: TreeNamespace, *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape)),
            measure="output_correlation",
        )     
 
        
"""All the analyses to perform in this part."""
ALL_ANALYSES = [
    CenterOutByEval,
    CenterOutSingleEval,
    CenterOutByReplicate,
    Aligned_IdxTrial,
    Aligned_IdxPertAmp,
    Aligned_IdxTrainStd,
    VelocityProfiles,
    Measures_ByTrainStd,
    # Measures_CompareReplicatesLoHi,
]