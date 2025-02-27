from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

from equinox import Module
import jax.tree as jt
from jaxtyping import PyTree

import feedbax.plotly as fbp
from jax_cookbook import is_module, is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.state_utils import get_aligned_vars, get_pos_endpoints
from rnns_learn_robust_motor_policies.colors import COLORSCALES, MEAN_LIGHTEN_FACTOR
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace
from rnns_learn_robust_motor_policies.types import RESPONSE_VAR_LABELS, Responses, PertAmpDict, TrainStdDict


WHERE_VARS_TO_ALIGN = lambda states, pos_endpoints: Responses(
    # Positions with respect to the origin
    states.mechanics.effector.pos - pos_endpoints[0][..., None, :],
    states.mechanics.effector.vel,
    states.efferent.output,
)


class AlignedVars(AbstractAnalysis):
    """Align spatial variable (e.g. position and velocity) coordinates with the reach direction."""
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    variant: Optional[str] = None
    conditions: tuple[str, ...] = ()

    def compute(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        trial_specs,
        **kwargs,
    ):
        return jt.map(
            lambda specs, states_by_std: jt.map(
                lambda states: get_aligned_vars(
                    states, WHERE_VARS_TO_ALIGN, get_pos_endpoints(specs),
                ),
                states_by_std,
                is_leaf=is_module,
            ),
            trial_specs,
            states,
            is_leaf=is_module,
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
        figs = jt.map(
            partial(
                plot_condition_trajectories,
                legend_title="Trial",
                colorscale=COLORSCALES['trial'],
                colorscale_axis=0,
                curves_mode='lines',
            ),
            aligned_vars[self.variant],
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
            lambda d: jtree.stack(list(d.values())),
            aligned_vars[self.variant],
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
                legend_labels=hps[self.variant].load.disturbance.std,
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