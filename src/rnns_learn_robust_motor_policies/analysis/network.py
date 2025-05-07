from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.train import SimpleTrainer
from jaxtyping import PyTree, PRNGKeyArray
import numpy as np

from jax_cookbook import is_module

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.measures import output_corr
from rnns_learn_robust_motor_policies.misc import center_and_rescale, ravel_except_last
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


def get_unit_pref_angles(model, task, *, key, n_iter=50, ts=slice(None), normalize_dirs=True):
    key_eval, key_fit = jr.split(key)

    _, states = task.eval(model, key=key_eval)
    hidden = states.net.hidden[:, ts]

    trial_specs, _ = task.trials_validation
    target = trial_specs.target.pos[:, ts]

    X, ys = prep_data(hidden, target)

    def fit_linear(X, y, n_iter=50):
        lin_model = jax.tree_map(
            jnp.zeros_like,
            eqx.nn.Linear(target.shape[-1], 1, key=key_fit),
        )
        trainer = SimpleTrainer()
        return trainer(lin_model, X.T, y, n_iter=n_iter)

    batch_fit_linear = jax.vmap(
        partial(fit_linear, n_iter=n_iter),
        in_axes=(None, 1)
    )
    linear_fits = batch_fit_linear(X, ys)

    pref_dirs = jnp.squeeze(linear_fits.weight)  # preferred directions
    # Vector length is irrelevant to angles
    pref_angles = jnp.arctan2(pref_dirs[:, 1], pref_dirs[:, 0])

    if normalize_dirs:
        pref_dirs = pref_dirs / jnp.linalg.norm(pref_dirs, axis=1, keepdims=True)

    return pref_angles, pref_dirs, states, trial_specs


def fit_linear(X, y, n_iter=50, *, key):
    lin_model = jax.tree_map(
        jnp.zeros_like,
        eqx.nn.Linear(X.shape[-1], 1, key=key),
    )
    trainer = SimpleTrainer()
    return trainer(lin_model, X.T, y, n_iter=n_iter)


class UnitPreferences(AbstractAnalysis):
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = "full"
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict())
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

    n_iter_fit: int = 50
    feature_fn: Callable = lambda task, states: task.trials_validation[0].target.pos
    key: PRNGKeyArray = eqx.field(default_factory=lambda: jr.PRNGKey(0))  # For linear fit -- not very important.

    def compute(
            self,
            data: AnalysisInputData,
            **kwargs,
    ):
        return jt.map(
            lambda task, states_by_task: jt.map(
                lambda states: self.get_prefs(task, states, self.key),
                states_by_task,
                is_leaf=is_module,
            ),
            data.tasks,
            data.states,
            is_leaf=is_module,
        )

    # We could also pass `model` and `hps` here, but I don't see why we'd ever be
    # treating them as features -- they don't have a time dimension.
    def get_prefs(self, task, states, key):
        activities = states.net.hidden
        features = self.feature_fn(task, states)
        # Generally, `activities` may have more axes than `features`, e.g. when 
        # the features are from the task, and we are evaluating the same conditions 
        # multiple times. However, any batch axes that are present in `features`
        # must be broadcastable with any that are present in `activities`.
        # We explicitly broadcast `features` here, because the trainer works with 
        # aggregated data.
        features_broadcast = jnp.broadcast_to(
            features,
            activities.shape[:-1] + (features.shape[-1],),
        )
        features_flat = center_and_rescale(ravel_except_last(features_broadcast))
        activities_flat = ravel_except_last(activities)
        return jnp.squeeze(self._batch_fit_linear(key=key)(
            features_flat, activities_flat
        ).weight)

    def _batch_fit_linear(self, key):
        return jax.vmap(
            partial(fit_linear, n_iter=self.n_iter_fit, key=key),
            in_axes=(None, 1),
        )
