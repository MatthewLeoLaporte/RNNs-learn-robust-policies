from collections import namedtuple
from collections.abc import Callable, Sequence
from functools import cached_property, partial, reduce
from types import MappingProxyType
from typing import ClassVar, Optional, Dict, Any, Literal as L, Type

import equinox as eqx
from equinox import Module
from equinox import filter_vmap as vmap
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PyTree

from feedbax.bodies import SimpleFeedbackState
from jax_cookbook import is_type, compose
import numpy as np

from rnns_learn_robust_motor_policies.types import Responses
from rnns_learn_robust_motor_policies.analysis.aligned import AlignedVars
from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.constants import EVAL_REACH_LENGTH, REPLICATE_CRITERION
from rnns_learn_robust_motor_policies.misc import lohi
from rnns_learn_robust_motor_policies.plot import get_measure_replicate_comparisons, get_violins
from rnns_learn_robust_motor_policies.tree_utils import TreeNamespace, subdict, tree_subset_ldict_level
from rnns_learn_robust_motor_policies.types import ResponseVar, Direction, DIRECTION_IDXS, LDict


frob = lambda x: jnp.linalg.norm(x, axis=(-1, -2), ord='fro')


subset_by_train_stds = partial(tree_subset_ldict_level, )


class Measure(Module):
    """Unified measure class for computing response metrics.
    
    Attributes:
        response_var: Which response variable to measure (pos, vel, force)
        agg_func: Function to aggregate over time axis (e.g. jnp.max, jnp.mean)
        direction: Optional direction to extract vector component
        timesteps: Optional slice to select specific timesteps
        transform_func: Optional function to transform values (e.g. jnp.linalg.norm)
        normalizer: Optional value to divide result by
    """
    response_var: ResponseVar
    agg_func: Optional[Callable] = None
    direction: Optional[Direction] = None
    timesteps: Optional[slice] = None
    transform_func: Optional[Callable] = None
    normalizer: Optional[float] = None
    
    @cached_property
    def _methods(self) -> dict[str, Callable]:
        return {
            'timesteps': self._select_timesteps,
            'direction': self._select_direction,
            'transform_func': self._apply_transform,
            'agg_func': self._aggregate,
            'normalizer': self._normalize,
        }
        
    @cached_property
    def _call_methods(self) -> list[Callable]:
        return [self._get_response_var] + [
            value for key, value in self._methods.items() 
            if getattr(self, key) is not None
        ]

    def _get_response_var(self, responses: Responses) -> Float[Array, "..."]:
        """Extract the specified response variable."""
        return getattr(responses, self.response_var.value)

    def _select_timesteps(self, values: Float[Array, "..."]) -> Float[Array, "..."]:
        """Select specified timesteps."""
        return values[..., self.timesteps, :]

    def _select_direction(self, values: Float[Array, "..."]) -> Float[Array, "..."]:
        """Select specified direction component."""
        assert self.direction is not None
        return values[..., DIRECTION_IDXS[self.direction]]

    def _aggregate(self, values: Float[Array, "..."]) -> Float[Array, "..."]:
        """Apply aggregation function over time axis."""
        assert self.agg_func is not None
        return self.agg_func(values, axis=-1)

    def _normalize(self, values: Float[Array, "..."]) -> Float[Array, "..."]:
        """Apply normalization."""
        return values / self.normalizer
    
    def _apply_transform(self, values: Float[Array, "..."]) -> Float[Array, "..."]:
        """Apply custom transformation function."""
        assert self.transform_func is not None
        return self.transform_func(values)

    def __call__(self, responses: Responses) -> Float[Array, "..."]:
        """Apply measure to response state.
        
        Args:
            responses: Response state containing trajectories
            
        Returns:
            Computed measure values
        """
        return compose(*self._call_methods)(responses)


# Common transformations
vector_magnitude = partial(jnp.linalg.norm, axis=-1)


def signed_max(x, axis=None, keepdims=False):
    """Return the value with the largest magnitude, positive or negative.
    """
    abs_x = jnp.abs(x)
    max_idx = jnp.argmax(abs_x, axis=axis)
    if axis is None:
        return x.flatten()[max_idx]
    else:
        return jnp.take_along_axis(x, jnp.expand_dims(max_idx, axis=axis), axis=axis)


# Force measures
max_net_force = Measure(
    response_var=ResponseVar.FORCE,
    transform_func=vector_magnitude,
    agg_func=jnp.max,
)
sum_net_force = Measure(
    response_var=ResponseVar.FORCE,
    transform_func=vector_magnitude,
    agg_func=jnp.sum,
)
max_parallel_force = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.PARALLEL,
    agg_func=jnp.max,
)
sum_parallel_force = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.PARALLEL,
    transform_func=jnp.abs,
    agg_func=jnp.sum,
)
max_orthogonal_force = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.ORTHOGONAL,
    agg_func=jnp.max,
)
sum_orthogonal_force_abs = Measure(
    response_var=ResponseVar.FORCE,
    direction=Direction.ORTHOGONAL,
    transform_func=jnp.abs,
    agg_func=jnp.sum,
)


# Velocity measures
max_parallel_vel = Measure(
    response_var=ResponseVar.VELOCITY,
    direction=Direction.PARALLEL,
    agg_func=jnp.max,
)
max_orthogonal_vel = Measure(
    response_var=ResponseVar.VELOCITY,
    direction=Direction.ORTHOGONAL,
    agg_func=jnp.max,
)
max_orthogonal_vel_signed = Measure(
    response_var=ResponseVar.VELOCITY,
    direction=Direction.ORTHOGONAL,
    agg_func=signed_max,
)


# Position measures
max_orthogonal_distance = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.ORTHOGONAL,
    agg_func=jnp.max,
    normalizer=EVAL_REACH_LENGTH / 100,
)
largest_orthogonal_distance = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.ORTHOGONAL,
    agg_func=signed_max,
    normalizer=EVAL_REACH_LENGTH / 100,
)
sum_orthogonal_distance = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.ORTHOGONAL,
    agg_func=jnp.sum,
)
sum_orthogonal_distance_abs = Measure(
    response_var=ResponseVar.POSITION,
    direction=Direction.ORTHOGONAL,
    transform_func=jnp.abs,
    agg_func=jnp.sum,
)
max_deviation = Measure(
    response_var=ResponseVar.POSITION,
    transform_func=vector_magnitude,
    agg_func=jnp.max,
)
sum_deviation = Measure(
    response_var=ResponseVar.POSITION,
    transform_func=vector_magnitude,
    agg_func=jnp.sum,
)


ENDPOINT_ERROR_STEPS = 10


def make_end_velocity_error(last_n_steps: int = ENDPOINT_ERROR_STEPS) -> Measure:
    return Measure(
        response_var=ResponseVar.VELOCITY,
        transform_func=vector_magnitude,
        agg_func=jnp.mean,
        timesteps=slice(-last_n_steps, None),
    )


def make_end_position_error(
    reach_length: float = EVAL_REACH_LENGTH, 
    last_n_steps: int = ENDPOINT_ERROR_STEPS,
) -> Measure:
    """Create measure for endpoint position error."""
    goal_pos = jnp.array([reach_length, 0.])
    return Measure(
        response_var=ResponseVar.POSITION,
        transform_func=lambda x: jnp.linalg.norm(x - goal_pos, axis=-1),
        agg_func=jnp.mean,
        timesteps=slice(-last_n_steps, None),
        normalizer=reach_length / 100,
    )
    
    
def reverse_measure(measure: Measure) -> Measure:
    """Create a new measure that inverts the sign of the states before computing.
    
    For example, use this to turn a measure of the maximum forward velocity into a 
    measure of the maximum reverse velocity.
    """
    if measure.transform_func is not None:
        transform_func = compose(measure.transform_func, jnp.negative)
    else:
        transform_func = jnp.negative
    
    return eqx.tree_at(
        lambda measure: measure.transform_func,
        measure,
        transform_func,
        is_leaf=lambda x: x is None,
    )
    

def set_timesteps(measure: Measure, timesteps) -> Measure:
    return eqx.tree_at(
        lambda measure: measure.timesteps,
        measure,
        timesteps,
        is_leaf=lambda x: x is None,
    )


MEASURES = LDict.of("measure")(dict(
    max_net_force=max_net_force,
    sum_net_force=sum_net_force,
    max_parallel_force_forward=max_parallel_force,
    max_parallel_force_reverse=reverse_measure(max_parallel_force),
    sum_parallel_force=sum_parallel_force,
    max_orthogonal_force_left=max_orthogonal_force,
    max_orthogonal_force_right=reverse_measure(max_orthogonal_force),
    sum_orthogonal_force_abs=sum_orthogonal_force_abs,
    max_parallel_vel_forward=max_parallel_vel,
    max_parallel_vel_reverse=reverse_measure(max_parallel_vel),
    max_orthogonal_vel_left=max_orthogonal_vel,
    max_orthogonal_vel_right=reverse_measure(max_orthogonal_vel),
    max_orthogonal_vel_signed=max_orthogonal_vel_signed,
    max_orthogonal_distance_left=max_orthogonal_distance,
    max_orthogonal_distance_right=reverse_measure(max_orthogonal_distance),
    largest_orthogonal_distance=largest_orthogonal_distance,
    sum_orthogonal_distance=sum_orthogonal_distance,
    sum_orthogonal_distance_abs=sum_orthogonal_distance_abs,
    max_deviation=max_deviation,
    sum_deviation=sum_deviation,
    end_velocity_error=make_end_velocity_error(),
    end_position_error=make_end_position_error(),
))


MEASURE_LABELS = LDict.of("measure")(dict(
    max_net_force="Max net control force",
    sum_net_force="Sum net control force",
    max_parallel_force_forward="Max forward force",
    max_parallel_force_reverse="Max reverse force",
    sum_parallel_force="Sum of absolute parallel forces",
    max_orthogonal_force_left="Max lateral force<br>(left)",
    max_orthogonal_force_right="Max lateral force<br>(right)",
    sum_orthogonal_force_abs="Sum of absolute lateral forces",
    max_parallel_vel_forward="Max forward velocity",
    max_parallel_vel_reverse="Max reverse velocity",
    max_orthogonal_vel_left="Max lateral velocity<br>(left)",
    max_orthogonal_vel_right="Max lateral velocity<br>(right)",
    max_orthogonal_vel_signed="Largest lateral velocity",
    max_orthogonal_distance_left="Max lateral distance<br>(left, % reach length)",
    max_orthogonal_distance_right="Max lateral distance<br>(right, % reach length)",
    largest_orthogonal_distance="Largest lateral distance<br>(% reach length)",
    sum_orthogonal_distance="Sum of signed lateral distances",
    sum_orthogonal_distance_abs="Sum of absolute lateral distances",
    max_deviation="Max deviation",  # From zero/origin! i.e. stabilization task
    sum_deviation="Sum of deviations",
    end_velocity_error=f"Mean velocity error<br>(last {ENDPOINT_ERROR_STEPS} steps)",
    end_position_error=f"Mean position error<br>(last {ENDPOINT_ERROR_STEPS} steps)",
))


def compute_all_measures(measures: PyTree[Measure], all_responses: PyTree[Responses]):
    """Maps the tree of measures over the tree of response conditions."""
    return jt.map(
        lambda func: jt.map(
            lambda responses: func(responses),
            all_responses,
            is_leaf=is_type(Responses),
        ),
        measures,
        is_leaf=is_type(Measure),
    )
    
    
def output_corr(
    activities: Float[Array, "evals replicates conditions time hidden"], 
    weights: Float[Array, "replicates outputs hidden"],
):
    # center the activities in time
    activities = activities - jnp.mean(activities, axis=-2, keepdims=True)
    
    def corr(x, w):
        z = jnp.dot(x, w.T)
        return frob(z) / (frob(w) * frob(x))

    corrs = vmap(
        # Vmap over evals and reach conditions (activities only)
        vmap(vmap(corr, in_axes=(0, None)), in_axes=(0, None)), 
        # Vmap over replicates (appears in both activities and weights)
        in_axes=(1, 0),
    )(activities, weights)
    
    # Return the replicate axis to the same position as in `activities`
    return jnp.moveaxis(corrs, 0, 1)


class Measures(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        aligned_vars=AlignedVars,
    ))
    measure_keys: Sequence[str]
    variant: Optional[str] = None
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
        all_measures: LDict[str, Measure] = subdict(MEASURES, self.measure_keys)  # type: ignore
        all_measure_values = compute_all_measures(all_measures, aligned_vars.get(self.variant, aligned_vars))
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
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()

    def make_figs(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        measure_values,
        colors_0,
        **kwargs,
    ):
        figs = get_violins_per_measure(
            measure_values[self.variant],
            colors=colors_0[self.variant]['pert_amp']['dark'],
        )
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape))
        )


def get_one_measure_plot_per_eval_condition(plot_func, measures, colors, **kwargs):
    return {
        key: LDict.of("pert__amp")({
            pert_amp: plot_func(
                measure[pert_amp],
                MEASURE_LABELS[key],
                colors,
                **kwargs,
            )
            for pert_amp in measure
        })
        for key, measure in measures.items()
    }


class MeasuresLoHiPertStd(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values=Measures,
    ))
    measure_keys: tuple[str, ...]
    variant: Optional[str] = None
    conditions: tuple[str, ...] = ()
        
    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        return dict(
            measure_values=dict(measure_keys=self.measure_keys)
        )

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
            lambda measure_values_by_std: LDict.of("train__pert__std")({
                std: measure_values
                for std, measure_values in measure_values_by_std.items()
                if std in (min(measure_values_by_std), max(measure_values_by_std))
            }),
            measure_values,
            is_leaf=LDict.is_of("train__pert__std"),
        )


class Measures_CompareReplicatesLoHi(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values_lohi_train_pert_std=MeasuresLoHiPertStd,
    ))
    measure_keys: tuple[str, ...]
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        return dict(
            measure_values_lohi_train_pert_std=dict(
                measure_keys=self.measure_keys
            )
        )

    def make_figs(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        measure_values_lohi_train_pert_std,
        colors_0,
        replicate_info,
        **kwargs,
    ):
        included_replicates = replicate_info['included_replicates'][REPLICATE_CRITERION]
        replicates_all_lohi_included = jt.reduce(jnp.logical_and, lohi(included_replicates))
        figs = get_one_measure_plot_per_eval_condition(
            get_measure_replicate_comparisons,
            measure_values_lohi_train_pert_std,
            lohi(colors_0[self.variant]["train__pert__std"]["dark"]),
            included_replicates=np.where(replicates_all_lohi_included)[0],
        )
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, measure_values_lohi_train_pert_std, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(measure_values_lohi_train_pert_std)[0].shape))
        )


class Measures_LoHiSummary(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        measure_values_lohi_train_pert_std=MeasuresLoHiPertStd,
    ))
    measure_keys: tuple[str, ...]
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    def dependency_kwargs(self) -> Dict[str, Dict[str, Any]]:
        return dict(
            measure_values_lohi_train_pert_std=dict(
                measure_keys=self.measure_keys,
            )
        )

    def compute(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        measure_values_lohi_train_pert_std,
        **kwargs,
    ):

        return LDict.of("measure")(**{
            key: subdict(measure, lohi(hps[self.variant].pert.amp))  # type: ignore
            # MeasuresLoHiPertStd returns `measure_values_lohi_train_pert_std` for all eval variants,
            # so we choose the right variant
            for key, measure in measure_values_lohi_train_pert_std[self.variant].items()
        })

    def make_figs(
        self,
        models: PyTree[Module],
        tasks: PyTree[Module],
        states: PyTree[Module],
        hps: PyTree[TreeNamespace],
        *,
        result,
        colors_0,
        **kwargs,
    ):
        figs = LDict.of("measure")({
            key: get_violins(
                measure,
                yaxis_title=MEASURE_LABELS[key],
                xaxis_title="Train field std.",
                legend_title="TODO",
                colors=colors_0[self.variant]['pert_amp']['dark'],
                layout_kws=dict(
                    width=300, height=300,
                )
            )
            for key, measure in result.items ()
        })
        return figs

    def _params_to_save(self, hps: PyTree[TreeNamespace], *, result, **kwargs):
        return dict(
            n=int(np.prod(jt.leaves(result)[0].shape))
        )