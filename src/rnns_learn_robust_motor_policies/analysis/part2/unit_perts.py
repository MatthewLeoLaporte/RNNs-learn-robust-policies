
from types import MappingProxyType
from typing import ClassVar, Optional

import jax.numpy as jnp
import jax.tree as jt
import equinox as eqx

from feedbax.intervene import ConstantInput,  NetworkConstantInput, TimeSeriesParam, schedule_intervenor
from jax_cookbook import is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
# from rnns_learn_robust_motor_policies.perturbations import random_unit_vector
from rnns_learn_robust_motor_policies.types import PertVarDict



def setup_eval_tasks_and_models(task_base, models_base, hps):
    # 1. Tasks are steady-state 
    # 2. `models_base` is a `TrainStdDict`
    # 3. Two types of tasks (plant vs. unit stim)
    
    all_tasks, all_models = {}, {}
    
    #! TODO: `task.eval_n_directions = 1`, and we need to (v)map over stim directions  
    all_tasks['plant_pert'], all_models['plant_pert'] = setup_ss_plant_pert_task(task_base, models_base, hps)
    all_tasks['unit_stim'], all_models['unit_stim'] = setup_ss_unit_stim_task(task_base, models_base, hps)
    
    return all_tasks, all_models, hps


def force_impulse(
    direction_idx,
    hps,
):
    idxs = slice(
        hps.disturbance.plant.start_step, 
        hps.disturbance.plant.start_step + hps.disturbance.plant.duration,
    )
    trial_mask = jnp.zeros((hps.load.model.n_steps - 1,), bool).at[idxs].set(True)
    
    angle = 2 * jnp.pi * direction_idx / hps.disturbance.plant.directions
    array = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    
    return ConstantInput.with_params(
        out_where=lambda channel_state: channel_state.output,
        scale=hps.disturbance.plant.amplitude,
        arrays=array,
        active=TimeSeriesParam(trial_mask),
    )


def activity_impulse(
    unit_idx,
    hps,
):
    idxs = slice(
        hps.disturbance.unit.start_step, 
        hps.disturbance.unit.start_step + hps.disturbance.unit.duration,
    )
    trial_mask = jnp.zeros((hps.load.model.n_steps - 1,), bool).at[idxs].set(True)
    
    unit_spec = jnp.full(hps.load.model.hidden_size, jnp.nan)
    unit_spec = unit_spec.at[unit_idx].set(hps.disturbance.unit.amplitude)
    
    return NetworkConstantInput.with_params(
        # out_where=lambda state: state.hidden,
        unit_spec=unit_spec,
        active=TimeSeriesParam(trial_mask),
    )


def setup_ss_plant_pert_task(task_base, models_base, hps):
    tasks, models = jtree.unzip([
            schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.efferent_channel,  
            force_impulse(direction_idx, hps),
            default_active=False,
            stage_name="update_queue",
        )
        for direction_idx in range(hps.disturbance.plant.directions)
    ])
    return tasks, models
    
    
def setup_ss_unit_stim_task(task_base, models_base, hps):
    tasks, models = jtree.unzip([
        schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.net,  
            activity_impulse(unit_idx, hps),
            default_active=False,
            stage_name=None,  # None -> before RNN forward pass; 'hidden' -> after 
        )
        for unit_idx in range(hps.load.model.hidden_size)
    ])
    return tasks, models


eval_func = vmap_eval_ensemble


class UnitPreferredDirections(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType({})
    variant: ClassVar[Optional[str]] = "full"
    conditions: tuple[str, ...] = ()
    
    def compute(self, models, tasks, states, hps, **dependencies):
        # 1. Get activities of all units at the time step of max forward force (accel)
        # 2. Compute instantaneous preference distribution/mode for each unit
        ...
    
    def make_figs(self, models, tasks, states, hps, **dependencies):
        # Plot the distribution of preferred directions, for each condition
        ... 
    

class UnitStimDirections(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType({})
    variant: ClassVar[Optional[str]] = "full"
    conditions: tuple[str, ...] = ()
    
    def compute(self, models, tasks, states, hps, **dependencies):
        # Compute the direction of maximum acceleration for each perturbed unit
        ...
    
    def make_figs(self, models, tasks, states, hps, **dependencies):
        # Plot the distribution of stim directions, for each condition
        ...


class UnitPreferenceAlignment(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        unit_preferred_directions=UnitPreferredDirections,
        unit_stim_directions=UnitStimDirections,
    ))
    variant: ClassVar[Optional[str]] = None
    conditions: tuple[str, ...] = ()

    def compute(self, models, tasks, states, hps, *, unit_preferred_directions, unit_stim_directions, **dependencies):
        # Compute the alignment between preferred and stim directions
        ...
    
    def make_figs(self, models, tasks, states, hps, **dependencies):
        # 1. Plot them together
        # 2. ...
        ...


ALL_ANALYSES = [
    UnitPreferenceAlignment,
]