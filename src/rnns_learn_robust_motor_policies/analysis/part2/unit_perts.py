
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import jax.numpy as jnp
import jax.tree as jt
import equinox as eqx

from feedbax.intervene import ConstantInput,  NetworkConstantInput, TimeSeriesParam, schedule_intervenor
from feedbax.task import TrialSpecDependency
from jax_cookbook import is_type
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
# from rnns_learn_robust_motor_policies.perturbations import random_unit_vector
from rnns_learn_robust_motor_policies.constants import INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.training.part2_context import CONTEXT_INPUT_FUNCS
from rnns_learn_robust_motor_policies.types import PertVarDict


COLOR_FUNCS = dict()


# def setup_eval_tasks_and_models(task_base, models_base, hps):
#     # 1. Tasks are steady-state 
#     # 2. `models_base` is a `TrainStdDict`
#     # 3. Two types of tasks (plant vs. unit stim)
    
#     all_tasks, all_models = {}, {}
    
#     all_tasks['plant_pert'], all_models['plant_pert'] = jtree.unzip(eqx.filter_vmap(
#         partial(setup_ss_plant_pert_task, task_base=task_base, models_base=models_base, hps=hps),
#     )(jnp.arange(hps.disturbance.plant.directions)))
    
#     all_tasks['unit_stim'], all_models['unit_stim'] = jtree.unzip(eqx.filter_vmap(
#         partial(setup_ss_unit_stim_task, task_base=task_base, models_base=models_base, hps=hps),
#     )(jnp.arange(hps.load.model.hidden_size)))

#     all_hps = {'plant_pert': hps, 'unit_stim': hps}

#     return all_tasks, all_models, all_hps

def setup_eval_tasks_and_models(task_base, models_base, hps):
    # 1. Tasks are steady-state 
    # 2. `models_base` is a `TrainStdDict`
    # 3. Two types of tasks (plant vs. unit stim)
    
    # TODO: Evaluate over different context inputs
    # all_tasks = ContextInputDict({
    #     context_input: eqx.tree_at(
    #         lambda task: task.input_dependencies,
    #         task, 
    #         {
    #             'context': TrialSpecDependency(get_context_input_func(
    #                 context_input, model_info_0.n_steps, task.n_validation_trials
    #             ))
    #         },
    #     )
    #     for context_input in context_inputs
    # })
    
    # Add the context input to the task dependencies, so that it is provided to the neural network
    #! Would be unnecessary if we used the task pytree constructed by the part2 setup function, 
    #! however we're using `setup_models_only` in `load_models`.
    task_base = eqx.tree_at(
        lambda task: task.input_dependencies,
        task_base,
        dict(context=TrialSpecDependency(CONTEXT_INPUT_FUNCS[hps.load.train.method]))
    )
    
    all_tasks, all_models = jtree.unzip({
        part_label: eqx.filter_vmap(
            partial(setup_part_func, task_base=task_base, models_base=models_base, hps=hps),
        )(jnp.arange(n))
        for part_label, (setup_part_func, n)  in {
            'plant_pert': (setup_ss_plant_pert_task, hps.disturbance.plant.directions),
            # 'unit_stim': (setup_ss_unit_stim_task, hps.load.model.hidden_size),
        }.items()
    })
    
    #! TODO: I'm not sure these should be the same in both cases. e.g. we should probably update 
    #! `hps.disturbance.amplitude` to contain the respective value.
    #! If so, it will be necessary to modify the `setup_part_func`s and return `all_hps` 
    #! along with the main unzip, above.
    all_hps = {'plant_pert': hps}#, 'unit_stim': hps}

    return all_tasks, all_models, all_hps


def force_impulse(direction_idx, *, hps):
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


def activity_impulse(unit_idx, *, hps):
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


def setup_ss_plant_pert_task(direction_idx, *, task_base, models_base, hps):
    pairs = schedule_intervenor(
        task_base, models_base,
        lambda model: model.step.efferent_channel,  
        force_impulse(direction_idx, hps=hps),
        default_active=False,
        stage_name="update_queue",
        label="PlantPert", 
    )
    return pairs
    
    
def setup_ss_unit_stim_task(unit_idx, *, task_base, models_base, hps):
    return schedule_intervenor(
        task_base, models_base,
        lambda model: model.step.net,  
        activity_impulse(unit_idx, hps=hps),
        default_active=False,
        stage_name=None,  # None -> before RNN forward pass; 'hidden' -> after 
        label="UnitStim", 
    )
    

def eval_func(models, task, hps, key):
    """Vmap over directions or units, depending on task."""
    return eqx.filter_vmap(
        partial(vmap_eval_ensemble, key, hps),
    )(models, task)


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