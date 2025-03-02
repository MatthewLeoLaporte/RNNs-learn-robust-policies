from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional, Literal as L

import jax.numpy as jnp
import jax.tree as jt
import equinox as eqx

from feedbax.bodies import SimpleFeedbackState
from feedbax.intervene import ConstantInput,  NetworkConstantInput, TimeSeriesParam, schedule_intervenor
from feedbax.task import TrialSpecDependency
from jax_cookbook import is_module
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.state_utils import angle_between_vectors, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.analysis.state_utils import get_constant_task_input
from rnns_learn_robust_motor_policies.types import LDict


COLOR_FUNCS = dict(
    context_input=lambda hps: hps.context_input,
)


def setup_eval_tasks_and_models(task_base, models_base, hps):
    # 1. Tasks are steady-state 
    # 2. `models_base` is a `disturbance_std` dict
    # 3. Two types of tasks (plant vs. unit stim)
    
    # Add the context input to the task dependencies, so that it is provided to the neural network
    #! Would be unnecessary if we used the task pytree constructed by the part2 setup function, 
    #! however we're using `setup_models_only` in `load_models`.
    # task_base = eqx.tree_at(
    #     # TODO: I think we can remove this since we just tree_at `input_dependencies` again, immediately
    #     lambda task: task.input_dependencies,
    #     task_base,
    #     dict(context=TrialSpecDependency(CONTEXT_INPUT_FUNCS[hps.load.train.method]))
    # )
    
    task_by_context = LDict.of("context_input")({
        context_input: eqx.tree_at(
            lambda task: task.input_dependencies,
            task_base, 
            {
                'context': TrialSpecDependency(get_constant_task_input(
                    context_input, 
                    hps.model.n_steps - 1, 
                    task_base.n_validation_trials,
                ))
            },
        )
        for context_input in hps.context_input
    })
    
    all_tasks, all_models = jtree.unzip({
        part_label: jt.map(
            lambda task: eqx.filter_vmap(
                partial(setup_part_func, task_base=task, models_base=models_base, hps=hps),
            )(jnp.arange(n)),
            task_by_context,
            is_leaf=is_module,
        )
        for part_label, (setup_part_func, n)  in {
            'plant_pert': (setup_ss_plant_pert_task, hps.disturbance.plant.directions),
            'unit_stim': (setup_ss_unit_stim_task, hps.load.model.hidden_size),
        }.items()
    })
    
    # #! TODO: I'm not sure hps should be the same in all cases. e.g. we should probably update 
    # #! `hps.disturbance.amplitude` to contain the respective value.
    # #! If so, it will be necessary to modify the `setup_part_func`s and return `all_hps` 
    # #! along with the main unzip, above.
    # hps_by_context = ContextInputDict.fromkeys(hps.context_input, hps)
    # all_hps = {'plant_pert': hps_by_context, 'unit_stim': hps_by_context}
    all_hps = jt.map(lambda _: hps, all_tasks, is_leaf=is_module)

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
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    # TODO: Separate into two `AbstractAnalysis` classes in serial
    def compute(self, models, tasks, states, hps, **dependencies):
        # 1. Get activities of all units at the time step of max forward force 
        def get_activity_at_max_force(state: SimpleFeedbackState):
            net_force = jnp.linalg.norm(state.efferent.output, axis=-1)  
            t_max_net_force = jnp.argmax(net_force, axis=-1)
            activity_max_force = state.net.hidden[
                *jnp.indices(t_max_net_force.shape), 
                t_max_net_force,
            ]
            # Remove the "condition" dimension, which is singleton in this module 
            return jnp.squeeze(activity_max_force)
        
        # Each array: (hps.disturbance.plant.directions, eval_n, n_replicates, n_units)
        activity_at_max_force = jt.map(
            lambda state: get_activity_at_max_force(state), 
            states['full']['plant_pert'], 
            is_leaf=is_module,
        )
        
        #! Collapse (eval_n, n_replicates, conditions) axes; for now, don't worry about non-aggregate statistics)
        #! Actually, this might be wrong; shouldn't we process `n_replicates` in parallel until the end? Otherwise 
        #! we assume that the preferred directions are similar across replicates, indexed by unit, which is clearly wrong.
        # Each array: (hps.disturbance.plant.directions, samples=(eval_n * n_replicates), n_units)
        # activities_at_max_accel = jt.map(
        #     lambda activity: jnp.reshape(activity, (activity.shape[0], -1, activity.shape[-1])),
        #     activities_at_max_accel,
        #     is_leaf=is_module,
        # )
        
        # 3. Compute instantaneous preference distribution/mode for each unit
        # Each array: (eval_n, n_replicates, conditions=1, n_units)
        # i.e. the index of the direction 
        preferred_directions = jt.map(
            lambda activities: jnp.argmax(activities, axis=0),
            activity_at_max_force,
            is_leaf=is_module,
        )
        
        # TODO: Convert from direction idxs to direction vectors here, instead of in `UnitPreferenceAlignment`?
        return dict(
            preferred_direction=preferred_directions,
            activity_at_max_force=activity_at_max_force,
        )
    
    def make_figs(self, models, tasks, states, hps, *, result, **dependencies):
        # Plot the distribution of preferred directions across units, for each context input
        # (Should remain uniform? However maybe the tuning curves get narrower.)
        preferred_directions = result['preferred_directions']
        activities_at_max_accel = result['activities_at_max_accel']
    

class UnitStimDirections(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType({})
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    def compute(self, models, tasks, states, hps, **dependencies):
        # Compute the direction of maximum force for each perturbed unit
        def get_angle_of_max_force(state: SimpleFeedbackState):
            net_force = jnp.linalg.norm(state.efferent.output, axis=-1)
            # TODO: Check the distribution of the time indices 
            t_max_net_force = jnp.argmax(net_force, axis=-1)
            forces_at_max_net_force = state.efferent.output[
                *jnp.indices(t_max_net_force.shape), 
                t_max_net_force,
            ]
            angle_of_max_net_force = jt.map(
                lambda forces: jnp.arctan2(forces[..., 1], forces[..., 0]),
                forces_at_max_net_force,
            )

            # Remove the "condition" dimension, which is singleton in this module 
            return jnp.squeeze(angle_of_max_net_force)
        
        # Each array: (hps.disturbance.plant.directions, eval_n, n_replicates, n_units)
        angle_of_max_force = jt.map(
            lambda state: get_angle_of_max_force(state), 
            states[self.variant]['unit_stim'], 
            is_leaf=is_module,
        )
        
        return dict(
            angle_of_max_force=angle_of_max_force,
        )
    
    def make_figs(self, models, tasks, states, hps, **dependencies):
        # Plot the distribution of stim directions, for each condition
        ...


def angle_to_direction(angle):
    return jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)


class UnitPreferenceAlignment(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        results1=UnitPreferredDirections,
        results2=UnitStimDirections,
    ))
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()

    def compute(self, models, tasks, states, hps, *, results1, results2, **dependencies):
        """Compute the alignment between preferred and stim directions"""
        unit_preferred_direction_idx = results1['preferred_direction']
        # activity_at_max_force = results1['activity_at_max_force']
        angle_of_max_force_on_unit_stim = results2['angle_of_max_force']
        
        # Either 
        # 1. Convert preferred direction from index to angle, and take the difference between the angles (works for all signs?)
        # 2. Convert both to vectors and use `angle_between_vectors` (trust this more)
        def get_angle_between_prefs_and_forces(unit_preferred_direction_idx, angle_of_max_force_on_unit_stim):
            n_directions = hps[self.variant]['plant_pert'][0].disturbance.plant.directions
            unit_preferred_angle = 2 * jnp.pi * unit_preferred_direction_idx / n_directions
            unit_preferred_direction = angle_to_direction(unit_preferred_angle)
            direction_of_max_force_on_unit_stim = angle_to_direction(angle_of_max_force_on_unit_stim)
            return angle_between_vectors(
                jnp.moveaxis(unit_preferred_direction, -2, 0), 
                direction_of_max_force_on_unit_stim,
            )
            
        angle_between_pref_and_stim = jt.map(
            lambda idx, angle: get_angle_between_prefs_and_forces(idx, angle),
            unit_preferred_direction_idx,
            angle_of_max_force_on_unit_stim,
        )
        
        mean_angle_between_pref_and_stim = jt.map(
            lambda angle: jnp.mean(angle, axis=0),
            angle_between_pref_and_stim,
        )
        
        std_angle_between_pref_and_stim = jt.map(
            lambda angle: jnp.std(angle, axis=0),
            angle_between_pref_and_stim,
        )
        
        return dict(
            angle_between_pref_and_stim=angle_between_pref_and_stim,
            mean_angle_between_pref_and_stim=mean_angle_between_pref_and_stim,
            std_angle_between_pref_and_stim=std_angle_between_pref_and_stim,
        )

    
    def make_figs(self, models, tasks, states, hps, *, result, **dependencies):
        # 1. Distribution of (absolute?) angles between pref and stim, versus context input
        ...
        

class AllResults(AbstractAnalysis):
    """Collect all the results for this analysis in one place, for interactive reasons."""
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        results1=UnitPreferredDirections,
        results2=UnitStimDirections,
        results3=UnitPreferenceAlignment,
    ))
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()

    def compute(
        self, 
        models, 
        tasks, 
        states, 
        hps, 
        *, 
        results1,
        results2, 
        results3,  
        **dependencies,
    ):
        updi = results1['preferred_direction']
        aatmf = results1['activity_at_max_force']
        aomfous = results2['angle_of_max_force']
        abpas = results3['angle_between_pref_and_stim']
        mabpas = results3['mean_angle_between_pref_and_stim']
        
        return dict(
            preferred_direction=updi,
            activity_at_max_force=aatmf,
            angle_of_max_force_on_unit_stim=aomfous,
            angle_between_pref_and_stim=abpas,
            mean_angle_between_pref_and_stim=mabpas,
        )


ALL_ANALYSES = [
    UnitPreferenceAlignment(),
    AllResults(),
]