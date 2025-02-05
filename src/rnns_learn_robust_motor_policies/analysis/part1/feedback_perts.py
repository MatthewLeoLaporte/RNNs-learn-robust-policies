
import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt 

from feedbax.intervene import schedule_intervenor
from jax_cookbook import is_type, is_module
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
from rnns_learn_robust_motor_policies.types import PertVarDict
from rnns_learn_robust_motor_policies.perturbations import feedback_impulse


ALL_ANALYSES = []

PERT_VAR_NAMES = ('pos', 'vel')
COORD_NAMES = ('x', 'y')


def _setup_rand(task_base, models_base, hps):
    """Impulses in random directions, i.e. uniform angles about the effector."""
    all_tasks, all_models = jtree.unzip(jt.map(
        lambda feedback_var_idx: schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.feedback_channels[0],  # type: ignore
            feedback_impulse(  
                hps.model.n_steps,
                1.0, #impulse_amplitude[pert_var_names[feedback_var_idx]],
                hps.disturbance.duration,
                feedback_var_idx,   
                hps.disturbance.start_step,
            ),
            default_active=False,
            stage_name="update_queue",
        ),
        PertVarDict(pos=0, vel=1),
        is_leaf=is_type(tuple),
    ))

    # Get the perturbation directions, for later:
    #? I think these values are equivalent to `line_vec` in the functions in `state_utils`
    impulse_directions = jt.map(
        lambda task: task.validation_trials.intervene['ConstantInput'].arrays[:, hps.disturbance.start_step],
        all_tasks,
        is_leaf=is_module,
    )
    return all_tasks, all_models, impulse_directions


def _setup_xy(task_base, models_base, hps):
    """Impulses only in the x and y directions."""
    feedback_var_idxs = PertVarDict(zip(PERT_VAR_NAMES, range(len(PERT_VAR_NAMES))))
    coord_idxs = dict(zip(COORD_NAMES, range(len(COORD_NAMES))))
    
    impulse_xy_conditions = PertVarDict.fromkeys(PERT_VAR_NAMES, dict.fromkeys(COORD_NAMES))
    impulse_xy_conditions_keys = jtree.key_tuples(
        impulse_xy_conditions, keys_to_strs=True, is_leaf=lambda x: x is None,
    )

    all_tasks, all_models = jtree.unzip(jt.map(
        lambda ks: schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.feedback_channels[0],  # type: ignore
            feedback_impulse(
                hps.model.n_steps,
                1.0, # impulse_amplitude[ks[0]],
                hps.disturbance.duration,
                feedback_var_idxs[ks[0]],  
                hps.disturbance.start_step,
                feedback_dim=coord_idxs[ks[1]],  
            ),
            default_active=False,
            stage_name="update_queue",
        ),
        impulse_xy_conditions_keys,
        is_leaf=is_type(tuple),
    ))

    impulse_directions = jt.map(
        lambda task, ks: jnp.zeros(
            (task.n_validation_trials, 2)
        # ).at[:, coord_idxs[ks[1]]].set(copysign(1, impulse_amplitude[ks[0]])),
        # Assume x-y impulses are in the positive direction.
        ).at[:, coord_idxs[ks[1]]].set(1),
        all_tasks, impulse_xy_conditions_keys,
        is_leaf=is_module,
    )
    
    return all_tasks, all_models, impulse_directions


SETUP_FUNCS_BY_DIRECTION = dict(
    rand=_setup_rand,
    xy=_setup_xy,
)


def setup_tasks_and_models(task_base, models_base, hps):
    impulse_amplitudes = jt.map(
        lambda max_amp: jnp.linspace(0, max_amp, hps.disturbance.n_amplitudes + 1)[1:],
        hps.disturbance.amplitude_max,
    )
    hps.disturbance.amplitude = impulse_amplitudes

    impulse_end_step = hps.disturbance.start_step + hps.disturbance.duration
    # TODO: Move extra info to another function? Or return it here.
    impulse_time_idxs = slice(hps.disturbance.start_step, impulse_end_step)

    # For the example trajectories and aligned profiles, we'll only plot one of the impulse amplitudes. 

    i_impulse_amp_plot = -1  # The largest amplitude perturbation
    impulse_amplitude_plot = {
        pert_var: v[i_impulse_amp_plot] for pert_var, v in impulse_amplitudes.items()
    }

    all_tasks, all_models, impulse_directions = SETUP_FUNCS_BY_DIRECTION[hps.disturbance.direction](
        task_base, models_base, hps
    )
    
    return all_tasks, all_models, hps


def task_with_imp_amplitude(task, impulse_amplitude):
    """Returns a task with the given disturbance amplitude."""
    return eqx.tree_at(
        lambda task: task.intervention_specs.validation['ConstantInput'].intervenor.params.scale,
        task,
        impulse_amplitude,
    ) 


def eval_func(models, task, hps, key_eval):
    """Vmap over impulse amplitude."""
    return eqx.filter_vmap(
        lambda amplitude: vmap_eval_ensemble(
            models, 
            task_with_imp_amplitude(task, amplitude), 
            hps,
            key_eval,
        )
    )(hps.disturbance.amplitudes)